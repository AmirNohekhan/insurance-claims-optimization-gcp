from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from insurance_claims_platform.cloud import LocalEventPublisher, OperationalEvent
from insurance_claims_platform.config import CostConfig
from insurance_claims_platform.features import build_features
from insurance_claims_platform.forecasting import rolling_origins
from insurance_claims_platform.monitoring import drift_summary
from insurance_claims_platform.optimization import optimize_workforce, realized_cost
from insurance_claims_platform.pipeline import build_demo_state
from insurance_claims_platform.scenarios import apply_scenario
from insurance_claims_platform.serving.app import _get_result, _store_result, app, state
from insurance_claims_platform.simulation import generate_portfolio, generate_weekly_claims


def test_simulation_reproducible_and_catastrophe_sensitive():
    p = generate_portfolio()
    a = generate_weekly_claims(p, periods=130, seed=3)
    b = generate_weekly_claims(p, periods=130, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert (
        a.loc[a.catastrophe == 1, "new_claim_count"].mean()
        > a.loc[a.catastrophe == 0, "new_claim_count"].mean()
    )
    assert (a.new_claim_count >= 0).all() and a.new_claim_count.var() > a.new_claim_count.mean()


def test_features_are_past_only():
    claims = generate_weekly_claims(generate_portfolio(), periods=70)
    feat = build_features(claims)
    market = feat[feat.market_id == "FL"].reset_index(drop=True)
    assert market.loc[1, "lag_1"] == market.loc[0, "new_claim_count"]
    changed = claims.copy()
    changed.loc[changed.index[-1], "new_claim_count"] = 999999
    before = feat.iloc[:-1].filter(regex="lag|rolling")
    after = build_features(changed).iloc[:-1].filter(regex="lag|rolling")
    pd.testing.assert_frame_equal(before, after)


def test_temporal_splits_and_quantile_order():
    state = build_demo_state(periods=150)
    for origin, train, test in rolling_origins(state["features"], origins=2):
        assert train.week_start.max() <= origin < test.week_start.min()
    q = state["forecasts"][["p10", "p50", "p75", "p90"]].to_numpy()
    assert np.all(np.diff(q, axis=1) >= 0)


def test_optimizer_integral_conserves_staff_and_scenario():
    state = build_demo_state(periods=150)
    plan = optimize_workforce(state["forecasts"], state["workforce"], state["backlog"], 0.75)
    assert (
        sum(x["recommended_adjusters"] for x in plan["markets"])
        == state["workforce"].current_adjusters.sum()
    )
    assert all(isinstance(x["recommended_adjusters"], int) for x in plan["markets"])
    cat = apply_scenario(state["forecasts"], catastrophe_market="FL")
    assert (
        cat.loc[cat.market_id == "FL", "p90"].sum()
        > state["forecasts"].loc[state["forecasts"].market_id == "FL", "p90"].sum()
    )


def test_optimizer_caps_overtime_by_each_markets_workforce():
    forecasts = pd.DataFrame(
        {
            "market_id": ["SMALL", "LARGE"],
            "p75": [200.0, 1.0],
            "workload_ratio": [1.0, 1.0],
        }
    )
    workforce = pd.DataFrame(
        {
            "market_id": ["SMALL", "LARGE"],
            "current_adjusters": [5, 20],
            "minimum_adjusters": [5, 20],
        }
    )
    plan = optimize_workforce(
        forecasts, workforce, quantile=0.75, allow_reassignment=False
    )
    small = next(item for item in plan["markets"] if item["market_id"] == "SMALL")
    assert small["overtime_hours"] <= 5 * 10


def test_optimizer_includes_transfer_cost_in_its_decision():
    forecasts = pd.DataFrame(
        {
            "market_id": ["A", "B"],
            "p75": [320.0, 0.0],
            "workload_ratio": [1.0, 1.0],
        }
    )
    workforce = pd.DataFrame(
        {
            "market_id": ["A", "B"],
            "current_adjusters": [10, 10],
            "minimum_adjusters": [5, 5],
        }
    )
    plan = optimize_workforce(
        forecasts,
        workforce,
        quantile=0.75,
        allow_overtime=False,
        costs=CostConfig(transfer_adjuster=1_000_000),
    )
    assert all(item["recommended_adjusters"] == 10 for item in plan["markets"])
    assert plan["transfer_cost"] == 0


def test_realized_cost_includes_opening_backlog():
    plan = {
        "transfer_cost": 0.0,
        "markets": [
            {
                "market_id": "A",
                "recommended_adjusters": 1,
                "overtime_hours": 0.0,
                "opening_backlog": 20.0,
            }
        ],
    }
    result = realized_cost(plan, pd.Series({"A": 0.0}))
    assert result["unserved_workload"] == 4.0
    assert result["service_level"] == 0.8


def test_local_publisher_is_idempotent(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    pub = LocalEventPublisher(path)
    event = OperationalEvent("claim.reported", "FL", {"claim_count": 1}, event_id="fixed")
    assert pub.publish(event) == pub.publish(event)
    assert LocalEventPublisher(path).publish(event) == "fixed"
    assert len(path.read_text().splitlines()) == 1


def test_local_publisher_rejects_a_corrupt_event_log(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    path.write_text("not-json\n", encoding="utf-8")
    with np.testing.assert_raises(ValueError):
        LocalEventPublisher(path)


def test_api_contract():
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    response = client.post(
        "/v1/forecast",
        json={"markets": ["FL", "TX"], "horizon_weeks": 2, "quantiles": [0.5, 0.75, 0.9]},
    )
    assert response.status_code == 200
    fid = response.json()["forecast_id"]
    plan = client.post(
        "/v1/optimize",
        json={
            "forecast_id": fid,
            "planning_quantile": 0.75,
            "allow_overtime": True,
            "allow_reassignment": True,
        },
    )
    assert plan.status_code == 200 and plan.json()["markets"]


def test_forecast_returns_only_requested_quantiles():
    client = TestClient(app)
    response = client.post(
        "/v1/forecast",
        json={"markets": ["FL"], "horizon_weeks": 1, "quantiles": [0.5, 0.9, 0.5]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["quantiles"] == [0.5, 0.9]
    assert {key for key in body["records"][0] if key.startswith("p")} == {"p50", "p90"}


def test_forecast_rejects_empty_quantiles():
    client = TestClient(app)
    response = client.post(
        "/v1/forecast", json={"markets": ["FL"], "horizon_weeks": 1, "quantiles": []}
    )
    assert response.status_code == 422


def test_optimize_requires_a_requested_supported_quantile():
    client = TestClient(app)
    forecast = client.post(
        "/v1/forecast",
        json={"markets": ["FL"], "horizon_weeks": 1, "quantiles": [0.5]},
    ).json()
    missing = client.post(
        "/v1/optimize", json={"forecast_id": forecast["forecast_id"], "planning_quantile": 0.9}
    )
    assert missing.status_code == 422
    assert missing.json()["detail"]["code"] == "QUANTILE_NOT_FORECAST"
    unsupported = client.post(
        "/v1/optimize", json={"forecast_id": forecast["forecast_id"], "planning_quantile": 0.8}
    )
    assert unsupported.status_code == 422


def test_scenario_supports_a_single_requested_quantile():
    client = TestClient(app)
    forecast = client.post(
        "/v1/forecast",
        json={"markets": ["FL"], "horizon_weeks": 1, "quantiles": [0.5]},
    ).json()
    response = client.post(
        "/v1/scenarios",
        json={
            "forecast_id": forecast["forecast_id"],
            "planning_quantile": 0.5,
            "catastrophe_market": "FL",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["plan"]["planning_quantile"] == 0.5
    decision = body["plan"]["markets"][0]
    assert decision["opening_backlog"] == round(float(state()["backlog"].loc["FL"]), 2)


def test_scenario_rejects_a_market_outside_the_forecast():
    client = TestClient(app)
    forecast = client.post(
        "/v1/forecast",
        json={"markets": ["FL"], "horizon_weeks": 1, "quantiles": [0.9]},
    ).json()
    response = client.post(
        "/v1/scenarios",
        json={"forecast_id": forecast["forecast_id"], "catastrophe_market": "TX"},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "CATASTROPHE_MARKET_NOT_FORECAST"


def test_result_store_evicts_least_recently_used_items():
    store: OrderedDict[str, dict] = OrderedDict()
    _store_result(store, "old", {"value": 1}, max_items=2)
    _store_result(store, "kept", {"value": 2}, max_items=2)
    assert _get_result(store, "old") == {"value": 1}
    _store_result(store, "new", {"value": 3}, max_items=2)
    assert list(store) == ["old", "new"]
    assert _get_result(store, "kept") is None


def test_drift_summary_counts_values_outside_reference_range():
    reference = np.arange(100, dtype=float)
    shifted = np.arange(1_000, 1_100, dtype=float)
    result = drift_summary(reference, shifted)
    assert result["psi"] > 1
    assert result["ks"] == 1
    assert result["wasserstein"] > 900


def test_drift_summary_rejects_invalid_samples():
    invalid_samples = (
        (np.array([]), np.array([1])),
        (np.array([1]), np.array([np.nan])),
    )
    for reference, current in invalid_samples:
        with np.testing.assert_raises(ValueError):
            drift_summary(reference, current)
    with np.testing.assert_raises(ValueError):
        drift_summary(np.array([1]), np.array([1]), bins=1)
