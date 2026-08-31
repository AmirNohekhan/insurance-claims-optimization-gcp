from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from insurance_claims_platform.cloud import LocalEventPublisher, OperationalEvent
from insurance_claims_platform.features import build_features
from insurance_claims_platform.forecasting import rolling_origins
from insurance_claims_platform.optimization import optimize_workforce
from insurance_claims_platform.pipeline import build_demo_state
from insurance_claims_platform.scenarios import apply_scenario
from insurance_claims_platform.serving.app import app, state
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


def test_local_publisher_is_idempotent(tmp_path: Path):
    pub = LocalEventPublisher(tmp_path / "events.jsonl")
    event = OperationalEvent("claim.reported", "FL", {"claim_count": 1}, event_id="fixed")
    assert pub.publish(event) == pub.publish(event)
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 1


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
