from __future__ import annotations

import json

from insurance_claims_platform.backtesting import decision_backtest
from insurance_claims_platform.forecasting import (
    baseline_predictions,
    evaluate_probabilistic,
    forecast_metrics,
)
from insurance_claims_platform.hierarchy import bottom_up
from insurance_claims_platform.optimization import optimize_workforce
from insurance_claims_platform.pipeline import build_demo_state
from insurance_claims_platform.scenarios import apply_scenario


def main() -> None:
    state = build_demo_state()
    future = state["future"]
    forecasts = state["forecasts"]
    actual = future.new_claim_count.to_numpy()
    pred = forecasts.sort_values(["target_week", "market_id"])
    seasonal_scale = float((future.new_claim_count - future.lag_52).abs().mean())
    metrics = forecast_metrics(actual, pred.p50.to_numpy(), seasonal_scale)
    baseline = forecast_metrics(
        actual, baseline_predictions(future, "seasonal_naive"), seasonal_scale
    )
    probabilistic = evaluate_probabilistic(actual, pred)
    region, national = bottom_up(forecasts)
    plans = {
        f"P{int(q * 100)}": optimize_workforce(forecasts, state["workforce"], state["backlog"], q)
        for q in (0.5, 0.75, 0.9)
    }
    realized = future.groupby("market_id").workload_units.mean()
    backtest = decision_backtest(forecasts, state["workforce"], realized)
    hurricane = apply_scenario(forecasts, catastrophe_market="FL")
    hurricane_plan = optimize_workforce(hurricane, state["workforce"], state["backlog"], 0.9)
    print("\nINSURANCE CLAIMS FORECAST -> UNCERTAINTY -> WORKFORCE DECISION")
    print("=" * 70)
    print(f"Generated {len(state['claims']):,} market-weeks; {len(state['portfolio'])} markets")
    print("\nForecast metrics")
    print(
        json.dumps(
            {"global_model": metrics, "seasonal_naive": baseline, "probabilistic": probabilistic},
            indent=2,
        )
    )
    print(f"\nHierarchy check: {len(region)} region-weeks, {len(national)} national weeks")
    print("\nRisk/cost trade-off")
    for name, plan in plans.items():
        shortage = sum(x["expected_shortage"] for x in plan["markets"])
        overtime = sum(x["overtime_hours"] for x in plan["markets"])
        print(f"{name:>4}: cost=${plan['total_expected_cost']:,.0f} ", end="")
        print(f"overtime={overtime:,.0f}h shortage={shortage:,.0f}")
    print("\nDecision backtest")
    print(backtest.to_string(index=False))
    fl = next(x for x in hurricane_plan["markets"] if x["market_id"] == "FL")
    print("\nFlorida hurricane scenario (P90)")
    print(json.dumps(fl, indent=2))


if __name__ == "__main__":
    main()
