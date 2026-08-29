from __future__ import annotations

import pandas as pd

from insurance_claims_platform.optimization import optimize_workforce, realized_cost


def decision_backtest(
    forecasts: pd.DataFrame,
    workforce: pd.DataFrame,
    realized: pd.Series,
    quantiles: tuple[float, ...] = (0.5, 0.75, 0.9),
) -> pd.DataFrame:
    rows = []
    for q in quantiles:
        plan = optimize_workforce(forecasts, workforce, quantile=q)
        rows.append(
            {
                "policy": f"ML P{int(q * 100)}",
                **realized_cost(plan, realized),
                "planned_cost": plan["total_expected_cost"],
            }
        )
    static = workforce.set_index("market_id").current_adjusters * 16
    shortage = (realized.reindex(static.index) - static).clip(lower=0)
    rows.append(
        {
            "policy": "Static staffing",
            "realized_cost": float(workforce.current_adjusters.sum() * 2200 + shortage.sum() * 840),
            "service_level": float(1 - shortage.sum() / realized.sum()),
            "unserved_workload": float(shortage.sum()),
            "planned_cost": float(workforce.current_adjusters.sum() * 2200),
        }
    )
    return pd.DataFrame(rows)
