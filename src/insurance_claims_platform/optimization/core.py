from __future__ import annotations

from dataclasses import asdict, dataclass
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from insurance_claims_platform.config import CostConfig


@dataclass
class MarketDecision:
    market_id: str
    forecasted_workload: float
    opening_backlog: float
    current_adjusters: int
    recommended_adjusters: int
    transfers_in: int
    transfers_out: int
    overtime_hours: float
    expected_shortage: float
    expected_closing_backlog: float
    utilization: float


def optimize_workforce(
    forecasts: pd.DataFrame,
    workforce: pd.DataFrame,
    backlog: pd.Series | None = None,
    quantile: float = 0.75,
    allow_overtime: bool = True,
    allow_reassignment: bool = True,
    costs: CostConfig | None = None,
) -> dict:
    """Solve a mixed-integer allocation with staffing, transfer, overtime and shortage variables."""
    costs = costs or CostConfig()
    qcol = f"p{int(quantile * 100)}"
    # Recommend a repeatable weekly staffing level over the requested horizon.
    demand = forecasts.groupby("market_id")[qcol].mean()
    ratio = (
        forecasts.groupby("market_id")["workload_ratio"].mean()
        if "workload_ratio" in forecasts
        else 1.15
    )
    demand = demand * ratio
    wf = workforce.set_index("market_id").loc[demand.index]
    opening = (
        backlog.reindex(demand.index).fillna(0)
        if backlog is not None
        else pd.Series(0.0, index=demand.index)
    )
    markets = list(demand.index)
    n = len(markets)
    # x assigned (integer), o overtime, s shortage, e excess. Transfers are derived vs current.
    c = np.r_[
        np.repeat(costs.regular_adjuster_week, n),
        np.repeat(costs.overtime_hour, n),
        np.repeat(costs.shortage_unit + costs.backlog_unit, n),
        np.repeat(costs.overcapacity_unit, n),
    ]
    integrality = np.r_[np.ones(n), np.zeros(3 * n)]
    lb = np.r_[wf.minimum_adjusters, np.zeros(3 * n)]
    max_staff = np.maximum(
        wf.current_adjusters + np.ceil(wf.current_adjusters * costs.max_transfer_fraction),
        wf.minimum_adjusters,
    )
    ub = np.r_[
        max_staff,
        costs.max_overtime_hours_per_adjuster * wf.current_adjusters.to_numpy(),
        np.repeat(np.inf, 2 * n),
    ]
    if not allow_overtime:
        ub[n : 2 * n] = 0
    if not allow_reassignment:
        lb[:n] = wf.current_adjusters
        ub[:n] = wf.current_adjusters
    # capacity*x + overtime_capacity*o + shortage - excess = demand + backlog
    Aeq = np.zeros((n, 4 * n))
    for i in range(n):
        Aeq[i, i] = costs.capacity_per_adjuster
        Aeq[i, n + i] = costs.overtime_capacity_per_hour
        Aeq[i, 2 * n + i] = 1
        Aeq[i, 3 * n + i] = -1
    constraints = [
        LinearConstraint(Aeq, (demand + opening).to_numpy(), (demand + opening).to_numpy())
    ]
    # Conserve total staff; reassignment cannot create employees.
    total = int(wf.current_adjusters.sum())
    conservation = np.zeros((1, 4 * n))
    conservation[0, :n] = 1
    constraints.append(LinearConstraint(conservation, total, total))
    result = milp(
        c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=constraints,
        options={"time_limit": 20},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Optimization infeasible: {result.message}")
    x, overtime, shortage, excess = np.split(result.x, 4)
    decisions = []
    transfer_cost = 0.0
    for i, market in enumerate(markets):
        assigned = int(round(x[i]))
        current = int(wf.current_adjusters.iloc[i])
        tin, tout = max(0, assigned - current), max(0, current - assigned)
        transfer_cost += tin * costs.transfer_adjuster
        available = (
            assigned * costs.capacity_per_adjuster + overtime[i] * costs.overtime_capacity_per_hour
        )
        decisions.append(
            MarketDecision(
                market,
                round(float(demand.iloc[i]), 2),
                round(float(opening.iloc[i]), 2),
                current,
                assigned,
                tin,
                tout,
                round(float(overtime[i]), 2),
                round(float(shortage[i]), 2),
                round(float(shortage[i]), 2),
                round(float(min(1.5, (demand.iloc[i] + opening.iloc[i]) / max(available, 1))), 3),
            )
        )
    return {
        "plan_id": str(uuid4()),
        "planning_quantile": quantile,
        "total_expected_cost": round(float(result.fun + transfer_cost), 2),
        "transfer_cost": round(transfer_cost, 2),
        "solver_status": result.message,
        "markets": [asdict(d) for d in decisions],
    }


def realized_cost(
    plan: dict, realized: pd.Series, costs: CostConfig | None = None
) -> dict[str, float]:
    costs = costs or CostConfig()
    decision = pd.DataFrame(plan["markets"]).set_index("market_id")
    capacity = (
        decision.recommended_adjusters * costs.capacity_per_adjuster
        + decision.overtime_hours * costs.overtime_capacity_per_hour
    )
    shortage = np.maximum(realized.reindex(decision.index).fillna(0) - capacity, 0)
    service = float((1 - shortage.sum() / max(realized.sum(), 1)).clip(0, 1))
    cost = (
        decision.recommended_adjusters.sum() * costs.regular_adjuster_week
        + decision.overtime_hours.sum() * costs.overtime_hour
        + shortage.sum() * (costs.shortage_unit + costs.backlog_unit)
        + plan["transfer_cost"]
    )
    return {
        "realized_cost": round(float(cost), 2),
        "service_level": round(service, 4),
        "unserved_workload": round(float(shortage.sum()), 2),
    }
