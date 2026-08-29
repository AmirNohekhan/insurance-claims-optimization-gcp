from __future__ import annotations

import pandas as pd


def apply_scenario(
    forecasts: pd.DataFrame,
    demand_multiplier: float = 1.0,
    catastrophe_market: str | None = None,
    catastrophe_multiplier: float = 1.65,
) -> pd.DataFrame:
    if demand_multiplier <= 0 or catastrophe_multiplier <= 0:
        raise ValueError("Scenario multipliers must be positive")
    out = forecasts.copy()
    cols = ["p10", "p50", "p75", "p90"]
    out[cols] = out[cols] * demand_multiplier
    if catastrophe_market:
        mask = out.market_id == catastrophe_market
        out.loc[mask, cols] *= catastrophe_multiplier
        # Widen the upper tail under rare-event uncertainty.
        out.loc[mask, "p75"] *= 1.06
        out.loc[mask, "p90"] *= 1.14
    out[cols] = out[cols].round(2)
    return out
