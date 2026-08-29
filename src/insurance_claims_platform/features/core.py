from __future__ import annotations

import pandas as pd

KNOWN_FUTURE = {"week_of_year", "month", "policy_count"}
FORECASTED_EXOGENOUS = {"precipitation_fcst", "wind_speed_fcst", "storm_risk_fcst"}
UNKNOWN_FUTURE = {"precipitation", "wind_speed", "storm_severity", "catastrophe"}


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Features use shifted outcomes only; current/future realized weather is excluded."""
    out = data.sort_values(["market_id", "week_start"]).copy()
    group = out.groupby("market_id", sort=False)["new_claim_count"]
    for lag in (1, 2, 4, 8, 52):
        out[f"lag_{lag}"] = group.shift(lag)
    shifted = group.shift(1)
    out["rolling_mean_4"] = shifted.groupby(out["market_id"]).transform(
        lambda s: s.rolling(4).mean()
    )
    out["rolling_std_8"] = shifted.groupby(out["market_id"]).transform(lambda s: s.rolling(8).std())
    out["week_of_year"] = out["week_start"].dt.isocalendar().week.astype(int)
    out["month"] = out["week_start"].dt.month
    # Operationally available proxies: noisy weather outlook made at origin.
    out["precipitation_fcst"] = out.groupby("market_id")["precipitation"].shift(1)
    out["wind_speed_fcst"] = out.groupby("market_id")["wind_speed"].shift(1)
    out["storm_risk_fcst"] = out.groupby("market_id")["storm_severity"].shift(1).clip(0, 5)
    return out
