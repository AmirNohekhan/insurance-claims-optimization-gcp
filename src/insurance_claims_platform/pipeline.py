from __future__ import annotations

import pandas as pd

from insurance_claims_platform.features import build_features
from insurance_claims_platform.forecasting import GlobalForecaster, format_forecasts
from insurance_claims_platform.simulation import (
    generate_portfolio,
    generate_weekly_claims,
    generate_workforce,
)


def build_demo_state(periods: int = 220, seed: int = 7) -> dict:
    portfolio = generate_portfolio(seed)
    claims = generate_weekly_claims(portfolio, periods=periods, seed=seed + 1)
    featured = build_features(claims)
    cutoff = sorted(featured.week_start.unique())[-9]
    train = featured[featured.week_start <= cutoff]
    future = featured[featured.week_start > cutoff].copy()
    model = GlobalForecaster.fit(train)
    pred = model.predict(future)
    ratios = claims.groupby("market_id").apply(
        lambda x: x.workload_units.sum() / x.new_claim_count.sum(), include_groups=False
    )
    forecasts = format_forecasts(pred, pd.Timestamp(cutoff), ratios)
    forecasts = forecasts.merge(portfolio[["market_id", "region"]], on="market_id")
    workforce = generate_workforce(portfolio, seed + 2)
    backlog = claims.groupby("market_id").tail(1).set_index("market_id").closing_backlog
    return {
        "portfolio": portfolio,
        "claims": claims,
        "features": featured,
        "model": model,
        "forecasts": forecasts,
        "workforce": workforce,
        "backlog": backlog,
        "future": future,
    }
