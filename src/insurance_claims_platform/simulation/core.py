from __future__ import annotations

import numpy as np
import pandas as pd

from insurance_claims_platform.config import MARKETS


def generate_portfolio(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    scales = {
        "FL": 1.35,
        "TX": 1.2,
        "CA": 1.15,
        "NY": 1.0,
        "PA": 0.75,
        "NJ": 0.7,
        "GA": 0.65,
        "NC": 0.62,
        "VA": 0.58,
        "MD": 0.5,
    }
    rows = []
    for market, region in MARKETS.items():
        policies = int(82_000 * scales[market] * rng.uniform(0.94, 1.06))
        rows.append(
            {
                "market_id": market,
                "state": market,
                "region": region,
                "policy_count": policies,
                "exposure_units": policies / 1_000,
                "average_insured_value": int(rng.normal(365_000, 45_000)),
                "policy_growth_rate": rng.uniform(-0.005, 0.02),
            }
        )
    return pd.DataFrame(rows)


def generate_weekly_claims(
    portfolio: pd.DataFrame, start: str = "2021-01-04", periods: int = 260, seed: int = 11
) -> pd.DataFrame:
    """Generate market-week aggregate claims from an overdispersed count process."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=periods, freq="W-MON")
    vulnerability = {"FL": 1.45, "TX": 1.25, "CA": 1.12, "NY": 1.0}
    rows: list[dict[str, object]] = []
    for rec in portfolio.to_dict("records"):
        market = str(rec["market_id"])
        exposure = float(rec["exposure_units"])
        backlog = 0.0
        for i, week in enumerate(dates):
            woy = int(week.isocalendar().week)
            hurricane = market in {"FL", "TX", "GA", "NC"} and 24 <= woy <= 44
            winter = market in {"NY", "NJ", "PA", "MD", "VA"} and (woy <= 10 or woy >= 49)
            fire = market == "CA" and 27 <= woy <= 45
            event_prob = 0.035 + (0.075 if hurricane or winter or fire else 0)
            cat = int(rng.random() < event_prob)
            storm_severity = float(cat * rng.gamma(2.2, 1.8))
            precipitation = max(0.0, rng.normal(1.0 + 0.45 * cat, 0.55))
            wind = max(3.0, rng.normal(12 + 7 * storm_severity, 5))
            seasonal = 1 + 0.15 * np.sin(2 * np.pi * (woy - 5) / 52)
            trend = 1 + float(rec["policy_growth_rate"]) * i / 52
            mu = exposure * 1.22 * vulnerability.get(market, 0.9) * seasonal * trend
            mu *= 1 + 0.08 * precipitation + 0.11 * max(wind - 18, 0) + 0.8 * storm_severity
            # Gamma-Poisson mixture produces realistic overdispersion.
            latent = rng.gamma(shape=12, scale=max(mu, 1) / 12)
            claims = int(rng.poisson(latent))
            cat_share = min(0.75, 0.08 + 0.1 * storm_severity)
            workload = claims * (1.05 + cat_share * 0.35)
            completed = max(0.0, min(backlog + workload, exposure * rng.uniform(4.0, 6.0)))
            opening = backlog
            backlog = max(0.0, backlog + workload - completed)
            rows.append(
                {
                    "market_id": market,
                    "region": rec["region"],
                    "week_start": week,
                    "new_claim_count": claims,
                    "workload_units": round(workload, 2),
                    "policy_count": int(float(rec["policy_count"]) * trend),
                    "precipitation": round(precipitation, 3),
                    "wind_speed": round(wind, 3),
                    "storm_severity": round(storm_severity, 3),
                    "catastrophe": cat,
                    "opening_backlog": round(opening, 2),
                    "claims_completed": round(completed, 2),
                    "closing_backlog": round(backlog, 2),
                }
            )
    return pd.DataFrame(rows).sort_values(["week_start", "market_id"]).reset_index(drop=True)


def generate_workforce(portfolio: pd.DataFrame, seed: int = 19) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for rec in portfolio.to_dict("records"):
        n = max(12, int(float(rec["exposure_units"]) * rng.uniform(0.065, 0.085)))
        rows.append(
            {
                "market_id": rec["market_id"],
                "region": rec["region"],
                "current_adjusters": n,
                "minimum_adjusters": max(5, int(n * 0.65)),
                "cat_qualified_adjusters": max(2, int(n * 0.3)),
            }
        )
    return pd.DataFrame(rows)
