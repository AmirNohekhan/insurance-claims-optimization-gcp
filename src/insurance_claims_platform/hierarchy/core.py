import pandas as pd


def bottom_up(forecasts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exact coherent reconciliation: regional and national totals sum bottom forecasts."""
    cols = ["p10", "p50", "p75", "p90"]
    region = forecasts.groupby(["region", "target_week"], as_index=False)[cols].sum()
    national = forecasts.groupby("target_week", as_index=False)[cols].sum()
    national["region"] = "National"
    return region, national
