from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES = [
    "market_id",
    "lag_1",
    "lag_2",
    "lag_4",
    "lag_8",
    "lag_52",
    "rolling_mean_4",
    "rolling_std_8",
    "week_of_year",
    "month",
    "policy_count",
    "precipitation_fcst",
    "wind_speed_fcst",
    "storm_risk_fcst",
]


@dataclass
class GlobalForecaster:
    model: Pipeline
    residual_quantiles: dict[float, float]
    version: str = "global-hgb-conformal-1"

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> GlobalForecaster:
        clean = frame.dropna(subset=FEATURES + ["new_claim_count"]).copy()
        if len(clean) < 100:
            raise ValueError("At least 100 complete observations are required")
        dates = sorted(clean["week_start"].unique())
        cutoff = dates[int(len(dates) * 0.82)]
        train = clean[clean.week_start < cutoff]
        calibration = clean[clean.week_start >= cutoff]
        prep = ColumnTransformer(
            [("market", OneHotEncoder(handle_unknown="ignore"), ["market_id"])],
            remainder="passthrough",
        )
        reg = HistGradientBoostingRegressor(
            loss="poisson",
            max_iter=180,
            max_leaf_nodes=24,
            learning_rate=0.06,
            l2_regularization=1.0,
            random_state=23,
        )
        model = Pipeline([("features", prep), ("model", reg)])
        model.fit(train[FEATURES], train["new_claim_count"])
        pred = np.maximum(0, model.predict(calibration[FEATURES]))
        residual = calibration["new_claim_count"].to_numpy() - pred
        rq = {q: float(np.quantile(residual, q)) for q in (0.1, 0.5, 0.75, 0.9)}
        return cls(model, rq)

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        median = np.maximum(0, self.model.predict(frame[FEATURES]))
        out = frame[["market_id", "week_start"]].copy()
        for q, label in ((0.1, "p10"), (0.5, "p50"), (0.75, "p75"), (0.9, "p90")):
            out[label] = np.maximum(0, median + self.residual_quantiles[q])
        values = np.sort(out[["p10", "p50", "p75", "p90"]].to_numpy(), axis=1)
        out[["p10", "p50", "p75", "p90"]] = values
        return out


def rolling_origins(
    data: pd.DataFrame, horizon: int = 8, origins: int = 4
) -> list[tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]]:
    dates = sorted(data["week_start"].unique())
    result = []
    for idx in range(len(dates) - horizon * origins, len(dates), horizon):
        origin = pd.Timestamp(dates[idx - 1])
        result.append(
            (
                origin,
                data[data.week_start <= origin],
                data[(data.week_start > origin) & (data.week_start <= dates[idx + horizon - 1])],
            )
        )
    return result


def baseline_predictions(test: pd.DataFrame, method: str = "seasonal_naive") -> np.ndarray:
    if method == "naive":
        return test["lag_1"].to_numpy()
    if method == "moving_average":
        return test["rolling_mean_4"].to_numpy()
    return test["lag_52"].fillna(test["rolling_mean_4"]).to_numpy()


def forecast_metrics(
    actual: np.ndarray, predicted: np.ndarray, seasonal_scale: float | None = None
) -> dict[str, float]:
    error = actual - predicted
    denom = max(float(np.abs(actual).sum()), 1e-9)
    smape_denom = np.abs(actual) + np.abs(predicted)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "wape": float(np.abs(error).sum() / denom),
        "smape": float(np.mean(2 * np.abs(error) / np.maximum(smape_denom, 1e-9))),
        "mase": float(np.mean(np.abs(error)) / max(seasonal_scale or 1, 1e-9)),
        "bias": float(np.mean(error)),
        "underforecast_rate": float(np.mean(error > 0)),
    }


def pinball(actual: np.ndarray, forecast: np.ndarray, q: float) -> float:
    delta = actual - forecast
    return float(np.mean(np.maximum(q * delta, (q - 1) * delta)))


def evaluate_probabilistic(actual: np.ndarray, forecasts: pd.DataFrame) -> dict[str, float]:
    result = {
        f"pinball_{q}": pinball(actual, forecasts[f"p{int(q * 100)}"].to_numpy(), q)
        for q in (0.1, 0.5, 0.75, 0.9)
    }
    result["p90_coverage"] = float(np.mean(actual <= forecasts["p90"].to_numpy()))
    result["p10_p90_coverage"] = float(
        np.mean((actual >= forecasts.p10) & (actual <= forecasts.p90))
    )
    result["p10_p90_width"] = float(np.mean(forecasts.p90 - forecasts.p10))
    return result


def format_forecasts(
    pred: pd.DataFrame, origin: pd.Timestamp, workload_ratio: pd.Series | None = None
) -> pd.DataFrame:
    out = pred.copy()
    out["forecast_id"] = str(uuid4())
    out["forecast_origin"] = origin
    out["target_week"] = out.pop("week_start")
    out["horizon"] = ((out.target_week - origin).dt.days // 7).astype(int)
    out["selected_model"] = "global_hgb_poisson_conformal"
    out["model_version"] = "global-hgb-conformal-1"
    out["generated_at"] = datetime.now(UTC).isoformat()
    ratio = (
        workload_ratio
        if workload_ratio is not None
        else pd.Series(1.15, index=out.market_id.unique())
    )
    out["workload_ratio"] = out.market_id.map(ratio).fillna(1.15)
    return out
