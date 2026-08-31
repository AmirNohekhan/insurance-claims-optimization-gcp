from __future__ import annotations

from functools import lru_cache
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from insurance_claims_platform.optimization import optimize_workforce
from insurance_claims_platform.pipeline import build_demo_state
from insurance_claims_platform.scenarios import apply_scenario

app = FastAPI(title="Insurance Claims Decision Platform", version="0.1.0")
FORECAST_STORE: dict[str, dict] = {}
PLAN_STORE: dict[str, dict] = {}
SUPPORTED_QUANTILES = {0.1, 0.5, 0.75, 0.9}


class ForecastRequest(BaseModel):
    markets: list[str] = Field(min_length=1)
    horizon_weeks: int = Field(ge=1, le=8)
    quantiles: list[float] = [0.5, 0.75, 0.9]

    @field_validator("quantiles")
    @classmethod
    def valid_quantiles(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("at least one quantile is required")
        if any(q not in SUPPORTED_QUANTILES for q in value):
            raise ValueError("supported quantiles are 0.1, 0.5, 0.75, 0.9")
        return list(dict.fromkeys(value))


class OptimizationRequest(BaseModel):
    forecast_id: str
    planning_quantile: float = 0.75
    allow_overtime: bool = True
    allow_reassignment: bool = True

    @field_validator("planning_quantile")
    @classmethod
    def valid_planning_quantile(cls, value: float) -> float:
        if value not in SUPPORTED_QUANTILES:
            raise ValueError("supported planning quantiles are 0.1, 0.5, 0.75, 0.9")
        return value


class ScenarioRequest(BaseModel):
    forecast_id: str
    demand_multiplier: float = Field(1.0, gt=0, le=3)
    catastrophe_market: str | None = None
    catastrophe_multiplier: float = Field(1.65, gt=0, le=5)


@lru_cache(maxsize=1)
def state() -> dict:
    return build_demo_state()


@app.middleware("http")
async def request_id(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.headers.get("X-Request-ID", str(uuid4()))
    return response


@app.get("/health")
def health() -> dict:
    return {"status": "healthy", "version": app.version}


@app.post("/v1/forecast")
def forecast(req: ForecastRequest) -> dict:
    available = set(state()["portfolio"].market_id)
    unknown = set(req.markets) - available
    if unknown:
        raise HTTPException(422, detail={"code": "UNKNOWN_MARKET", "markets": sorted(unknown)})
    frame = state()["forecasts"]
    frame = frame[frame.market_id.isin(req.markets) & (frame.horizon <= req.horizon_weeks)]
    forecast_id = str(uuid4())
    stored_payload = {
        "forecast_id": forecast_id,
        "model_version": "global-hgb-conformal-1",
        "records": frame.to_dict("records"),
    }
    quantile_columns = {f"p{int(q * 100)}" for q in req.quantiles}
    records = [
        {key: value for key, value in record.items() if not key.startswith("p") or key in quantile_columns}
        for record in stored_payload["records"]
    ]
    payload = {**stored_payload, "quantiles": req.quantiles, "records": records}
    FORECAST_STORE[forecast_id] = payload
    return payload


@app.post("/v1/optimize")
def optimize(req: OptimizationRequest) -> dict:
    if req.forecast_id not in FORECAST_STORE:
        raise HTTPException(404, detail={"code": "FORECAST_NOT_FOUND"})
    payload = FORECAST_STORE[req.forecast_id]
    if req.planning_quantile not in payload["quantiles"]:
        raise HTTPException(
            422,
            detail={
                "code": "QUANTILE_NOT_FORECAST",
                "planning_quantile": req.planning_quantile,
                "available_quantiles": payload["quantiles"],
            },
        )
    import pandas as pd

    forecasts = pd.DataFrame(payload["records"])
    workforce = state()["workforce"]
    workforce = workforce[workforce.market_id.isin(forecasts.market_id)]
    plan = optimize_workforce(
        forecasts,
        workforce,
        state()["backlog"],
        req.planning_quantile,
        req.allow_overtime,
        req.allow_reassignment,
    )
    PLAN_STORE[plan["plan_id"]] = plan
    return plan


@app.post("/v1/scenarios")
def scenarios(req: ScenarioRequest) -> dict:
    if req.forecast_id not in FORECAST_STORE:
        raise HTTPException(404, detail={"code": "FORECAST_NOT_FOUND"})
    import pandas as pd

    base = pd.DataFrame(FORECAST_STORE[req.forecast_id]["records"])
    adjusted = apply_scenario(
        base, req.demand_multiplier, req.catastrophe_market, req.catastrophe_multiplier
    )
    workforce = state()["workforce"]
    workforce = workforce[workforce.market_id.isin(adjusted.market_id)]
    return {
        "scenario": req.model_dump(),
        "plan": optimize_workforce(adjusted, workforce, quantile=0.9),
    }


@app.get("/v1/forecasts/{forecast_id}")
def get_forecast(forecast_id: str) -> dict:
    if forecast_id not in FORECAST_STORE:
        raise HTTPException(404, "forecast not found")
    return FORECAST_STORE[forecast_id]


@app.get("/v1/plans/{plan_id}")
def get_plan(plan_id: str) -> dict:
    if plan_id not in PLAN_STORE:
        raise HTTPException(404, "plan not found")
    return PLAN_STORE[plan_id]
