import asyncio
from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd

from src.optimization.constrained_optimizer import ConstrainedFuelOptimizer
from src.utils.constants import AIRCRAFT_ENVELOPES, DEFAULT_ENVELOPE
from app.dependencies import load_model_and_features

router = APIRouter()
templates = Jinja2Templates(directory=Path("app/templates"))


class OptimizeRequest(BaseModel):
    aircraft_type: str = "A320"
    flight_dist_nm: float = 500.0
    max_delay_min: float = 8.0
    max_fl_step: float = 40.0


def _run_optimization(req: OptimizeRequest):
    model, feature_names = load_model_and_features()
    env = AIRCRAFT_ENVELOPES.get(req.aircraft_type, DEFAULT_ENVELOPE)

    base_speed = env.get("nominal_mach", 0.78) * 661.47 * 0.85  # rough kts at ~35k ft
    base_fl = float(env.get("opt_fl", 350))
    dur_min = (req.flight_dist_nm / base_speed) * 60.0

    baseline = pd.Series({
        "duration_min": dur_min,
        "mean_flight_level": base_fl,
        "mean_altitude_m": base_fl * 100.0 * 0.3048,
        "max_altitude_m": base_fl * 100.0 * 0.3048,
        "min_altitude_m": base_fl * 100.0 * 0.3048,
        "altitude_change_m": 0.0,
        "altitude_std_m": 10.0,
        "mean_groundspeed_mps": base_speed * 0.514444,
        "mean_groundspeed_kts": base_speed,
        "max_groundspeed_kts": base_speed * 1.05,
        "groundspeed_std_kts": 3.0,
        "mean_vertical_rate_mps": 0.0,
        "vertical_rate_abs_mean": 0.1,
        "distance_flown_nm": req.flight_dist_nm,
        "distance_flown_km": req.flight_dist_nm * 1.852,
        "track_change_deg_per_min": 0.5,
        "is_climb": 0.0, "is_cruise": 1.0, "is_descent": 0.0, "is_approach": 0.0,
        "ref_mass_kg": env.get("ref_mass_kg", 65000.0),
        "nominal_mach": env.get("nominal_mach", 0.78),
        "is_narrowbody": 1.0, "is_widebody": 0.0, "is_freighter": 0.0,
    })

    opt = ConstrainedFuelOptimizer(model, feature_names)

    result = opt.optimize_cruise(
        baseline_features=baseline,
        aircraft_type=req.aircraft_type,
        max_time_delay_min=req.max_delay_min,
        max_fl_change=req.max_fl_step,
    )

    # pareto frontier for the trade-off chart
    pareto_df = opt.compute_pareto_frontier(baseline, req.aircraft_type)
    pareto_data = pareto_df.to_dict(orient="records")

    return {
        "optimal": result["optimal_scenario"],
        "evaluated_count": result["evaluated_count"],
        "pareto": pareto_data,
    }


@router.get("/optimizer", response_class=HTMLResponse)
async def optimizer_page(request: Request):
    aircraft_types = list(AIRCRAFT_ENVELOPES.keys())
    return templates.TemplateResponse(
        "optimizer.html",
        {"request": request, "aircraft_types": aircraft_types},
    )


@router.post("/api/optimize")
async def api_optimize(req: OptimizeRequest):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _run_optimization, req)
    return JSONResponse(result)
