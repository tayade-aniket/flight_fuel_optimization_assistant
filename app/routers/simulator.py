import asyncio
from pathlib import Path
from typing import List
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd

from src.optimization.scenario_simulator import WhatIfSimulator
from src.utils.constants import AIRCRAFT_ENVELOPES, DEFAULT_ENVELOPE
from app.dependencies import load_model_and_features

router = APIRouter()
templates = Jinja2Templates(directory=Path("app/templates"))


class ScenarioDef(BaseModel):
    fl_delta: float = 0.0
    speed_delta: float = 0.0
    dist_delta: float = 0.0
    name: str = "Scenario"


class SimulateRequest(BaseModel):
    aircraft_type: str = "A320"
    base_dist: float = 200.0
    base_fl: float = 350.0
    base_speed: float = 450.0
    fuel_price: float = 0.82
    scenarios: List[ScenarioDef] = []


def _run_simulation(req: SimulateRequest):
    model, feature_names = load_model_and_features()
    env = AIRCRAFT_ENVELOPES.get(req.aircraft_type, DEFAULT_ENVELOPE)

    # build baseline feature series
    base = {
        "duration_min": (req.base_dist / req.base_speed) * 60.0,
        "mean_flight_level": req.base_fl,
        "mean_altitude_m": req.base_fl * 100.0 * 0.3048,
        "max_altitude_m": req.base_fl * 100.0 * 0.3048,
        "min_altitude_m": req.base_fl * 100.0 * 0.3048,
        "altitude_change_m": 0.0,
        "altitude_std_m": 10.0,
        "mean_groundspeed_mps": req.base_speed * 0.514444,
        "mean_groundspeed_kts": req.base_speed,
        "max_groundspeed_kts": req.base_speed * 1.05,
        "groundspeed_std_kts": 3.0,
        "mean_vertical_rate_mps": 0.0,
        "vertical_rate_abs_mean": 0.1,
        "distance_flown_nm": req.base_dist,
        "distance_flown_km": req.base_dist * 1.852,
        "track_change_deg_per_min": 0.5,
        "is_climb": 0.0, "is_cruise": 1.0, "is_descent": 0.0, "is_approach": 0.0,
        "ref_mass_kg": env.get("ref_mass_kg", 65000.0),
        "nominal_mach": env.get("nominal_mach", 0.78),
        "is_narrowbody": 1.0, "is_widebody": 0.0, "is_freighter": 0.0,
    }
    baseline_series = pd.Series(base)
    sim = WhatIfSimulator(model, feature_names)

    results = []
    # always include baseline
    baseline_result = sim.simulate_scenario(baseline_series, scenario_name="Baseline", fuel_price_per_kg=req.fuel_price)
    results.append(baseline_result)

    for s in req.scenarios:
        r = sim.simulate_scenario(
            baseline_series,
            delta_flight_level=s.fl_delta,
            delta_speed_kts=s.speed_delta,
            delta_distance_nm=s.dist_delta,
            scenario_name=s.name,
            fuel_price_per_kg=req.fuel_price,
        )
        results.append(r)

    return results


@router.get("/simulator", response_class=HTMLResponse)
async def simulator_page(request: Request):
    aircraft_types = list(AIRCRAFT_ENVELOPES.keys())
    return templates.TemplateResponse(
        "simulator.html",
        {"request": request, "aircraft_types": aircraft_types},
    )


@router.post("/api/simulate")
async def api_simulate(req: SimulateRequest):
    # CPU-bound work goes off the event loop
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, _run_simulation, req)
    return JSONResponse({"scenarios": results})
