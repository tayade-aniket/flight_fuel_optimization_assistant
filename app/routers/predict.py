import numpy as np
import pandas as pd
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from pydantic import BaseModel

from src.utils.constants import (
    AIRCRAFT_ENVELOPES,
    DEFAULT_ENVELOPE,
    AIRCRAFT_CATEGORIES,
    CO2_PER_KG_FUEL,
    METERS_PER_FOOT,
    KNOTS_PER_MPS,
    NM_PER_METER,
)
from src.features.trajectory_features import FeatureExtractor
from app.dependencies import load_model_and_features

router = APIRouter()
templates = Jinja2Templates(directory=Path("app/templates"))

fe = FeatureExtractor()


class PredictRequest(BaseModel):
    aircraft_type: str = "A320"
    flight_phase: str = "Cruise"
    duration_min: float = 15.0
    flight_level: float = 350.0
    groundspeed_kts: float = 450.0
    altitude_change_m: float = 0.0
    distance_flown_nm: float = 112.5


@router.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    aircraft_types = list(AIRCRAFT_ENVELOPES.keys())
    return templates.TemplateResponse(
        "predict.html",
        {"request": request, "aircraft_types": aircraft_types},
    )


@router.post("/api/predict")
async def api_predict(req: PredictRequest):
    model, feature_names = load_model_and_features()

    env = AIRCRAFT_ENVELOPES.get(req.aircraft_type, DEFAULT_ENVELOPE)

    # build feature dict from user inputs
    feats = fe._fallback_features(req.aircraft_type, req.duration_min * 60.0)
    feats["duration_min"] = req.duration_min
    feats["mean_flight_level"] = req.flight_level
    feats["mean_altitude_m"] = req.flight_level * 100.0 * METERS_PER_FOOT
    feats["mean_groundspeed_kts"] = req.groundspeed_kts
    feats["mean_groundspeed_mps"] = req.groundspeed_kts / KNOTS_PER_MPS
    feats["altitude_change_m"] = req.altitude_change_m
    feats["distance_flown_nm"] = req.distance_flown_nm
    feats["distance_flown_km"] = req.distance_flown_nm * 1.852
    feats["is_climb"] = 1.0 if req.flight_phase == "Climb" else 0.0
    feats["is_cruise"] = 1.0 if req.flight_phase == "Cruise" else 0.0
    feats["is_descent"] = 1.0 if req.flight_phase == "Descent" else 0.0
    feats["is_approach"] = 1.0 if req.flight_phase == "Approach" else 0.0

    X = pd.DataFrame([feats])[feature_names].fillna(0)
    fuel_kg = float(model.predict(X)[0])

    # simple uncertainty: +/- 6.5%
    ci_half = fuel_kg * 0.065
    co2_kg = fuel_kg * CO2_PER_KG_FUEL

    # approximate top drivers
    drivers = [
        {
            "factor": "Interval Duration",
            "value": f"{req.duration_min:.0f} min",
            "impact": f"+{req.duration_min * 18.5:.1f} kg",
            "direction": "positive",
        },
        {
            "factor": "Aircraft Reference Mass",
            "value": f"{feats['ref_mass_kg']:,.0f} kg",
            "impact": f"+{feats['ref_mass_kg'] * 0.003:.1f} kg",
            "direction": "positive",
        },
        {
            "factor": "Flight Level (Atmospheric Drag)",
            "value": f"FL{req.flight_level:.0f}",
            "impact": f"-{max(0, (req.flight_level - 300) * 1.2):.1f} kg",
            "direction": "negative",
        },
        {
            "factor": "Flight Phase",
            "value": req.flight_phase,
            "impact": "+45.0 kg" if req.flight_phase == "Climb" else "-30.0 kg",
            "direction": "positive" if req.flight_phase == "Climb" else "negative",
        },
    ]

    return JSONResponse({
        "fuel_kg": round(fuel_kg, 1),
        "ci_low": round(max(0.0, fuel_kg - ci_half), 1),
        "ci_high": round(fuel_kg + ci_half, 1),
        "co2_kg": round(co2_kg, 1),
        "drivers": drivers,
    })
