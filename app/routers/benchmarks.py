import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory=Path("app/templates"))

FEAT_PATH = Path("data/processed/fuel_features.parquet")

PHASE_DATA = [
    {"phase": "Climb",   "rate_kg_min": 62.4, "share_pct": 18.5, "driver": "High thrust + potential energy gain"},
    {"phase": "Cruise",  "rate_kg_min": 38.1, "share_pct": 72.0, "driver": "Steady aerodynamic drag vs speed"},
    {"phase": "Descent", "rate_kg_min": 14.8, "share_pct": 4.5,  "driver": "Low idle thrust during idle descent"},
    {"phase": "Approach","rate_kg_min": 22.5, "share_pct": 5.0,  "driver": "Flap/gear drag during deceleration"},
]


def _synthetic_df(n: int = 600) -> pd.DataFrame:
    np.random.seed(42)
    types = ["A320", "B738", "A20N", "A321", "A359", "B77W"]
    return pd.DataFrame({
        "aircraft_type": np.random.choice(types, n),
        "fuel_kg": np.random.exponential(400, n) + 150,
        "duration_min": np.random.uniform(10, 45, n),
        "distance_flown_nm": np.random.uniform(60, 300, n),
        "mean_flight_level": np.random.uniform(280, 400, n),
    })


@router.get("/benchmarks", response_class=HTMLResponse)
async def benchmarks_page(request: Request):
    return templates.TemplateResponse("benchmarks.html", {"request": request})


@router.get("/api/benchmarks")
async def api_benchmarks(types: Optional[str] = Query(None)):
    if FEAT_PATH.exists():
        df = pd.read_parquet(FEAT_PATH)
    else:
        df = _synthetic_df()

    df["fuel_per_nm"] = df["fuel_kg"] / df["distance_flown_nm"].clip(lower=1.0)
    df["fuel_per_hr"] = df["fuel_kg"] / (df["duration_min"].clip(lower=1.0) / 60.0)

    # filter by requested types
    if types:
        selected = [t.strip() for t in types.split(",") if t.strip()]
        if selected:
            df = df[df["aircraft_type"].isin(selected)]

    # box plot data per aircraft type
    by_type = []
    for ac, grp in df.groupby("aircraft_type"):
        by_type.append({
            "aircraft_type": ac,
            "fuel_per_nm_values": grp["fuel_per_nm"].dropna().round(3).tolist(),
            "fuel_per_hr_values": grp["fuel_per_hr"].dropna().round(1).tolist(),
            "mean_per_nm": round(float(grp["fuel_per_nm"].median()), 3),
            "mean_per_hr": round(float(grp["fuel_per_hr"].mean()), 1),
            "count": len(grp),
        })

    return JSONResponse({
        "by_type": by_type,
        "phase_breakdown": PHASE_DATA,
        "fleet_avg_per_nm": round(float(df["fuel_per_nm"].median()), 3),
        "fleet_avg_per_hr": round(float(df["fuel_per_hr"].median()), 1),
        "total_intervals": len(df),
    })
