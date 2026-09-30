import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from src.data_processing.segmentation import TrajectorySegmenter

router = APIRouter()
templates = Jinja2Templates(directory=Path("app/templates"))
segmenter = TrajectorySegmenter()

TRAJ_DIR = Path("data/raw/flights_rank/flights_rank")


def _make_synthetic_traj(flight_id: str) -> pd.DataFrame:
    # fallback when real data isn't available
    n = 180
    np.random.seed(abs(hash(flight_id)) % 2**31)
    lats = np.linspace(48.85, 41.30, n) + np.sin(np.linspace(0, 3, n)) * 0.2
    lons = np.linspace(2.35, 2.08, n) + np.cos(np.linspace(0, 3, n)) * 0.15
    alts = np.concatenate([
        np.linspace(300, 10500, 40),
        np.full(100, 10500.0) + np.random.normal(0, 10, 100),
        np.linspace(10500, 1500, 30),
        np.linspace(1500, 20, 10),
    ])
    speeds = np.concatenate([
        np.linspace(140, 235, 40),
        np.full(100, 235.0) + np.random.normal(0, 2, 100),
        np.linspace(235, 160, 30),
        np.linspace(160, 70, 10),
    ])
    return pd.DataFrame({
        "latitude": lats,
        "longitude": lons,
        "altitude": alts,
        "groundspeed": speeds,
        "vertical_rate": np.gradient(alts, 60.0),
        "flight_id": flight_id,
    })


@router.get("/trajectory", response_class=HTMLResponse)
async def trajectory_page(request: Request):
    # list available flight ids if data dir exists
    flight_ids = []
    if TRAJ_DIR.exists():
        flight_ids = [f.stem for f in sorted(TRAJ_DIR.glob("*.parquet"))[:100]]
    if not flight_ids:
        flight_ids = ["prc_demo_flight_A320", "prc_demo_flight_B738", "prc_demo_flight_A359"]
    return templates.TemplateResponse(
        "trajectory.html",
        {"request": request, "flight_ids": flight_ids},
    )


@router.get("/api/trajectory")
async def api_trajectory(flight_id: str = Query("prc_demo_flight_A320")):
    parquet_path = TRAJ_DIR / f"{flight_id}.parquet"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    else:
        df = _make_synthetic_traj(flight_id)

    # segment into phases
    df = segmenter.classify_trajectory_points(df)

    # compute speed in kts for display
    df["speed_kts"] = df["groundspeed"] * 1.94384

    # cap points for reasonable JSON size
    if len(df) > 500:
        df = df.iloc[::len(df) // 500]

    points = df[["latitude", "longitude", "altitude", "speed_kts", "flight_phase"]].rename(
        columns={"speed_kts": "groundspeed"}
    ).fillna(0).to_dict(orient="records")

    max_alt = float(df["altitude"].max())
    cruise_ratio = float((df["flight_phase"] == "Cruise").mean() * 100)

    return JSONResponse({
        "flight_id": flight_id,
        "points": points,
        "metrics": {
            "total_points": len(df),
            "max_altitude_fl": round(max_alt * 3.28084 / 100, 0),
            "max_speed_kts": round(float(df["speed_kts"].max()), 1),
            "cruise_ratio_pct": round(cruise_ratio, 1),
        },
    })
