import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.dependencies import load_model_and_features

router = APIRouter()
templates = Jinja2Templates(directory=Path("app/templates"))

FEAT_PATH = Path("data/processed/fuel_features.parquet")


def _build_synthetic_anomaly_df(n: int = 400) -> pd.DataFrame:
    np.random.seed(99)
    df = pd.DataFrame({
        "flight_id": [f"prc_{i:06d}" for i in range(n)],
        "aircraft_type": np.random.choice(["A320", "B738", "A359", "B77W"], n),
        "duration_min": np.random.uniform(10, 40, n),
        "distance_flown_nm": np.random.uniform(80, 280, n),
        "observed_fuel_kg": np.random.uniform(300, 1500, n),
        "predicted_fuel_kg": np.random.uniform(300, 1500, n),
    })
    df["residual_kg"] = df["observed_fuel_kg"] - df["predicted_fuel_kg"]
    # inject a few realistic anomalies
    for idx, val in [(12, 820.0), (88, 730.0), (45, -650.0), (201, 910.0)]:
        df.loc[idx, "residual_kg"] = val
        df.loc[idx, "observed_fuel_kg"] = df.loc[idx, "predicted_fuel_kg"] + val
    return df


@router.get("/anomaly", response_class=HTMLResponse)
async def anomaly_page(request: Request):
    return templates.TemplateResponse("anomaly.html", {"request": request})


@router.post("/api/anomaly")
async def api_anomaly():
    model, feature_names = load_model_and_features()

    if FEAT_PATH.exists():
        df_feat = pd.read_parquet(FEAT_PATH).head(1000)
        X = df_feat[feature_names].fillna(0)
        y_true = df_feat["fuel_kg"].values
        y_pred = model.predict(X)

        df = pd.DataFrame({
            "flight_id": df_feat.get("flight_id", pd.Series([f"prc_{i:06d}" for i in range(len(df_feat))])),
            "aircraft_type": df_feat.get("aircraft_type", "Unknown"),
            "observed_fuel_kg": y_true.round(1),
            "predicted_fuel_kg": y_pred.round(1),
            "residual_kg": (y_true - y_pred).round(1),
        })
    else:
        df = _build_synthetic_anomaly_df()

    mean_res = float(df["residual_kg"].mean())
    std_res = max(float(df["residual_kg"].std()), 1.0)
    df["z_score"] = ((df["residual_kg"] - mean_res) / std_res).round(3)

    df["severity"] = np.where(
        df["z_score"].abs() > 4.0, "High",
        np.where(df["z_score"].abs() > 2.5, "Medium", "Normal"),
    )
    df["status"] = np.where(df["z_score"].abs() > 2.5, "Flagged", "Nominal")

    flagged = df[df["status"] == "Flagged"]

    # send all points for scatter, but only top 50 flagged for the table
    all_points = df[["flight_id", "aircraft_type", "observed_fuel_kg", "predicted_fuel_kg", "residual_kg", "z_score", "severity", "status"]].to_dict(orient="records")
    top_flagged = flagged.sort_values("z_score", key=abs, ascending=False).head(50).to_dict(orient="records")

    return JSONResponse({
        "total_screened": len(df),
        "flagged_count": int(len(flagged)),
        "anomaly_rate_pct": round(len(flagged) / max(len(df), 1) * 100, 2),
        "mean_residual": round(mean_res, 1),
        "points": all_points,
        "flagged_table": top_flagged,
    })
