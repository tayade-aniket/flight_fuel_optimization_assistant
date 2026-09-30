from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory=Path("app/templates"))

# hardcoded SHAP values from training run (same as Streamlit page)
SHAP_DATA = [
    {"feature": "duration_min",            "shap_value": 0.412, "direction": "positive"},
    {"feature": "ref_mass_kg",             "shap_value": 0.198, "direction": "positive"},
    {"feature": "distance_flown_nm",       "shap_value": 0.187, "direction": "positive"},
    {"feature": "mean_groundspeed_kts",    "shap_value": 0.143, "direction": "positive"},
    {"feature": "mean_flight_level",       "shap_value": -0.131, "direction": "negative"},
    {"feature": "altitude_change_m",       "shap_value": 0.089, "direction": "positive"},
    {"feature": "is_climb",                "shap_value": 0.082, "direction": "positive"},
    {"feature": "vertical_rate_abs_mean",  "shap_value": 0.071, "direction": "positive"},
    {"feature": "is_widebody",             "shap_value": 0.064, "direction": "positive"},
    {"feature": "nominal_mach",            "shap_value": 0.053, "direction": "positive"},
    {"feature": "groundspeed_std_kts",     "shap_value": -0.041, "direction": "negative"},
    {"feature": "mean_vertical_rate_mps",  "shap_value": 0.038, "direction": "positive"},
    {"feature": "is_cruise",               "shap_value": -0.033, "direction": "negative"},
    {"feature": "track_change_deg_per_min","shap_value": 0.021, "direction": "positive"},
    {"feature": "altitude_std_m",          "shap_value": 0.017, "direction": "positive"},
]


@router.get("/explainability", response_class=HTMLResponse)
async def explainability_page(request: Request):
    return templates.TemplateResponse("explainability.html", {"request": request})


@router.get("/api/shap")
async def api_shap():
    return JSONResponse({"shap_values": SHAP_DATA})
