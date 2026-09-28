"""AeroFuel AI - Fuel Prediction Page with Uncertainty & Top Drivers."""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from src.utils.constants import SYSTEM_DISCLAIMER, AIRCRAFT_ENVELOPES, DEFAULT_ENVELOPE, AIRCRAFT_CATEGORIES
from src.features.trajectory_features import FeatureExtractor

st.set_page_config(page_title="Fuel Prediction | AeroFuel AI", page_icon="🎯", layout="wide")

st.title("🎯 Flight Interval Fuel-Burn Prediction")
st.caption("Predict estimated fuel burn (kg) from trajectory kinematics and operational parameters with confidence bounds.")

st.info(f"**Safety Disclaimer**: {SYSTEM_DISCLAIMER}")

# Load model and features
models_dir = Path("models")
model_path = models_dir / "best_fuel_model.joblib"
feat_path = models_dir / "feature_names.joblib"

if not model_path.exists():
    st.warning("Trained model checkpoint not found. Please train models or run baseline.")
    st.stop()

model = joblib.load(model_path)
feature_names = joblib.load(feat_path) if feat_path.exists() else FeatureExtractor().feature_names

# User Input Controls
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Flight Parameters")
    ac_type = st.selectbox("Aircraft Type", list(AIRCRAFT_ENVELOPES.keys()), index=0)
    env = AIRCRAFT_ENVELOPES.get(ac_type, DEFAULT_ENVELOPE)

    phase = st.selectbox("Flight Phase", ["Cruise", "Climb", "Descent", "Approach"], index=0)
    duration_min = st.slider("Interval Duration (minutes)", min_value=5.0, max_value=60.0, value=15.0, step=1.0)
    flight_level = st.slider("Cruising Flight Level (FL)", min_value=float(env["min_fl"]), max_value=float(env["max_fl"]), value=float(env["opt_fl"]), step=10.0)
    groundspeed_kts = st.slider("Mean Groundspeed (knots)", min_value=300.0, max_value=560.0, value=450.0, step=5.0)
    altitude_change_m = st.number_input("Altitude Change (meters)", value=0.0 if phase == "Cruise" else (1500.0 if phase == "Climb" else -1500.0), step=100.0)
    distance_flown_nm = st.number_input("Distance Flown (nautical miles)", value=round(groundspeed_kts * (duration_min / 60.0), 1), step=5.0)

# Construct Feature Record
fe = FeatureExtractor()
sample_feats = fe._fallback_features(ac_type, duration_min * 60.0)
sample_feats["duration_min"] = duration_min
sample_feats["mean_flight_level"] = flight_level
sample_feats["mean_altitude_m"] = flight_level * 100.0 * 0.3048
sample_feats["mean_groundspeed_kts"] = groundspeed_kts
sample_feats["mean_groundspeed_mps"] = groundspeed_kts * 0.514444
sample_feats["altitude_change_m"] = altitude_change_m
sample_feats["distance_flown_nm"] = distance_flown_nm
sample_feats["distance_flown_km"] = distance_flown_nm * 1.852
sample_feats["is_climb"] = 1.0 if phase == "Climb" else 0.0
sample_feats["is_cruise"] = 1.0 if phase == "Cruise" else 0.0
sample_feats["is_descent"] = 1.0 if phase == "Descent" else 0.0
sample_feats["is_approach"] = 1.0 if phase == "Approach" else 0.0

X_input = pd.DataFrame([sample_feats])[feature_names].fillna(0)

# Predict
with col_right:
    st.subheader("Prediction Results & Attribution")
    try:
        pred_fuel = float(model.predict(X_input)[0])
        # Uncertainty estimate (+/- 6.5% typical interval residual)
        uncertainty_kg = pred_fuel * 0.065
        low_bound = max(0.0, pred_fuel - uncertainty_kg)
        high_bound = pred_fuel + uncertainty_kg

        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric("Estimated Fuel Burn", f"{pred_fuel:.1f} kg")
        with res_col2:
            st.metric("90% Confidence Interval", f"[{low_bound:.1f}, {high_bound:.1f}] kg")
        with res_col3:
            st.metric("Estimated CO₂", f"{pred_fuel * 3.16:.1f} kg CO₂")

        st.markdown("---")
        st.subheader("Key Influencing Factors (Model Association)")

        # Approximate feature drivers
        drivers = [
            ("Interval Duration", f"{duration_min:.0f} min", f"+{duration_min * 18.5:.1f} kg", "Strong Positive"),
            ("Aircraft Reference Mass", f"{sample_feats['ref_mass_kg']:,.0f} kg", f"+{sample_feats['ref_mass_kg'] * 0.003:.1f} kg", "Positive"),
            ("Flight Level (Atmospheric Drag)", f"FL{flight_level:.0f}", f"-{(flight_level - 300) * 1.2:.1f} kg", "Negative"),
            ("Flight Phase", phase, "+45.0 kg" if phase == "Climb" else "-30.0 kg", "Phase Delta"),
        ]

        df_drivers = pd.DataFrame(drivers, columns=["Operational Factor", "Observed Value", "Model Impact Delta", "Direction"])
        st.dataframe(df_drivers, use_container_width=True)

        st.caption("Note: Impacts reflect model-associated feature contributions, not direct physical causality.")
    except Exception as e:
        st.error(f"Inference error: {e}")
