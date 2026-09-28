"""AeroFuel AI - What-If Scenario Simulator."""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from src.utils.constants import SYSTEM_DISCLAIMER, AIRCRAFT_ENVELOPES, DEFAULT_ENVELOPE, DEFAULT_FUEL_PRICE_USD_PER_KG
from src.features.trajectory_features import FeatureExtractor
from src.optimization.scenario_simulator import WhatIfSimulator
from src.models.baseline import PhysicsInspiredBaseline

st.set_page_config(page_title="What-If Simulator | AeroFuel AI", page_icon="🔬", layout="wide")

st.title("🔬 Operational What-If Scenario Simulator")
st.caption("Simulate candidate operational decisions: evaluate trade-offs across fuel burn, flight duration, and CO₂ emissions.")

st.info(f"**Safety Disclaimer**: {SYSTEM_DISCLAIMER}")

# Load model
model_path = Path("models/best_fuel_model.joblib")
feat_path = Path("models/feature_names.joblib")

fe = FeatureExtractor()
if model_path.exists():
    model = joblib.load(model_path)
    feature_names = joblib.load(feat_path) if feat_path.exists() else fe.feature_names
else:
    model = PhysicsInspiredBaseline()
    feature_names = fe.feature_names

simulator = WhatIfSimulator(model, feature_names)

# Scenario Configuration Controls
st.subheader("1. Baseline Flight Profile")
col_b1, col_b2, col_b3, col_b4 = st.columns(4)

with col_b1:
    ac_type = st.selectbox("Aircraft Type", list(AIRCRAFT_ENVELOPES.keys()), index=0)
    env = AIRCRAFT_ENVELOPES.get(ac_type, DEFAULT_ENVELOPE)
with col_b2:
    base_dist = st.number_input("Segment Distance (nm)", value=350.0, step=25.0)
with col_b3:
    base_fl = st.slider("Baseline Flight Level (FL)", min_value=float(env["min_fl"]), max_value=float(env["max_fl"]), value=340.0, step=10.0)
with col_b4:
    base_speed = st.slider("Baseline Groundspeed (kts)", min_value=380.0, max_value=520.0, value=450.0, step=5.0)

fuel_price = st.number_input("Assumed Jet-A1 Fuel Price ($/kg)", value=DEFAULT_FUEL_PRICE_USD_PER_KG, step=0.05)

# Construct baseline series
base_feats = pd.Series(fe._fallback_features(ac_type, (base_dist / base_speed) * 3600.0))
base_feats["duration_min"] = (base_dist / base_speed) * 60.0
base_feats["distance_flown_nm"] = base_dist
base_feats["distance_flown_km"] = base_dist * 1.852
base_feats["mean_flight_level"] = base_fl
base_feats["mean_altitude_m"] = base_fl * 100.0 * 0.3048
base_feats["mean_groundspeed_kts"] = base_speed
base_feats["mean_groundspeed_mps"] = base_speed * 0.514444

st.markdown("---")
st.subheader("2. Operational Scenarios to Compare")

col_s1, col_s2 = st.columns(2)

with col_s1:
    st.markdown("#### Scenario A (e.g. Higher Altitude / Step Climb)")
    fl_delta_a = st.slider("FL Change (Scenario A)", -40.0, 40.0, 20.0, 10.0, key="fl_a")
    speed_delta_a = st.slider("Speed Change kts (Scenario A)", -30.0, 30.0, 0.0, 5.0, key="spd_a")
    dist_delta_a = st.number_input("Distance Change nm (Scenario A)", value=0.0, step=5.0, key="dst_a")

with col_s2:
    st.markdown("#### Scenario B (e.g. Speed Reduction / Cost Index Shift)")
    fl_delta_b = st.slider("FL Change (Scenario B)", -40.0, 40.0, 0.0, 10.0, key="fl_b")
    speed_delta_b = st.slider("Speed Change kts (Scenario B)", -30.0, 30.0, -15.0, 5.0, key="spd_b")
    dist_delta_b = st.number_input("Distance Change nm (Scenario B)", value=0.0, step=5.0, key="dst_b")

# Run Simulations
res_base = simulator.simulate_scenario(base_feats, 0, 0, 0, "Baseline", fuel_price)
res_a = simulator.simulate_scenario(base_feats, fl_delta_a, speed_delta_a, dist_delta_a, "Scenario A", fuel_price)
res_b = simulator.simulate_scenario(base_feats, fl_delta_b, speed_delta_b, dist_delta_b, "Scenario B", fuel_price)

# Comparison Display
st.markdown("---")
st.subheader("3. Scenario Evaluation Matrix")

matrix = [
    {
        "Scenario": "Baseline",
        "Flight Level": f"FL{res_base['flight_level']:.0f}",
        "Speed (kts)": f"{res_base['groundspeed_kts']:.0f}",
        "Est. Fuel (kg)": f"{res_base['scenario_fuel_kg']:.1f}",
        "Fuel Delta": "0.0 kg (0.0%)",
        "Flight Time": f"{res_base['scenario_time_min']:.1f} min",
        "Time Delta": "0.0 min",
        "CO₂ Delta": "0.0 kg",
        "Cost Delta ($)": "$0.00",
    },
    {
        "Scenario": "Scenario A",
        "Flight Level": f"FL{res_a['flight_level']:.0f}",
        "Speed (kts)": f"{res_a['groundspeed_kts']:.0f}",
        "Est. Fuel (kg)": f"{res_a['scenario_fuel_kg']:.1f}",
        "Fuel Delta": f"{res_a['fuel_delta_kg']:+.1f} kg ({res_a['fuel_delta_pct']:+.2f}%)",
        "Flight Time": f"{res_a['scenario_time_min']:.1f} min",
        "Time Delta": f"{res_a['time_delta_min']:+.1f} min",
        "CO₂ Delta": f"{res_a['co2_delta_kg']:+.1f} kg",
        "Cost Delta ($)": f"${res_a['cost_delta_usd']:+.2f}",
    },
    {
        "Scenario": "Scenario B",
        "Flight Level": f"FL{res_b['flight_level']:.0f}",
        "Speed (kts)": f"{res_b['groundspeed_kts']:.0f}",
        "Est. Fuel (kg)": f"{res_b['scenario_fuel_kg']:.1f}",
        "Fuel Delta": f"{res_b['fuel_delta_kg']:+.1f} kg ({res_b['fuel_delta_pct']:+.2f}%)",
        "Flight Time": f"{res_b['scenario_time_min']:.1f} min",
        "Time Delta": f"{res_b['time_delta_min']:+.1f} min",
        "CO₂ Delta": f"{res_b['co2_delta_kg']:+.1f} kg",
        "Cost Delta ($)": f"${res_b['cost_delta_usd']:+.2f}",
    },
]

st.dataframe(pd.DataFrame(matrix), use_container_width=True)

st.caption("Operational Insight: Reductions in cruise speed save fuel but incur flight time penalties, illustrating the critical airline trade-off between fuel cost and crew/schedule reliability.")
