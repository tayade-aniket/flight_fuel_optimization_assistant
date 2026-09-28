"""AeroFuel AI - Constrained Optimization & Pareto Frontier."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib

from src.utils.constants import SYSTEM_DISCLAIMER, AIRCRAFT_ENVELOPES, DEFAULT_ENVELOPE
from src.features.trajectory_features import FeatureExtractor
from src.optimization.constrained_optimizer import ConstrainedFuelOptimizer
from src.models.baseline import PhysicsInspiredBaseline

st.set_page_config(page_title="Constrained Optimization | AeroFuel AI", page_icon="⚡", layout="wide")

st.title("⚡ Constrained Fuel Optimization & Pareto Trade-offs")
st.caption("Minimize estimated fuel burn subject to explicitly bounded operational constraints (flight envelope, speed limits, and schedule delay tolerance).")

st.warning(f"**Research Prototype Notice**: {SYSTEM_DISCLAIMER}")

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

optimizer = ConstrainedFuelOptimizer(model, feature_names)

# Sidebar / Constraints Form
st.subheader("1. Operational Boundary Constraints")
col_c1, col_c2, col_c3, col_c4 = st.columns(4)

with col_c1:
    ac_type = st.selectbox("Aircraft Type", list(AIRCRAFT_ENVELOPES.keys()), index=0)
    env = AIRCRAFT_ENVELOPES.get(ac_type, DEFAULT_ENVELOPE)
with col_c2:
    flight_dist_nm = st.number_input("Segment Distance (nm)", value=400.0, step=25.0)
with col_c3:
    max_delay = st.slider("Max Allowable Time Delay (min)", min_value=0.0, max_value=20.0, value=6.0, step=1.0)
with col_c4:
    max_fl_step = st.slider("Max Flight Level Deviation (FL)", min_value=10.0, max_value=60.0, value=40.0, step=10.0)

# Construct baseline
base_speed = 450.0
base_fl = env["opt_fl"]
base_feats = pd.Series(fe._fallback_features(ac_type, (flight_dist_nm / base_speed) * 3600.0))
base_feats["duration_min"] = (flight_dist_nm / base_speed) * 60.0
base_feats["distance_flown_nm"] = flight_dist_nm
base_feats["distance_flown_km"] = flight_dist_nm * 1.852
base_feats["mean_flight_level"] = base_fl
base_feats["mean_altitude_m"] = base_fl * 100.0 * 0.3048
base_feats["mean_groundspeed_kts"] = base_speed
base_feats["mean_groundspeed_mps"] = base_speed * 0.514444

# Solve Optimization
opt_res = optimizer.optimize_cruise(
    baseline_features=base_feats,
    aircraft_type=ac_type,
    max_time_delay_min=max_delay,
    max_fl_change=max_fl_step,
)
opt = opt_res["optimal_scenario"]

st.markdown("---")
st.subheader("2. Optimal Feasible Solution")

res_col1, res_col2, res_col3, res_col4 = st.columns(4)
with res_col1:
    st.metric("Baseline Fuel", f"{opt['baseline_fuel_kg']:.1f} kg", f"FL{base_fl:.0f} @ {base_speed:.0f} kts")
with res_col2:
    st.metric("Optimized Fuel", f"{opt['scenario_fuel_kg']:.1f} kg", f"FL{opt['flight_level']:.0f} @ {opt['groundspeed_kts']:.0f} kts")
with res_col3:
    st.metric("Estimated Fuel Saved", f"{-opt['fuel_delta_kg']:.1f} kg", f"{-opt['fuel_delta_pct']:.2f}%", delta_color="normal")
with res_col4:
    st.metric("Flight Time Impact", f"{opt['time_delta_min']:+.1f} min", f"Delay limit: {max_delay:.0f} min")

st.markdown("<br>", unsafe_allow_html=True)

# Pareto Frontier
st.subheader("3. Multi-Objective Fuel vs Time Trade-Off (Pareto Frontier)")
df_pareto = optimizer.compute_pareto_frontier(base_feats, ac_type, delay_thresholds=[0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 15.0])

fig_pareto = px.line(
    df_pareto,
    x="time_penalty_min",
    y="fuel_saved_kg",
    markers=True,
    title="Pareto Optimal Frontier: Fuel Savings vs Allowable Schedule Delay",
    labels={"time_penalty_min": "Allowable Schedule Delay (minutes)", "fuel_saved_kg": "Estimated Fuel Savings (kg)"},
    hover_data=["optimal_fl", "optimal_speed_kts", "co2_saved_kg"],
)
fig_pareto.update_traces(line=dict(color="#10B981", width=3), marker=dict(size=10, color="#059669"))
st.plotly_chart(fig_pareto, use_container_width=True)

st.dataframe(df_pareto, use_container_width=True)
st.caption("Pareto Analysis: Beyond ~8 minutes of flight time penalty, marginal fuel savings diminish significantly due to fixed aerodynamic parasitic drag.")
