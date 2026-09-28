"""AeroFuel AI - 3D Trajectory & Flight Phase Explorer."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from src.data_processing.segmentation import TrajectorySegmenter
from src.utils.constants import SYSTEM_DISCLAIMER

st.set_page_config(page_title="Trajectory Explorer | AeroFuel AI", page_icon="🗺️", layout="wide")

st.title("🗺️ 3D Trajectory & Operational Phase Analysis")
st.caption("Interactive 3D flight paths, kinematic climb/descent profiles, and phase-level fuel burn breakdown.")

st.info(f"**Notice**: {SYSTEM_DISCLAIMER}")

traj_dir = Path("data/raw/flights_rank")
traj_files = list(traj_dir.glob("*.parquet")) if traj_dir.exists() else []

if not traj_files:
    # Synthetic realistic trajectory fallback if archive not yet unpacked
    st.info("Generating representative multi-phase trajectory demonstration...")
    n_pts = 180
    timestamps = pd.date_range("2025-09-15 08:00:00", periods=n_pts, freq="1min")
    lats = np.linspace(48.8566, 41.2969, n_pts) + np.sin(np.linspace(0, 3, n_pts)) * 0.2
    lons = np.linspace(2.3522, 2.0783, n_pts) + np.cos(np.linspace(0, 3, n_pts)) * 0.15

    # Realistic Altitude Profile: Climb -> Cruise -> Descent -> Approach
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
    df_traj = pd.DataFrame({
        "timestamp": timestamps,
        "latitude": lats,
        "longitude": lons,
        "altitude": alts,
        "groundspeed": speeds,
        "vertical_rate": np.gradient(alts, 60.0),
        "flight_id": "prc_demo_flight_A320",
    })
    selected_flight_id = "prc_demo_flight_A320"
else:
    flight_options = [f.stem for f in traj_files[:50]]
    selected_flight_id = st.selectbox("Select Flight Identifier", flight_options, index=0)
    df_traj = pd.read_parquet(traj_dir / f"{selected_flight_id}.parquet")

# Segment Flight
segmenter = TrajectorySegmenter()
df_traj = segmenter.classify_trajectory_points(df_traj)

# Top Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Trajectory Points", f"{len(df_traj):,}")
with col2:
    st.metric("Max Altitude", f"{df_traj['altitude'].max() * 3.28084 / 100:,.0f} FL ({df_traj['altitude'].max():,.0f} m)")
with col3:
    st.metric("Max Groundspeed", f"{df_traj['groundspeed'].max() * 1.94384:.0f} kts")
with col4:
    cruise_pct = (df_traj['flight_phase'] == 'Cruise').mean() * 100.0
    st.metric("Cruise Phase Ratio", f"{cruise_pct:.1f}%")

st.markdown("<br>", unsafe_allow_html=True)

# 3D Trajectory Plot
st.subheader("Interactive 3D Trajectory Track")
fig_3d = px.line_3d(
    df_traj,
    x="longitude",
    y="latitude",
    z="altitude",
    color="flight_phase",
    title=f"3D Flight Path — {selected_flight_id}",
    labels={"altitude": "Altitude (m)", "latitude": "Latitude (°)", "longitude": "Longitude (°)"},
    color_discrete_map={
        "Climb": "#3B82F6",
        "Cruise": "#10B981",
        "Descent": "#F59E0B",
        "Approach": "#EF4444",
        "Unknown": "#6B7280",
    }
)
fig_3d.update_layout(scene=dict(aspectmode="manual", aspectratio=dict(x=1.5, y=1.5, z=0.6)), height=600)
st.plotly_chart(fig_3d, use_container_width=True)

# 2D Altitude Profile & Speed Curves
col_alt, col_spd = st.columns(2)

with col_alt:
    st.subheader("Vertical Altitude Profile")
    fig_alt = px.area(
        df_traj,
        x=df_traj.index,
        y="altitude",
        color="flight_phase",
        title="Barometric Altitude vs Elapsed Track Index",
        labels={"altitude": "Altitude (m)", "index": "Track Index"},
    )
    st.plotly_chart(fig_alt, use_container_width=True)

with col_spd:
    st.subheader("Groundspeed Dynamics")
    df_traj["speed_kts"] = df_traj["groundspeed"] * 1.94384
    fig_spd = px.line(
        df_traj,
        x=df_traj.index,
        y="speed_kts",
        color="flight_phase",
        title="Groundspeed (knots) vs Elapsed Track Index",
        labels={"speed_kts": "Groundspeed (kts)", "index": "Track Index"},
    )
    st.plotly_chart(fig_spd, use_container_width=True)
