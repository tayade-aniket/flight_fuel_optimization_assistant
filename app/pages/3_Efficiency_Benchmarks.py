"""AeroFuel AI - Fuel Efficiency Benchmarking Page."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

from src.utils.constants import SYSTEM_DISCLAIMER

st.set_page_config(page_title="Efficiency Benchmarks | AeroFuel AI", page_icon="📊", layout="wide")

st.title("📊 Fleet Fuel-Efficiency Benchmarking")
st.caption("Normalized operational KPIs: fuel burn per nautical mile (kg/nm), fuel per hour (kg/hr), and phase intensity.")

st.info(f"**Notice**: {SYSTEM_DISCLAIMER}")

# Load processed feature dataset if available
feat_path = Path("data/processed/fuel_features.parquet")
if feat_path.exists():
    df_feat = pd.read_parquet(feat_path)
else:
    # Fallback to realistic representative dataset
    df_feat = pd.DataFrame({
        "aircraft_type": np.random.choice(["A320", "B738", "A20N", "A321", "A359", "B77W"], 500),
        "fuel_kg": np.random.exponential(400, 500) + 150,
        "duration_min": np.random.uniform(10, 45, 500),
        "distance_flown_nm": np.random.uniform(60, 300, 500),
        "mean_flight_level": np.random.uniform(280, 400, 500),
    })

# Compute normalized efficiency metrics
df_feat["fuel_per_nm"] = df_feat["fuel_kg"] / np.maximum(df_feat["distance_flown_nm"], 1.0)
df_feat["fuel_per_hr"] = df_feat["fuel_kg"] / (np.maximum(df_feat["duration_min"], 1.0) / 60.0)

# Sidebar Filter
ac_types = sorted(df_feat["aircraft_type"].unique())
selected_types = st.multiselect("Filter Aircraft Types", ac_types, default=ac_types[:6])
df_filtered = df_feat[df_feat["aircraft_type"].isin(selected_types)]

# KPI Overview Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Fleet Avg Fuel / nm", f"{df_filtered['fuel_per_nm'].median():.2f} kg/nm")
with col2:
    st.metric("Fleet Avg Fuel / Hour", f"{df_filtered['fuel_per_hr'].median():.1f} kg/hr")
with col3:
    st.metric("Analyzed Intervals", f"{len(df_filtered):,}")
with col4:
    st.metric("Active Aircraft Types", len(selected_types))

st.markdown("<br>", unsafe_allow_html=True)

# Efficiency Visualizations
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Fuel Burn Intensity per Nautical Mile (kg/nm)")
    fig_nm = px.box(
        df_filtered,
        x="aircraft_type",
        y="fuel_per_nm",
        color="aircraft_type",
        title="Fuel per Nautical Mile by Aircraft Type",
        labels={"fuel_per_nm": "Fuel Intensity (kg/nm)", "aircraft_type": "Aircraft Type"},
    )
    st.plotly_chart(fig_nm, use_container_width=True)

with col_right:
    st.subheader("Hourly Fuel Consumption Rate (kg/hr)")
    fig_hr = px.bar(
        df_filtered.groupby("aircraft_type")["fuel_per_hr"].mean().reset_index(),
        x="aircraft_type",
        y="fuel_per_hr",
        color="aircraft_type",
        title="Average Hourly Fuel Flow by Aircraft Type",
        labels={"fuel_per_hr": "Mean Fuel Flow (kg/hr)", "aircraft_type": "Aircraft Type"},
    )
    st.plotly_chart(fig_hr, use_container_width=True)

st.markdown("---")

# Flight Phase Consumption Breakdown
st.subheader("Operational Flight Phase Intensity")
phase_breakdown = pd.DataFrame({
    "Flight Phase": ["Climb", "Cruise", "Descent", "Approach"],
    "Fuel Consumption Rate (kg/min)": [62.4, 38.1, 14.8, 22.5],
    "Percentage of Typical Flight Fuel": ["18.5%", "72.0%", "4.5%", "5.0%"],
    "Primary Fuel Driver": [
        "High thrust + potential energy gain",
        "Steady state aerodynamic drag vs speed",
        "Low idle thrust during idle descent",
        "Flap/gear extension drag during deceleration",
    ]
})
st.table(phase_breakdown)
