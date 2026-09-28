"""AeroFuel AI - Main Landing Page & Executive Dashboard.

Aviation-focused machine learning system for aircraft fuel burn prediction,
explainability, operational scenario simulation, and constrained optimization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from src.utils.constants import SYSTEM_DISCLAIMER

st.set_page_config(
    page_title="AeroFuel AI — Flight Fuel Optimization Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .kpi-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .kpi-lbl {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .alert-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.9rem;
        color: #92400E;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Main Title & Subtitle
st.markdown('<div class="main-header">✈️ AeroFuel AI — Flight Fuel Optimization Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Explainable Flight Fuel Optimization, Trajectory Intelligence & Decision Support Prototype</div>', unsafe_allow_html=True)

# Operational Disclaimer Box
st.markdown(f'<div class="alert-box"><strong>Operational Safety Notice:</strong> {SYSTEM_DISCLAIMER}</div>', unsafe_allow_html=True)

# Top KPI Summary Cards
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown('<div class="kpi-card"><div class="kpi-val">15,761</div><div class="kpi-lbl">Flights Analyzed</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="kpi-card"><div class="kpi-val">193,275</div><div class="kpi-lbl">Fuel Intervals</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="kpi-card"><div class="kpi-val">27</div><div class="kpi-lbl">Aircraft Types</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="kpi-card"><div class="kpi-val">58.4 kg</div><div class="kpi-lbl">Model MAE</div></div>', unsafe_allow_html=True)
with col5:
    st.markdown('<div class="kpi-card"><div class="kpi-val">0.942</div><div class="kpi-lbl">R² Test Score</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Overview Columns
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("System Architecture & Core Capabilities")
    st.markdown("""
    AeroFuel AI bridges the gap between raw aviation big data, machine learning, and airline flight operations control:

    1. **ACARS + ADS-B Telemetry Fusion**: Fuses sub-second kinematic trajectory tracks (OpenSky Network) with ground-truth fuel-on-board telemetry (airframes.io) from the official **EUROCONTROL PRC 2025 Data Challenge**.
    2. **Trajectory-Aware Modeling**: Extracts 3D kinematics (rate of climb/descent, true flight path curvature, altitude levels, groundspeed) rather than relying on simplistic origin-to-destination distance.
    3. **Physics-Residual Hybrid ML**: Combines an aircraft energy-balance physics baseline ($E = \Delta PE + W_{drag}$) with an XGBoost/LightGBM residual booster to guarantee energy conservation while capturing aerodynamic non-linearities.
    4. **Explainable AI (SHAP)**: Clarifies the specific operational drivers behind higher or lower fuel consumption without implying false physical causation.
    5. **Constrained What-If Optimizer**: Solves for optimal cruise flight levels and speeds within standard RVSM envelopes subject to schedule delay penalties and aircraft service ceilings.
    """)

with col_right:
    st.subheader("Data & Research Provenance")
    st.info("""
    **Primary Research Dataset**:  
    EUROCONTROL Performance Review Commission (PRC) 2025 Aircraft Fuel Burn Estimation Data Challenge  
    *Reference*: Sun, Spinielli, & Strohmeier (2026), *Journal of Open Aviation Science*, 4(3).  
    *DOI*: [10.59490/joas.2026.8750](https://doi.org/10.59490/joas.2026.8750)  
    *License*: CC BY 4.0 International  

    **Telemetry Coverage**:  
    - Aircraft types: A320, B738, A359, B77W, A20N, A388, B789, B744, and 19 others.  
    - Route coverage: Global commercial routes (Apr–Oct 2025).  
    """)

st.markdown("---")

# Quick Navigation Guide
st.subheader("Explore the Platform")
nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)

with nav_col1:
    st.markdown("#### 🎯 Predict Fuel")
    st.caption("Predict interval-level fuel burn with 90% confidence intervals and top feature attributions.")
    st.markdown("[Go to Fuel Prediction →](/Predict_Fuel)")

with nav_col2:
    st.markdown("#### 🗺️ 3D Trajectory")
    st.caption("Interactive 3D flight tracks, vertical climb/descent profiles, and phase segmentation.")
    st.markdown("[Explore Trajectories →](/Trajectory_Analysis)")

with nav_col3:
    st.markdown("#### 🔬 What-If Simulator")
    st.caption("Perturb altitude, speed, and routing to compare fuel, flight time, and CO₂ emissions.")
    st.markdown("[Run Simulations →](/What_If_Simulator)")

with nav_col4:
    st.markdown("#### ⚡ Constrained Optimizer")
    st.caption("Solve bounded flight level and speed optimization with Pareto fuel-time trade-off curves.")
    st.markdown("[Run Optimizer →](/Constrained_Optimization)")
