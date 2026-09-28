"""AeroFuel AI - Fuel Anomaly Detection Page."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import joblib

from src.utils.constants import SYSTEM_DISCLAIMER
from src.models.anomaly import FuelAnomalyDetector

st.set_page_config(page_title="Anomaly Detection | AeroFuel AI", page_icon="⚠️", layout="wide")

st.title("⚠️ Operational Fuel-Burn Anomaly Detection")
st.caption("Identify intervals where observed fuel telemetry deviates materially from model expectations.")

st.warning(
    "**Operational Protocol**: Observations flagged by this system indicate statistical deviation "
    "and are marked strictly for **Analytical Review**. They must NOT be automatically diagnosed as aircraft faults. "
    "Common drivers include unexpected atmospheric headwinds/tailwind shifts, ATC holding, step climbs, or telemetry noise."
)

feat_path = Path("data/processed/fuel_features.parquet")
model_path = Path("models/best_fuel_model.joblib")
feat_names_path = Path("models/feature_names.joblib")

if not feat_path.exists() or not model_path.exists():
    st.info("Generating synthetic anomaly demonstration data...")
    n_samples = 400
    df_data = pd.DataFrame({
        "flight_id": [f"prc_{i:06d}" for i in range(n_samples)],
        "aircraft_type": np.random.choice(["A320", "B738", "A359", "B77W"], n_samples),
        "duration_min": np.random.uniform(10, 40, n_samples),
        "distance_flown_nm": np.random.uniform(80, 280, n_samples),
        "observed_fuel_kg": np.random.uniform(300, 1500, n_samples),
        "predicted_fuel_kg": np.random.uniform(300, 1500, n_samples),
    })
    df_data["residual_kg"] = df_data["observed_fuel_kg"] - df_data["predicted_fuel_kg"]
    # Inject 5 realistic anomalies
    df_data.loc[12, "residual_kg"] = 820.0
    df_data.loc[12, "observed_fuel_kg"] = df_data.loc[12, "predicted_fuel_kg"] + 820.0
    df_data.loc[45, "residual_kg"] = -650.0
    df_data.loc[45, "observed_fuel_kg"] = max(50.0, df_data.loc[45, "predicted_fuel_kg"] - 650.0)
    df_data.loc[88, "residual_kg"] = 730.0
    df_data.loc[88, "observed_fuel_kg"] = df_data.loc[88, "predicted_fuel_kg"] + 730.0
else:
    df_feat = pd.read_parquet(feat_path).head(1000)
    model = joblib.load(model_path)
    feature_names = joblib.load(feat_names_path)
    X = df_feat[feature_names].fillna(0)
    y_true = df_feat["fuel_kg"].values
    y_pred = model.predict(X)

    detector = FuelAnomalyDetector()
    df_data = df_feat[["flight_id", "aircraft_type", "duration_min", "distance_flown_nm"]].copy()
    df_data["observed_fuel_kg"] = y_true
    df_data["predicted_fuel_kg"] = y_pred
    df_data["residual_kg"] = y_true - y_pred

# Compute Residual Statistics & Severity
mean_res = df_data["residual_kg"].mean()
std_res = df_data["residual_kg"].std() if df_data["residual_kg"].std() > 0 else 1.0
df_data["z_score"] = (df_data["residual_kg"] - mean_res) / std_res

df_data["status"] = np.where(df_data["z_score"].abs() > 2.5, "Flagged for Review", "Nominal")
df_data["severity"] = np.where(
    df_data["z_score"].abs() > 4.0, "High",
    np.where(df_data["z_score"].abs() > 2.5, "Medium", "Normal")
)

# KPI Summary
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Intervals Screened", f"{len(df_data):,}")
with col2:
    flagged_count = (df_data["status"] == "Flagged for Review").sum()
    st.metric("Flagged Intervals", flagged_count)
with col3:
    st.metric("Anomaly Rate", f"{(flagged_count / len(df_data)) * 100:.2f}%")
with col4:
    st.metric("Mean Model Residual", f"{mean_res:.1f} kg")

st.markdown("<br>", unsafe_allow_html=True)

# Scatter Plot: Observed vs Predicted with Anomaly Outliers
fig_scatter = px.scatter(
    df_data,
    x="predicted_fuel_kg",
    y="observed_fuel_kg",
    color="status",
    symbol="severity",
    hover_data=["flight_id", "aircraft_type", "residual_kg", "duration_min"],
    title="Observed vs Model-Predicted Fuel Burn (Outliers Highlighted)",
    labels={"predicted_fuel_kg": "Model Predicted Fuel (kg)", "observed_fuel_kg": "Actual ACARS Fuel Burn (kg)"},
    color_discrete_map={"Nominal": "#3B82F6", "Flagged for Review": "#EF4444"},
)
# Add 1:1 parity line
max_val = max(df_data["predicted_fuel_kg"].max(), df_data["observed_fuel_kg"].max())
fig_scatter.add_trace(px.line(x=[0, max_val], y=[0, max_val]).data[0])
st.plotly_chart(fig_scatter, use_container_width=True)

# Flagged Observations Table
st.subheader("Flagged Intervals Queue for Analytical Review")
flagged_df = df_data[df_data["status"] == "Flagged for Review"].sort_values(by="residual_kg", ascending=False)

if not flagged_df.empty:
    st.dataframe(
        flagged_df[["flight_id", "aircraft_type", "duration_min", "observed_fuel_kg", "predicted_fuel_kg", "residual_kg", "severity"]],
        use_container_width=True
    )
else:
    st.success("No critical statistical anomalies detected in current sample.")
