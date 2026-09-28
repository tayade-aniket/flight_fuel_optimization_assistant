"""AeroFuel AI - Methodology, Dataset Provenance & Safety Principles."""

import streamlit as st
import pandas as pd
from src.utils.constants import SYSTEM_DISCLAIMER

st.set_page_config(page_title="Methodology & Safety | AeroFuel AI", page_icon="📜", layout="wide")

st.title("📜 Methodology, Dataset Provenance & Safety Principles")
st.caption("Complete scientific documentation, research citations, data leakage audit, and operational safety boundaries.")

# Prominent Safety Notice
st.error(f"**Safety & Compliance Statement**: {SYSTEM_DISCLAIMER}")

st.markdown("---")

# 1. Dataset & Provenance
st.subheader("1. Primary Dataset & Academic Citations")
st.markdown("""
AeroFuel AI is built on the official **EUROCONTROL Performance Review Commission (PRC) 2025 Aircraft Fuel Burn Estimation Data Challenge**, released under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

- **Official Zenodo Archive**: [zenodo.org/records/19184662](https://zenodo.org/records/19184662)
- **Academic Publication**:  
  *Sun, J., Spinielli, E., & Strohmeier, M. (2026). Aircraft Fuel Burn Estimation: The EUROCONTROL PRC 2025 Data Challenge. Journal of Open Aviation Science, 4(3).*  
  *DOI*: [10.59490/joas.2026.8750](https://doi.org/10.59490/joas.2026.8750)

**Dataset Architecture**:
- **ACARS Fuel Telemetry**: Crowdsourced Fuel-on-Board (FOB) reports collected via consumer-grade VHF receivers through [airframes.io](https://airframes.io).
- **ADS-B Kinematic Trajectories**: Sub-second flight tracks re-decoded from raw Mode S messages using `pyModeS` through the [OpenSky Network](https://opensky-network.org).
- **Flight Metadata**: Flight plans, origin/destination airports, and schedule times compiled from EUROCONTROL Network Manager.
- **Reporting Unit Inference**: Curators validated fuel reports against TU Delft's OpenAP physics reference model to resolve ambiguous imperial (lbs) vs metric (kg) telemetry.
""")

st.markdown("---")

# 2. Data Leakage Prevention Audit Table
st.subheader("2. Strict Data Leakage Audit Table")
st.markdown("""
In operational machine learning, feature leakage occurs when information unavailable at inference time is improperly fed to the model. 
Every feature in AeroFuel AI was audited against operational availability:
""")

leakage_data = [
    {"Feature": "Aircraft Type / Family", "Available at Prediction Time?": "Yes (Filed flight plan)", "Allowed?": "Yes", "Audit Result": "Clean"},
    {"Feature": "Departure / Destination Airports", "Available at Prediction Time?": "Yes (Flight plan)", "Allowed?": "Yes", "Audit Result": "Clean"},
    {"Feature": "Interval Planned Duration", "Available at Prediction Time?": "Yes (Target interval slice)", "Allowed?": "Yes", "Audit Result": "Clean"},
    {"Feature": "Interval Planned Flight Level", "Available at Prediction Time?": "Yes (Target interval slice)", "Allowed?": "Yes", "Audit Result": "Clean"},
    {"Feature": "Interval Planned Groundspeed", "Available at Prediction Time?": "Yes (Target interval slice)", "Allowed?": "Yes", "Audit Result": "Clean"},
    {"Feature": "Historical Fleet Fuel Consumption", "Available at Prediction Time?": "Yes (Prior training sets)", "Allowed?": "Yes", "Audit Result": "Clean"},
    {"Feature": "Future Flight Trajectory", "Available at Prediction Time?": "No (Occurs after interval)", "Allowed?": "Strictly Prohibited", "Audit Result": "Excluded"},
    {"Feature": "Subsequent ACARS FOB Telemetry", "Available at Prediction Time?": "No (Future telemetry)", "Allowed?": "Strictly Prohibited", "Audit Result": "Excluded"},
    {"Feature": "Actual Completed Flight Total Fuel", "Available at Prediction Time?": "No (Post-flight)", "Allowed?": "Strictly Prohibited", "Audit Result": "Excluded"},
]

st.table(pd.DataFrame(leakage_data))

st.markdown("---")

# 3. Aircraft Performance Modeling (BADA Context)
st.subheader("3. Aircraft Performance Modeling & BADA Context")
st.markdown("""
EUROCONTROL's Base of Aircraft Data (**BADA**) is the industry benchmark for aircraft performance modeling. 
BADA models thrust, aerodynamic drag polar curves, and fuel consumption across the entire operational flight envelope.

**Licensing & Ethics Compliance**:
- Access to proprietary BADA data files is strictly restricted and requires direct organizational licensing from EUROCONTROL.
- AeroFuel AI does **NOT** copy, redistribute, or use unauthorized BADA tables.
- Instead, AeroFuel AI implements open physics-based energy formulations ($E = \Delta PE + W_{drag}$) calibrated against open literature and the public EUROCONTROL PRC dataset.
""")

st.markdown("---")

# 4. Operational Boundaries
st.subheader("4. Analytical Prototype Boundaries")
st.markdown("""
AeroFuel AI is designed as a **decision-support research prototype** for airline operations research and machine learning engineering portfolios.
- It does **not** command autopilot or FMS systems.
- It does **not** generate certified minimum dispatch fuel or legally binding operational flight plans (OFP).
- All airlines must plan fuel in strict adherence to ICAO Annex 6, FAA FAR 121, EASA Part-CAT, and manufacturer approved flight manuals.
""")
