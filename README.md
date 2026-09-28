# ✈️ AeroFuel AI — Explainable Flight Fuel Optimization & Decision Support

[![CI Status](https://github.com/tayade-aniket/flight_fuel_optimization_assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/tayade-aniket/flight_fuel_optimization_assistant/actions)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Dataset: Zenodo](https://img.shields.io/badge/Dataset-Zenodo_19184662-orange.svg)](https://zenodo.org/records/19184662)
[![Streamlit App](https://img.shields.io/badge/Application-Streamlit-FF4B4B.svg)](https://streamlit.io/)

> **Portfolio Project ADS-04**: An aviation-focused machine learning decision-support system that predicts aircraft fuel burn from flight trajectories and operational conditions, explains the major fuel drivers using TreeSHAP, identifies telemetry anomalies, and evaluates fuel-efficient operational scenarios while keeping safety and operational constraints explicit.

---

> [!IMPORTANT]
> **Operational Safety Notice**: This system is an analytical decision-support prototype. It does **not** provide operational flight-planning or safety-critical recommendations, command autopilot/FMS systems, or provide certified minimum fuel calculations. Actual airline fuel planning must follow approved aircraft performance data, operational procedures, dispatch requirements, regulatory requirements, and flight crew/operations-control decisions.

---

## 1. Why This Project Matters to Airlines

Fuel represents **25%–35% of an airline's total operating costs** and is the primary driver of airline carbon emissions ($3.16\text{ kg CO}_2 / \text{kg Jet-A1}$). Even marginal flight efficiency gains ($1\%–2\%$) translate into millions of dollars in annual savings and thousands of tons of averted emissions.

Unlike generic machine learning projects that naively predict fuel from origin, destination, and aircraft type, **AeroFuel AI** operates on **dense 3D trajectory kinematics** (altitude changes, vertical climb/descent rates, true flight path curvature, and groundspeed profiles) paired with real-world ACARS fuel telemetry from over **15,000 commercial flights**.

```
ADS-B Trajectories + ACARS Fuel Telemetry + Flight Metadata
                      ↓ Data Engineering
        Schema Validation & Unit Inference (OpenAP)
                      ↓ Feature Engineering
           Trajectory Kinematics & Phase Segmentation
                      ↓ Fuel-Burn Prediction
   Baselines + Tree Ensembles (XGBoost/LightGBM) + Physics Hybrid
                      ↓ Explainable AI
                TreeSHAP Feature Attributions
                      ↓ Optimization
       Constrained Bounded Solver (Fuel vs Time Trade-off)
                      ↓ Decision Support
            Multi-Page Interactive Streamlit Application
```

---

## 2. Research Dataset & Provenance

AeroFuel AI is built on the official **EUROCONTROL Performance Review Commission (PRC) 2025 Aircraft Fuel Burn Estimation Data Challenge**, organized in collaboration with TU Delft and the OpenSky Network.

- **Dataset Archive**: [Zenodo Record 19184662](https://zenodo.org/records/19184662) (CC BY 4.0)
- **Scientific Publication**:  
  *Sun, J., Spinielli, E., & Strohmeier, M. (2026). Aircraft Fuel Burn Estimation: The EUROCONTROL PRC 2025 Data Challenge. Journal of Open Aviation Science, 4(3).* [doi:10.59490/joas.2026.8750](https://doi.org/10.59490/joas.2026.8750)
- **Telemetry Scale**: **193,275 real ACARS fuel consumption intervals** and dense ADS-B trajectory vectors derived from **15,761 flights** across **27 aircraft types** (April–October 2025).
- **Reporting Unit Inference**: Raw ACARS VHF reports (which mix lbs, kg, and non-standard telemetry scaling) were verified against TU Delft's OpenAP aerodynamic model to ensure strictly decreasing Fuel-on-Board (FOB) measurements.

---

## 3. Machine Learning Modeling & Benchmarks

All models were evaluated using **5-Fold Flight-Grouped Cross-Validation (`GroupKFold` on `flight_id`)** to strictly eliminate intra-flight data leakage across folds.

| Model Architecture | MAE (kg) | RMSE (kg) | R² Score | MedAE (kg) | Rel. Error (%) | Operational Characteristics |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Mean Baseline** | 687.99 | 1,022.87 | -0.0003 | 610.2 | 186.4% | Unacceptable error lower bound |
| **Physics Energy Baseline** | 277.82 | 549.21 | 0.7116 | 185.0 | 48.2% | Unparameterized first-principles energy balance ($E = \Delta PE + W_{drag}$) |
| **Ridge Linear Regression** | 266.71 | 495.05 | 0.7657 | 178.4 | 42.1% | Captures duration linearity; fails on transonic drag rise |
| **Random Forest** | 160.12 | 395.37 | 0.8505 | 88.5 | 16.8% | Tree depth 12; segments narrow-body vs wide-body regimes |
| **XGBoost Regressor** | 153.92 | 373.08 | 0.8669 | 81.2 | 14.1% | Gradient boosted trees; robust on continuous kinematic features |
| **Physics-Residual Hybrid** | 153.15 | 371.31 | 0.8682 | 80.5 | 13.9% | Physics Energy Baseline + XGBoost Residual Booster |
| **LightGBM Regressor** 🏆 | **149.55** | **364.54** | **0.8729** | **78.4** | **13.2%** | **Top Model**: 43.9% MAE reduction over linear baseline |

*Evaluation Target: Interval Fuel Burn (`fuel_kg`) across 15,000 validated test observations.*

---

## 4. Strict Data Leakage Prevention

In aviation machine learning, subtle target leakage can produce deceptive 99% test accuracy. AeroFuel AI enforces strict execution-time auditing:

| Feature | Available at Prediction Time? | Status | Audit Enforcement |
| :--- | :--- | :--- | :--- |
| **Aircraft Type / Engine Family** | Yes (Filed Flight Plan) | Allowed | Static flight plan metadata |
| **Planned Interval Duration** | Yes (Trajectory Slice) | Allowed | Target interval definition |
| **Interval Flight Level (FL)** | Yes (Trajectory Slice) | Allowed | Target kinematic state |
| **Interval Groundspeed** | Yes (Trajectory Slice) | Allowed | Target kinematic state |
| **Historical Fleet Fuel Rates** | Yes (Historical Training Folds) | Allowed | Grouped historical statistics |
| **Future Flight Trajectory** | **No** (Occurs after interval) | **PROHIBITED** | Features computed strictly up to $t_{end}$ |
| **Subsequent ACARS FOB Telemetry**| **No** (Future telemetry) | **PROHIBITED** | Excluded from input feature store |
| **Actual Full-Flight Total Fuel** | **No** (Post-flight actual) | **PROHIBITED** | Excluded from input feature store |

---

## 5. Constrained Multi-Objective Optimization & Pareto Frontier

In real airline operations, **fuel is never minimized in a vacuum**. Minimizing fuel unconditionally selects the slowest clean airspeed, causing severe flight delays that trigger missed passenger connections, crew duty-hour overtime, and airport slot penalties.

AeroFuel AI formulates **Bounded Constrained Optimization**:
$$\min_{\text{FL}, V} \text{Estimated Fuel Burn}(\text{FL}, V)$$
$$\text{subject to } \text{FL} \in [\text{FL}_{\min}, \text{FL}_{\max}], \quad V \in [V_{\min}, V_{\max}], \quad \Delta t \le \Delta t_{\max}$$

The system computes the **Pareto Optimal Frontier**:

| Max Allowable Delay ($\Delta t$) | Optimal Flight Level | Optimal Cruise Speed | Est. Fuel Burn | Fuel Saved | Flight Delay | CO₂ Averted |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0.0 min (On-Time)** | FL350 | 450 kts | 1,420.5 kg | 0.0 kg (0.0%) | 0.0 min | 0.0 kg |
| **2.0 min** | FL360 | 442 kts | 1,385.2 kg | 35.3 kg (2.5%) | +1.8 min | 111.5 kg |
| **4.0 min** | FL370 | 435 kts | 1,360.8 kg | 59.7 kg (4.2%) | +3.6 min | 188.7 kg |
| **6.0 min (Sweet Spot)** | FL370 | 428 kts | 1,348.1 kg | 72.4 kg (5.1%) | +5.5 min | 228.8 kg |
| **10.0 min** | FL380 | 420 kts | 1,342.0 kg | 78.5 kg (5.5%) | +9.2 min | 247.1 kg |

*Operational Takeaway*: Beyond ~6 minutes of delay tolerance, marginal fuel savings flatten significantly due to baseline aerodynamic parasite drag.

---

## 6. Interactive Web Application (Streamlit)

AeroFuel AI features a comprehensive 8-page interactive web application:

1. **Executive Dashboard (`streamlit_app.py`)**: High-level aviation KPIs, system architecture diagram, fleet distribution, and safety notices.
2. **🎯 Predict Fuel (`1_Predict_Fuel.py`)**: Interval-level fuel prediction with calibrated 90% confidence intervals and top model-associated drivers.
3. **🗺️ 3D Trajectory (`2_Trajectory_Analysis.py`)**: Interactive 3D flight paths (Plotly), vertical climb/descent profiles, and phase segmentation.
4. **📊 Efficiency Benchmarks (`3_Efficiency_Benchmarks.py`)**: Normalized fleet KPIs ($kg/nm$, $kg/hr$) across aircraft types and operational phases.
5. **⚠️ Anomaly Detection (`4_Anomaly_Detection.py`)**: Residual tracking and Isolation Forest flagging unexpected fuel burn for operational review.
6. **🔬 What-If Simulator (`5_What_If_Simulator.py`)**: Real-time simulation of flight level, speed, and distance variations with cost and CO₂ deltas.
7. **⚡ Constrained Optimizer (`6_Constrained_Optimization.py`)**: Bounded solver generating Pareto fuel-time trade-off curves.
8. **💡 Explainable AI (`7_Explainable_AI.py`)**: TreeSHAP global feature attributions and partial dependence insights.
9. **📜 Methodology & Safety (`8_Methodology_Safety.py`)**: Dataset provenance, anti-leakage audit, BADA context, and non-operational disclaimers.

---

## 7. 13-Part Jupyter Notebook Series

The `notebooks/` directory contains 13 standard-compliant research notebooks, each structured with the mandatory 8-part junior developer standard:

| Notebook | Title & Topic | Key Finding |
| :--- | :--- | :--- |
| `01_dataset_understanding.ipynb` | Ingestion & Schema Pairing | Ingested 15,761 flights with 193k fuel intervals |
| `02_data_quality_analysis.ipynb` | Quality Audit & Unit Inference | Cleaned 82,674 valid intervals [5, 60] min |
| `03_trajectory_processing.ipynb` | Flight Phase Segmentation | Segmented Climb, Cruise, Descent, and Approach |
| `04_exploratory_fuel_analysis.ipynb` | Aviation KPI Analysis | Widebody fuel intensity is ~3.2x narrowbody |
| `05_feature_engineering.ipynb` | Kinematic Features & Anti-Leakage | 25 trajectory features; verified 0 future leakage |
| `06_baseline_fuel_model.ipynb` | Physics Energy Balance | Physics baseline explains 71.2% variance |
| `07_ml_model_comparison.ipynb` | Multi-Model Benchmark | LightGBM achieves top MAE (149.5 kg, R² 0.873) |
| `08_fuel_prediction.ipynb` | Segment-Level Predictions | Cruise error lowest (11.2%); descent variance highest |
| `09_shap_explainability.ipynb` | TreeSHAP Feature Attribution | Duration, distance, and altitude explain 82% SHAP |
| `10_anomaly_detection.ipynb` | Fuel Anomaly Detection | Flagged 3.4% intervals ($|z| > 2.5$) for review |
| `11_scenario_simulation.ipynb` | What-If Simulator | Perturbed FL and speed; evaluated fuel/time/CO₂ |
| `12_fuel_optimization.ipynb` | Constrained Optimization | Solved bounded Pareto trade-off curve |
| `13_final_evaluation.ipynb` | Cross-Type Generalization | Documented domain shift across airframe types |

---

## 8. Quickstart & Deployment

### Local Installation
```bash
# Clone the repository
git clone https://github.com/tayade-aniket/flight_fuel_optimization_assistant.git
cd flight_fuel_optimization_assistant

# Install dependencies
pip install -r requirements.txt

# Run automated tests
pytest -v tests/

# Launch the interactive Streamlit application
streamlit run app/streamlit_app.py
```

### Docker Deployment
```bash
# Build the Docker image
docker build -t aerofuel-ai .

# Run the container
docker run -p 8501:8501 aerofuel-ai
```

Access the dashboard at `http://localhost:8501`.

---

## 9. Developer Profile & Authenticity

This project was developed independently as a portfolio demonstration for **Junior Machine Learning Engineer / Aviation Data Scientist** roles (~1.5 years experience). 

- **Learning Journal**: See [`reports/learning_journal.md`](reports/learning_journal.md) for authentic documentation of early failures, data leakage troubleshooting, and operational lessons learned.
- **Experiment Log**: See [`reports/experiment_log.md`](reports/experiment_log.md) for iteration tracking from baseline to hybrid models.
- **Technical Interview Prep**: See [`reports/interview_qa.md`](reports/interview_qa.md) for in-depth answers covering ML, aviation physics, explainability, and production engineering.
- **Senior ML Review**: See [`reports/senior_ml_review.md`](reports/senior_ml_review.md) for the complete engineering audit.
