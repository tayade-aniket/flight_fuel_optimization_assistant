# AeroFuel AI — Experiment Log

Structured tracking of modeling iterations, validation strategies, and performance metrics across the development lifecycle.

---

## Experiment 01 — Baseline Mean Prediction
- **Date**: Week 1
- **Hypothesis**: A dummy model predicting the training population mean provides the baseline error lower bound.
- **Model**: `DummyBaseline(strategy="mean")`
- **Features**: None
- **Validation**: 5-Fold Flight-Grouped CV
- **Results**:
  - MAE: 687.99 kg
  - RMSE: 1,022.87 kg
  - R²: -0.0003
- **Conclusion**: Unacceptable variance. Serves solely as reference benchmark.

---

## Experiment 02 — Ordinary Ridge Regression
- **Date**: Week 1
- **Hypothesis**: Linear combination of duration, groundspeed, and distance will capture first-order fuel burn.
- **Model**: `Ridge(alpha=1.0)`
- **Features**: `duration_min`, `distance_flown_km`, `mean_groundspeed_kts`, `mean_flight_level`
- **Validation**: 5-Fold Flight-Grouped CV
- **Results**:
  - MAE: 266.71 kg
  - RMSE: 495.05 kg
  - R²: 0.7657
- **Conclusion**: 61% reduction in MAE compared to Dummy Mean. Captures duration linearity, but fails on step climbs and aircraft weight variations.

---

## Experiment 03 — First-Principles Physics Energy Baseline
- **Date**: Week 2
- **Hypothesis**: An unparameterized physical energy balance ($E = \Delta PE + W_{drag}$) will explain major fuel variance based on Newton's laws.
- **Model**: `PhysicsInspiredBaseline(lod=16.5, eta=0.32)`
- **Features**: `ref_mass_kg`, `distance_flown_km`, `altitude_change_m`
- **Validation**: 5-Fold Flight-Grouped CV
- **Results**:
  - MAE: 277.82 kg
  - RMSE: 549.21 kg
  - R²: 0.7116
- **Conclusion**: Remarkable result: 71.2% of fuel variance explained purely from physics constants without a single trained gradient step.

---

## Experiment 04 — Random Forest with Trajectory Features
- **Date**: Week 2
- **Hypothesis**: Decision tree ensembles will capture non-linear relationships between flight level, vertical rates, and airframe categories.
- **Model**: `RandomForestRegressor(n_estimators=100, max_depth=12)`
- **Features**: 25 trajectory kinematic features + aircraft categorical flags
- **Validation**: 5-Fold Flight-Grouped CV
- **Results**:
  - MAE: 160.12 kg
  - RMSE: 395.37 kg
  - R²: 0.8505
- **Conclusion**: 39.9% MAE improvement over Ridge regression. Trees effectively segment narrow-body vs wide-body regimes.

---

## Experiment 05 — Gradient Boosting (LightGBM & XGBoost)
- **Date**: Week 3
- **Hypothesis**: Gradient-boosted decision trees using histogram binning will optimize split thresholds on continuous kinematic rates (ROCD, track curvature).
- **Models**: `LGBMRegressor` and `XGBRegressor`
- **Features**: Full 25 trajectory features
- **Validation**: 5-Fold Flight-Grouped CV
- **Results**:
  - **LightGBM**: MAE: 149.55 kg | RMSE: 364.54 kg | R²: 0.8729 | MedAE: 78.4 kg
  - **XGBoost**: MAE: 153.92 kg | RMSE: 373.08 kg | R²: 0.8669 | MedAE: 81.2 kg
- **Conclusion**: LightGBM achieved the top overall score, cutting MAE by 43.9% relative to linear regression.

---

## Experiment 06 — Physics-Residual Hybrid Model
- **Date**: Week 3
- **Hypothesis**: Predicting the residual ($Fuel_{actual} - Fuel_{physics}$) using an XGBoost booster guarantees physical baseline positivity and robust out-of-distribution stability.
- **Model**: `PhysicsResidualHybrid` (Physics Baseline + XGBoost Residual Booster)
- **Validation**: 5-Fold Flight-Grouped CV
- **Results**:
  - MAE: 153.15 kg
  - RMSE: 371.31 kg
  - R²: 0.8682
- **Conclusion**: Comparable performance to standalone XGBoost with significantly higher physical interpretability and guaranteed non-zero lower bounds.

---

## Experiment 07 — TreeSHAP Explainability & Bounded Constrained Optimization
- **Date**: Week 4
- **Objective**: Compute exact Shapley values and formulate SciPy bounded cruise optimization with Pareto fuel-time trade-off frontiers.
- **Findings**:
  - Duration and distance account for 58% of global Shapley impact.
  - Cruising at FL360 vs FL320 yields ~4% fuel savings at steady speeds.
  - Pareto frontier proves that speed reductions beyond 6 minutes of schedule delay yield sharply diminishing fuel returns.
