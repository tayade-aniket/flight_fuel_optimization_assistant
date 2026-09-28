# AeroFuel AI - Model Benchmark & Comparative Evaluation

**Validation Strategy**: 5-Fold Flight-Grouped Split (Zero flight leakage across folds)  
**Evaluation Target**: Interval Fuel Burn (`fuel_kg`) from ACARS Ground Truth  

---

## 1. Comparative Results Table

| Model | MAE (kg) | RMSE (kg) | R² Score | MedAE (kg) | Mean Rel. Error (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LightGBM** | 149.55 | 364.54 | 0.8729 | 65.30 | 44.09% |
| **Physics-Residual Hybrid** | 153.15 | 371.31 | 0.8682 | 65.77 | 44.69% |
| **XGBoost** | 153.92 | 373.08 | 0.8669 | 66.54 | 45.24% |
| **Random Forest** | 160.12 | 395.37 | 0.8505 | 68.15 | 46.84% |
| **Ridge Regression** | 266.71 | 495.05 | 0.7657 | 136.08 | 75.90% |
| **Physics Energy Baseline** | 277.82 | 549.21 | 0.7116 | 123.28 | 76.26% |
| **Mean Baseline** | 687.99 | 1022.87 | -0.0003 | 564.21 | 329.11% |

---

## 2. Key Insights for Aviation ML Engineers

1. **Baseline vs Ensembles**: Tree-based gradient boosters (XGBoost, LightGBM) significantly outperform ordinary linear regression and mean baselines by capturing non-linear aerodynamic drag rises and altitude-dependent thrust efficiency.
2. **Physics-Informed Hybrid**: The physics energy baseline provides robust bounds and positive energy conservation, while the ML residual booster models aerodynamic variations and operational noise.
3. **Leakage Audit**: All features were verified at interval start and duration, guaranteeing zero future target leakage.
