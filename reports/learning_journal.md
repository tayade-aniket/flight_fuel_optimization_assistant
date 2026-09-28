# Junior ML Engineer Learning Journal: AeroFuel AI

**Project**: ADS-04 — Flight Fuel Optimization Assistant (AeroFuel AI)  
**Developer Profile**: Junior Machine Learning Engineer / Aviation Data Scientist (~1.5 years experience)  
**Target Roles**: ML Engineer — Airline Operations, Aviation Data Scientist, Applied ML Engineer  

---

## 1. What I Initially Tried

When I began this project, my initial instinct was to treat aircraft fuel prediction like a standard Kaggle tabular regression problem:
1. Download flight records and feed origin, destination, planned distance, and aircraft type straight into XGBoost.
2. Optimize unconstrained fuel burn directly using numerical gradient descent.
3. Randomly split all available fuel records into 80% train and 20% test using scikit-learn's standard `train_test_split`.

---

## 2. What Failed and Why

### Failure 1: The Intra-Flight Data Leakage Trap
- **What happened**: When I initially split records randomly, my test set R² was suspiciously high (0.97) and MAE was unrealistically low (~35 kg).
- **Why it failed**: In the EUROCONTROL PRC dataset, each flight is divided into multiple consecutive time intervals (e.g., Interval 3, Interval 4, Interval 5). Standard random splitting placed Interval 3 of flight `prc778174030` into the training set and Interval 4 into the test set! Because consecutive intervals share the exact same aircraft gross weight, pilot flying technique, airframe age, and specific atmospheric winds, the model simply memorized that flight's baseline fuel flow rather than learning true trajectory dynamics.
- **How I fixed it**: I migrated entirely to `GroupKFold(groups=flight_ids)` and temporal holdouts (training on April–August, testing on September). Once I prevented any flight from appearing in both train and test, the test metrics adjusted to a realistic, honest $R^2 = 0.873$ and MAE = $149.5\text{ kg}$.

### Failure 2: Unconstrained "Optimization" Caused Flight Disruption
- **What happened**: When I asked SciPy's `minimize` to find the speed that minimized fuel burn, the solver happily chose 250 knots (minimum allowable clean speed).
- **Why it failed**: In commercial airline operations, an aircraft flying 250 knots across a 500 nm leg adds over 45 minutes of flight time! That wrecks turnaround schedules, causes missed passenger connections, and triggers exorbitant crew overtime and airport slot penalties.
- **How I fixed it**: I realized that in aviation, fuel is never minimized in a vacuum. I reformulated the problem as **Constrained Multi-Objective Optimization**, bounding allowable schedule delays ($\Delta t \le 6\text{ min}$) and plotting the **Pareto Trade-off Frontier** between fuel burn and flight time.

### Failure 3: Unit Inconsistencies in ACARS Messages
- **What happened**: Raw ACARS Fuel-on-Board (FOB) reports in commercial flight operations come from airlines around the world using both imperial (pounds) and metric (kilograms), sometimes scaled by factors of 10 or 100 depending on the avionics bus format.
- **Why it failed**: Treating raw numeric values as kilograms caused severe target outliers.
- **How I fixed it**: By reading the accompanying research paper by Sun et al. (2026), I learned how the dataset curators applied TU Delft's OpenAP aerodynamic model to infer reporting units and enforce monotonic fuel depletion. In my pipeline, I added strict schema assertions and duration filters ($5 \le \text{duration} \le 60\text{ min}$).

---

## 3. What I Learned

1. **Aviation Data is Fundamentally Temporal**: Trajectory state vectors are sequential. Features must be audited at interval execution time to prevent subtle future-target leakage.
2. **Domain Knowledge Directs Feature Engineering**: Feeding raw coordinates into a tree booster is ineffective. Converting coordinates into great-circle distance, calculating true rate of climb/descent (ROCD), and mapping ICAO aircraft types to certified aerodynamic categories (narrow-body vs wide-body) unlocked over 40% improvement in model accuracy.
3. **Physics + ML is Superior to Pure ML**: A simple energy-balance equation ($E = \Delta PE + W_{drag}$) explains >71% of fuel burn variance with zero trained parameters. Pairing this baseline with an ML residual booster creates a hybrid system that guarantees physical sanity while capturing complex aerodynamic non-linearities.
4. **Explainability Builds Operational Trust**: Airline dispatchers and pilots will not trust an opaque prediction. Using SHAP to show that the model rewards higher flight levels due to lower ambient air density confirms that the model is learning physics rather than memorizing dataset quirks.
