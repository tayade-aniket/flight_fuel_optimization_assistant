# AeroFuel AI — Airline ML Engineer Technical Interview Guide

Comprehensive interview preparation guide with authentic, in-depth technical answers tailored for roles at major airlines (United, Delta, British Airways, Lufthansa, Singapore Airlines, Emirates, Air India, IndiGo) and aviation technology companies.

---

## Part 1: Machine Learning & Modeling

### Q1: Why did you choose LightGBM / XGBoost over deep learning or simple linear regression?
> **Answer**:  
> "I started with first-principles baselines: an empirical mean baseline (MAE: 688 kg), a physics energy balance model (MAE: 278 kg, R²: 0.71), and a Ridge linear regression (MAE: 267 kg). While linear regression captures the primary duration-to-fuel relationship, it fails on non-linear aerodynamic phenomena—such as the quadratic rise of parasitic drag with speed, transonic wave drag, and altitude-dependent engine thrust specific fuel consumption (TSFC).  
> Tree-based gradient boosters (LightGBM and XGBoost) cut the baseline MAE down to 149.5 kg (a 43.9% improvement) because decision trees naturally create piecewise step functions that segment different operational regimes (e.g. narrow-body vs wide-body, climb thrust vs idle descent). I evaluated deep learning, but tabular gradient boosting outperformed neural architectures on this tabular feature matrix while training in seconds and offering native tree-based Shapley value computation via TreeSHAP."

### Q2: Why did you evaluate MAE alongside RMSE?
> **Answer**:  
> "RMSE squares errors before averaging, making it disproportionately sensitive to extreme outliers—such as heavy wide-body freighters (B744, B77W) burning 4,000 kg in an interval or flights experiencing long holding patterns. MAE provides an intuitive, robust operational metric representing the average kilogram error an airline dispatcher can expect. Evaluating MedAE and Relative Percentage Error (11.2% in cruise) allowed me to compare accuracy fairly across both short-haul A320 flights and long-haul A350 flights."

### Q3: How did you rigorously prevent data leakage?
> **Answer**:  
> "Data leakage is the biggest trap in aviation telemetry. In the EUROCONTROL dataset, each flight is divided into multiple consecutive intervals. If you use standard random K-Fold splitting, Interval 2 of flight X ends up in train while Interval 3 ends up in test. Because both intervals share identical aircraft empty weight, pilot technique, engine degradation, and ambient winds, the model simply memorizes the flight's baseline fuel rate.  
> To guarantee zero leakage:  
> 1. **Flight-Grouped Splitting**: I used `GroupKFold` grouped strictly by `flight_id`, ensuring no flight ever spans both train and test sets.  
> 2. **Temporal Split**: I validated chronologically (April–August train, September test).  
> 3. **Execution-Time Feature Auditing**: I conducted a feature availability audit ensuring all kinematic features are computed strictly up to interval completion time ($t_{end}$), prohibiting future trajectory points or completed flight totals."

---

## Part 2: Aviation Domain Knowledge & Operations

### Q4: What are the primary factors influencing aircraft fuel burn?
> **Answer**:  
> "Aircraft fuel burn is governed by the Breguet Range equation and flight physics:  
> 1. **Aircraft Mass / Weight**: Lift must equal weight ($L = W$). Heavier aircraft require higher lift coefficients ($C_L$), generating higher lift-induced drag ($D_i \propto W^2$).  
> 2. **Cruising Altitude / Air Density**: Air density ($\rho$) decreases exponentially with altitude. In the upper troposphere (FL330–FL390), lower density drastically reduces parasite drag ($D_p \propto \rho V^2$) and allows jet engines to operate at optimal compressor pressure ratios.  
> 3. **Speed (Mach / True Airspeed)**: Total drag is the sum of induced drag (high at slow speeds) and parasite/wave drag (steeply increasing at high speeds). The minimum drag speed defines maximum endurance.  
> 4. **Flight Phase**: Climb requires maximum continuous thrust to overcome gravity ($\Delta PE$), consuming ~60 kg/min. Cruise requires only enough thrust to balance aerodynamic drag (~38 kg/min). Descent can be flown at flight idle thrust (<15 kg/min) if ATC allows a Continuous Descent Operation (CDO).  
> 5. **Atmospheric Conditions**: Headwinds decrease groundspeed, extending flight duration and increasing trip fuel; tailwinds do the opposite."

### Q5: What is the difference between ADS-B and ACARS in your dataset?
> **Answer**:  
> "They represent complementary data streams:  
> - **ADS-B (Automatic Dependent Surveillance-Broadcast)**: Dense kinematic tracking broadcast by the aircraft transponder at 1090 MHz. Decoded via OpenSky Network, it provides sub-second GPS position, barometric altitude, groundspeed, and track angle.  
> - **ACARS (Aircraft Communications Addressing and Reporting System)**: Digital datalink over VHF or satellite between aircraft avionics and airline ground systems (FOC). It provides operational reports—such as Fuel-on-Board (FOB) readings, wind reports, and gross weight—at coarse intervals (every 5 to 30 minutes).  
> The core data engineering challenge of the EUROCONTROL challenge was pairing sparse ACARS fuel telemetry with continuous OpenSky ADS-B state vectors."

### Q6: Why is flight fuel optimization not simply 'minimizing fuel'?
> **Answer**:  
> "Because airlines do not optimize fuel in isolation; they optimize **total operating cost** (governed by the aircraft Cost Index, $CI = \text{Time Cost} / \text{Fuel Cost}$).  
> If an algorithm minimizes fuel unconditionally, it will choose the slowest aerodynamic speed (minimum drag speed). Across a 500 nm sector, flying 40 knots slower saves ~60 kg of fuel (~$50), but adds 15 minutes of flight time. That delay can trigger missed passenger connections, crew duty-hour overtime ($500+/hr), missed runway arrival slots, and disruption across downstream legs. That is why AeroFuel AI implements **Constrained Multi-Objective Optimization** and plots the **Pareto Trade-off Frontier** between fuel savings and schedule delay."

---

## Part 3: Explainable AI (SHAP)

### Q7: Why use SHAP, and does a high SHAP value prove that changing that feature will physically change fuel burn?
> **Answer**:  
> "I used TreeSHAP because it satisfies the Shapley efficiency, symmetry, and additivity axioms, providing mathematically grounded feature attributions that sum exactly to the difference between the model's prediction and the base expected value.  
> However, **SHAP does not prove physical causation**. SHAP measures the feature's contribution within the model's learned conditional distribution. In aviation, confusing statistical association with physical causation can be hazardous. For example, if a model observes that lower cruising altitudes correlate with lower fuel burn on certain routes, it may merely be capturing the fact that short 100 nm flights never climb above FL240! True physical causation requires aerodynamic wind-tunnel testing, certified aircraft flight manuals (AFM), or BADA performance tables."

---

## Part 4: Production Engineering & Scale

### Q8: Why use Apache Parquet and DuckDB rather than CSVs or a traditional SQL database?
> **Answer**:  
> "Trajectory datasets are columnar and large. A single flight trajectory contains thousands of state vectors, and 15,000 flights produce millions of rows. Storing this in CSV wastes storage and forces slow row-by-row I/O. Apache Parquet uses Snappy/ZSTD compression and columnar layout, reducing file size by >80% and enabling column projection (reading only `altitude` and `groundspeed` without scanning entire files). DuckDB allows in-process SQL execution directly on Parquet files with zero serialization overhead, eliminating the need to maintain an expensive, running database server for local analytics."

### Q9: How would you monitor model drift in airline production?
> **Answer**:  
> "I would monitor four distinct drift categories:  
> 1. **Covariate / Data Drift**: Track Kolmogorov-Smirnov tests and Population Stability Index (PSI) on input features (e.g. shifts in assigned cruising altitudes during winter jet stream seasons).  
> 2. **Concept Drift / Fleet Aging**: Engine thermal efficiency degrades over time, and airframe surface roughness increases aerodynamic drag. Tracking rolling 30-day mean residual ($Observed - Predicted$) by specific airframe tail number will detect airframe-specific performance degradation.  
> 3. **Prediction Drift**: Monitoring the output fuel distribution for unexpected distributional shifts.  
> 4. **Missing Telemetry Alerts**: Monitoring VHF packet dropouts or corrupt ACARS messages."

---

## Part 5: Behavioral & Motivational Answers

### Q10: Why are you interested in Aviation Data Science / Airline ML?
> **Answer**:  
> *"I chose aviation because machine learning here has immediate, tangible physical and economic context. Unlike generic web click-through prediction, flight operations connects aeronautical physics, high-consequence safety constraints, environmental sustainability, and razor-thin airline operating margins. Building AeroFuel AI taught me that creating an impactful model in aviation is not just about hyperparameter tuning to get a lower test RMSE; it requires understanding trajectory timing, respecting operational flight planning constraints, preventing temporal data leakage, and delivering explainable decision support that operations controllers can actually trust."*
