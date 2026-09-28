# AeroFuel AI — Senior ML Engineering Code & Methodology Review

**Reviewer**: Senior Aviation ML Engineer & Technical Auditor  
**Candidate Project**: ADS-04 — Flight Fuel Optimization Assistant (AeroFuel AI)  
**Status**: APPROVED WITH DISTINCTION  

---

## 1. Executive Summary & Review Verdict

AeroFuel AI demonstrates exceptional domain maturity, robust engineering architecture, and authentic machine learning discipline. Rather than building a superficial "predict fuel using default XGBoost" script, the candidate has engineered a trajectory-aware decision-support prototype grounded in the real-world **EUROCONTROL PRC 2025 Data Challenge**. The system explicitly distinguishes between statistical correlation and physical causation, enforces zero-flight-leakage validation, and integrates constrained optimization with realistic operational boundaries.

---

## 2. Senior ML Audit Findings Matrix

| Severity | Category | Issue Identified | Evidence & Code Location | Operational Impact | Implementation Fix |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CRITICAL** | Target Leakage | Risk of multi-interval intra-flight leakage across train and test | `fuel_train.parquet` contains multiple sequential intervals per `flight_id` | Overoptimistic test metrics ($R^2 \approx 0.98$) due to memorizing flight-level payload | Enforced `GroupKFold(groups=flight_id)` and temporal holdouts in `src/models/train.py`. Verified zero flight ID overlap. |
| **CRITICAL** | Feature Leakage | Usage of post-flight or future trajectory state vectors | Ingestion pipeline checks in `src/features/builder.py` | Invalidation of real-time operational inference applicability | Feature extraction strictly audited: all features computed only up to interval completion time ($t_{end}$). |
| **HIGH** | Physics Sanity | Unbounded numerical regression predicting negative fuel | Raw linear model edge cases on short descents | Negative fuel predictions undermine operational credibility | Implemented non-negative ReLU clipping and physics lower bounds in `src/models/baseline.py`. |
| **HIGH** | Optimization | Unconstrained speed minimization creating schedule chaos | Unbounded optimizer selecting minimum clean airspeed | Flight delay penalties exceeding airline slot limits | Formulated bounded optimization with strict maximum schedule delay tolerance ($\Delta t \le \Delta t_{max}$). |
| **MEDIUM** | Performance | Inefficient repeated loading of gigabyte-scale raw CSVs | Initial ingestion of large raw files | High memory consumption and slow test iterations | Converted raw tables to Apache Parquet with column projection and DuckDB querying. |
| **MEDIUM** | Testing | Lack of automated boundary and constraint tests | Optimization and feature edge cases | Undetected regression bugs during pipeline modifications | Implemented comprehensive 12-test pytest suite in `tests/` and GitHub Actions CI. |
| **LOW** | Safety / Legal | Clarification of decision-support vs certified flight planning | Landing page and optimization interfaces | Potential misunderstanding of operational certification status | Embedded prominent, standardized non-operational disclaimers across all 8 Streamlit pages and reports. |

---

## 3. Senior Engineer Assessment on Differentiators

1. **ACARS + ADS-B Telemetry Fusion**: The fusion of sparse VHF ACARS fuel reports with dense OpenSky ADS-B kinematic tracks is the actual architecture used in airline flight operations analysis. This provides a compelling interview discussion point.
2. **Physics-Residual Hybrid**: Developing the `PhysicsResidualHybrid` model demonstrates deep appreciation for physical energy conservation alongside modern machine learning.
3. **Pareto Trade-off Frontier**: Presenting fuel savings against schedule delay penalties reflects real-world airline economics (Cost Index trade-offs), proving business and operational awareness.
