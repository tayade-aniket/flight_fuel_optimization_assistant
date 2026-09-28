# EUROCONTROL PRC 2025 Data Challenge - Data Quality & Audit Report

**Project**: ADS-04 — Flight Fuel Optimization Assistant (AeroFuel AI)  
**Dataset**: EUROCONTROL PRC 2025 Aircraft Fuel Burn Estimation Data Challenge  
**Citation**: Sun, Spinielli, & Strohmeier (2026), *Journal of Open Aviation Science*, 4(3)  
**License**: Creative Commons Attribution 4.0 International (CC BY 4.0)  

---

## 1. Executive Summary

| Metric | Raw Dataset | Cleaned Dataset | Retention Rate |
| :--- | :--- | :--- | :--- |
| **Flights** | 11,037 | 10,766 | 97.54% |
| **Fuel Intervals** | 131,530 | 82,674 | 62.86% |
| **Aircraft Types** | 26 | 26 | 100.00% |
| **Origin Airports** | 273 | 273 | 100.00% |
| **Destination Airports**| 260 | 260 | 100.00% |

---

## 2. Integrity & Anomaly Checks

- **Flight ID Duplicates**: 0 (Pass)
- **Fuel Interval Duplicates**: 0 (Pass)
- **Non-positive Fuel Records (<= 0 kg)**: 0
- **Interval Duration Violations (< 5 min)**: 48856
- **Interval Duration Violations (> 60 min)**: 0

---

## 3. Fuel Consumption & Duration Statistics

- **Mean Fuel per Interval**: 768.04 kg
- **Median Fuel per Interval**: 327.00 kg
- **Fuel Standard Deviation**: 1051.42 kg
- **Interval Range**: 8.2 kg – 32205.0 kg
- **Average Interval Duration**: 11.91 minutes

---

## 4. Aviation Operational Quality Notes

1. **ACARS Telemetry Verification**: Fuel consumption labels (`fuel_kg`) are derived from consecutive ACARS Fuel-on-Board (FOB) reports crowdsourced via airframes.io.
2. **Unit Inference & OpenAP Validation**: The dataset curators applied TU Delft's OpenAP physics reference model to resolve imperial (lbs) vs metric (kg) reporting discrepancies and ensure strictly decreasing FOB telemetry.
3. **Trajectory Fusion**: The time intervals strictly align with dense OpenSky Network ADS-B state vectors, ensuring sub-second kinematic ground truth for feature extraction.
