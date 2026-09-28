"""Fuel-Burn Anomaly Detection Engine for AeroFuel AI.

Identifies intervals and flights where observed fuel consumption deviates
materially from model predictions. Uses residual analysis and Isolation Forest.
Flags observations strictly for "Analytical Review" rather than diagnosing mechanical faults.
"""

import logging
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger("AnomalyDetector")


class FuelAnomalyDetector:
    """Detects unusual fuel-burn observations using model residuals and feature isolation."""

    def __init__(self, z_threshold: float = 2.5, contamination: float = 0.03):
        self.z_threshold = z_threshold
        self.contamination = contamination
        self.iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
        )
        self.residual_mean_: float = 0.0
        self.residual_std_: float = 1.0

    def fit(self, X: pd.DataFrame, residuals: np.ndarray):
        """Fit Isolation Forest on feature space combined with prediction residuals."""
        self.residual_mean_ = float(np.mean(residuals))
        self.residual_std_ = float(np.std(residuals)) if np.std(residuals) > 0 else 1.0

        features_with_res = X.copy()
        features_with_res["residual"] = residuals
        self.iso_forest.fit(features_with_res.fillna(0))
        return self

    def detect(self, X: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        """Score each observation and return actionable review flags."""
        residuals = y_true - y_pred
        z_scores = (residuals - self.residual_mean_) / self.residual_std_

        features_with_res = X.copy()
        features_with_res["residual"] = residuals
        iso_preds = self.iso_forest.predict(features_with_res.fillna(0))
        iso_scores = self.iso_forest.score_samples(features_with_res.fillna(0))

        # Severity categorization
        abs_z = np.abs(z_scores)
        severity = []
        is_anomaly = []
        notes = []

        for i, z in enumerate(z_scores):
            is_anom = (abs(z) > self.z_threshold) or (iso_preds[i] == -1)
            is_anomaly.append(is_anom)

            if abs(z) > 4.0:
                sev = "High"
            elif abs(z) > 2.5:
                sev = "Medium"
            elif abs(z) > 1.8:
                sev = "Low"
            else:
                sev = "Normal"
            severity.append(sev)

            # Diagnostic hypothesis for operational review
            if is_anom:
                if residuals[i] > 0:
                    notes.append("Higher than expected burn: Check strong headwinds, ATC vectors, or holding")
                else:
                    notes.append("Lower than expected burn: Check strong tailwinds or continuous descent")
            else:
                notes.append("Nominal operational envelope")

        return pd.DataFrame({
            "observed_fuel_kg": y_true,
            "predicted_fuel_kg": y_pred,
            "residual_kg": residuals,
            "residual_z_score": z_scores,
            "anomaly_score": -iso_scores,  # Higher score = more anomalous
            "is_anomaly": is_anomaly,
            "severity": severity,
            "review_recommendation": notes,
        })
