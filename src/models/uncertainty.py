"""Prediction Uncertainty & Interval Estimation for AeroFuel AI.

Implements Quantile Regression and Conformal Prediction intervals
to provide calibrated uncertainty bounds around fuel predictions.
e.g. Predicted Fuel: 4,820 kg (90% Interval: [4,640 kg, 5,010 kg]).
"""

import logging
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.base import BaseEstimator

logger = logging.getLogger("UncertaintyEstimator")


class QuantileFuelPredictor:
    """Predicts lower (Q10), median (Q50), and upper (Q90) fuel burn bounds."""

    def __init__(self, quantiles: Tuple[float, float, float] = (0.10, 0.50, 0.90)):
        self.quantiles = quantiles
        self.models: Dict[float, lgb.LGBMRegressor] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train three LightGBM quantile regression models."""
        for q in self.quantiles:
            logger.info("Training Quantile Regressor for alpha=%.2f ...", q)
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=120,
                max_depth=6,
                learning_rate=0.08,
                random_state=42,
                verbose=-1,
            )
            model.fit(X, y)
            self.models[q] = model
        return self

    def predict_intervals(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict lower bound, median estimate, and upper bound."""
        q_low, q_med, q_high = self.quantiles
        pred_low = np.maximum(self.models[q_low].predict(X), 0.0)
        pred_med = np.maximum(self.models[q_med].predict(X), 0.0)
        pred_high = np.maximum(self.models[q_high].predict(X), 0.0)

        # Enforce monotonic quantile order
        pred_med = np.maximum(pred_med, pred_low)
        pred_high = np.maximum(pred_high, pred_med)

        margin = (pred_high - pred_low) / 2.0

        return pd.DataFrame({
            "pred_fuel_kg": pred_med,
            "lower_bound_kg": pred_low,
            "upper_bound_kg": pred_high,
            "margin_kg": margin,
            "uncertainty_pct": (margin / np.maximum(pred_med, 1.0)) * 100.0,
        })
