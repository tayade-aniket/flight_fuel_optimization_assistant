"""Baseline Models for Aircraft Fuel Consumption Prediction.

Implements:
1. Mean / Median Target Baselines
2. Ordinary Least Squares (OLS) Linear Regression
3. Simplified Aircraft Physics Energy Baseline (Lift-to-Drag, Potential Energy change)
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from src.utils.constants import (
    GRAVITY_ACCELERATION,
    JET_A1_LHV_MJ_KG,
)


class DummyBaseline:
    """Predicts historical mean or median fuel burn."""

    def __init__(self, strategy: str = "mean"):
        self.strategy = strategy
        self.value_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if self.strategy == "median":
            self.value_ = float(np.median(y))
        else:
            self.value_ = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.value_)


class PhysicsInspiredBaseline:
    """Simplified physics energy-balance baseline for aircraft interval fuel burn.

    Fuel burn = (Drag * Distance + Delta_PE) / (Engine_Efficiency * Fuel_LHV)
    Where:
      Drag approx Weight / (L/D)
      Delta_PE = Weight * g * Delta_Altitude
    """

    def __init__(self, overall_efficiency: float = 0.32, nominal_lift_to_drag: float = 16.5):
        self.eta = overall_efficiency
        self.lod = nominal_lift_to_drag
        self.lhv_j_kg = JET_A1_LHV_MJ_KG * 1e6  # 43.15 MJ/kg in Joules

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Physics baseline is parameter-free or can calibrate overall efficiency eta
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        mass = X["ref_mass_kg"].values if "ref_mass_kg" in X.columns else np.full(len(X), 65000.0)
        dist_m = (X["distance_flown_km"].values * 1000.0) if "distance_flown_km" in X.columns else (X["mean_groundspeed_mps"].values * X["duration_min"].values * 60.0)
        alt_change = X["altitude_change_m"].values if "altitude_change_m" in X.columns else np.zeros(len(X))

        weight = mass * GRAVITY_ACCELERATION
        drag = weight / self.lod

        work_drag = drag * dist_m
        work_climb = np.maximum(weight * alt_change, 0.0)

        total_work_j = work_drag + work_climb
        fuel_est_kg = total_work_j / (self.eta * self.lhv_j_kg)

        # Baseline clamp to non-negative realistic interval bounds
        return np.maximum(fuel_est_kg, 10.0)


class LinearBaseline:
    """Standardized linear regression baseline for benchmark comparison."""

    def __init__(self):
        self.model = Ridge(alpha=1.0)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X)
        return np.maximum(preds, 0.0)
