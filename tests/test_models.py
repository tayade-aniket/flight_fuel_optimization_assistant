"""Unit Tests for Models, Physics Baselines, and Anomaly Detection."""

import pytest
import numpy as np
import pandas as pd

from src.models.baseline import DummyBaseline, PhysicsInspiredBaseline, LinearBaseline
from src.models.anomaly import FuelAnomalyDetector


def test_physics_baseline_positive_output():
    """Verify that physics energy model produces realistic non-negative fuel values."""
    pb = PhysicsInspiredBaseline()
    df_sample = pd.DataFrame({
        "ref_mass_kg": [65000.0, 75000.0, 200000.0],
        "distance_flown_km": [200.0, 500.0, 1000.0],
        "altitude_change_m": [0.0, 2000.0, -1000.0],
        "duration_min": [15.0, 35.0, 70.0],
        "mean_groundspeed_mps": [220.0, 240.0, 250.0],
    })
    preds = pb.predict(df_sample)
    assert len(preds) == 3
    assert (preds > 0).all()
    # Wide-body (200,000 kg) on 1000km should consume substantially more fuel than narrow-body on 200km
    assert preds[2] > preds[0]


def test_linear_baseline_fit_predict():
    """Verify Linear baseline fits and outputs positive fuel predictions."""
    lb = LinearBaseline()
    X = pd.DataFrame({
        "duration_min": [10.0, 20.0, 30.0, 40.0],
        "distance_flown_km": [100.0, 200.0, 300.0, 400.0],
        "mean_altitude_m": [10000.0, 10000.0, 10000.0, 10000.0],
    })
    y = pd.Series([300.0, 600.0, 900.0, 1200.0])
    lb.fit(X, y)
    preds = lb.predict(X)
    assert len(preds) == 4
    assert (preds >= 0).all()


def test_fuel_anomaly_detector():
    """Verify anomaly detector flags severe deviations."""
    ad = FuelAnomalyDetector(z_threshold=2.0)
    X = pd.DataFrame({
        "duration_min": [15.0, 15.0, 15.0, 15.0, 15.0],
        "distance_flown_km": [150.0, 150.0, 150.0, 150.0, 150.0],
    })
    # residuals with one high outlier
    residuals = np.array([10.0, -15.0, 5.0, -5.0, 800.0])
    y_true = np.array([510.0, 485.0, 505.0, 495.0, 1300.0])
    y_pred = np.array([500.0, 500.0, 500.0, 500.0, 500.0])

    ad.fit(X, residuals)
    results = ad.detect(X, y_true, y_pred)
    assert results.iloc[4]["is_anomaly"] == True
    assert results.iloc[4]["severity"] in ["Medium", "High"]
