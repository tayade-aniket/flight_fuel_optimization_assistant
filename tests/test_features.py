"""Unit Tests for Feature Extraction, Anti-Leakage, and Kinematic Calculations."""

import pytest
import numpy as np
import pandas as pd

from src.features.trajectory_features import FeatureExtractor, haversine_distance, compute_cumulative_distance


def test_haversine_distance():
    """Verify great-circle distance between known coordinates (London to Paris ~344 km)."""
    lat_lon_london = (51.5074, -0.1278)
    lat_lon_paris = (48.8566, 2.3522)
    dist_m = haversine_distance(lat_lon_london[0], lat_lon_london[1], lat_lon_paris[0], lat_lon_paris[1])
    dist_km = dist_m / 1000.0
    assert 340.0 <= dist_km <= 350.0


def test_feature_extractor_output_format():
    """Verify that feature extractor produces all required features and valid types."""
    fe = FeatureExtractor()
    features = fe._fallback_features(aircraft_type="A320", duration_sec=900.0)

    for req_col in fe.feature_names:
        assert req_col in features, f"Missing feature: {req_col}"
        assert not np.isnan(features[req_col]), f"NaN found in feature: {req_col}"

    assert features["duration_min"] == 15.0
    assert features["is_narrowbody"] == 1.0
    assert features["is_widebody"] == 0.0


def test_feature_extractor_from_dense_points():
    """Verify feature calculation from synthetic trajectory dataframe."""
    fe = FeatureExtractor()
    df_traj = pd.DataFrame({
        "timestamp": pd.date_range("2025-06-01 10:00:00", periods=60, freq="10s"),
        "altitude": np.linspace(10000, 10200, 60),
        "groundspeed": np.full(60, 230.0),
        "vertical_rate": np.full(60, 0.33),
        "latitude": np.linspace(48.0, 49.0, 60),
        "longitude": np.linspace(2.0, 3.0, 60),
        "track": np.full(60, 45.0),
    })

    feats = fe.extract_features_from_trajectory(df_traj, "A320", duration_sec=600.0)
    assert feats["duration_min"] == 10.0
    assert feats["mean_altitude_m"] > 10000.0
    assert feats["mean_groundspeed_mps"] == 230.0
    assert feats["distance_flown_nm"] > 50.0
