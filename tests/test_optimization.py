"""Unit Tests for Constrained Optimization and Scenario Simulation."""

import pytest
import numpy as np
import pandas as pd

from src.models.baseline import PhysicsInspiredBaseline
from src.optimization.scenario_simulator import WhatIfSimulator
from src.optimization.constrained_optimizer import ConstrainedFuelOptimizer
from src.features.trajectory_features import FeatureExtractor


def test_what_if_simulator_delta_consistency():
    """Verify that what-if simulator produces consistent fuel and time deltas."""
    model = PhysicsInspiredBaseline()
    fe = FeatureExtractor()
    sim = WhatIfSimulator(model, fe.feature_names)

    base = pd.Series(fe._fallback_features("A320", 1800.0))
    res = sim.simulate_scenario(
        baseline_features=base,
        delta_flight_level=20.0,
        delta_speed_kts=-10.0,
        scenario_name="Test Scenario",
    )

    assert "baseline_fuel_kg" in res
    assert "scenario_fuel_kg" in res
    assert res["fuel_delta_kg"] == round(res["scenario_fuel_kg"] - res["baseline_fuel_kg"], 1)
    assert res["co2_delta_kg"] == round(res["fuel_delta_kg"] * 3.16, 1)


def test_constrained_optimizer_delay_bound():
    """Verify that optimizer does not exceed the user-defined maximum time delay constraint."""
    model = PhysicsInspiredBaseline()
    fe = FeatureExtractor()
    optimizer = ConstrainedFuelOptimizer(model, fe.feature_names)

    base = pd.Series(fe._fallback_features("A320", 1800.0))
    max_delay = 5.0  # max 5 minutes delay
    res = optimizer.optimize_cruise(
        baseline_features=base,
        aircraft_type="A320",
        max_time_delay_min=max_delay,
    )

    opt = res["optimal_scenario"]
    assert opt["time_delta_min"] <= max_delay + 0.1
