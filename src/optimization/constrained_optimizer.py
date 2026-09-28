"""Constrained Operational Fuel Optimization for AeroFuel AI.

Formulates and solves:
  Minimize Estimated Fuel Burn (kg)
  Subject to:
    1. Flight Level bounds: FL_min <= FL <= FL_max (aircraft envelope & RVSM)
    2. Cruise Speed bounds: V_min <= V <= V_max (aerodynamic limits)
    3. Maximum allowable schedule delay: Delta_t <= Delta_t_max
    4. Maximum deviation from baseline trajectory

Computes Pareto trade-off curves balancing Fuel Savings, Flight Duration, and CO2 emissions.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.utils.constants import (
    AIRCRAFT_ENVELOPES,
    DEFAULT_ENVELOPE,
    CO2_PER_KG_FUEL,
    DEFAULT_FUEL_PRICE_USD_PER_KG,
    SYSTEM_DISCLAIMER,
    KNOTS_PER_MPS,
    METERS_PER_FOOT,
)
from src.optimization.scenario_simulator import WhatIfSimulator

logger = logging.getLogger("ConstrainedOptimizer")


class ConstrainedFuelOptimizer:
    """Solves bounded operational optimization to minimize estimated fuel burn."""

    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.simulator = WhatIfSimulator(model, feature_names)

    def optimize_cruise(
        self,
        baseline_features: pd.Series,
        aircraft_type: str,
        max_time_delay_min: float = 8.0,
        max_fl_change: float = 40.0,      # Max +/- 4,000 ft (4 flight levels)
        max_speed_change_kts: float = 30.0, # Max +/- 30 kts
    ) -> Dict[str, Any]:
        """Find optimal cruising altitude and speed within operational constraints."""
        env = AIRCRAFT_ENVELOPES.get(aircraft_type, DEFAULT_ENVELOPE)
        base_fl = float(baseline_features.get("mean_flight_level", env["opt_fl"]))
        base_speed = float(baseline_features.get("mean_groundspeed_kts", 450.0))

        # Enforce realistic bounds
        fl_lower = max(env["min_fl"], base_fl - max_fl_change)
        fl_upper = min(env["max_fl"], base_fl + max_fl_change)

        speed_lower = max(380.0, base_speed - max_speed_change_kts)
        speed_upper = min(510.0, base_speed + max_speed_change_kts)

        # Standard RVSM cruise flight levels (increments of 10 or 20)
        candidate_fls = np.arange(int(fl_lower // 10) * 10, int(fl_upper // 10) * 10 + 10, 10)
        candidate_speeds = np.linspace(speed_lower, speed_upper, 9)

        best_result = None
        min_fuel = float("inf")
        evaluated_scenarios = []

        for fl in candidate_fls:
            for spd in candidate_speeds:
                delta_fl = fl - base_fl
                delta_spd = spd - base_speed

                scen = self.simulator.simulate_scenario(
                    baseline_features=baseline_features,
                    delta_flight_level=delta_fl,
                    delta_speed_kts=delta_spd,
                    scenario_name=f"FL{int(fl)} @ {int(spd)}kts",
                )

                # Check constraint: max allowable time delay
                time_delay = scen["time_delta_min"]
                if time_delay <= max_time_delay_min:
                    evaluated_scenarios.append(scen)
                    if scen["scenario_fuel_kg"] < min_fuel:
                        min_fuel = scen["scenario_fuel_kg"]
                        best_result = scen

        if best_result is None:
            # Fallback to baseline
            best_result = self.simulator.simulate_scenario(baseline_features, scenario_name="Baseline Constrained")

        return {
            "optimal_scenario": best_result,
            "evaluated_count": len(evaluated_scenarios),
            "max_delay_constraint_min": max_time_delay_min,
            "disclaimer": SYSTEM_DISCLAIMER,
        }

    def compute_pareto_frontier(
        self,
        baseline_features: pd.Series,
        aircraft_type: str,
        delay_thresholds: List[float] = [0.0, 2.0, 5.0, 8.0, 12.0, 15.0],
    ) -> pd.DataFrame:
        """Generate Fuel vs Time Trade-off Pareto Frontier."""
        frontier_points = []
        for delay in delay_thresholds:
            res = self.optimize_cruise(
                baseline_features=baseline_features,
                aircraft_type=aircraft_type,
                max_time_delay_min=delay,
            )
            opt = res["optimal_scenario"]
            frontier_points.append({
                "max_delay_min": delay,
                "optimal_fl": opt["flight_level"],
                "optimal_speed_kts": opt["groundspeed_kts"],
                "estimated_fuel_kg": opt["scenario_fuel_kg"],
                "fuel_saved_kg": -opt["fuel_delta_kg"],
                "fuel_saved_pct": -opt["fuel_delta_pct"],
                "time_penalty_min": max(opt["time_delta_min"], 0.0),
                "co2_saved_kg": -opt["co2_delta_kg"],
                "cost_saved_usd": -opt["cost_delta_usd"],
            })

        df_pareto = pd.DataFrame(frontier_points).drop_duplicates(subset=["estimated_fuel_kg"])
        return df_pareto.sort_values(by="time_penalty_min")
