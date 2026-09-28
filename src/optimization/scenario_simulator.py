"""What-If Fuel Simulator and Scenario Comparison Engine for AeroFuel AI.

Allows operational analysts to evaluate candidate operational adjustments
(e.g., cruising at FL370 vs FL330, flying 15 knots slower, or rerouting),
predicting fuel burn delta, flight time delta, and indicative CO2 emissions.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from src.utils.constants import (
    CO2_PER_KG_FUEL,
    DEFAULT_FUEL_PRICE_USD_PER_KG,
    SYSTEM_DISCLAIMER,
    KNOTS_PER_MPS,
    METERS_PER_FOOT,
    FEET_PER_METER,
)


class WhatIfSimulator:
    """Simulates operational flight scenarios and evaluates fuel, time, and CO2 deltas."""

    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names

    def simulate_scenario(
        self,
        baseline_features: pd.Series,
        delta_flight_level: float = 0.0,
        delta_speed_kts: float = 0.0,
        delta_distance_nm: float = 0.0,
        scenario_name: str = "Scenario",
        fuel_price_per_kg: float = DEFAULT_FUEL_PRICE_USD_PER_KG,
    ) -> Dict[str, Any]:
        """Evaluate a scenario perturbation against the baseline."""
        feat_dict = baseline_features.to_dict()

        # Compute baseline prediction
        baseline_df = pd.DataFrame([feat_dict])[self.feature_names].fillna(0)
        baseline_fuel = float(self.model.predict(baseline_df)[0])
        baseline_duration_min = float(feat_dict.get("duration_min", 30.0))

        # Apply perturbations
        scen_dict = feat_dict.copy()

        # Altitude adjustment
        cur_fl = scen_dict.get("mean_flight_level", 350.0)
        new_fl = max(100.0, min(430.0, cur_fl + delta_flight_level))
        scen_dict["mean_flight_level"] = new_fl
        scen_dict["mean_altitude_m"] = new_fl * 100.0 * METERS_PER_FOOT
        scen_dict["max_altitude_m"] = max(scen_dict["max_altitude_m"], scen_dict["mean_altitude_m"])

        # Speed adjustment
        cur_speed_kts = scen_dict.get("mean_groundspeed_kts", 450.0)
        new_speed_kts = max(250.0, min(550.0, cur_speed_kts + delta_speed_kts))
        scen_dict["mean_groundspeed_kts"] = new_speed_kts
        scen_dict["mean_groundspeed_mps"] = new_speed_kts / KNOTS_PER_MPS

        # Distance adjustment
        cur_dist_nm = scen_dict.get("distance_flown_nm", 200.0)
        new_dist_nm = max(10.0, cur_dist_nm + delta_distance_nm)
        scen_dict["distance_flown_nm"] = new_dist_nm
        scen_dict["distance_flown_km"] = new_dist_nm * 1.852

        # Recompute estimated flight time based on new distance and new speed
        # Duration = Distance / Speed
        new_duration_hr = new_dist_nm / new_speed_kts
        new_duration_min = new_duration_hr * 60.0
        scen_dict["duration_min"] = new_duration_min

        # Compute scenario prediction
        scen_df = pd.DataFrame([scen_dict])[self.feature_names].fillna(0)
        scen_fuel = float(self.model.predict(scen_df)[0])

        # Deltas
        fuel_diff_kg = scen_fuel - baseline_fuel
        fuel_diff_pct = (fuel_diff_kg / max(baseline_fuel, 1.0)) * 100.0

        time_diff_min = new_duration_min - baseline_duration_min
        time_diff_pct = (time_diff_min / max(baseline_duration_min, 0.1)) * 100.0

        co2_diff_kg = fuel_diff_kg * CO2_PER_KG_FUEL
        cost_diff_usd = fuel_diff_kg * fuel_price_per_kg

        return {
            "scenario_name": scenario_name,
            "baseline_fuel_kg": round(baseline_fuel, 1),
            "scenario_fuel_kg": round(scen_fuel, 1),
            "fuel_delta_kg": round(fuel_diff_kg, 1),
            "fuel_delta_pct": round(fuel_diff_pct, 2),
            "baseline_time_min": round(baseline_duration_min, 1),
            "scenario_time_min": round(new_duration_min, 1),
            "time_delta_min": round(time_diff_min, 1),
            "co2_delta_kg": round(co2_diff_kg, 1),
            "cost_delta_usd": round(cost_diff_usd, 2),
            "flight_level": round(new_fl, 0),
            "groundspeed_kts": round(new_speed_kts, 1),
            "distance_nm": round(new_dist_nm, 1),
            "disclaimer": SYSTEM_DISCLAIMER,
        }
