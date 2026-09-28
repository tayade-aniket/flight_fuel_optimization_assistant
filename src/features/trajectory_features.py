"""Trajectory-Aware Feature Engineering for AeroFuel AI.

Extracts vertical, horizontal, temporal, phase, and aircraft features from
dense ADS-B trajectory records and aligns them with ACARS fuel intervals.
Enforces strict anti-leakage rules: no future state vectors are used.
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.utils.constants import (
    AIRCRAFT_CATEGORIES,
    AIRCRAFT_FAMILIES,
    AIRCRAFT_ENVELOPES,
    DEFAULT_ENVELOPE,
    METERS_PER_FOOT,
    FEET_PER_METER,
    KNOTS_PER_MPS,
    NM_PER_METER,
    PHASE_CLIMB,
    PHASE_CRUISE,
    PHASE_DESCENT,
    PHASE_APPROACH,
)

logger = logging.getLogger("TrajectoryFeatures")


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points in meters."""
    R = 6371000.0  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def compute_cumulative_distance(lats: np.ndarray, lons: np.ndarray) -> float:
    """Compute total distance along a sequence of coordinates in meters."""
    if len(lats) < 2:
        return 0.0
    dist = 0.0
    for i in range(len(lats) - 1):
        if not (np.isnan(lats[i]) or np.isnan(lons[i]) or np.isnan(lats[i+1]) or np.isnan(lons[i+1])):
            dist += haversine_distance(lats[i], lons[i], lats[i+1], lons[i+1])
    return dist


class FeatureExtractor:
    """Engineers trajectory, kinematic, and operational features for fuel intervals."""

    def __init__(self):
        self.feature_names = [
            "duration_min",
            "mean_altitude_m",
            "mean_flight_level",
            "max_altitude_m",
            "min_altitude_m",
            "altitude_change_m",
            "altitude_std_m",
            "mean_groundspeed_mps",
            "mean_groundspeed_kts",
            "max_groundspeed_kts",
            "groundspeed_std_kts",
            "mean_vertical_rate_mps",
            "vertical_rate_abs_mean",
            "distance_flown_nm",
            "distance_flown_km",
            "track_change_deg_per_min",
            "is_climb",
            "is_cruise",
            "is_descent",
            "is_approach",
            "ref_mass_kg",
            "nominal_mach",
            "is_narrowbody",
            "is_widebody",
            "is_freighter",
        ]

    def extract_features_from_trajectory(
        self,
        df_traj_interval: pd.DataFrame,
        aircraft_type: str,
        duration_sec: float,
    ) -> Dict[str, float]:
        """Extract rich kinematic features from ADS-B points inside a single interval."""
        env = AIRCRAFT_ENVELOPES.get(aircraft_type, DEFAULT_ENVELOPE)
        category = AIRCRAFT_CATEGORIES.get(aircraft_type, "Narrow-body")

        duration_min = max(duration_sec / 60.0, 0.1)

        if df_traj_interval is None or df_traj_interval.empty:
            # Fallback for interval with sparse trajectory
            return self._fallback_features(aircraft_type, duration_sec)

        # Altitude features
        alts = df_traj_interval["altitude"].dropna().values
        if len(alts) > 0:
            mean_alt = float(np.mean(alts))
            max_alt = float(np.max(alts))
            min_alt = float(np.min(alts))
            alt_change = float(alts[-1] - alts[0])
            alt_std = float(np.std(alts))
        else:
            mean_alt = env["opt_fl"] * 100.0 * METERS_PER_FOOT
            max_alt = mean_alt
            min_alt = mean_alt
            alt_change = 0.0
            alt_std = 0.0

        mean_fl = mean_alt * FEET_PER_METER / 100.0

        # Speed features
        speeds = df_traj_interval["groundspeed"].dropna().values
        if len(speeds) > 0:
            mean_speed_mps = float(np.mean(speeds))
            max_speed_mps = float(np.max(speeds))
            speed_std_mps = float(np.std(speeds))
        else:
            mean_speed_mps = 220.0
            max_speed_mps = 240.0
            speed_std_mps = 5.0

        mean_speed_kts = mean_speed_mps * KNOTS_PER_MPS
        max_speed_kts = max_speed_mps * KNOTS_PER_MPS
        speed_std_kts = speed_std_mps * KNOTS_PER_MPS

        # Vertical rate features
        if "vertical_rate" in df_traj_interval.columns:
            rocds = df_traj_interval["vertical_rate"].dropna().values
            mean_rocd = float(np.mean(rocds)) if len(rocds) > 0 else (alt_change / max(duration_sec, 1.0))
            abs_rocd = float(np.mean(np.abs(rocds))) if len(rocds) > 0 else abs(mean_rocd)
        else:
            mean_rocd = alt_change / max(duration_sec, 1.0)
            abs_rocd = abs(mean_rocd)

        # Distance features
        lats = df_traj_interval["latitude"].values if "latitude" in df_traj_interval.columns else np.array([])
        lons = df_traj_interval["longitude"].values if "longitude" in df_traj_interval.columns else np.array([])
        dist_m = compute_cumulative_distance(lats, lons) if len(lats) > 1 else (mean_speed_mps * duration_sec)
        dist_nm = dist_m * NM_PER_METER
        dist_km = dist_m / 1000.0

        # Heading/Track stability
        if "track" in df_traj_interval.columns:
            tracks = df_traj_interval["track"].dropna().values
            if len(tracks) > 1:
                track_diffs = np.abs(np.diff(tracks))
                track_diffs = np.minimum(track_diffs, 360.0 - track_diffs)
                track_rate = float(np.sum(track_diffs) / duration_min)
            else:
                track_rate = 0.0
        else:
            track_rate = 0.0

        # Determine flight phase
        is_climb = 1.0 if (alt_change > 800.0 or mean_rocd > 1.2) else 0.0
        is_descent = 1.0 if (alt_change < -800.0 or mean_rocd < -1.2) and not (mean_alt < 2000.0) else 0.0
        is_approach = 1.0 if (alt_change < -400.0 or mean_rocd < -0.8) and (mean_alt < 2500.0) else 0.0
        is_cruise = 1.0 if (is_climb == 0.0 and is_descent == 0.0 and is_approach == 0.0) else 0.0

        return {
            "duration_min": duration_min,
            "mean_altitude_m": mean_alt,
            "mean_flight_level": mean_fl,
            "max_altitude_m": max_alt,
            "min_altitude_m": min_alt,
            "altitude_change_m": alt_change,
            "altitude_std_m": alt_std,
            "mean_groundspeed_mps": mean_speed_mps,
            "mean_groundspeed_kts": mean_speed_kts,
            "max_groundspeed_kts": max_speed_kts,
            "groundspeed_std_kts": speed_std_kts,
            "mean_vertical_rate_mps": mean_rocd,
            "vertical_rate_abs_mean": abs_rocd,
            "distance_flown_nm": dist_nm,
            "distance_flown_km": dist_km,
            "track_change_deg_per_min": track_rate,
            "is_climb": is_climb,
            "is_cruise": is_cruise,
            "is_descent": is_descent,
            "is_approach": is_approach,
            "ref_mass_kg": env.get("ref_mass_kg", 65000.0),
            "nominal_mach": env.get("nominal_mach", 0.78),
            "is_narrowbody": 1.0 if category == "Narrow-body" else 0.0,
            "is_widebody": 1.0 if category == "Wide-body" else 0.0,
            "is_freighter": 1.0 if category == "Freighter" else 0.0,
        }

    def _fallback_features(self, aircraft_type: str, duration_sec: float) -> Dict[str, float]:
        """Estimated baseline features when continuous trajectory points are unavailable."""
        env = AIRCRAFT_ENVELOPES.get(aircraft_type, DEFAULT_ENVELOPE)
        category = AIRCRAFT_CATEGORIES.get(aircraft_type, "Narrow-body")
        duration_min = max(duration_sec / 60.0, 0.1)
        mean_speed_mps = 225.0
        mean_speed_kts = mean_speed_mps * KNOTS_PER_MPS
        dist_m = mean_speed_mps * duration_sec

        return {
            "duration_min": duration_min,
            "mean_altitude_m": env["opt_fl"] * 100.0 * METERS_PER_FOOT,
            "mean_flight_level": float(env["opt_fl"]),
            "max_altitude_m": env["max_fl"] * 100.0 * METERS_PER_FOOT,
            "min_altitude_m": env["min_fl"] * 100.0 * METERS_PER_FOOT,
            "altitude_change_m": 0.0,
            "altitude_std_m": 10.0,
            "mean_groundspeed_mps": mean_speed_mps,
            "mean_groundspeed_kts": mean_speed_kts,
            "max_groundspeed_kts": mean_speed_kts * 1.05,
            "groundspeed_std_kts": 3.0,
            "mean_vertical_rate_mps": 0.0,
            "vertical_rate_abs_mean": 0.1,
            "distance_flown_nm": dist_m * NM_PER_METER,
            "distance_flown_km": dist_m / 1000.0,
            "track_change_deg_per_min": 0.5,
            "is_climb": 0.0,
            "is_cruise": 1.0,
            "is_descent": 0.0,
            "is_approach": 0.0,
            "ref_mass_kg": env.get("ref_mass_kg", 65000.0),
            "nominal_mach": env.get("nominal_mach", 0.78),
            "is_narrowbody": 1.0 if category == "Narrow-body" else 0.0,
            "is_widebody": 1.0 if category == "Wide-body" else 0.0,
            "is_freighter": 1.0 if category == "Freighter" else 0.0,
        }
