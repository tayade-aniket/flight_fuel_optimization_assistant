"""Flight Trajectory Segmentation and Phase Identification for AeroFuel AI.

Classifies continuous trajectory points and intervals into operational flight phases:
Climb, Cruise, Descent, and Approach, based on barometric altitude, vertical rate (ROCD),
and groundspeed thresholds according to ICAO / EUROCONTROL operational definitions.
"""

import logging
from typing import Tuple, Optional
import numpy as np
import pandas as pd

from src.utils.constants import (
    PHASE_CLIMB,
    PHASE_CRUISE,
    PHASE_DESCENT,
    PHASE_APPROACH,
    PHASE_UNKNOWN,
)

logger = logging.getLogger("TrajectorySegmentation")


class TrajectorySegmenter:
    """Segments continuous flight trajectories into standard operational phases."""

    def __init__(
        self,
        rocd_climb_threshold_mps: float = 1.5,
        rocd_descent_threshold_mps: float = -1.5,
        cruise_min_altitude_m: float = 6000.0,  # ~FL200
        approach_max_altitude_m: float = 2000.0,
    ):
        self.rocd_climb_thresh = rocd_climb_threshold_mps
        self.rocd_descent_thresh = rocd_descent_threshold_mps
        self.cruise_min_alt = cruise_min_altitude_m
        self.approach_max_alt = approach_max_altitude_m

    def classify_trajectory_points(self, df_traj: pd.DataFrame) -> pd.DataFrame:
        """Classify each point in a trajectory DataFrame into a flight phase."""
        df = df_traj.copy()
        if "vertical_rate" not in df.columns or "altitude" not in df.columns:
            df["flight_phase"] = PHASE_UNKNOWN
            return df

        # Fill missing vertical rate from diff of altitude if necessary
        if df["vertical_rate"].isnull().all() and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            dt = df["timestamp"].diff().dt.total_seconds().replace(0, np.nan)
            df["vertical_rate"] = df["altitude"].diff() / dt

        # Smooth vertical rate with a rolling window to filter high-frequency ADS-B noise
        rocd_smooth = df["vertical_rate"].rolling(window=5, min_periods=1, center=True).mean()

        conditions = [
            # Approach: Low altitude and descending
            (df["altitude"] <= self.approach_max_alt) & (rocd_smooth <= self.rocd_descent_thresh),
            # Climb: Positive vertical rate
            (rocd_smooth >= self.rocd_climb_thresh),
            # Descent: Negative vertical rate
            (rocd_smooth <= self.rocd_descent_thresh),
            # Cruise: High altitude and near-zero vertical rate
            (df["altitude"] >= self.cruise_min_alt) & (rocd_smooth.abs() < self.rocd_climb_thresh),
        ]

        choices = [
            PHASE_APPROACH,
            PHASE_CLIMB,
            PHASE_DESCENT,
            PHASE_CRUISE,
        ]

        df["flight_phase"] = np.select(conditions, choices, default=PHASE_CRUISE)
        return df

    def classify_interval(self, alt_start: float, alt_end: float, mean_rocd: float, duration_sec: float) -> str:
        """Classify a fuel measurement interval based on boundary altitudes and mean ROCD."""
        delta_alt = alt_end - alt_start
        mean_alt = (alt_start + alt_end) / 2.0

        if delta_alt > 1000.0 or mean_rocd > 1.0:
            return PHASE_CLIMB
        elif delta_alt < -1000.0 or mean_rocd < -1.0:
            if mean_alt < self.approach_max_alt:
                return PHASE_APPROACH
            return PHASE_DESCENT
        else:
            return PHASE_CRUISE


def segment_flight_file(parquet_path: str) -> pd.DataFrame:
    """Utility to load a single flight trajectory parquet and classify its phases."""
    df = pd.read_parquet(parquet_path)
    segmenter = TrajectorySegmenter()
    return segmenter.classify_trajectory_points(df)
