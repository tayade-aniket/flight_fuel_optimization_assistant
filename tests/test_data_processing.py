"""Unit and Integration Tests for Data Processing and Segmentation."""

import pytest
import numpy as np
import pandas as pd

from src.data_processing.segmentation import TrajectorySegmenter
from src.utils.constants import PHASE_CLIMB, PHASE_CRUISE, PHASE_DESCENT, PHASE_APPROACH


def test_trajectory_segmenter_climb():
    """Verify that high positive vertical rate is classified as climb."""
    segmenter = TrajectorySegmenter()
    df_climb = pd.DataFrame({
        "altitude": [1000.0, 1500.0, 2000.0, 2500.0, 3000.0],
        "vertical_rate": [3.0, 3.5, 4.0, 3.8, 3.2],
        "groundspeed": [150.0, 160.0, 170.0, 180.0, 190.0],
    })
    classified = segmenter.classify_trajectory_points(df_climb)
    assert (classified["flight_phase"] == PHASE_CLIMB).all()


def test_trajectory_segmenter_cruise():
    """Verify that high altitude with stable vertical rate is classified as cruise."""
    segmenter = TrajectorySegmenter()
    df_cruise = pd.DataFrame({
        "altitude": [10500.0, 10505.0, 10502.0, 10508.0, 10500.0],
        "vertical_rate": [0.1, -0.1, 0.0, 0.2, -0.2],
        "groundspeed": [230.0, 230.0, 231.0, 229.0, 230.0],
    })
    classified = segmenter.classify_trajectory_points(df_cruise)
    assert (classified["flight_phase"] == PHASE_CRUISE).all()


def test_trajectory_segmenter_descent():
    """Verify that negative vertical rate above approach altitude is descent."""
    segmenter = TrajectorySegmenter()
    df_descent = pd.DataFrame({
        "altitude": [8000.0, 7500.0, 7000.0, 6500.0, 6000.0],
        "vertical_rate": [-3.0, -3.5, -3.2, -3.8, -3.4],
        "groundspeed": [210.0, 205.0, 200.0, 195.0, 190.0],
    })
    classified = segmenter.classify_trajectory_points(df_descent)
    assert (classified["flight_phase"] == PHASE_DESCENT).all()


def test_interval_classification():
    """Verify interval level phase tagging."""
    segmenter = TrajectorySegmenter()
    assert segmenter.classify_interval(alt_start=2000, alt_end=8000, mean_rocd=3.0, duration_sec=600) == PHASE_CLIMB
    assert segmenter.classify_interval(alt_start=10000, alt_end=10000, mean_rocd=0.0, duration_sec=600) == PHASE_CRUISE
    assert segmenter.classify_interval(alt_start=9000, alt_end=4000, mean_rocd=-2.5, duration_sec=600) == PHASE_DESCENT
