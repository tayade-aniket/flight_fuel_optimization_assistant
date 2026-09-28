"""Feature Dataset Assembly Pipeline for AeroFuel AI.

Merges flight metadata, ACARS fuel telemetry ground-truth intervals,
and OpenSky ADS-B trajectory points into a high-performance Parquet feature store.
"""

import os
import logging
from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.features.trajectory_features import FeatureExtractor
from src.utils.constants import AIRCRAFT_CATEGORIES, AIRCRAFT_ENVELOPES, DEFAULT_ENVELOPE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FeatureBuilder")


class FeatureDatasetBuilder:
    """Builds unified ML-ready feature datasets from raw tables and trajectory files."""

    def __init__(
        self,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        sample_size: Optional[int] = None,
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.sample_size = sample_size
        self.extractor = FeatureExtractor()

    def build(self) -> pd.DataFrame:
        """Construct the complete tabular feature store."""
        fl_path = self.processed_dir / "flightlist_cleaned.parquet"
        fuel_path = self.processed_dir / "fuel_cleaned.parquet"
        traj_dir = self.raw_dir / "flights_rank"

        if not fl_path.exists() or not fuel_path.exists():
            # Fallback to raw if cleaned not yet generated
            fl_path = self.raw_dir / "flightlist_train.parquet"
            fuel_path = self.raw_dir / "fuel_train.parquet"

        logger.info("Loading flight metadata from %s and fuel labels from %s ...", fl_path, fuel_path)
        df_fl = pd.read_parquet(fl_path)
        df_fuel = pd.read_parquet(fuel_path)

        # Merge metadata into fuel intervals
        df_merged = df_fuel.merge(
            df_fl[["flight_id", "aircraft_type", "origin_icao", "destination_icao"]],
            on="flight_id",
            how="inner",
        )

        if self.sample_size is not None and len(df_merged) > self.sample_size:
            logger.info("Subsampling %d intervals for rapid training...", self.sample_size)
            df_merged = df_merged.sample(n=self.sample_size, random_state=42).reset_index(drop=True)

        logger.info("Extracting trajectory features for %d fuel intervals...", len(df_merged))

        # Check available trajectory files
        available_traj_files = set()
        if traj_dir.exists():
            available_traj_files = {p.stem: p for p in traj_dir.glob("*.parquet")}
            logger.info("Found %d extracted trajectory parquet files.", len(available_traj_files))

        feature_records = []
        for _, row in tqdm(df_merged.iterrows(), total=len(df_merged), desc="Building features"):
            fid = row["flight_id"]
            ac_type = row["aircraft_type"]
            dur_sec = row.get("duration_sec", row.get("duration_min", 15.0) * 60.0)

            # Check if trajectory file exists for this flight
            traj_file = available_traj_files.get(fid)
            df_traj_interval = None

            if traj_file:
                try:
                    df_traj = pd.read_parquet(traj_file)
                    df_traj["timestamp"] = pd.to_datetime(df_traj["timestamp"])
                    t_start = pd.to_datetime(row["start"])
                    t_end = pd.to_datetime(row["end"])
                    mask = (df_traj["timestamp"] >= t_start) & (df_traj["timestamp"] <= t_end)
                    df_traj_interval = df_traj[mask]
                except Exception:
                    df_traj_interval = None

            feat = self.extractor.extract_features_from_trajectory(df_traj_interval, ac_type, dur_sec)

            # Add flight & target metadata
            feat["flight_id"] = fid
            feat["aircraft_type"] = ac_type
            feat["fuel_kg"] = float(row["fuel_kg"])
            feat["start"] = str(row["start"])
            feat["end"] = str(row["end"])
            feat["origin_icao"] = row.get("origin_icao", "UNKNOWN")
            feat["destination_icao"] = row.get("destination_icao", "UNKNOWN")

            feature_records.append(feat)

        df_features = pd.DataFrame(feature_records)

        # Save to Parquet
        output_path = self.processed_dir / "fuel_features.parquet"
        df_features.to_parquet(output_path, index=False)
        logger.info("Feature dataset successfully built with %d rows, saved to %s", len(df_features), output_path)

        return df_features


if __name__ == "__main__":
    builder = FeatureDatasetBuilder(sample_size=15000)
    builder.build()
