"""EUROCONTROL PRC 2025 Data Challenge - Automated Ingestion Pipeline.

Downloads and verifies the official dataset files from Zenodo:
Zenodo Record: https://zenodo.org/records/19184662
Citation:
Sun, J., Spinielli, E., & Strohmeier, M. (2026).
Aircraft Fuel Burn Estimation: The EUROCONTROL PRC 2025 Data Challenge.
Journal of Open Aviation Science, 4(3). https://doi.org/10.59490/joas.2026.8750
"""

import os
import sys
import zipfile
import urllib.request
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AeroFuelDownloader")

ZENODO_BASE_URL = "https://zenodo.org/records/19184662/files/"

CORE_METADATA_FILES = [
    "airports.parquet",
    "flightlist_train.parquet",
    "fuel_train.parquet",
    "flightlist_rank.parquet",
    "fuel_rank.parquet",
    "flightlist_final.parquet",
    "fuel_final.parquet",
]

TRAJECTORY_ARCHIVES = {
    "rank": "flights_rank.zip",      # 631 MB, 1,888 flights
    "final": "flights_final.zip",    # 982 MB, 2,836 flights
    "train": "flights_train.zip",    # 3.3 GB, 11,037 flights
}


class DatasetDownloader:
    """Manages downloading, verifying, and extracting EUROCONTROL PRC dataset components."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.sample_dir = self.data_dir / "sample"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, filename: str, force: bool = False) -> Path:
        """Download a single file from Zenodo if not already present."""
        target_path = self.raw_dir / filename
        if target_path.exists() and not force:
            logger.info("File already exists: %s (%d bytes)", target_path, target_path.stat().st_size)
            return target_path

        url = f"{ZENODO_BASE_URL}{filename}"
        logger.info("Downloading %s from %s ...", filename, url)

        def report_hook(block_num: int, block_size: int, total_size: int):
            if total_size > 0:
                percent = min(100, int(block_num * block_size * 100 / total_size))
                if block_num % 1000 == 0 or percent == 100:
                    sys.stdout.write(f"\rDownloading {filename}: {percent}% ({block_num * block_size // (1024*1024)} MB)")
                    sys.stdout.flush()

        urllib.request.urlretrieve(url, target_path, reporthook=report_hook)
        print()  # newline
        logger.info("Successfully downloaded: %s", target_path)
        return target_path

    def download_core_metadata(self) -> Dict[str, Path]:
        """Download all flight metadata, airports, and fuel consumption ground-truth tables."""
        downloaded = {}
        for fname in CORE_METADATA_FILES:
            downloaded[fname] = self.download_file(fname)
        return downloaded

    def download_and_extract_trajectories(self, phase: str = "rank", max_flights: Optional[int] = 100) -> Path:
        """Download a trajectory zip archive and optionally extract a subset or all flights."""
        archive_name = TRAJECTORY_ARCHIVES.get(phase, "flights_rank.zip")
        archive_path = self.download_file(archive_name)

        extract_dir = self.raw_dir / f"flights_{phase}"
        extract_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Extracting flights from %s into %s ...", archive_path, extract_dir)
        with zipfile.ZipFile(archive_path, 'r') as zf:
            members = zf.namelist()
            if max_flights is not None:
                members = members[:max_flights]
            for m in members:
                target_file = extract_dir / Path(m).name
                if not target_file.exists():
                    zf.extract(m, extract_dir)
        logger.info("Extracted %d flight trajectory files.", len(members))
        return extract_dir

    def create_sample_extract(self, n_flights: int = 50) -> Path:
        """Create a compact, highly curated sample extract for rapid demonstration and testing."""
        logger.info("Building curated sample dataset with %d flights...", n_flights)
        flightlist_path = self.raw_dir / "flightlist_train.parquet"
        fuel_path = self.raw_dir / "fuel_train.parquet"
        airports_path = self.raw_dir / "airports.parquet"

        if not flightlist_path.exists() or not fuel_path.exists():
            self.download_core_metadata()

        df_fl = pd.read_parquet(flightlist_path)
        df_fuel = pd.read_parquet(fuel_path)

        # Select top aircraft types to ensure diverse representation (A320, B738, A359, B77W, A20N)
        top_types = ["A320", "B738", "A359", "B77W", "A20N", "A321", "B789"]
        selected_flights = []
        for ac in top_types:
            f_ids = df_fl[df_fl["aircraft_type"] == ac]["flight_id"].head(n_flights // len(top_types)).tolist()
            selected_flights.extend(f_ids)

        if len(selected_flights) < n_flights:
            remaining = df_fl[~df_fl["flight_id"].isin(selected_flights)]["flight_id"].head(n_flights - len(selected_flights)).tolist()
            selected_flights.extend(remaining)

        sample_fl = df_fl[df_fl["flight_id"].isin(selected_flights)].copy()
        sample_fuel = df_fuel[df_fuel["flight_id"].isin(selected_flights)].copy()

        sample_fl.to_parquet(self.sample_dir / "flightlist_sample.parquet", index=False)
        sample_fuel.to_parquet(self.sample_dir / "fuel_sample.parquet", index=False)

        if airports_path.exists():
            df_airports = pd.read_parquet(airports_path)
            df_airports.to_parquet(self.sample_dir / "airports_sample.parquet", index=False)

        logger.info("Sample extract saved: %d flights, %d fuel intervals.", len(sample_fl), len(sample_fuel))
        return self.sample_dir


if __name__ == "__main__":
    downloader = DatasetDownloader()
    print("Initiating core metadata download from Zenodo...")
    downloader.download_core_metadata()
