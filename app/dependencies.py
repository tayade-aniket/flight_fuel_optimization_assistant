from functools import lru_cache
from pathlib import Path
from typing import Optional, List, Tuple, Any

import joblib

from src.models.baseline import PhysicsInspiredBaseline
from src.features.trajectory_features import FeatureExtractor


MODEL_PATH = Path("models/best_fuel_model.joblib")
FEAT_NAMES_PATH = Path("models/feature_names.joblib")


@lru_cache(maxsize=1)
def load_model_and_features() -> Tuple[Any, List[str]]:
    # try the trained checkpoint first, fall back to physics baseline
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    else:
        model = PhysicsInspiredBaseline()

    if FEAT_NAMES_PATH.exists():
        feature_names = joblib.load(FEAT_NAMES_PATH)
    else:
        feature_names = FeatureExtractor().feature_names

    return model, feature_names
