"""In-memory data and model cache for the web application.

Loads all feature data and the production model once at startup.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PROCESSED_DIR = Path("data/processed")
_MODEL_DIR = Path("data/models")

# Populated at startup
_features: pd.DataFrame | None = None
_model: object | None = None
_meta: object | None = None
_feature_cols: list[str] = []

# Retrosheet code → full name
TEAM_NAMES: dict[str, str] = {
    "ARI": "Arizona Diamondbacks",   "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles",      "BOS": "Boston Red Sox",
    "CHA": "Chicago White Sox",      "CHN": "Chicago Cubs",
    "CIN": "Cincinnati Reds",        "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies",       "DET": "Detroit Tigers",
    "HOU": "Houston Astros",         "KCA": "Kansas City Royals",
    "LAN": "Los Angeles Dodgers",    "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers",      "MIN": "Minnesota Twins",
    "NYA": "New York Yankees",       "NYN": "New York Mets",
    "OAK": "Oakland Athletics",      "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates",     "SDN": "San Diego Padres",
    "SEA": "Seattle Mariners",       "SFN": "San Francisco Giants",
    "SLN": "St. Louis Cardinals",    "TBA": "Tampa Bay Rays",
    "TEX": "Texas Rangers",          "TOR": "Toronto Blue Jays",
    "WAS": "Washington Nationals",   "ANA": "Los Angeles Angels",
    "ATH": "Athletics",              "FLO": "Florida Marlins",
    "MON": "Montreal Expos",
}

TEAM_ABBREVS: dict[str, str] = {
    v: k for k, v in TEAM_NAMES.items()
}  # name → retro code


def startup(model_type: str = "logistic") -> None:
    """Load features and model into memory.  Called once at application startup."""
    global _features, _model, _meta, _feature_cols

    logger.info("Loading feature data…")
    frames = [
        pd.read_parquet(f)
        for f in sorted((_PROCESSED_DIR / "features").glob("features_*.parquet"))
    ]
    if not frames:
        raise RuntimeError("No feature files found.  Run build_features.py first.")
    _features = pd.concat(frames, ignore_index=True)

    # Precompute predictions
    from winprob.model.artifacts import latest_artifact, load_model
    from winprob.model.train import _predict_proba

    art = latest_artifact(model_type, model_dir=_MODEL_DIR, version="v3")
    if art is None:
        raise RuntimeError(f"No production model found for type '{model_type}'.")
    _model, _meta = load_model(art)
    _feature_cols = _meta.feature_cols

    clean = _features.dropna(subset=_feature_cols)
    X = clean[_feature_cols].astype(float)
    probs = _predict_proba(_model, X)
    _features = _features.copy()
    _features["prob"] = np.nan
    _features.loc[clean.index, "prob"] = probs

    logger.info(
        "Loaded %d games (%.0f with probabilities), model=%s",
        len(_features),
        _features["prob"].notna().sum(),
        model_type,
    )


def get_features() -> pd.DataFrame:
    if _features is None:
        raise RuntimeError("Data not loaded.  Call startup() first.")
    return _features


def get_model():
    return _model, _meta, _feature_cols
