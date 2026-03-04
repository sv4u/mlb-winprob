"""In-memory data and model cache for the web application.

Loads all feature data and the production model once at startup.
Uses a threading lock to prevent concurrent-request corruption during
hot reloads.
"""

from __future__ import annotations

import logging
import subprocess
import threading
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Resolve paths relative to this file so the app works regardless of CWD.
# Layout: src/winprob/app/data_cache.py → repo root is four levels up.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
_MODEL_DIR = _REPO_ROOT / "data" / "models"

_lock = threading.Lock()

# Populated at startup
_features: pd.DataFrame | None = None
_model: object | None = None
_meta: object | None = None
_feature_cols: list[str] = []
_git_commit: str = "unknown"


def _resolve_git_commit() -> str:
    """Read the current HEAD commit hash (short) from git or a baked-in file."""
    stamp_file = _REPO_ROOT / "GIT_COMMIT"
    if stamp_file.exists():
        return stamp_file.read_text().strip()[:12]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def get_git_commit() -> str:
    """Return the resolved git commit hash."""
    return _git_commit


# Retrosheet code → full name
TEAM_NAMES: dict[str, str] = {
    "ARI": "Arizona Diamondbacks",
    "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox",
    "CHA": "Chicago White Sox",
    "CHN": "Chicago Cubs",
    "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies",
    "DET": "Detroit Tigers",
    "HOU": "Houston Astros",
    "KCA": "Kansas City Royals",
    "LAN": "Los Angeles Dodgers",
    "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins",
    "NYA": "New York Yankees",
    "NYN": "New York Mets",
    "OAK": "Oakland Athletics",
    "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates",
    "SDN": "San Diego Padres",
    "SEA": "Seattle Mariners",
    "SFN": "San Francisco Giants",
    "SLN": "St. Louis Cardinals",
    "TBA": "Tampa Bay Rays",
    "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays",
    "WAS": "Washington Nationals",
    "ANA": "Los Angeles Angels",
    "ATH": "Athletics",
    "FLO": "Florida Marlins",
    "MON": "Montreal Expos",
}

TEAM_ABBREVS: dict[str, str] = {v: k for k, v in TEAM_NAMES.items()}  # name → retro code


def startup(model_type: str = "logistic") -> None:
    """Load features and model into memory.  Called once at application startup."""
    global _features, _model, _meta, _feature_cols, _git_commit

    _git_commit = _resolve_git_commit()

    logger.info("Loading feature data…")
    with _lock:
        frames = [
            pd.read_parquet(f)
            for f in sorted((_PROCESSED_DIR / "features").glob("features_*.parquet"))
        ]
        if not frames:
            raise RuntimeError("No feature files found.  Run build_features.py first.")
        _features = pd.concat(frames, ignore_index=True)
        # Normalize date column to datetime.date across all seasons (guards against
        # features_2026.parquet having string dates from a previous run).
        _features["date"] = pd.to_datetime(_features["date"], errors="coerce").dt.date

        from winprob.model.artifacts import latest_artifact, load_model
        from winprob.model.train import _predict_proba

        art = latest_artifact(model_type, model_dir=_MODEL_DIR, version="v3")
        if art is None:
            raise RuntimeError(f"No production model found for type '{model_type}'.")
        _model, _meta = load_model(art)
        _feature_cols = _meta.feature_cols

        _features = _features.copy()
        _features["prob"] = np.nan

        has_all_cols = all(c in _features.columns for c in _feature_cols)
        if has_all_cols:
            predictable = _features[_feature_cols].notna().all(axis=1)
            if predictable.any():
                X = _features.loc[predictable, _feature_cols].astype(float)
                probs = _predict_proba(_model, X)
                _features.loc[predictable, "prob"] = probs
        else:
            missing = [c for c in _feature_cols if c not in _features.columns]
            logger.warning("Feature columns missing from data: %s", missing)

        logger.info(
            "Loaded %d games (%.0f with probabilities), model=%s",
            len(_features),
            _features["prob"].notna().sum(),
            model_type,
        )


def get_features() -> pd.DataFrame:
    """Return the cached features DataFrame."""
    with _lock:
        if _features is None:
            raise RuntimeError("Data not loaded.  Call startup() first.")
        return _features


def get_model() -> tuple[object | None, object | None, list[str]]:
    """Return the cached (model, metadata, feature_cols) tuple."""
    with _lock:
        return _model, _meta, _feature_cols
