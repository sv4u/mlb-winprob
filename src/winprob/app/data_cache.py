"""In-memory data and model cache for the web application.

Loads all feature data and the production model once at startup.
Uses a threading lock to prevent concurrent-request corruption during
hot reloads.  Supports runtime model switching via ``switch_model()``.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
_MODEL_DIR = _REPO_ROOT / "data" / "models"

_lock = threading.Lock()

_features: pd.DataFrame | None = None
_model: object | None = None
_meta: object | None = None
_feature_cols: list[str] = []
_git_commit: str = "unknown"
_active_model_type: str = "stacked"
_app_ready: bool = False


def _resolve_git_commit() -> str:
    """Read the current HEAD commit hash (short) from env, baked-in file, or git."""
    env_val = os.environ.get("GIT_COMMIT", "").strip()
    if env_val and env_val != "unknown":
        return env_val[:12]
    stamp_file = _REPO_ROOT / "GIT_COMMIT"
    if stamp_file.exists():
        file_val = stamp_file.read_text().strip()[:12]
        if file_val and file_val != "unknown":
            return file_val
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


def get_active_model_type() -> str:
    """Return the model type currently loaded in memory."""
    return _active_model_type


def available_model_types() -> list[str]:
    """Return the list of model types that have trained artifacts on disk."""
    from winprob.model.artifacts import latest_artifact

    types = []
    for mt in ("logistic", "lightgbm", "xgboost", "catboost", "mlp", "stacked"):
        if latest_artifact(mt, model_dir=_MODEL_DIR, version="v3") is not None:
            types.append(mt)
    return types


def switch_model(model_type: str) -> None:
    """Hot-swap the active model and re-score all games.

    Thread-safe: acquires the global lock, loads new model artifacts,
    recomputes probabilities, then releases the lock.
    Clears response and game-detail caches so new model is reflected.
    """
    global _features, _model, _meta, _feature_cols, _active_model_type

    t0 = time.monotonic()
    logger.info("Switching active model to '%s' …", model_type)
    from winprob.model.artifacts import latest_artifact, load_model
    from winprob.model.train import _predict_proba

    art = latest_artifact(model_type, model_dir=_MODEL_DIR, version="v3")
    if art is None:
        raise RuntimeError(f"No trained artifact found for model type '{model_type}'.")

    with _lock:
        prev_model, prev_meta = _model, _meta
        prev_feature_cols = _feature_cols
        prev_active = _active_model_type

        try:
            _model, _meta = load_model(art)
            _feature_cols = _meta.feature_cols
            _active_model_type = model_type

            if _features is not None:
                updated = _features.copy()
                updated["prob"] = np.nan
                has_all_cols = all(c in updated.columns for c in _feature_cols)
                if has_all_cols:
                    predictable = updated[_feature_cols].notna().all(axis=1)
                    if predictable.any():
                        X = updated.loc[predictable, _feature_cols].astype(float)
                        probs = _predict_proba(_model, X)
                        updated.loc[predictable, "prob"] = probs
                _features = updated
        except Exception:
            _model, _meta = prev_model, prev_meta
            _feature_cols = prev_feature_cols
            _active_model_type = prev_active
            raise

    # Only invalidate caches after a successful switch so failed attempts don't wipe them
    from winprob.app.game_detail_cache import clear_game_detail_cache
    from winprob.app.response_cache import clear_response_cache

    clear_response_cache()
    clear_game_detail_cache()

    elapsed_ms = (time.monotonic() - t0) * 1000
    logger.info(
        "Model switched to '%s' — %d games re-scored in %.0fms.",
        model_type,
        int(_features["prob"].notna().sum()) if _features is not None else 0,
        elapsed_ms,
    )


def is_ready() -> bool:
    """Return True if data and model are loaded and the app can serve requests."""
    return _app_ready


def startup(model_type: str = "logistic") -> None:
    """Load features and model into memory.  Called once at application startup."""
    global _features, _model, _meta, _feature_cols, _git_commit, _active_model_type, _app_ready

    t0 = time.monotonic()
    _git_commit = _resolve_git_commit()
    _active_model_type = model_type

    logger.info("Loading feature data…")
    with _lock:
        frames = [
            pd.read_parquet(f)
            for f in sorted((_PROCESSED_DIR / "features").glob("features_*.parquet"))
        ]
        if not frames:
            raise RuntimeError("No feature files found.  Run build_features.py first.")
        _features = pd.concat(frames, ignore_index=True)
        _features["date"] = pd.to_datetime(_features["date"], errors="coerce").dt.date
        if "game_type" not in _features.columns:
            _features["game_type"] = "R"
        else:
            _features["game_type"] = _features["game_type"].fillna("R")
        if "is_spring" not in _features.columns:
            _features["is_spring"] = 0.0
        else:
            _features["is_spring"] = _features["is_spring"].fillna(0.0)

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

        _app_ready = True
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "Loaded %d games (%.0f with probabilities), model=%s in %.0fms",
            len(_features),
            _features["prob"].notna().sum(),
            model_type,
            elapsed_ms,
        )

    # Invalidate caches so responses reflect this load (reload after pipeline / auto-bootstrap)
    from winprob.app.game_detail_cache import clear_game_detail_cache
    from winprob.app.response_cache import clear_response_cache

    clear_response_cache()
    clear_game_detail_cache()


def try_startup(model_type: str = "logistic") -> bool:
    """Attempt startup; return True if data loaded, False if data/model missing."""
    global _git_commit
    _git_commit = _resolve_git_commit()
    try:
        startup(model_type)
        return True
    except RuntimeError as exc:
        logger.warning("Startup deferred — data not ready: %s", exc)
        return False


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
