"""Model artifact persistence: save and load trained models with metadata."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import joblib

_MODEL_DIR_DEFAULT = Path("data/models")


@dataclass
class ModelMetadata:
    """Provenance record saved alongside every model artifact.

    Supports all six model types: logistic, lightgbm, xgboost, catboost,
    mlp, and stacked.
    """

    model_version: str
    model_type: str
    training_seasons: list[int]
    hyperparameters: dict[str, Any]
    feature_set_version: str
    feature_cols: list[str]
    eval_brier: float
    train_n_games: int
    train_brier: float | None = None  # legacy compat for old artifacts


def save_model(
    model: Any,
    meta: ModelMetadata,
    *,
    model_dir: Path = _MODEL_DIR_DEFAULT,
) -> Path:
    """Persist a fitted model and its metadata to *model_dir*.

    Returns
    -------
    Path
        Directory that contains the saved artifact files.
    """
    tag = f"{meta.model_type}_{meta.model_version}_train{meta.training_seasons[-1]}"
    out_dir = model_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_dir / "model.joblib")
    (out_dir / "metadata.json").write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
    return out_dir


def _register_winprob_compat() -> None:
    """Register sys.modules so pickles saved as 'winprob.*' resolve to mlb_predict.*."""
    if "winprob" in sys.modules:
        return
    # Ensure top-level package is loaded so getattr(winprob, 'model') etc. work
    import mlb_predict  # noqa: F401

    sys.modules["winprob"] = sys.modules["mlb_predict"]


def load_model(artifact_dir: Path) -> tuple[Any, ModelMetadata]:
    """Load a model and its metadata from *artifact_dir*.

    Handles legacy metadata that used ``train_brier`` instead of
    ``eval_brier`` by mapping the old field name.

    Supports models pickled when the package was named ``winprob`` by
    registering ``winprob`` -> ``mlb_predict`` in sys.modules before load.

    Returns
    -------
    tuple[model, ModelMetadata]
    """
    _register_winprob_compat()
    model = joblib.load(artifact_dir / "model.joblib")
    raw = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    # Legacy compat: old artifacts have train_brier but not eval_brier
    if "eval_brier" not in raw and "train_brier" in raw:
        raw["eval_brier"] = raw.pop("train_brier")
    meta = ModelMetadata(**raw)
    return model, meta


def latest_artifact(
    model_type: str,
    *,
    model_dir: Path = _MODEL_DIR_DEFAULT,
    version: str = "v1",
) -> Path | None:
    """Return the most recently trained artifact directory for *model_type*.

    Parameters
    ----------
    model_type:
        One of: logistic, lightgbm, xgboost, catboost, mlp, stacked.
    version:
        Model version prefix to match (e.g. ``"v3"``).
    """
    pattern = f"{model_type}_{version}_train*"
    candidates = sorted(model_dir.glob(pattern))
    return candidates[-1] if candidates else None
