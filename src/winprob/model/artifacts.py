"""Model artifact persistence: save and load trained models with metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import joblib

_MODEL_DIR_DEFAULT = Path("data/models")


@dataclass
class ModelMetadata:
    """Provenance record saved alongside every model artifact."""

    model_version: str
    model_type: str          # "logistic" | "lightgbm"
    training_seasons: list[int]
    hyperparameters: dict[str, Any]
    feature_set_version: str
    feature_cols: list[str]
    train_brier: float
    train_n_games: int


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
    (out_dir / "metadata.json").write_text(
        json.dumps(asdict(meta), indent=2), encoding="utf-8"
    )
    return out_dir


def load_model(artifact_dir: Path) -> tuple[Any, ModelMetadata]:
    """Load a model and its metadata from *artifact_dir*.

    Returns
    -------
    tuple[model, ModelMetadata]
    """
    model = joblib.load(artifact_dir / "model.joblib")
    raw = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
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
        ``"logistic"`` or ``"lightgbm"``.
    version:
        Model version prefix to match (e.g. ``"v1"``).
    """
    pattern = f"{model_type}_{version}_train*"
    candidates = sorted(model_dir.glob(pattern))
    return candidates[-1] if candidates else None
