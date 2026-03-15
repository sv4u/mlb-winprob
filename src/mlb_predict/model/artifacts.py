"""Model artifact persistence: save and load trained models with metadata.

Supports two training tiers:
- **quick**: bootstrap training (``--skip-cv --no-stage1``), version tag ``v4q``.
  Stored under ``data/models/quick/``.
- **full**: complete CV + Stage 1 pipeline, version tag ``v4``.
  Stored under ``data/models/full/``.

Old models are archived (not deleted) to ``data/models/archive/`` for drift analysis.
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import joblib

_MODEL_DIR_DEFAULT = Path("data/models")


class TrainingTier(str, Enum):
    """Distinguishes quick-bootstrap models from full-pipeline models."""

    QUICK = "quick"
    FULL = "full"


TIER_VERSION_TAG: dict[TrainingTier, str] = {
    TrainingTier.QUICK: "v4q",
    TrainingTier.FULL: "v4",
}


def tier_subdir(model_dir: Path, tier: TrainingTier) -> Path:
    """Return the tier-specific subdirectory for model storage."""
    return model_dir / tier.value


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
    training_tier: str = "full"
    trained_at: str = ""
    train_brier: float | None = None  # legacy compat for old artifacts


def save_model(
    model: Any,
    meta: ModelMetadata,
    *,
    model_dir: Path = _MODEL_DIR_DEFAULT,
    training_tier: TrainingTier = TrainingTier.FULL,
) -> Path:
    """Persist a fitted model and its metadata to the tier-specific subdirectory.

    Quick models are saved to ``model_dir/quick/``, full models to ``model_dir/full/``.
    The version tag in the directory name reflects the tier (``v4q`` vs ``v4``).

    Returns
    -------
    Path
        Directory that contains the saved artifact files.
    """
    meta.training_tier = training_tier.value
    if not meta.trained_at:
        meta.trained_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    version_tag = TIER_VERSION_TAG[training_tier]
    meta.model_version = version_tag

    dest_dir = tier_subdir(model_dir, training_tier)
    tag = f"{meta.model_type}_{version_tag}_train{meta.training_seasons[-1]}"
    out_dir = dest_dir / tag
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


def archive_models(
    model_dir: Path = _MODEL_DIR_DEFAULT,
    tier: TrainingTier | None = None,
) -> int:
    """Move existing model artifacts to ``model_dir/archive/`` for drift analysis.

    When *tier* is given, only archives models in that tier's subdirectory.
    When *tier* is ``None``, archives legacy top-level models.
    Returns the number of artifacts archived.
    """
    archive_dir = model_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    source_dir = tier_subdir(model_dir, tier) if tier else model_dir
    if not source_dir.exists():
        return 0

    count = 0
    for item in sorted(source_dir.iterdir()):
        if not item.is_dir():
            continue
        if item.name in ("quick", "full", "archive"):
            continue
        if not (item / "model.joblib").exists():
            continue
        dest = archive_dir / f"{item.name}_{timestamp}"
        shutil.move(str(item), str(dest))
        count += 1

    return count


def has_trained_models(
    model_dir: Path = _MODEL_DIR_DEFAULT,
    tier: TrainingTier | None = None,
) -> bool:
    """Return True if any trained model artifacts exist.

    When *tier* is given, checks only that tier's subdirectory.
    When ``None``, checks full, then quick, then legacy top-level.
    """
    search_dirs: list[Path] = []
    if tier is not None:
        search_dirs.append(tier_subdir(model_dir, tier))
    else:
        search_dirs.extend(
            [
                tier_subdir(model_dir, TrainingTier.FULL),
                tier_subdir(model_dir, TrainingTier.QUICK),
                model_dir,
            ]
        )

    for d in search_dirs:
        if not d.exists():
            continue
        for item in d.iterdir():
            if item.is_dir() and (item / "model.joblib").exists():
                return True
    return False


def latest_artifact(
    model_type: str,
    *,
    model_dir: Path = _MODEL_DIR_DEFAULT,
    version: str = "v1",
    tier: TrainingTier | None = None,
) -> Path | None:
    """Return the most recently trained artifact directory for *model_type*.

    Parameters
    ----------
    model_type:
        One of: logistic, lightgbm, xgboost, catboost, mlp, stacked.
    version:
        Model version prefix to match (e.g. ``"v3"``).
    tier:
        When given, searches only that tier's subdirectory.
        When ``None``, searches the given *model_dir* directly (legacy behavior).
    """
    search_dir = tier_subdir(model_dir, tier) if tier else model_dir
    pattern = f"{model_type}_{version}_train*"
    candidates = sorted(search_dir.glob(pattern))
    return candidates[-1] if candidates else None


def latest_artifact_best_tier(
    model_type: str,
    *,
    model_dir: Path = _MODEL_DIR_DEFAULT,
) -> tuple[Path | None, TrainingTier | None]:
    """Find the best available artifact, preferring full over quick over legacy.

    Returns (artifact_path, tier) or (None, None) if nothing found.
    """
    full_v = TIER_VERSION_TAG[TrainingTier.FULL]
    quick_v = TIER_VERSION_TAG[TrainingTier.QUICK]

    art = latest_artifact(model_type, model_dir=model_dir, version=full_v, tier=TrainingTier.FULL)
    if art is not None:
        return art, TrainingTier.FULL

    art = latest_artifact(model_type, model_dir=model_dir, version=quick_v, tier=TrainingTier.QUICK)
    if art is not None:
        return art, TrainingTier.QUICK

    for v in (full_v, "v3"):
        art = latest_artifact(model_type, model_dir=model_dir, version=v)
        if art is not None:
            return art, None

    return None, None
