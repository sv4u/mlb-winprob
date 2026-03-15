"""Unit tests for tiered model artifact storage (TrainingTier, save/load, archive).

Covers the two-tier model storage system: quick (bootstrap, v4q) and
full (complete pipeline, v4), plus archiving for drift analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from mlb_predict.model.artifacts import (
    ModelMetadata,
    TrainingTier,
    TIER_VERSION_TAG,
    archive_models,
    has_trained_models,
    latest_artifact,
    latest_artifact_best_tier,
    load_model,
    save_model,
    tier_subdir,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metadata(
    model_type: str = "logistic",
    version: str = "v4",
    seasons: list[int] | None = None,
) -> ModelMetadata:
    """Create a minimal ModelMetadata for testing."""
    return ModelMetadata(
        model_version=version,
        model_type=model_type,
        training_seasons=seasons or [2020, 2021, 2022],
        hyperparameters={"C": 1.0},
        feature_set_version="v4",
        feature_cols=["home_elo", "away_elo"],
        eval_brier=0.24,
        train_n_games=5000,
    )


class _DummyModel:
    """Minimal picklable model stand-in for save/load tests."""

    def __init__(self, label: str = "test") -> None:
        self.label = label

    def predict_proba(self, X):  # noqa: N803
        import numpy as np
        return np.column_stack([1 - X[:, 0], X[:, 0]])


# ---------------------------------------------------------------------------
# TrainingTier enum
# ---------------------------------------------------------------------------


class TestTrainingTier:
    """Tests for the TrainingTier enum and version tag mapping."""

    def test_enum_values(self) -> None:
        """TrainingTier has 'quick' and 'full' string values."""
        assert TrainingTier.QUICK.value == "quick"
        assert TrainingTier.FULL.value == "full"

    def test_construct_from_string(self) -> None:
        """TrainingTier can be constructed from string values."""
        assert TrainingTier("quick") is TrainingTier.QUICK
        assert TrainingTier("full") is TrainingTier.FULL

    def test_invalid_tier_raises(self) -> None:
        """Invalid tier strings raise ValueError."""
        with pytest.raises(ValueError):
            TrainingTier("invalid")

    def test_version_tag_mapping(self) -> None:
        """TIER_VERSION_TAG maps quick→v4q and full→v4."""
        assert TIER_VERSION_TAG[TrainingTier.QUICK] == "v4q"
        assert TIER_VERSION_TAG[TrainingTier.FULL] == "v4"


# ---------------------------------------------------------------------------
# tier_subdir
# ---------------------------------------------------------------------------


class TestTierSubdir:
    """Tests for tier_subdir path construction."""

    def test_quick_subdir(self, tmp_path: Path) -> None:
        """Quick tier goes to model_dir/quick/."""
        result = tier_subdir(tmp_path, TrainingTier.QUICK)
        assert result == tmp_path / "quick"

    def test_full_subdir(self, tmp_path: Path) -> None:
        """Full tier goes to model_dir/full/."""
        result = tier_subdir(tmp_path, TrainingTier.FULL)
        assert result == tmp_path / "full"


# ---------------------------------------------------------------------------
# save_model with tier
# ---------------------------------------------------------------------------


class TestSaveModelTier:
    """Tests for save_model with training tier routing."""

    def test_save_quick_model_goes_to_quick_dir(self, tmp_path: Path) -> None:
        """Quick models are saved under model_dir/quick/ with v4q version tag."""
        meta = _make_metadata(model_type="logistic", seasons=[2020, 2021, 2022])
        model = _DummyModel("quick-lr")

        out = save_model(model, meta, model_dir=tmp_path, training_tier=TrainingTier.QUICK)

        assert out.parent == tmp_path / "quick"
        assert "v4q" in out.name
        assert "logistic" in out.name
        assert (out / "model.joblib").exists()
        assert (out / "metadata.json").exists()

    def test_save_full_model_goes_to_full_dir(self, tmp_path: Path) -> None:
        """Full models are saved under model_dir/full/ with v4 version tag."""
        meta = _make_metadata(model_type="xgboost", seasons=[2020, 2021, 2022])
        model = _DummyModel("full-xgb")

        out = save_model(model, meta, model_dir=tmp_path, training_tier=TrainingTier.FULL)

        assert out.parent == tmp_path / "full"
        assert "v4_train" in out.name
        assert "xgboost" in out.name

    def test_metadata_has_tier_and_timestamp(self, tmp_path: Path) -> None:
        """Saved metadata includes training_tier and trained_at fields."""
        meta = _make_metadata()
        save_model(_DummyModel(), meta, model_dir=tmp_path, training_tier=TrainingTier.QUICK)

        saved = json.loads((tmp_path / "quick" / "logistic_v4q_train2022" / "metadata.json").read_text())
        assert saved["training_tier"] == "quick"
        assert saved["trained_at"] != ""
        assert saved["model_version"] == "v4q"

    def test_save_overwrites_version_tag(self, tmp_path: Path) -> None:
        """save_model updates model_version in metadata to match the tier."""
        meta = _make_metadata(version="v3")
        save_model(_DummyModel(), meta, model_dir=tmp_path, training_tier=TrainingTier.FULL)

        assert meta.model_version == "v4"
        assert meta.training_tier == "full"

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """A model saved with a tier can be loaded back with correct metadata."""
        meta = _make_metadata(model_type="stacked", seasons=[2020, 2021, 2025])
        original = _DummyModel("roundtrip")

        out = save_model(original, meta, model_dir=tmp_path, training_tier=TrainingTier.FULL)
        loaded_model, loaded_meta = load_model(out)

        assert loaded_model.label == "roundtrip"
        assert loaded_meta.model_type == "stacked"
        assert loaded_meta.training_tier == "full"
        assert loaded_meta.model_version == "v4"
        assert loaded_meta.training_seasons == [2020, 2021, 2025]


# ---------------------------------------------------------------------------
# latest_artifact with tier
# ---------------------------------------------------------------------------


class TestLatestArtifactTier:
    """Tests for latest_artifact with tier-specific search."""

    @pytest.fixture
    def model_dir(self, tmp_path: Path) -> Path:
        """Populate a model directory with quick and full artifacts."""
        for tier, version in [("quick", "v4q"), ("full", "v4")]:
            for mt in ["logistic", "stacked"]:
                d = tmp_path / tier / f"{mt}_{version}_train2024"
                d.mkdir(parents=True)
                (d / "model.joblib").write_text("mock")
                (d / "metadata.json").write_text("{}")
        return tmp_path

    def test_find_quick_artifact(self, model_dir: Path) -> None:
        """latest_artifact with tier=QUICK finds models in quick/."""
        art = latest_artifact("logistic", model_dir=model_dir, version="v4q", tier=TrainingTier.QUICK)
        assert art is not None
        assert "quick" in str(art)
        assert "v4q" in art.name

    def test_find_full_artifact(self, model_dir: Path) -> None:
        """latest_artifact with tier=FULL finds models in full/."""
        art = latest_artifact("stacked", model_dir=model_dir, version="v4", tier=TrainingTier.FULL)
        assert art is not None
        assert "full" in str(art)

    def test_no_match_returns_none(self, model_dir: Path) -> None:
        """Returns None when no matching artifact exists in the tier."""
        art = latest_artifact("catboost", model_dir=model_dir, version="v4", tier=TrainingTier.FULL)
        assert art is None

    def test_legacy_search_without_tier(self, tmp_path: Path) -> None:
        """Without tier, searches the model_dir root (legacy behavior)."""
        d = tmp_path / "logistic_v4_train2023"
        d.mkdir(parents=True)
        (d / "model.joblib").write_text("mock")

        art = latest_artifact("logistic", model_dir=tmp_path, version="v4")
        assert art is not None
        assert art.name == "logistic_v4_train2023"


# ---------------------------------------------------------------------------
# latest_artifact_best_tier
# ---------------------------------------------------------------------------


class TestLatestArtifactBestTier:
    """Tests for latest_artifact_best_tier tier preference logic."""

    def test_prefers_full_over_quick(self, tmp_path: Path) -> None:
        """When both tiers have artifacts, prefers full."""
        for tier, version in [("quick", "v4q"), ("full", "v4")]:
            d = tmp_path / tier / f"logistic_{version}_train2024"
            d.mkdir(parents=True)
            (d / "model.joblib").write_text("mock")

        art, tier = latest_artifact_best_tier("logistic", model_dir=tmp_path)
        assert art is not None
        assert tier == TrainingTier.FULL
        assert "full" in str(art)

    def test_falls_back_to_quick(self, tmp_path: Path) -> None:
        """When only quick artifacts exist, returns quick tier."""
        d = tmp_path / "quick" / "stacked_v4q_train2024"
        d.mkdir(parents=True)
        (d / "model.joblib").write_text("mock")

        art, tier = latest_artifact_best_tier("stacked", model_dir=tmp_path)
        assert art is not None
        assert tier == TrainingTier.QUICK

    def test_falls_back_to_legacy(self, tmp_path: Path) -> None:
        """When only legacy top-level artifacts exist, returns them with tier=None."""
        d = tmp_path / "xgboost_v4_train2023"
        d.mkdir(parents=True)
        (d / "model.joblib").write_text("mock")

        art, tier = latest_artifact_best_tier("xgboost", model_dir=tmp_path)
        assert art is not None
        assert tier is None

    def test_returns_none_when_nothing_exists(self, tmp_path: Path) -> None:
        """Returns (None, None) when no artifacts exist at all."""
        art, tier = latest_artifact_best_tier("logistic", model_dir=tmp_path)
        assert art is None
        assert tier is None

    def test_legacy_v3_fallback(self, tmp_path: Path) -> None:
        """Falls back to legacy v3 artifacts when no v4 exist."""
        d = tmp_path / "lightgbm_v3_train2022"
        d.mkdir(parents=True)
        (d / "model.joblib").write_text("mock")

        art, tier = latest_artifact_best_tier("lightgbm", model_dir=tmp_path)
        assert art is not None
        assert art.name == "lightgbm_v3_train2022"


# ---------------------------------------------------------------------------
# archive_models
# ---------------------------------------------------------------------------


class TestArchiveModels:
    """Tests for archive_models drift-preserving archival."""

    def test_archive_quick_tier(self, tmp_path: Path) -> None:
        """Archiving quick tier moves models to archive/ with timestamp."""
        d = tmp_path / "quick" / "logistic_v4q_train2024"
        d.mkdir(parents=True)
        (d / "model.joblib").write_text("mock")

        count = archive_models(tmp_path, tier=TrainingTier.QUICK)

        assert count == 1
        assert not d.exists()
        archived = list((tmp_path / "archive").iterdir())
        assert len(archived) == 1
        assert "logistic_v4q_train2024" in archived[0].name

    def test_archive_full_tier(self, tmp_path: Path) -> None:
        """Archiving full tier moves models to archive/ with timestamp."""
        d = tmp_path / "full" / "stacked_v4_train2025"
        d.mkdir(parents=True)
        (d / "model.joblib").write_text("mock")
        (d / "metadata.json").write_text("{}")

        count = archive_models(tmp_path, tier=TrainingTier.FULL)

        assert count == 1
        assert not d.exists()
        archived = list((tmp_path / "archive").iterdir())
        assert len(archived) == 1

    def test_archive_does_not_touch_other_tier(self, tmp_path: Path) -> None:
        """Archiving one tier leaves the other intact."""
        for tier, version in [("quick", "v4q"), ("full", "v4")]:
            d = tmp_path / tier / f"logistic_{version}_train2024"
            d.mkdir(parents=True)
            (d / "model.joblib").write_text("mock")

        archive_models(tmp_path, tier=TrainingTier.QUICK)

        assert not (tmp_path / "quick" / "logistic_v4q_train2024").exists()
        assert (tmp_path / "full" / "logistic_v4_train2024").exists()

    def test_archive_skips_non_model_dirs(self, tmp_path: Path) -> None:
        """Directories without model.joblib are not archived."""
        (tmp_path / "quick" / "not_a_model").mkdir(parents=True)
        (tmp_path / "quick" / "not_a_model" / "readme.txt").write_text("hi")

        count = archive_models(tmp_path, tier=TrainingTier.QUICK)
        assert count == 0

    def test_archive_empty_dir_returns_zero(self, tmp_path: Path) -> None:
        """Returns 0 when the tier directory is empty or doesn't exist."""
        count = archive_models(tmp_path, tier=TrainingTier.QUICK)
        assert count == 0

    def test_archive_legacy_models(self, tmp_path: Path) -> None:
        """Archiving with tier=None archives legacy top-level models."""
        d = tmp_path / "logistic_v4_train2023"
        d.mkdir(parents=True)
        (d / "model.joblib").write_text("mock")

        count = archive_models(tmp_path, tier=None)

        assert count == 1
        assert not d.exists()

    def test_archive_preserves_special_dirs(self, tmp_path: Path) -> None:
        """quick/, full/, and archive/ directories are not themselves archived."""
        for name in ["quick", "full", "archive"]:
            (tmp_path / name).mkdir(parents=True)

        count = archive_models(tmp_path, tier=None)
        assert count == 0
        assert (tmp_path / "quick").exists()
        assert (tmp_path / "full").exists()
        assert (tmp_path / "archive").exists()

    def test_multiple_archives_get_unique_names(self, tmp_path: Path) -> None:
        """Multiple archives of the same model produce distinct archive entries."""
        import time

        d = tmp_path / "quick" / "logistic_v4q_train2024"
        d.mkdir(parents=True)
        (d / "model.joblib").write_text("mock-1")
        archive_models(tmp_path, tier=TrainingTier.QUICK)

        time.sleep(0.01)  # ensure different timestamp
        d.mkdir(parents=True)
        (d / "model.joblib").write_text("mock-2")
        archive_models(tmp_path, tier=TrainingTier.QUICK)

        archived = list((tmp_path / "archive").iterdir())
        assert len(archived) >= 1


# ---------------------------------------------------------------------------
# has_trained_models
# ---------------------------------------------------------------------------


class TestHasTrainedModels:
    """Tests for has_trained_models detection across tiers."""

    def test_no_models_returns_false(self, tmp_path: Path) -> None:
        """Returns False when no model directories exist."""
        assert has_trained_models(tmp_path) is False

    def test_full_model_detected(self, tmp_path: Path) -> None:
        """Returns True when a full-tier model exists."""
        d = tmp_path / "full" / "logistic_v4_train2024"
        d.mkdir(parents=True)
        (d / "model.joblib").write_text("mock")

        assert has_trained_models(tmp_path) is True

    def test_quick_model_detected(self, tmp_path: Path) -> None:
        """Returns True when a quick-tier model exists."""
        d = tmp_path / "quick" / "logistic_v4q_train2024"
        d.mkdir(parents=True)
        (d / "model.joblib").write_text("mock")

        assert has_trained_models(tmp_path) is True

    def test_legacy_model_detected(self, tmp_path: Path) -> None:
        """Returns True when a legacy top-level model exists."""
        d = tmp_path / "logistic_v4_train2023"
        d.mkdir(parents=True)
        (d / "model.joblib").write_text("mock")

        assert has_trained_models(tmp_path) is True

    def test_specific_tier_check(self, tmp_path: Path) -> None:
        """Only checks the specified tier when tier argument is given."""
        d = tmp_path / "full" / "logistic_v4_train2024"
        d.mkdir(parents=True)
        (d / "model.joblib").write_text("mock")

        assert has_trained_models(tmp_path, tier=TrainingTier.FULL) is True
        assert has_trained_models(tmp_path, tier=TrainingTier.QUICK) is False

    def test_archive_not_counted(self, tmp_path: Path) -> None:
        """Models in archive/ are not counted as active trained models."""
        d = tmp_path / "archive" / "logistic_v4_train2024_20260315"
        d.mkdir(parents=True)
        (d / "model.joblib").write_text("mock")

        assert has_trained_models(tmp_path) is False


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Tests for loading legacy metadata without tier/timestamp fields."""

    def test_load_legacy_metadata_missing_tier(self, tmp_path: Path) -> None:
        """Legacy metadata without training_tier defaults to 'full'."""
        d = tmp_path / "logistic_v4_train2023"
        d.mkdir(parents=True)

        import joblib
        joblib.dump(_DummyModel("legacy"), d / "model.joblib")
        legacy_meta = {
            "model_version": "v4",
            "model_type": "logistic",
            "training_seasons": [2020, 2021, 2022, 2023],
            "hyperparameters": {"C": 1.0},
            "feature_set_version": "v4",
            "feature_cols": ["home_elo"],
            "eval_brier": 0.24,
            "train_n_games": 5000,
        }
        (d / "metadata.json").write_text(json.dumps(legacy_meta))

        model, meta = load_model(d)
        assert meta.training_tier == "full"
        assert meta.trained_at == ""
        assert model.label == "legacy"

    def test_load_legacy_train_brier_migration(self, tmp_path: Path) -> None:
        """Legacy metadata with train_brier is migrated to eval_brier."""
        d = tmp_path / "logistic_v3_train2022"
        d.mkdir(parents=True)

        import joblib
        joblib.dump(_DummyModel("old"), d / "model.joblib")
        legacy_meta = {
            "model_version": "v3",
            "model_type": "logistic",
            "training_seasons": [2020, 2021, 2022],
            "hyperparameters": {},
            "feature_set_version": "v3",
            "feature_cols": ["home_elo"],
            "train_brier": 0.25,
            "train_n_games": 3000,
        }
        (d / "metadata.json").write_text(json.dumps(legacy_meta))

        _, meta = load_model(d)
        assert meta.eval_brier == 0.25
