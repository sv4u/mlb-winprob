"""Unit tests for admin tier-aware pipeline functions.

Covers _archive_models, has_processed_data, has_trained_models,
and _retrain_commands with training tier support.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from mlb_predict.app.admin import (
    PipelineKind,
    PipelineOptions,
    PipelineState,
    PipelineStatus,
    _retrain_commands,
    has_processed_data,
    has_trained_models,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# has_processed_data
# ---------------------------------------------------------------------------


class TestHasProcessedData:
    """Tests for detecting whether processed feature files exist."""

    def test_no_features_dir(self, tmp_path: Path) -> None:
        """Returns False when the features directory doesn't exist."""
        with patch("mlb_predict.app.admin._PROCESSED_DIR", tmp_path / "processed"):
            assert has_processed_data() is False

    def test_empty_features_dir(self, tmp_path: Path) -> None:
        """Returns False when features directory exists but has no parquet files."""
        features_dir = tmp_path / "processed" / "features"
        features_dir.mkdir(parents=True)
        with patch("mlb_predict.app.admin._PROCESSED_DIR", tmp_path / "processed"):
            assert has_processed_data() is False

    def test_features_present(self, tmp_path: Path) -> None:
        """Returns True when feature parquet files exist."""
        features_dir = tmp_path / "processed" / "features"
        features_dir.mkdir(parents=True)
        (features_dir / "features_2024.parquet").write_text("mock")
        with patch("mlb_predict.app.admin._PROCESSED_DIR", tmp_path / "processed"):
            assert has_processed_data() is True


# ---------------------------------------------------------------------------
# has_trained_models (admin wrapper)
# ---------------------------------------------------------------------------


class TestAdminHasTrainedModels:
    """Tests for the admin module's has_trained_models wrapper."""

    def test_no_models(self, tmp_path: Path) -> None:
        """Returns False when no model artifacts exist."""
        with patch("mlb_predict.app.admin._MODEL_DIR", tmp_path / "models"):
            assert has_trained_models() is False

    def test_quick_model_detected(self, tmp_path: Path) -> None:
        """Returns True when a quick-tier model exists."""
        d = tmp_path / "models" / "quick" / "logistic_v4q_train2024"
        d.mkdir(parents=True)
        (d / "model.joblib").write_text("mock")
        with patch("mlb_predict.app.admin._MODEL_DIR", tmp_path / "models"):
            assert has_trained_models() is True


# ---------------------------------------------------------------------------
# _archive_models (admin wrapper)
# ---------------------------------------------------------------------------


class TestAdminArchiveModels:
    """Tests for the admin _archive_models function."""

    def test_archive_logs_count(self, tmp_path: Path) -> None:
        """_archive_models appends log line with archived count."""
        from mlb_predict.app.admin import _archive_models

        d = tmp_path / "models" / "quick" / "logistic_v4q_train2024"
        d.mkdir(parents=True)
        (d / "model.joblib").write_text("mock")

        state = PipelineState(kind=PipelineKind.RETRAIN)
        with patch("mlb_predict.app.admin._MODEL_DIR", tmp_path / "models"):
            _archive_models(state, tier="quick")

        assert any("Archived 1" in line for line in state.log_lines)

    def test_archive_with_no_models_logs_zero(self, tmp_path: Path) -> None:
        """_archive_models logs zero when no models exist for the tier."""
        from mlb_predict.app.admin import _archive_models

        state = PipelineState(kind=PipelineKind.RETRAIN)
        with patch("mlb_predict.app.admin._MODEL_DIR", tmp_path / "models"):
            _archive_models(state, tier="full")

        assert any("Archived 0" in line for line in state.log_lines)


# ---------------------------------------------------------------------------
# _retrain_commands with tier
# ---------------------------------------------------------------------------


class TestRetrainCommandsTier:
    """Tests for _retrain_commands with training tier parameter."""

    def test_default_tier_is_full(self) -> None:
        """Default retrain commands use full tier."""
        cmds = _retrain_commands()
        assert len(cmds) >= 1
        desc, cmd = cmds[0]
        assert "--tier full" in cmd
        assert "(full)" in desc

    def test_quick_tier_skips_cv_and_stage1(self) -> None:
        """Quick tier auto-adds --skip-cv and --no-stage1 flags."""
        cmds = _retrain_commands(training_tier="quick")
        _, cmd = cmds[0]
        assert "--tier quick" in cmd
        assert "--skip-cv" in cmd
        assert "--no-stage1" in cmd

    def test_bootstrap_forces_quick(self) -> None:
        """Bootstrap=True forces quick tier regardless of training_tier arg."""
        cmds = _retrain_commands(training_tier="full", bootstrap=True)
        _, cmd = cmds[0]
        assert "--tier quick" in cmd
        assert "--skip-cv" in cmd

    def test_full_tier_no_skip_by_default(self) -> None:
        """Full tier does not add --skip-cv or --no-stage1 by default."""
        cmds = _retrain_commands(training_tier="full")
        _, cmd = cmds[0]
        assert "--skip-cv" not in cmd
        assert "--no-stage1" not in cmd

    def test_opts_override_full_tier(self) -> None:
        """PipelineOptions can add --skip-cv even in full tier."""
        opts = PipelineOptions(skip_cv=True)
        cmds = _retrain_commands(opts, training_tier="full")
        _, cmd = cmds[0]
        assert "--tier full" in cmd
        assert "--skip-cv" in cmd

    def test_command_includes_all_model_types(self) -> None:
        """Retrain command includes all 6 model types."""
        cmds = _retrain_commands()
        _, cmd = cmds[0]
        for mt in ["logistic", "lightgbm", "xgboost", "catboost", "mlp", "stacked"]:
            assert mt in cmd
