"""Unit tests for admin pipeline state tracking (StepInfo, PipelineState).

Covers the step-by-step progress tracking added for the bootstrap dashboard.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mlb_predict.app.admin import (
    PipelineKind,
    PipelineState,
    PipelineStatus,
    StepInfo,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# StepInfo tests
# ---------------------------------------------------------------------------


class TestStepInfo:
    """Tests for the StepInfo dataclass."""

    def test_default_values(self) -> None:
        """StepInfo defaults to pending status with no elapsed time."""
        step = StepInfo(description="Ingest schedules")
        assert step.description == "Ingest schedules"
        assert step.status == "pending"
        assert step.elapsed_seconds is None

    def test_custom_values(self) -> None:
        """StepInfo accepts custom status and elapsed time."""
        step = StepInfo(description="Train models", status="complete", elapsed_seconds=42.5)
        assert step.description == "Train models"
        assert step.status == "complete"
        assert step.elapsed_seconds == 42.5


# ---------------------------------------------------------------------------
# PipelineState step tracking tests
# ---------------------------------------------------------------------------


class TestPipelineStateSteps:
    """Tests for step tracking methods on PipelineState."""

    @pytest.fixture
    def state(self) -> PipelineState:
        """Fresh PipelineState for ingest."""
        return PipelineState(kind=PipelineKind.INGEST)

    def test_init_steps_creates_pending_list(self, state: PipelineState) -> None:
        """init_steps populates the steps list with pending StepInfo objects."""
        descriptions = ["Step A", "Step B", "Step C"]
        state.init_steps(descriptions)

        assert len(state.steps) == 3
        for i, desc in enumerate(descriptions):
            assert state.steps[i].description == desc
            assert state.steps[i].status == "pending"
            assert state.steps[i].elapsed_seconds is None

    def test_begin_step_marks_running(self, state: PipelineState) -> None:
        """begin_step transitions a step to running and updates current_step_index."""
        state.init_steps(["A", "B"])
        state.begin_step(0)

        assert state.steps[0].status == "running"
        assert state.current_step_index == 0

    def test_complete_step_marks_complete_with_elapsed(self, state: PipelineState) -> None:
        """complete_step transitions a step to complete and records duration."""
        state.init_steps(["A", "B"])
        state.begin_step(0)
        state.complete_step(0, elapsed=12.3)

        assert state.steps[0].status == "complete"
        assert state.steps[0].elapsed_seconds == 12.3

    def test_fail_step_marks_failed(self, state: PipelineState) -> None:
        """fail_step transitions a step to failed status."""
        state.init_steps(["A", "B"])
        state.begin_step(1)
        state.fail_step(1)

        assert state.steps[1].status == "failed"

    def test_out_of_bounds_step_operations_are_safe(self, state: PipelineState) -> None:
        """Step operations on invalid indices do not raise exceptions."""
        state.init_steps(["A"])
        state.begin_step(5)
        state.complete_step(5, elapsed=1.0)
        state.fail_step(5)
        assert state.current_step_index == 5

    def test_reset_clears_steps(self, state: PipelineState) -> None:
        """reset() clears all step tracking data."""
        state.init_steps(["A", "B", "C"])
        state.begin_step(0)
        state.complete_step(0, elapsed=5.0)
        state.begin_step(1)

        state.reset()

        assert state.steps == []
        assert state.current_step_index == -1
        assert state.status == PipelineStatus.RUNNING

    def test_full_lifecycle(self, state: PipelineState) -> None:
        """Simulate a full pipeline lifecycle: reset → init → steps → finish."""
        state.reset()
        state.init_steps(["Step 1", "Step 2", "Step 3"])

        state.begin_step(0)
        state.complete_step(0, elapsed=10.0)

        state.begin_step(1)
        state.complete_step(1, elapsed=20.0)

        state.begin_step(2)
        state.complete_step(2, elapsed=15.0)

        state.finish(ok=True)

        assert state.status == PipelineStatus.SUCCESS
        assert all(s.status == "complete" for s in state.steps)
        assert state.elapsed_seconds is not None

    def test_lifecycle_with_failure(self, state: PipelineState) -> None:
        """Pipeline fails mid-way: first step complete, second fails."""
        state.reset()
        state.init_steps(["Step 1", "Step 2", "Step 3"])

        state.begin_step(0)
        state.complete_step(0, elapsed=5.0)

        state.begin_step(1)
        state.fail_step(1)
        state.finish(ok=False, error="Step 'Step 2' exited with code 1")

        assert state.status == PipelineStatus.FAILED
        assert state.steps[0].status == "complete"
        assert state.steps[1].status == "failed"
        assert state.steps[2].status == "pending"
        assert state.error == "Step 'Step 2' exited with code 1"


# ---------------------------------------------------------------------------
# PipelineState.to_dict() tests
# ---------------------------------------------------------------------------


class TestPipelineStateToDict:
    """Tests for the to_dict serialization including step data."""

    def test_to_dict_includes_step_fields(self) -> None:
        """to_dict returns steps, current_step_index, and total_steps."""
        state = PipelineState(kind=PipelineKind.RETRAIN)
        state.reset()
        state.init_steps(["Train models"])
        state.begin_step(0)

        d = state.to_dict()

        assert "steps" in d
        assert "current_step_index" in d
        assert "total_steps" in d
        assert d["total_steps"] == 1
        assert d["current_step_index"] == 0
        assert d["steps"][0]["description"] == "Train models"
        assert d["steps"][0]["status"] == "running"

    def test_to_dict_idle_state_has_empty_steps(self) -> None:
        """to_dict for an idle pipeline returns empty steps list."""
        state = PipelineState(kind=PipelineKind.INGEST)
        d = state.to_dict()

        assert d["steps"] == []
        assert d["total_steps"] == 0
        assert d["current_step_index"] == -1
        assert d["status"] == "idle"

    def test_to_dict_step_elapsed_serialized(self) -> None:
        """Step elapsed seconds are included in the serialized output."""
        state = PipelineState(kind=PipelineKind.INGEST)
        state.reset()
        state.init_steps(["A", "B"])
        state.begin_step(0)
        state.complete_step(0, elapsed=7.5)
        state.begin_step(1)

        d = state.to_dict()

        assert d["steps"][0]["elapsed_seconds"] == 7.5
        assert d["steps"][1]["elapsed_seconds"] is None


# ---------------------------------------------------------------------------
# PipelineState core behavior tests
# ---------------------------------------------------------------------------


class TestPipelineStateCore:
    """Tests for existing PipelineState behavior (log lines, finish, etc.)."""

    def test_append_log_respects_max(self) -> None:
        """Log lines are capped at _MAX_LOG_LINES."""
        from mlb_predict.app.admin import _MAX_LOG_LINES

        state = PipelineState(kind=PipelineKind.INGEST)
        for i in range(_MAX_LOG_LINES + 50):
            state.append_log(f"Line {i}")

        assert len(state.log_lines) == _MAX_LOG_LINES

    def test_finish_success_sets_elapsed(self) -> None:
        """finish(ok=True) computes elapsed_seconds from started_at."""
        state = PipelineState(kind=PipelineKind.INGEST)
        state.reset()
        state.finish(ok=True)

        assert state.status == PipelineStatus.SUCCESS
        assert state.elapsed_seconds is not None
        assert state.error is None

    def test_finish_failure_captures_error(self) -> None:
        """finish(ok=False) records the error message."""
        state = PipelineState(kind=PipelineKind.INGEST)
        state.reset()
        state.finish(ok=False, error="Something broke")

        assert state.status == PipelineStatus.FAILED
        assert state.error == "Something broke"

    def test_to_dict_log_tail_truncated(self) -> None:
        """to_dict returns only the last 80 log lines."""
        state = PipelineState(kind=PipelineKind.INGEST)
        for i in range(200):
            state.append_log(f"Line {i}")

        d = state.to_dict()
        assert len(d["log_tail"]) == 80
        assert d["log_line_count"] == 200


# ---------------------------------------------------------------------------
# UPDATE pipeline default season (matches UI)
# ---------------------------------------------------------------------------


class TestDefaultUpdateSeasonYear:
    """``_default_update_season_year`` must use the same resolver as the dashboard."""

    def test_uses_resolve_api_season_with_on_disk_seasons(self) -> None:
        """Resolver receives None and seasons from feature Parquet stems."""
        from unittest.mock import patch

        from mlb_predict.app.admin import _default_update_season_year

        with patch("mlb_predict.app.admin._seasons_from_feature_parquets", return_value=[2025, 2024]):
            with patch("mlb_predict.season.resolve_api_season", return_value=2025) as mock_resolve:
                y = _default_update_season_year()

        assert y == 2025
        mock_resolve.assert_called_once_with(None, available_seasons=[2025, 2024])

    def test_update_commands_default_year_in_ingest_schedule_cmd(self) -> None:
        """_update_commands without opts.seasons embeds the resolved year in shell steps."""
        from unittest.mock import patch

        from mlb_predict.app.admin import _update_commands

        with patch("mlb_predict.app.admin._default_update_season_year", return_value=2027):
            cmds = _update_commands(None)
        first_cmd = cmds[0][1]
        assert "--seasons 2027" in first_cmd or "2027" in first_cmd
