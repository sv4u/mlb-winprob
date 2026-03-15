"""Unit tests for the 4-way startup decision logic in app/main.py.

Validates the correct bootstrap path is chosen based on data/model availability:
- No data + No models → full bootstrap (ingest + quick-train)
- Data + No models → quick-train only
- No data + Models → ingest only
- Data + Models → normal startup
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# 4-way startup decision matrix
# ---------------------------------------------------------------------------


class TestStartupDecisionMatrix:
    """Tests for the 4-way startup decision in _lifespan.

    Each test patches has_processed_data and has_trained_models to simulate
    the four possible states, then verifies the correct bootstrap path is taken.
    """

    @pytest.mark.asyncio
    async def test_data_and_models_exist_normal_startup(self) -> None:
        """When both data and models exist, try_startup is called (no bootstrap)."""
        with (
            patch("mlb_predict.app.main.has_processed_data", return_value=True),
            patch("mlb_predict.app.main.has_trained_models", return_value=True),
            patch("mlb_predict.app.main.try_startup", return_value=True) as mock_startup,
            patch("mlb_predict.app.main._auto_bootstrap") as mock_bootstrap,
            patch("mlb_predict.app.main._quick_train_bootstrap") as mock_quick,
            patch("mlb_predict.app.main._data_ingest_only") as mock_ingest,
        ):
            from mlb_predict.app.main import _lifespan, app

            async with _lifespan(app):
                pass

            mock_startup.assert_called_once()
            mock_bootstrap.assert_not_called()
            mock_quick.assert_not_called()
            mock_ingest.assert_not_called()

    @pytest.mark.asyncio
    async def test_data_exists_no_models_quick_train(self) -> None:
        """When data exists but no models, quick-train bootstrap is started."""
        mock_task = AsyncMock()

        with (
            patch("mlb_predict.app.main.has_processed_data", return_value=True),
            patch("mlb_predict.app.main.has_trained_models", return_value=False),
            patch("mlb_predict.app.main._quick_train_bootstrap", return_value=mock_task),
            patch("mlb_predict.app.main._auto_bootstrap") as mock_bootstrap,
            patch("mlb_predict.app.main._data_ingest_only") as mock_ingest,
            patch("asyncio.create_task") as mock_create_task,
        ):
            from mlb_predict.app.main import _lifespan, app

            async with _lifespan(app):
                pass

            mock_create_task.assert_called()
            mock_bootstrap.assert_not_called()
            mock_ingest.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_data_models_exist_ingest_only(self) -> None:
        """When models exist but no data, data ingest is started (models preserved)."""
        mock_task = AsyncMock()

        with (
            patch("mlb_predict.app.main.has_processed_data", return_value=False),
            patch("mlb_predict.app.main.has_trained_models", return_value=True),
            patch("mlb_predict.app.main._data_ingest_only", return_value=mock_task),
            patch("mlb_predict.app.main._auto_bootstrap") as mock_bootstrap,
            patch("mlb_predict.app.main._quick_train_bootstrap") as mock_quick,
            patch("asyncio.create_task") as mock_create_task,
        ):
            from mlb_predict.app.main import _lifespan, app

            async with _lifespan(app):
                pass

            mock_create_task.assert_called()
            mock_bootstrap.assert_not_called()
            mock_quick.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_data_no_models_full_bootstrap(self) -> None:
        """When neither data nor models exist, full bootstrap is started."""
        mock_task = AsyncMock()

        with (
            patch("mlb_predict.app.main.has_processed_data", return_value=False),
            patch("mlb_predict.app.main.has_trained_models", return_value=False),
            patch("mlb_predict.app.main._auto_bootstrap", return_value=mock_task),
            patch("mlb_predict.app.main._quick_train_bootstrap") as mock_quick,
            patch("mlb_predict.app.main._data_ingest_only") as mock_ingest,
            patch("asyncio.create_task") as mock_create_task,
        ):
            from mlb_predict.app.main import _lifespan, app

            async with _lifespan(app):
                pass

            mock_create_task.assert_called()
            mock_quick.assert_not_called()
            mock_ingest.assert_not_called()


# ---------------------------------------------------------------------------
# Bootstrap function behavior
# ---------------------------------------------------------------------------


class TestAutoBootstrap:
    """Tests for _auto_bootstrap (no data + no models path)."""

    @pytest.mark.asyncio
    async def test_auto_bootstrap_runs_ingest_then_retrain(self) -> None:
        """Full bootstrap runs ingest first, then quick retrain."""
        ingest_state = MagicMock()
        ingest_state.status.value = "success"

        with (
            patch("mlb_predict.app.main.run_pipeline", new_callable=AsyncMock) as mock_pipeline,
            patch("mlb_predict.app.main.get_state", return_value=ingest_state),
        ):
            from mlb_predict.app.main import _auto_bootstrap, PipelineKind

            await _auto_bootstrap()

            assert mock_pipeline.call_count == 2
            first_call = mock_pipeline.call_args_list[0]
            assert first_call[0][0] == PipelineKind.INGEST
            second_call = mock_pipeline.call_args_list[1]
            assert second_call[0][0] == PipelineKind.RETRAIN
            assert second_call[1].get("bootstrap") is True
            assert second_call[1].get("training_tier") == "quick"

    @pytest.mark.asyncio
    async def test_auto_bootstrap_skips_retrain_on_ingest_failure(self) -> None:
        """If ingest fails, retrain is skipped."""
        ingest_state = MagicMock()
        ingest_state.status.value = "failed"

        with (
            patch("mlb_predict.app.main.run_pipeline", new_callable=AsyncMock) as mock_pipeline,
            patch("mlb_predict.app.main.get_state", return_value=ingest_state),
        ):
            from mlb_predict.app.main import _auto_bootstrap

            await _auto_bootstrap()

            assert mock_pipeline.call_count == 1


class TestQuickTrainBootstrap:
    """Tests for _quick_train_bootstrap (data exists, no models path)."""

    @pytest.mark.asyncio
    async def test_quick_train_populates_duckdb_then_retrains(self) -> None:
        """Quick train populates DuckDB first, then runs quick retrain."""
        with (
            patch("mlb_predict.app.main._populate_duckdb") as mock_duckdb,
            patch("mlb_predict.app.main.run_pipeline", new_callable=AsyncMock) as mock_pipeline,
        ):
            from mlb_predict.app.main import _quick_train_bootstrap

            await _quick_train_bootstrap()

            mock_duckdb.assert_called_once()
            mock_pipeline.assert_called_once()
            call_kwargs = mock_pipeline.call_args[1]
            assert call_kwargs.get("bootstrap") is True
            assert call_kwargs.get("training_tier") == "quick"

    @pytest.mark.asyncio
    async def test_quick_train_does_not_ingest(self) -> None:
        """Quick train bootstrap skips data ingestion entirely."""
        with (
            patch("mlb_predict.app.main._populate_duckdb"),
            patch("mlb_predict.app.main.run_pipeline", new_callable=AsyncMock) as mock_pipeline,
        ):
            from mlb_predict.app.main import PipelineKind, _quick_train_bootstrap

            await _quick_train_bootstrap()

            call_args = mock_pipeline.call_args[0]
            assert call_args[0] == PipelineKind.RETRAIN


class TestDataIngestOnly:
    """Tests for _data_ingest_only (models exist, no data path)."""

    @pytest.mark.asyncio
    async def test_ingest_only_runs_ingest_pipeline(self) -> None:
        """Data ingest only runs the INGEST pipeline."""
        with patch("mlb_predict.app.main.run_pipeline", new_callable=AsyncMock) as mock_pipeline:
            from mlb_predict.app.main import _data_ingest_only, PipelineKind

            await _data_ingest_only()

            mock_pipeline.assert_called_once()
            call_args = mock_pipeline.call_args[0]
            assert call_args[0] == PipelineKind.INGEST

    @pytest.mark.asyncio
    async def test_ingest_only_has_reload_callback(self) -> None:
        """Data ingest passes a reload callback for post-ingest startup."""
        with patch("mlb_predict.app.main.run_pipeline", new_callable=AsyncMock) as mock_pipeline:
            from mlb_predict.app.main import _data_ingest_only

            await _data_ingest_only()

            call_kwargs = mock_pipeline.call_args[1]
            assert call_kwargs.get("on_success") is not None
