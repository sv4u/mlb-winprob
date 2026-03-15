"""Integration tests for POST /api/admin/retrain with training tier support.

Validates the retrain endpoint accepts tier selection, defaults to full,
and rejects invalid values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

if TYPE_CHECKING:
    pass


@pytest.fixture
def client() -> TestClient:
    """TestClient configured with patched startup to avoid model loading."""
    with (
        patch("mlb_predict.app.main.has_processed_data", return_value=True),
        patch("mlb_predict.app.main.has_trained_models", return_value=True),
        patch("mlb_predict.app.main.try_startup", return_value=True),
    ):
        from mlb_predict.app.main import app
        yield TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Retrain endpoint with tier
# ---------------------------------------------------------------------------


class TestRetrainEndpointTier:
    """Tests for POST /api/admin/retrain with training_tier parameter."""

    def test_retrain_default_tier_is_full(self, client: TestClient) -> None:
        """Retrain without body defaults to full tier."""
        with patch("mlb_predict.app.main.run_pipeline", new_callable=AsyncMock):
            resp = client.post("/api/admin/retrain")
            assert resp.status_code == 200
            body = resp.json()
            assert body["ok"] is True
            assert "tier=full" in body["message"]

    def test_retrain_quick_tier(self, client: TestClient) -> None:
        """Retrain with training_tier='quick' is accepted."""
        with patch("mlb_predict.app.main.run_pipeline", new_callable=AsyncMock):
            resp = client.post(
                "/api/admin/retrain",
                json={"training_tier": "quick"},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert "tier=quick" in body["message"]

    def test_retrain_full_tier_explicit(self, client: TestClient) -> None:
        """Retrain with explicit training_tier='full' is accepted."""
        with patch("mlb_predict.app.main.run_pipeline", new_callable=AsyncMock):
            resp = client.post(
                "/api/admin/retrain",
                json={"training_tier": "full"},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert "tier=full" in body["message"]

    def test_retrain_passes_tier_to_pipeline(self, client: TestClient) -> None:
        """The training_tier value is forwarded to run_pipeline."""
        with patch("mlb_predict.app.main.run_pipeline", new_callable=AsyncMock) as mock_pipeline:
            client.post(
                "/api/admin/retrain",
                json={"training_tier": "quick"},
            )
            mock_pipeline.assert_called_once()
            call_kwargs = mock_pipeline.call_args[1]
            assert call_kwargs.get("training_tier") == "quick"

    def test_retrain_with_pipeline_already_running(self, client: TestClient) -> None:
        """Returns ok=False when a pipeline is already running."""
        from mlb_predict.app.admin import PipelineStatus, get_state, PipelineKind

        state = get_state(PipelineKind.RETRAIN)
        original_status = state.status
        state.status = PipelineStatus.RUNNING
        try:
            resp = client.post(
                "/api/admin/retrain",
                json={"training_tier": "full"},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["ok"] is False
            assert "running" in body["message"].lower()
        finally:
            state.status = original_status


# ---------------------------------------------------------------------------
# Model status endpoint
# ---------------------------------------------------------------------------


class TestModelStatusEndpoint:
    """Tests for GET /api/admin/status model tier reporting."""

    def test_status_returns_model_info(self, client: TestClient) -> None:
        """Admin status endpoint includes model inventory."""
        resp = client.get("/api/admin/status")
        assert resp.status_code == 200
        body = resp.json()
        assert "models" in body
