"""Integration tests for the /api/odds endpoint and live_odds in game detail."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from starlette.testclient import TestClient

if TYPE_CHECKING:
    pass


@pytest.fixture
def api_client() -> TestClient:
    """TestClient for the FastAPI app."""
    from mlb_predict.app.main import app

    return TestClient(app)


def test_api_odds_response_shape(api_client: TestClient) -> None:
    """GET /api/odds returns the expected top-level keys."""
    r = api_client.get("/api/odds")
    assert r.status_code == 200
    data = r.json()
    assert "configured" in data
    assert "count" in data
    assert "events" in data
    assert isinstance(data["configured"], bool)
    assert isinstance(data["count"], int)
    assert isinstance(data["events"], list)


def test_api_odds_count_matches_events(api_client: TestClient) -> None:
    """The count field matches the length of the events list."""
    r = api_client.get("/api/odds")
    data = r.json()
    assert data["count"] == len(data["events"])


def test_game_detail_has_odds_fields(api_client: TestClient) -> None:
    """GET /api/games/{pk} includes live_odds and odds_configured fields.

    Uses the /api/games endpoint to find a valid game_pk first.
    If no data is loaded, the test is skipped.
    """
    games_r = api_client.get("/api/games", params={"limit": "1"})
    if games_r.status_code != 200:
        pytest.skip("App data not loaded; skipping game detail test")
    games_data = games_r.json()
    if not games_data.get("games"):
        pytest.skip("No games available in test data")

    game_pk = games_data["games"][0]["game_pk"]
    r = api_client.get(f"/api/games/{game_pk}")
    assert r.status_code == 200
    data = r.json()
    assert "live_odds" in data
    assert "odds_configured" in data
    assert isinstance(data["odds_configured"], bool)


def test_api_ev_opportunities_response_shape(api_client: TestClient) -> None:
    """GET /api/ev-opportunities returns the expected top-level keys."""
    r = api_client.get("/api/ev-opportunities")
    assert r.status_code == 200
    data = r.json()
    assert "configured" in data
    assert "count" in data
    assert "opportunities" in data
    assert isinstance(data["configured"], bool)
    assert isinstance(data["count"], int)
    assert isinstance(data["opportunities"], list)
    assert data["count"] == len(data["opportunities"])


def test_ev_calculator_redirects_to_odds_hub(api_client: TestClient) -> None:
    """GET /tools/ev-calculator returns a 301 redirect to /odds."""
    r = api_client.get("/tools/ev-calculator", follow_redirects=False)
    assert r.status_code == 301
    assert "/odds" in r.headers.get("location", "")
