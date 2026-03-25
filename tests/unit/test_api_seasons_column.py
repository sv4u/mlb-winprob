"""Tests for /api/seasons when the features DataFrame lacks a season column."""

from __future__ import annotations

import pandas as pd
import pytest
from starlette.testclient import TestClient


def test_api_seasons_missing_season_column_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    """api_seasons must not raise KeyError if season is absent from features."""
    from mlb_predict.app import main as app_main
    from mlb_predict.app.response_cache import clear_response_cache

    clear_response_cache()
    monkeypatch.setattr(app_main, "is_ready", lambda: True)
    monkeypatch.setattr(
        app_main,
        "get_features",
        lambda: pd.DataFrame({"game_pk": [1], "home_retro": ["NYA"], "away_retro": ["BOS"]}),
    )

    client = TestClient(app_main.app)
    response = client.get("/api/seasons")
    assert response.status_code == 503
    err = response.json().get("error", "")
    assert "season" in err.lower()
