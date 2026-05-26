"""Tests for player advanced stats loader and API."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mlb_predict.player.player_advanced import load_player_advanced_stats


def test_load_player_advanced_batters(tmp_path: Path) -> None:
    """Batter Parquet rows include names, stats, and percentiles."""
    df = pd.DataFrame(
        [
            {
                "player_id": 592450,
                "name": "Aaron Judge",
                "team_abbrev": "NYY",
                "woba": 0.450,
                "wrc_plus": 200,
                "ops": 1.100,
                "iso": 0.300,
                "xwoba": 0.440,
                "barrel_pct": 0.20,
                "k_pct": 0.25,
                "bb_pct": 0.15,
            },
            {
                "player_id": 545361,
                "name": "Mike Trout",
                "team_abbrev": "LAA",
                "woba": 0.380,
                "wrc_plus": 150,
                "ops": 0.950,
                "iso": 0.220,
                "xwoba": 0.370,
                "barrel_pct": 0.12,
                "k_pct": 0.22,
                "bb_pct": 0.12,
            },
        ]
    )
    df.to_parquet(tmp_path / "batter_stats_2025.parquet", index=False)

    payload = load_player_advanced_stats(2025, "hitting", player_dir=tmp_path)
    assert payload["season"] == 2025
    assert payload["total"] == 2
    assert len(payload["players"]) == 2
    top = payload["players"][0]
    assert top["name"] == "Aaron Judge"
    assert top["stats"]["woba"] == 0.450
    assert top["stats"]["woba_pctile"] == 100


def test_load_player_advanced_search(tmp_path: Path) -> None:
    """Name search filters the result set."""
    df = pd.DataFrame(
        [
            {"player_id": 1, "name": "Aaron Judge", "woba": 0.4},
            {"player_id": 2, "name": "Mike Trout", "woba": 0.35},
        ]
    )
    df.to_parquet(tmp_path / "batter_stats_2025.parquet", index=False)

    payload = load_player_advanced_stats(
        2025, "hitting", search="judge", player_dir=tmp_path
    )
    assert payload["total"] == 1
    assert payload["players"][0]["name"] == "Aaron Judge"


def test_load_player_advanced_missing_file(tmp_path: Path) -> None:
    """Missing Parquet returns ingest guidance."""
    payload = load_player_advanced_stats(2024, "hitting", player_dir=tmp_path)
    assert payload["players"] == []
    assert "ingest_player_data" in payload.get("message", "")


def test_api_player_advanced_stats_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GET /api/player-advanced-stats returns JSON from player Parquet."""
    from starlette.testclient import TestClient

    from mlb_predict.app import main as app_main
    from mlb_predict.app.response_cache import clear_response_cache

    df = pd.DataFrame(
        [
            {
                "player_id": 592450,
                "name": "Aaron Judge",
                "team_abbrev": "NYY",
                "fip": 3.5,
                "xfip": 3.6,
                "k_pct": 0.28,
            }
        ]
    )
    df.to_parquet(tmp_path / "pitcher_stats_2025.parquet", index=False)

    clear_response_cache()
    monkeypatch.setattr(
        "mlb_predict.player.player_advanced.default_player_dir",
        lambda: tmp_path,
    )
    monkeypatch.setattr(app_main, "resolve_season", lambda requested: requested or 2025)

    client = TestClient(app_main.app)
    response = client.get("/api/player-advanced-stats?season=2025&group=pitching")
    assert response.status_code == 200
    body = response.json()
    assert body["season"] == 2025
    assert body["group"] == "pitching"
    assert len(body["players"]) == 1
    assert body["players"][0]["name"] == "Aaron Judge"
