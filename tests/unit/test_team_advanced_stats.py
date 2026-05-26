"""Tests for FanGraphs team advanced stats loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mlb_predict.statcast.fangraphs import fg_to_retro_code
from mlb_predict.statcast.team_advanced import load_team_advanced_stats


def test_fg_to_retro_code_prefers_canonical_franchise() -> None:
    """FanGraphs NYY maps to Retrosheet NYA."""
    assert fg_to_retro_code("NYY") == "NYA"
    assert fg_to_retro_code("LAD") == "LAN"
    assert fg_to_retro_code("TBR") == "TBA"


def test_load_team_advanced_stats_from_parquet(tmp_path: Path) -> None:
    """Parquet rows are enriched with retro codes, names, and percentiles."""
    df = pd.DataFrame(
        [
            {
                "team_fg": "NYY",
                "season": 2025,
                "bat_woba": 0.330,
                "bat_xwoba": 0.335,
                "bat_iso": 0.190,
                "bat_babip": 0.300,
                "bat_barrel_pct": 0.09,
                "bat_hard_pct": 0.40,
                "pit_fip": 3.80,
                "pit_xfip": 3.90,
                "pit_k_pct": 0.24,
                "pit_bb_pct": 0.08,
                "pit_hr_fb": 0.10,
                "pit_whip": 1.20,
            },
            {
                "team_fg": "BOS",
                "season": 2025,
                "bat_woba": 0.310,
                "bat_xwoba": 0.312,
                "bat_iso": 0.160,
                "bat_babip": 0.290,
                "bat_barrel_pct": 0.07,
                "bat_hard_pct": 0.36,
                "pit_fip": 4.20,
                "pit_xfip": 4.10,
                "pit_k_pct": 0.22,
                "pit_bb_pct": 0.09,
                "pit_hr_fb": 0.12,
                "pit_whip": 1.35,
            },
        ]
    )
    df.to_parquet(tmp_path / "fangraphs_2025.parquet", index=False)

    payload = load_team_advanced_stats(2025, fg_dir=tmp_path)
    assert payload["season"] == 2025
    assert len(payload["teams"]) == 2

    nyy = next(t for t in payload["teams"] if t["retro_code"] == "NYA")
    assert nyy["team_name"] == "New York Yankees"
    assert nyy["batting"]["woba"] == 0.330
    assert nyy["batting"]["barrel_pct"] == 9.0
    assert nyy["batting"]["woba_pctile"] == 100
    assert nyy["pitching"]["fip"] == 3.80
    assert nyy["pitching"]["fip_pctile"] == 100


def test_load_team_advanced_stats_missing_file(tmp_path: Path) -> None:
    """Missing Parquet returns an empty teams list with ingest guidance."""
    payload = load_team_advanced_stats(2024, fg_dir=tmp_path)
    assert payload["teams"] == []
    assert "ingest_fangraphs" in payload.get("message", "")


def test_api_team_advanced_stats_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GET /api/team-advanced-stats returns JSON from FanGraphs Parquet."""
    from starlette.testclient import TestClient

    from mlb_predict.app import main as app_main
    from mlb_predict.app.response_cache import clear_response_cache

    df = pd.DataFrame(
        [
            {
                "team_fg": "NYY",
                "season": 2025,
                "bat_woba": 0.320,
                "bat_xwoba": 0.320,
                "bat_iso": 0.170,
                "bat_babip": 0.300,
                "bat_barrel_pct": 0.08,
                "bat_hard_pct": 0.38,
                "pit_fip": 4.00,
                "pit_xfip": 4.00,
                "pit_k_pct": 0.22,
                "pit_bb_pct": 0.085,
                "pit_hr_fb": 0.11,
                "pit_whip": 1.30,
            }
        ]
    )
    df.to_parquet(tmp_path / "fangraphs_2025.parquet", index=False)

    clear_response_cache()
    monkeypatch.setattr(
        "mlb_predict.statcast.team_advanced.default_fangraphs_dir",
        lambda: tmp_path,
    )
    monkeypatch.setattr(app_main, "resolve_season", lambda requested: requested or 2025)

    client = TestClient(app_main.app)
    response = client.get("/api/team-advanced-stats?season=2025")
    assert response.status_code == 200
    body = response.json()
    assert body["season"] == 2025
    assert len(body["teams"]) == 1
    assert body["teams"][0]["retro_code"] == "NYA"
