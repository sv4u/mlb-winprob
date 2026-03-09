"""Tests for mlb_predict.mlbapi.leaders — league leaders and player stats normalization."""

from __future__ import annotations

from mlb_predict.mlbapi.leaders import (
    HITTING_LEADER_CATEGORIES,
    PITCHING_LEADER_CATEGORIES,
    _normalize_leader_entry,
    _normalize_player_stat_row,
)


def test_normalize_leader_entry() -> None:
    """_normalize_leader_entry returns expected keys."""
    entry = {
        "person": {"id": 1, "fullName": "Player One"},
        "team": {"id": 10, "name": "Team A", "abbreviation": "TMA"},
        "value": 42,
        "stat": {"name": "homeRuns"},
    }
    out = _normalize_leader_entry(entry, 1, "homeRuns")
    assert out["rank"] == 1
    assert out["category"] == "homeRuns"
    assert out["person_id"] == 1
    assert out["name"] == "Player One"
    assert out["team_abbrev"] == "TMA"
    assert out["value"] == 42


def test_hitting_categories_non_empty() -> None:
    """Hitting leader categories list is non-empty."""
    assert len(HITTING_LEADER_CATEGORIES) > 0
    assert "homeRuns" in HITTING_LEADER_CATEGORIES
    assert "battingAverage" in HITTING_LEADER_CATEGORIES


def test_pitching_categories_non_empty() -> None:
    """Pitching leader categories list is non-empty."""
    assert len(PITCHING_LEADER_CATEGORIES) > 0
    assert "earnedRunAverage" in PITCHING_LEADER_CATEGORIES
    assert "strikeOuts" in PITCHING_LEADER_CATEGORIES


def test_normalize_player_stat_row() -> None:
    """_normalize_player_stat_row merges person, team, and stat."""
    stat = {"avg": ".310", "homeRuns": 25, "rbi": 80}
    person = {"id": 1, "fullName": "Batter One"}
    team = {"id": 10, "name": "Team A", "abbreviation": "TMA"}
    out = _normalize_player_stat_row(stat, person, team, "hitting")
    assert out["person_id"] == 1
    assert out["name"] == "Batter One"
    assert out["team_abbrev"] == "TMA"
    assert out["group"] == "hitting"
    assert out["homeRuns"] == 25
    assert out["rbi"] == 80
