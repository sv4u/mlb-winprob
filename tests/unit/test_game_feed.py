"""Tests for mlb_predict.mlbapi.game_feed — play-by-play normalization."""

from __future__ import annotations

from mlb_predict.mlbapi.game_feed import _normalize_play, _normalize_plays


def test_normalize_play_minimal() -> None:
    """_normalize_play returns expected keys for a minimal play."""
    play = {
        "atBatIndex": 0,
        "about": {
            "inning": 1,
            "halfInning": "Top",
            "outs": 0,
            "runs": 0,
            "homeScore": 0,
            "awayScore": 0,
        },
        "result": {"description": "Flyout.", "event": "Flyout", "eventType": "out"},
        "matchup": {
            "batter": {"id": 1, "fullName": "Batter One"},
            "pitcher": {"id": 2, "fullName": "Pitcher Two"},
        },
    }
    out = _normalize_play(play, 0)
    assert out["index"] == 0
    assert out["inning"] == 1
    assert out["half"] == "top"
    assert out["outs"] == 0
    assert out["description"] == "Flyout."
    assert out["batter_name"] == "Batter One"
    assert out["pitcher_name"] == "Pitcher Two"
    assert out["home_score"] == 0
    assert out["away_score"] == 0


def test_normalize_plays_empty() -> None:
    """_normalize_plays returns empty list for empty input."""
    assert _normalize_plays([]) == []


def test_normalize_plays_skips_invalid() -> None:
    """_normalize_plays skips entries that raise during normalization."""
    plays = [
        {"about": {"inning": 1}, "result": {}, "matchup": {}},
        "not a dict",
        {"about": {}, "result": {}, "matchup": {}},
    ]
    result = _normalize_plays(plays)  # type: ignore[arg-type]
    assert len(result) == 2
    assert result[0]["inning"] == 1
    assert result[1]["inning"] == 0
