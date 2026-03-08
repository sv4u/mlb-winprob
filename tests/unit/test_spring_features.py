"""Tests for scripts/build_spring_features.py."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pytest

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Helpers — import the module under test
# ---------------------------------------------------------------------------


from scripts.build_spring_features import (
    _build_game_row,
    _build_team_state,
    _extract_state_from_role,
    _feature_hash,
    _last_game_dates,
    _park_factor_by_home_team,
    _resolve,
    build_spring_features_for_season,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_features_df() -> pd.DataFrame:
    """Minimal historical feature DataFrame with two teams across two seasons."""
    rows = []
    for season in [2023, 2024]:
        for date_offset in range(3):
            d = datetime.date(season, 4, 1 + date_offset)
            rows.append(
                {
                    "date": d,
                    "season": season,
                    "home_retro": "BOS",
                    "away_retro": "NYA",
                    "home_elo": 1520.0 + season - 2023,
                    "away_elo": 1510.0 + season - 2023,
                    "home_win_pct_15": 0.55,
                    "away_win_pct_15": 0.48,
                    "home_pythag_30": 0.54,
                    "away_pythag_30": 0.47,
                    "home_win_pct_home_only": 0.60,
                    "away_win_pct_away_only": 0.45,
                    "park_run_factor": 1.05,
                    "home_win": 1.0 if date_offset % 2 == 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def team_state() -> dict[str, dict[str, float]]:
    """Pre-built team state for testing _build_game_row."""
    return {
        "BOS": {"elo": 1525.0, "win_pct_15": 0.55, "pythag_30": 0.54, "sp_era": 3.80},
        "NYA": {"elo": 1515.0, "win_pct_15": 0.48, "pythag_30": 0.47, "sp_era": 4.10},
    }


@pytest.fixture
def spring_schedule_df() -> pd.DataFrame:
    """Minimal spring training schedule with completed games."""
    return pd.DataFrame(
        {
            "game_pk": [900001, 900002, 900003],
            "game_date_utc": [
                "2025-03-01T18:00:00Z",
                "2025-03-02T18:00:00Z",
                "2025-03-03T18:00:00Z",
            ],
            "game_date_local": [
                "2025-03-01T13:00:00",
                "2025-03-02T13:00:00",
                "2025-03-03T13:00:00",
            ],
            "home_mlb_id": [111, 147, 111],
            "away_mlb_id": [147, 111, 147],
            "home_score": [5, 3, None],
            "away_score": [3, 7, None],
            "status": ["Final", "Final", "Scheduled"],
            "game_type": ["S", "S", "S"],
        }
    )


# ---------------------------------------------------------------------------
# _resolve
# ---------------------------------------------------------------------------


def test_resolve_returns_state_value() -> None:
    """_resolve returns the value from state when present and non-NaN."""
    state = {"elo": 1520.0, "sp_era": 3.50}
    assert _resolve(state, "elo") == 1520.0
    assert _resolve(state, "sp_era") == 3.50


def test_resolve_returns_default_for_nan() -> None:
    """_resolve returns the league-average default when state value is NaN."""
    state: dict[str, float] = {"elo": float("nan")}
    result = _resolve(state, "elo")
    assert result == 1500.0


def test_resolve_returns_default_for_missing_key() -> None:
    """_resolve returns the default when key is not in state dict."""
    result = _resolve({}, "sp_era")
    assert result == 4.50


# ---------------------------------------------------------------------------
# _feature_hash
# ---------------------------------------------------------------------------


def test_feature_hash_deterministic() -> None:
    """_feature_hash must produce identical hashes for identical inputs."""
    row = {"home_elo": 1520.0, "away_elo": 1510.0, "elo_diff": 10.0}
    h1 = _feature_hash(row)
    h2 = _feature_hash(row)
    assert h1 == h2
    assert len(h1) == 16


def test_feature_hash_ignores_non_float() -> None:
    """_feature_hash must ignore non-float values and NaNs."""
    row_a = {"home_elo": 1520.0, "game_pk": 12345}
    row_b = {"home_elo": 1520.0, "game_pk": 99999}
    assert _feature_hash(row_a) == _feature_hash(row_b)


def test_feature_hash_different_for_different_values() -> None:
    """_feature_hash must differ when float values differ."""
    row_a = {"home_elo": 1520.0}
    row_b = {"home_elo": 1530.0}
    assert _feature_hash(row_a) != _feature_hash(row_b)


# ---------------------------------------------------------------------------
# _extract_state_from_role
# ---------------------------------------------------------------------------


def test_extract_state_from_home_role(minimal_features_df: pd.DataFrame) -> None:
    """_extract_state_from_role must extract last-known home state per team."""
    from scripts.build_spring_features import _HOME_COL_MAP

    states = _extract_state_from_role(minimal_features_df, "home_retro", _HOME_COL_MAP)
    assert "BOS" in states
    assert states["BOS"]["elo"] == pytest.approx(1521.0, abs=0.1)


def test_extract_state_from_away_role(minimal_features_df: pd.DataFrame) -> None:
    """_extract_state_from_role must extract last-known away state per team."""
    from scripts.build_spring_features import _AWAY_COL_MAP

    states = _extract_state_from_role(minimal_features_df, "away_retro", _AWAY_COL_MAP)
    assert "NYA" in states


# ---------------------------------------------------------------------------
# _build_team_state
# ---------------------------------------------------------------------------


def test_build_team_state_merges_roles(minimal_features_df: pd.DataFrame) -> None:
    """_build_team_state must merge home and away role states per team."""
    state = _build_team_state(minimal_features_df)
    assert "BOS" in state
    assert "NYA" in state
    assert "elo" in state["BOS"]


def test_build_team_state_preserves_splits(minimal_features_df: pd.DataFrame) -> None:
    """_build_team_state must preserve home-only and away-only splits."""
    state = _build_team_state(minimal_features_df)
    assert "win_pct_home_only" in state["BOS"]
    assert "win_pct_away_only" in state["NYA"]


def test_build_team_state_prefers_more_recent_role() -> None:
    """When a team has both home and away games, role-agnostic stats should come from the more recent game."""
    rows = [
        {
            "date": datetime.date(2024, 9, 25),
            "season": 2024,
            "home_retro": "BOS",
            "away_retro": "TBA",
            "home_elo": 1520.0,
            "away_elo": 1490.0,
            "home_win_pct_15": 0.55,
            "away_win_pct_15": 0.42,
            "home_pythag_30": 0.54,
            "away_pythag_30": 0.44,
            "home_win_pct_home_only": 0.60,
            "away_win_pct_away_only": 0.40,
            "home_win": 1.0,
        },
        {
            "date": datetime.date(2024, 9, 28),
            "season": 2024,
            "home_retro": "NYA",
            "away_retro": "BOS",
            "home_elo": 1540.0,
            "away_elo": 1535.0,
            "home_win_pct_15": 0.60,
            "away_win_pct_15": 0.58,
            "home_pythag_30": 0.59,
            "away_pythag_30": 0.57,
            "home_win_pct_home_only": 0.62,
            "away_win_pct_away_only": 0.56,
            "home_win": 0.0,
        },
    ]
    df = pd.DataFrame(rows)
    state = _build_team_state(df)

    # BOS: last home Sep 25, last away Sep 28 — away is more recent
    assert state["BOS"]["elo"] == pytest.approx(1535.0)
    assert state["BOS"]["win_pct_15"] == pytest.approx(0.58)
    assert state["BOS"]["pythag_30"] == pytest.approx(0.57)
    # Role-specific splits stay with their respective roles
    assert state["BOS"]["win_pct_home_only"] == pytest.approx(0.60)
    assert state["BOS"]["win_pct_away_only"] == pytest.approx(0.56)

    # NYA: only appears as home (Sep 28) — home value used
    assert state["NYA"]["elo"] == pytest.approx(1540.0)
    assert state["NYA"]["win_pct_15"] == pytest.approx(0.60)


def test_build_team_state_prefers_home_when_home_is_later() -> None:
    """When a team's last home game is more recent than their last away, home state wins."""
    rows = [
        {
            "date": datetime.date(2024, 9, 20),
            "season": 2024,
            "home_retro": "NYA",
            "away_retro": "BOS",
            "home_elo": 1500.0,
            "away_elo": 1505.0,
            "home_win_pct_15": 0.50,
            "away_win_pct_15": 0.51,
            "home_pythag_30": 0.49,
            "away_pythag_30": 0.52,
            "home_win_pct_home_only": 0.48,
            "away_win_pct_away_only": 0.53,
            "home_win": 1.0,
        },
        {
            "date": datetime.date(2024, 9, 29),
            "season": 2024,
            "home_retro": "BOS",
            "away_retro": "TBA",
            "home_elo": 1550.0,
            "away_elo": 1480.0,
            "home_win_pct_15": 0.65,
            "away_win_pct_15": 0.40,
            "home_pythag_30": 0.63,
            "away_pythag_30": 0.42,
            "home_win_pct_home_only": 0.68,
            "away_win_pct_away_only": 0.38,
            "home_win": 1.0,
        },
    ]
    df = pd.DataFrame(rows)
    state = _build_team_state(df)

    # BOS: last away Sep 20, last home Sep 29 — home is more recent
    assert state["BOS"]["elo"] == pytest.approx(1550.0)
    assert state["BOS"]["win_pct_15"] == pytest.approx(0.65)
    # Role-specific splits come from their own role
    assert state["BOS"]["win_pct_home_only"] == pytest.approx(0.68)
    assert state["BOS"]["win_pct_away_only"] == pytest.approx(0.53)


# ---------------------------------------------------------------------------
# _last_game_dates
# ---------------------------------------------------------------------------


def test_last_game_dates_returns_chronological_last(minimal_features_df: pd.DataFrame) -> None:
    """_last_game_dates should return the most recent date per team."""
    dates = _last_game_dates(minimal_features_df, "home_retro")
    assert dates["BOS"] == datetime.date(2024, 4, 3)


# ---------------------------------------------------------------------------
# _park_factor_by_home_team
# ---------------------------------------------------------------------------


def test_park_factor_by_home_team(minimal_features_df: pd.DataFrame) -> None:
    """_park_factor_by_home_team must return median park factor per home team."""
    pf = _park_factor_by_home_team(minimal_features_df)
    assert "BOS" in pf
    assert pf["BOS"] == pytest.approx(1.05, abs=0.01)


def test_park_factor_empty_without_column() -> None:
    """_park_factor_by_home_team returns empty dict if column is missing."""
    df = pd.DataFrame({"home_retro": ["BOS"], "season": [2024]})
    pf = _park_factor_by_home_team(df)
    assert pf == {}


# ---------------------------------------------------------------------------
# _build_game_row
# ---------------------------------------------------------------------------


def test_build_game_row_complete(team_state: dict) -> None:
    """_build_game_row must return a dict with is_spring=1.0, game_type='S', and correct home_win."""
    game = pd.Series(
        {
            "game_pk": 900001,
            "game_date_local": "2025-03-01T13:00:00",
            "game_date_utc": "2025-03-01T18:00:00Z",
            "home_mlb_id": 111,
            "away_mlb_id": 147,
            "home_score": 5,
            "away_score": 3,
            "status": "Final",
            "game_type": "S",
        }
    )
    row = _build_game_row(
        game,
        season=2025,
        idx=0,
        n_games=10,
        team_state=team_state,
        park_factors={"BOS": 1.05},
        mlb_to_retro={111: "BOS", 147: "NYA"},
    )
    assert row is not None
    assert row["is_spring"] == 1.0
    assert row["game_type"] == "S"
    assert row["home_win"] == 1.0
    assert row["home_elo"] == 1525.0
    assert row["elo_diff"] == pytest.approx(10.0)
    assert "feature_hash" in row


def test_build_game_row_away_win(team_state: dict) -> None:
    """_build_game_row must correctly flag an away win."""
    game = pd.Series(
        {
            "game_pk": 900002,
            "game_date_local": "2025-03-02T13:00:00",
            "game_date_utc": "2025-03-02T18:00:00Z",
            "home_mlb_id": 147,
            "away_mlb_id": 111,
            "home_score": 3,
            "away_score": 7,
        }
    )
    row = _build_game_row(
        game,
        season=2025,
        idx=1,
        n_games=10,
        team_state=team_state,
        park_factors={},
        mlb_to_retro={111: "BOS", 147: "NYA"},
    )
    assert row is not None
    assert row["home_win"] == 0.0


def test_build_game_row_returns_none_for_missing_scores(team_state: dict) -> None:
    """_build_game_row must return None when scores are NaN."""
    game = pd.Series(
        {
            "game_pk": 900003,
            "game_date_local": "2025-03-03T13:00:00",
            "game_date_utc": "2025-03-03T18:00:00Z",
            "home_mlb_id": 111,
            "away_mlb_id": 147,
            "home_score": None,
            "away_score": None,
        }
    )
    row = _build_game_row(
        game,
        season=2025,
        idx=2,
        n_games=10,
        team_state=team_state,
        park_factors={},
        mlb_to_retro={111: "BOS", 147: "NYA"},
    )
    assert row is None


def test_build_game_row_returns_none_for_tied_game(team_state: dict) -> None:
    """_build_game_row must return None for tied spring training games."""
    game = pd.Series(
        {
            "game_pk": 900004,
            "game_date_local": "2025-03-04T13:00:00",
            "game_date_utc": "2025-03-04T18:00:00Z",
            "home_mlb_id": 111,
            "away_mlb_id": 147,
            "home_score": 4,
            "away_score": 4,
        }
    )
    row = _build_game_row(
        game,
        season=2025,
        idx=3,
        n_games=10,
        team_state=team_state,
        park_factors={},
        mlb_to_retro={111: "BOS", 147: "NYA"},
    )
    assert row is None


# ---------------------------------------------------------------------------
# build_spring_features_for_season
# ---------------------------------------------------------------------------


def test_build_spring_features_no_schedule(
    tmp_path: Path,
    minimal_features_df: pd.DataFrame,
) -> None:
    """build_spring_features_for_season returns None when schedule file is missing."""
    result = build_spring_features_for_season(
        2025,
        minimal_features_df,
        {},
        {},
        {},
        processed_dir=tmp_path,
    )
    assert result is None


def test_build_spring_features_no_spring_games(
    tmp_path: Path,
    minimal_features_df: pd.DataFrame,
) -> None:
    """build_spring_features_for_season returns None if schedule has no spring games."""
    sched_dir = tmp_path / "schedule"
    sched_dir.mkdir(parents=True)
    regular_sched = pd.DataFrame(
        {
            "game_pk": [1],
            "game_type": ["R"],
            "status": ["Final"],
            "home_mlb_id": [111],
            "away_mlb_id": [147],
            "home_score": [5],
            "away_score": [3],
            "game_date_local": ["2025-04-01T13:00:00"],
        }
    )
    regular_sched.to_parquet(sched_dir / "games_2025.parquet", index=False)

    result = build_spring_features_for_season(
        2025,
        minimal_features_df,
        {},
        {},
        {},
        processed_dir=tmp_path,
    )
    assert result is None


def test_build_spring_features_produces_dataframe(
    tmp_path: Path,
    minimal_features_df: pd.DataFrame,
    spring_schedule_df: pd.DataFrame,
) -> None:
    """build_spring_features_for_season produces a DataFrame for completed spring games."""
    sched_dir = tmp_path / "schedule"
    sched_dir.mkdir(parents=True)
    spring_schedule_df.to_parquet(sched_dir / "games_2025.parquet", index=False)

    team_state = _build_team_state(minimal_features_df)
    park_factors = _park_factor_by_home_team(minimal_features_df)
    mlb_to_retro = {111: "BOS", 147: "NYA"}

    result = build_spring_features_for_season(
        2025,
        minimal_features_df,
        team_state,
        park_factors,
        mlb_to_retro,
        processed_dir=tmp_path,
    )

    assert result is not None
    assert len(result) == 2
    assert (result["is_spring"] == 1.0).all()
    assert (result["game_type"] == "S").all()
    assert result["home_win"].notna().all()


def test_build_spring_features_skips_tied_games(
    tmp_path: Path,
    minimal_features_df: pd.DataFrame,
) -> None:
    """build_spring_features_for_season must exclude tied spring training games."""
    sched_dir = tmp_path / "schedule"
    sched_dir.mkdir(parents=True)
    sched = pd.DataFrame(
        {
            "game_pk": [900001, 900002],
            "game_date_utc": ["2025-03-01T18:00:00Z", "2025-03-02T18:00:00Z"],
            "game_date_local": ["2025-03-01T13:00:00", "2025-03-02T13:00:00"],
            "home_mlb_id": [111, 111],
            "away_mlb_id": [147, 147],
            "home_score": [5, 3],
            "away_score": [5, 1],
            "status": ["Final", "Final"],
            "game_type": ["S", "S"],
        }
    )
    sched.to_parquet(sched_dir / "games_2025.parquet", index=False)

    team_state = _build_team_state(minimal_features_df)
    park_factors = _park_factor_by_home_team(minimal_features_df)
    mlb_to_retro = {111: "BOS", 147: "NYA"}

    result = build_spring_features_for_season(
        2025,
        minimal_features_df,
        team_state,
        park_factors,
        mlb_to_retro,
        processed_dir=tmp_path,
    )

    assert result is not None
    assert len(result) == 1
    assert result.iloc[0]["home_win"] == 1.0
