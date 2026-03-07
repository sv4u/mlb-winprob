"""Tests for winprob.crosswalk.build."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass

from winprob.crosswalk.build import CrosswalkResult, _prep_schedule, build_crosswalk
from winprob.ingest.id_map import load_retro_team_map


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_schedule(*games: dict) -> pd.DataFrame:
    """Build a minimal schedule DataFrame from dicts of game fields."""
    rows = []
    for g in games:
        rows.append(
            {
                "game_pk": g.get("game_pk", 1),
                "game_date_utc": g.get("game_date_utc", "2024-04-01T17:10:00Z"),
                "home_mlb_id": g.get("home_mlb_id", 111),
                "away_mlb_id": g.get("away_mlb_id", 133),
                "venue_id": g.get("venue_id", 3),
                "local_timezone": g.get("local_timezone", "America/New_York"),
                "double_header": g.get("double_header", "N"),
                "game_number": g.get("game_number", 1),
                "status": g.get("status", "Final"),
                "game_type": g.get("game_type", "R"),
            }
        )
    return pd.DataFrame(rows)


def _make_gamelogs(*games: dict) -> pd.DataFrame:
    """Build a minimal gamelogs DataFrame from dicts of game fields."""
    rows = []
    for g in games:
        rows.append(
            {
                "date": g.get("date", datetime.date(2024, 4, 1)),
                "game_num": g.get("game_num", 0),
                "visiting_team": g.get("visiting_team", "OAK"),
                "home_team": g.get("home_team", "BOS"),
                "visiting_score": g.get("visiting_score", 3),
                "home_score": g.get("home_score", 5),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _prep_schedule
# ---------------------------------------------------------------------------


def test_prep_schedule_extracts_date() -> None:
    """_prep_schedule must extract a date column from game_date_utc."""
    sched = _make_schedule({"game_date_utc": "2024-04-01T17:10:00Z"})
    out = _prep_schedule(sched)
    assert out["date"].iloc[0] == datetime.date(2024, 4, 1)


def test_prep_schedule_handles_z_suffix() -> None:
    """_prep_schedule must correctly handle the trailing Z in UTC timestamps."""
    sched = _make_schedule({"game_date_utc": "2024-07-04T20:10:00Z"})
    out = _prep_schedule(sched)
    assert out["date"].iloc[0] == datetime.date(2024, 7, 4)


def test_prep_schedule_game_number_to_int64() -> None:
    """_prep_schedule must coerce game_number to nullable Int64."""
    sched = _make_schedule({"game_number": "2"})
    out = _prep_schedule(sched)
    assert out["game_number"].dtype == pd.Int64Dtype()
    assert out["game_number"].iloc[0] == 2


def test_prep_schedule_preserves_other_columns() -> None:
    """_prep_schedule must not drop the original columns."""
    sched = _make_schedule({"game_pk": 99999})
    out = _prep_schedule(sched)
    assert "game_pk" in out.columns
    assert out["game_pk"].iloc[0] == 99999


# ---------------------------------------------------------------------------
# build_crosswalk — result schema
# ---------------------------------------------------------------------------


def test_build_crosswalk_output_columns(retro_team_map_csv: Path) -> None:
    """build_crosswalk must produce a DataFrame with all required output columns."""
    retro_map = load_retro_team_map(retro_team_map_csv)
    sched = _make_schedule({"game_pk": 1, "home_mlb_id": 111, "away_mlb_id": 133})
    logs = _make_gamelogs({"home_team": "BOS", "visiting_team": "OAK"})

    result = build_crosswalk(season=2024, schedule=sched, gamelogs=logs, retro_team_map=retro_map)

    expected_cols = {
        "date",
        "home_mlb_id",
        "away_mlb_id",
        "home_retro",
        "away_retro",
        "dh_game_num",
        "status",
        "mlb_game_pk",
        "match_confidence",
        "notes",
    }
    assert expected_cols.issubset(set(result.df.columns))


def test_build_crosswalk_returns_crosswalk_result(retro_team_map_csv: Path) -> None:
    """build_crosswalk must return a CrosswalkResult dataclass."""
    retro_map = load_retro_team_map(retro_team_map_csv)
    sched = _make_schedule({"game_pk": 1, "home_mlb_id": 111, "away_mlb_id": 133})
    logs = _make_gamelogs({"home_team": "BOS", "visiting_team": "OAK"})

    result = build_crosswalk(season=2024, schedule=sched, gamelogs=logs, retro_team_map=retro_map)
    assert isinstance(result, CrosswalkResult)


# ---------------------------------------------------------------------------
# build_crosswalk — match scenarios
# ---------------------------------------------------------------------------


def test_build_crosswalk_matched_unique(retro_team_map_csv: Path) -> None:
    """A game present in both datasets with a unique game_pk must be 'matched'."""
    retro_map = load_retro_team_map(retro_team_map_csv)
    sched = _make_schedule(
        {"game_pk": 745803, "home_mlb_id": 111, "away_mlb_id": 133, "game_number": 1}
    )
    logs = _make_gamelogs({"home_team": "BOS", "visiting_team": "OAK", "game_num": 0})

    result = build_crosswalk(season=2024, schedule=sched, gamelogs=logs, retro_team_map=retro_map)

    assert result.matched == 1
    assert result.missing == 0
    assert result.ambiguous == 0
    row = result.df.iloc[0]
    assert row["status"] == "matched"
    assert row["mlb_game_pk"] == 745803
    assert row["match_confidence"] == 1.0


def test_build_crosswalk_missing(retro_team_map_csv: Path) -> None:
    """A gamelog game with no schedule match must be marked 'missing'."""
    retro_map = load_retro_team_map(retro_team_map_csv)
    # Schedule has NYY vs OAK, but gamelog has BOS vs OAK
    sched = _make_schedule({"game_pk": 1, "home_mlb_id": 147, "away_mlb_id": 133})
    logs = _make_gamelogs({"home_team": "BOS", "visiting_team": "OAK"})

    result = build_crosswalk(season=2024, schedule=sched, gamelogs=logs, retro_team_map=retro_map)

    assert result.missing == 1
    assert result.matched == 0
    row = result.df.iloc[0]
    assert row["status"] == "missing"
    assert pd.isna(row["mlb_game_pk"])
    assert row["match_confidence"] == 0.0


def test_build_crosswalk_doubleheader_disambiguation(retro_team_map_csv: Path) -> None:
    """Two DH games must be matched by game_num vs game_number."""
    retro_map = load_retro_team_map(retro_team_map_csv)
    sched = _make_schedule(
        {
            "game_pk": 100,
            "home_mlb_id": 111,
            "away_mlb_id": 133,
            "game_number": 1,
            "game_date_utc": "2024-04-01T17:10:00Z",
        },
        {
            "game_pk": 101,
            "home_mlb_id": 111,
            "away_mlb_id": 133,
            "game_number": 2,
            "game_date_utc": "2024-04-01T20:10:00Z",
        },
    )
    # Gamelog game_num=2 should match schedule game_number=2
    logs = _make_gamelogs(
        {"home_team": "BOS", "visiting_team": "OAK", "game_num": 2},
    )

    result = build_crosswalk(season=2024, schedule=sched, gamelogs=logs, retro_team_map=retro_map)

    assert result.matched == 1
    row = result.df.iloc[0]
    assert row["status"] == "matched"
    assert row["mlb_game_pk"] == 101
    assert row["match_confidence"] == 0.9
    assert row["notes"] == "matched_on_game_number"


# ---------------------------------------------------------------------------
# build_crosswalk — coverage metrics
# ---------------------------------------------------------------------------


def test_build_crosswalk_coverage_pct_all_matched(retro_team_map_csv: Path) -> None:
    """Coverage must be 100.0 when every game is matched."""
    retro_map = load_retro_team_map(retro_team_map_csv)
    sched = _make_schedule(
        {
            "game_pk": 1,
            "home_mlb_id": 111,
            "away_mlb_id": 133,
            "game_date_utc": "2024-04-01T17:10:00Z",
        },
        {
            "game_pk": 2,
            "home_mlb_id": 116,
            "away_mlb_id": 145,
            "game_date_utc": "2024-04-01T19:00:00Z",
        },
    )
    logs = _make_gamelogs(
        {"home_team": "BOS", "visiting_team": "OAK", "date": datetime.date(2024, 4, 1)},
        {"home_team": "DET", "visiting_team": "CWS", "date": datetime.date(2024, 4, 1)},
    )

    result = build_crosswalk(season=2024, schedule=sched, gamelogs=logs, retro_team_map=retro_map)
    assert result.coverage_pct == 100.0


def test_build_crosswalk_coverage_pct_none_matched(retro_team_map_csv: Path) -> None:
    """Coverage must be 0.0 when no games are matched."""
    retro_map = load_retro_team_map(retro_team_map_csv)
    # Schedule has NYY vs OAK but gamelog has BOS vs OAK
    sched = _make_schedule({"game_pk": 1, "home_mlb_id": 147, "away_mlb_id": 133})
    logs = _make_gamelogs({"home_team": "BOS", "visiting_team": "OAK"})

    result = build_crosswalk(season=2024, schedule=sched, gamelogs=logs, retro_team_map=retro_map)
    assert result.coverage_pct == 0.0


def test_build_crosswalk_coverage_in_range(retro_team_map_csv: Path) -> None:
    """Coverage percentage must always be in [0.0, 100.0]."""
    retro_map = load_retro_team_map(retro_team_map_csv)
    sched = _make_schedule(
        {"game_pk": 1, "home_mlb_id": 111, "away_mlb_id": 133},
        {"game_pk": 2, "home_mlb_id": 147, "away_mlb_id": 133},  # won't match
    )
    logs = _make_gamelogs(
        {"home_team": "BOS", "visiting_team": "OAK"},  # matches game 1
        {"home_team": "DET", "visiting_team": "OAK"},  # no schedule match
    )

    result = build_crosswalk(season=2024, schedule=sched, gamelogs=logs, retro_team_map=retro_map)
    assert 0.0 <= result.coverage_pct <= 100.0


# ---------------------------------------------------------------------------
# build_crosswalk — edge cases
# ---------------------------------------------------------------------------


def test_build_crosswalk_empty_gamelogs(retro_team_map_csv: Path) -> None:
    """An empty gamelogs DataFrame must produce an empty result with 100% coverage."""
    retro_map = load_retro_team_map(retro_team_map_csv)
    sched = _make_schedule({"game_pk": 1})
    logs = pd.DataFrame(
        columns=["date", "game_num", "visiting_team", "home_team", "visiting_score", "home_score"]
    )

    result = build_crosswalk(season=2024, schedule=sched, gamelogs=logs, retro_team_map=retro_map)
    assert len(result.df) == 0
    # With 0 games total, coverage = 0/max(0,1)*100 = 0
    assert result.coverage_pct == 0.0


def test_build_crosswalk_retro_columns_renamed(retro_team_map_csv: Path) -> None:
    """home_team and visiting_team must be renamed to home_retro and away_retro."""
    retro_map = load_retro_team_map(retro_team_map_csv)
    sched = _make_schedule({"game_pk": 1, "home_mlb_id": 111, "away_mlb_id": 133})
    logs = _make_gamelogs({"home_team": "BOS", "visiting_team": "OAK"})

    result = build_crosswalk(season=2024, schedule=sched, gamelogs=logs, retro_team_map=retro_map)

    assert "home_retro" in result.df.columns
    assert "away_retro" in result.df.columns
    assert "home_team" not in result.df.columns
    assert "visiting_team" not in result.df.columns


def test_crosswalk_result_counts_sum_to_total(retro_team_map_csv: Path) -> None:
    """matched + missing + ambiguous must equal the number of rows in result.df."""
    retro_map = load_retro_team_map(retro_team_map_csv)
    sched = _make_schedule(
        {"game_pk": 1, "home_mlb_id": 111, "away_mlb_id": 133},
    )
    logs = _make_gamelogs(
        {"home_team": "BOS", "visiting_team": "OAK"},
        {"home_team": "DET", "visiting_team": "CWS"},  # no schedule match
    )

    result = build_crosswalk(season=2024, schedule=sched, gamelogs=logs, retro_team_map=retro_map)
    total = result.matched + result.missing + result.ambiguous
    assert total == len(result.df)
