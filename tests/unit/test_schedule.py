"""Tests for mlb_predict.mlbapi.schedule."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pandas as pd
import pytest

if TYPE_CHECKING:
    pass

from mlb_predict.mlbapi.schedule import (
    GAME_TYPE_REGULAR,
    GAME_TYPE_SPRING,
    SCHEDULE_FIELDS_MIN,
    fetch_schedule_chunk,
    normalize_schedule,
    parse_utc_iso,
    schedule_bounds,
    schedule_bounds_regular_season,
)


# ---------------------------------------------------------------------------
# parse_utc_iso
# ---------------------------------------------------------------------------


def test_parse_utc_iso_z_suffix() -> None:
    """parse_utc_iso must handle trailing Z and return an aware UTC datetime."""
    dt = parse_utc_iso("2024-04-01T17:10:00Z")
    assert dt == datetime(2024, 4, 1, 17, 10, 0, tzinfo=timezone.utc)


def test_parse_utc_iso_offset_format() -> None:
    """parse_utc_iso must pass through an explicit +00:00 offset unchanged."""
    dt = parse_utc_iso("2024-04-01T17:10:00+00:00")
    assert dt == datetime(2024, 4, 1, 17, 10, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# normalize_schedule
# ---------------------------------------------------------------------------


def test_normalize_schedule_basic(raw_schedule_response: dict) -> None:
    """normalize_schedule must produce one row per game with correct columns."""
    df = normalize_schedule(raw_schedule_response)
    assert len(df) == 2
    expected_cols = {
        "game_pk",
        "game_date_utc",
        "home_mlb_id",
        "away_mlb_id",
        "venue_id",
        "local_timezone",
        "double_header",
        "game_number",
        "status",
        "game_type",
        "home_score",
        "away_score",
    }
    assert expected_cols.issubset(set(df.columns))


def test_normalize_schedule_dtypes(raw_schedule_response: dict) -> None:
    """game_pk, home_mlb_id, away_mlb_id must be integer typed."""
    df = normalize_schedule(raw_schedule_response)
    assert df["game_pk"].dtype == int
    assert df["home_mlb_id"].dtype == int
    assert df["away_mlb_id"].dtype == int


def test_normalize_schedule_correct_values(raw_schedule_response: dict) -> None:
    """normalize_schedule must extract the correct game_pk and team IDs."""
    df = normalize_schedule(raw_schedule_response).sort_values("game_pk").reset_index(drop=True)
    assert df.loc[0, "game_pk"] == 745803
    assert df.loc[0, "home_mlb_id"] == 111
    assert df.loc[0, "away_mlb_id"] == 133


def test_normalize_schedule_drops_missing_game_pk() -> None:
    """Rows missing game_pk must be silently dropped."""
    raw = {
        "dates": [
            {
                "date": "2024-04-01",
                "games": [
                    {
                        "gamePk": None,
                        "gameDate": "2024-04-01T17:10:00Z",
                        "teams": {"home": {"team": {"id": 111}}, "away": {"team": {"id": 133}}},
                        "venue": {},
                        "doubleHeader": "N",
                        "gameNumber": 1,
                        "status": {},
                    },
                    {
                        "gamePk": 745804,
                        "gameDate": "2024-04-01T19:40:00Z",
                        "teams": {"home": {"team": {"id": 147}}, "away": {"team": {"id": 139}}},
                        "venue": {},
                        "doubleHeader": "N",
                        "gameNumber": 1,
                        "status": {},
                    },
                ],
            }
        ]
    }
    df = normalize_schedule(raw)
    assert len(df) == 1
    assert df.iloc[0]["game_pk"] == 745804


def test_normalize_schedule_drops_missing_team_ids() -> None:
    """Rows missing home or away team ID must be silently dropped."""
    raw = {
        "dates": [
            {
                "date": "2024-04-01",
                "games": [
                    # Missing away team ID
                    {
                        "gamePk": 1,
                        "gameDate": "2024-04-01T17:10:00Z",
                        "teams": {"home": {"team": {"id": 111}}, "away": {"team": {}}},
                        "venue": {},
                        "doubleHeader": "N",
                        "gameNumber": 1,
                        "status": {},
                    },
                ],
            }
        ]
    }
    df = normalize_schedule(raw)
    assert len(df) == 0


def test_normalize_schedule_empty_dates() -> None:
    """An empty dates list must produce an empty DataFrame."""
    df = normalize_schedule({"dates": []})
    assert len(df) == 0


def test_normalize_schedule_null_venue() -> None:
    """A None venue must be tolerated with venue_id and local_timezone set to None."""
    raw = {
        "dates": [
            {
                "date": "2024-04-01",
                "games": [
                    {
                        "gamePk": 1,
                        "gameDate": "2024-04-01T17:10:00Z",
                        "teams": {"home": {"team": {"id": 111}}, "away": {"team": {"id": 133}}},
                        "venue": None,
                        "doubleHeader": "N",
                        "gameNumber": 1,
                        "status": {},
                    },
                ],
            }
        ]
    }
    df = normalize_schedule(raw)
    assert len(df) == 1
    assert pd.isna(df.iloc[0]["venue_id"])
    assert pd.isna(df.iloc[0]["local_timezone"])


def test_normalize_schedule_no_duplicate_game_pks(raw_schedule_response: dict) -> None:
    """normalize_schedule must not produce duplicate game_pk rows from a single response."""
    df = normalize_schedule(raw_schedule_response)
    assert df["game_pk"].nunique() == len(df)


# ---------------------------------------------------------------------------
# SCHEDULE_FIELDS_MIN
# ---------------------------------------------------------------------------


def test_schedule_fields_min_is_nonempty_string() -> None:
    """SCHEDULE_FIELDS_MIN must be a non-empty comma-separated string."""
    assert isinstance(SCHEDULE_FIELDS_MIN, str)
    assert len(SCHEDULE_FIELDS_MIN) > 0
    assert "gamePk" in SCHEDULE_FIELDS_MIN


# ---------------------------------------------------------------------------
# schedule_bounds_regular_season (async, mocked client)
# ---------------------------------------------------------------------------


async def test_schedule_bounds_returns_first_and_last() -> None:
    """schedule_bounds_regular_season must return (first_date, last_date)."""
    raw = {
        "dates": [
            {"date": "2024-04-01"},
            {"date": "2024-09-29"},
            {"date": "2024-07-04"},
        ]
    }
    client = AsyncMock()
    client.get_json = AsyncMock(return_value=raw)
    start, end = await schedule_bounds_regular_season(client, season=2024)
    assert start == date(2024, 4, 1)
    assert end == date(2024, 9, 29)


async def test_schedule_bounds_empty_raises() -> None:
    """schedule_bounds_regular_season must raise RuntimeError when no dates returned."""
    client = AsyncMock()
    client.get_json = AsyncMock(return_value={"dates": []})
    with pytest.raises(RuntimeError, match="No dates returned"):
        await schedule_bounds_regular_season(client, season=2024)


async def test_schedule_bounds_single_date() -> None:
    """schedule_bounds_regular_season with a single date must return that date for both."""
    client = AsyncMock()
    client.get_json = AsyncMock(return_value={"dates": [{"date": "2024-04-01"}]})
    start, end = await schedule_bounds_regular_season(client, season=2024)
    assert start == end == date(2024, 4, 1)


# ---------------------------------------------------------------------------
# fetch_schedule_chunk (async, mocked client)
# ---------------------------------------------------------------------------


async def test_fetch_schedule_chunk_returns_dataframe(raw_schedule_response: dict) -> None:
    """fetch_schedule_chunk must return a normalized DataFrame."""
    client = AsyncMock()
    client.get_json = AsyncMock(return_value=raw_schedule_response)
    df = await fetch_schedule_chunk(
        client,
        season=2024,
        start_date=date(2024, 4, 1),
        end_date=date(2024, 4, 30),
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "game_pk" in df.columns
    assert "game_type" in df.columns
    assert (df["game_type"] == GAME_TYPE_REGULAR).all()


async def test_fetch_schedule_chunk_spring_training(raw_schedule_response: dict) -> None:
    """fetch_schedule_chunk with game_type='S' must tag all rows as spring training."""
    client = AsyncMock()
    client.get_json = AsyncMock(return_value=raw_schedule_response)
    df = await fetch_schedule_chunk(
        client,
        season=2024,
        start_date=date(2024, 2, 20),
        end_date=date(2024, 3, 24),
        game_type=GAME_TYPE_SPRING,
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert (df["game_type"] == GAME_TYPE_SPRING).all()


async def test_fetch_schedule_chunk_empty_response() -> None:
    """fetch_schedule_chunk with no games must return an empty DataFrame."""
    client = AsyncMock()
    client.get_json = AsyncMock(return_value={"dates": []})
    df = await fetch_schedule_chunk(
        client,
        season=2024,
        start_date=date(2024, 4, 1),
        end_date=date(2024, 4, 30),
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


# ---------------------------------------------------------------------------
# schedule_bounds (generic, async)
# ---------------------------------------------------------------------------


async def test_schedule_bounds_with_game_type() -> None:
    """schedule_bounds must pass game_type to the API and return correct dates."""
    raw = {"dates": [{"date": "2024-02-22"}, {"date": "2024-03-24"}]}
    client = AsyncMock()
    client.get_json = AsyncMock(return_value=raw)
    start, end = await schedule_bounds(client, season=2024, game_type=GAME_TYPE_SPRING)
    assert start == date(2024, 2, 22)
    assert end == date(2024, 3, 24)
    call_params = client.get_json.call_args[0][1]
    assert call_params["gameType"] == GAME_TYPE_SPRING


async def test_schedule_bounds_generic_empty_raises() -> None:
    """schedule_bounds must raise RuntimeError when no dates returned."""
    client = AsyncMock()
    client.get_json = AsyncMock(return_value={"dates": []})
    with pytest.raises(RuntimeError, match="No dates returned"):
        await schedule_bounds(client, season=2024, game_type=GAME_TYPE_REGULAR)


# ---------------------------------------------------------------------------
# normalize_schedule — game_type handling
# ---------------------------------------------------------------------------


def test_normalize_schedule_game_type_from_response() -> None:
    """normalize_schedule reads gameType from the API response when no override."""
    raw = {
        "dates": [
            {
                "date": "2024-02-22",
                "games": [
                    {
                        "gamePk": 1,
                        "gameDate": "2024-02-22T18:00:00Z",
                        "gameType": "S",
                        "teams": {"home": {"team": {"id": 111}}, "away": {"team": {"id": 133}}},
                        "venue": {},
                        "doubleHeader": "N",
                        "gameNumber": 1,
                        "status": {"detailedState": "Final"},
                    },
                ],
            }
        ]
    }
    df = normalize_schedule(raw)
    assert len(df) == 1
    assert df.iloc[0]["game_type"] == "S"


def test_normalize_schedule_game_type_override() -> None:
    """game_type_override must take precedence over the API response value."""
    raw = {
        "dates": [
            {
                "date": "2024-02-22",
                "games": [
                    {
                        "gamePk": 1,
                        "gameDate": "2024-02-22T18:00:00Z",
                        "gameType": "R",
                        "teams": {"home": {"team": {"id": 111}}, "away": {"team": {"id": 133}}},
                        "venue": {},
                        "doubleHeader": "N",
                        "gameNumber": 1,
                        "status": {"detailedState": "Final"},
                    },
                ],
            }
        ]
    }
    df = normalize_schedule(raw, game_type_override="S")
    assert df.iloc[0]["game_type"] == "S"


def test_normalize_schedule_extracts_scores(raw_schedule_response: dict) -> None:
    """normalize_schedule must extract home_score and away_score from the API response."""
    df = normalize_schedule(raw_schedule_response).sort_values("game_pk").reset_index(drop=True)
    assert df.loc[0, "home_score"] == 5
    assert df.loc[0, "away_score"] == 3
    assert df.loc[1, "home_score"] == 4
    assert df.loc[1, "away_score"] == 1


def test_normalize_schedule_scores_none_when_missing() -> None:
    """home_score and away_score must be None when scores are absent from the response."""
    raw = {
        "dates": [
            {
                "date": "2024-04-01",
                "games": [
                    {
                        "gamePk": 1,
                        "gameDate": "2024-04-01T18:00:00Z",
                        "teams": {
                            "home": {"team": {"id": 111}},
                            "away": {"team": {"id": 133}},
                        },
                        "venue": {},
                        "doubleHeader": "N",
                        "gameNumber": 1,
                        "status": {"detailedState": "Scheduled"},
                    },
                ],
            }
        ]
    }
    df = normalize_schedule(raw)
    assert len(df) == 1
    assert pd.isna(df.iloc[0]["home_score"])
    assert pd.isna(df.iloc[0]["away_score"])


def test_schedule_fields_min_includes_score_fields() -> None:
    """SCHEDULE_FIELDS_MIN must request score data from the API."""
    assert "teams,home,score" in SCHEDULE_FIELDS_MIN
    assert "teams,away,score" in SCHEDULE_FIELDS_MIN


def test_normalize_schedule_game_type_default() -> None:
    """When gameType is missing from response and no override, default to 'R'."""
    raw = {
        "dates": [
            {
                "date": "2024-04-01",
                "games": [
                    {
                        "gamePk": 1,
                        "gameDate": "2024-04-01T18:00:00Z",
                        "teams": {"home": {"team": {"id": 111}}, "away": {"team": {"id": 133}}},
                        "venue": {},
                        "doubleHeader": "N",
                        "gameNumber": 1,
                        "status": {"detailedState": "Final"},
                    },
                ],
            }
        ]
    }
    df = normalize_schedule(raw)
    assert df.iloc[0]["game_type"] == GAME_TYPE_REGULAR
