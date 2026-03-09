"""Tests for mlb_predict.mlbapi.teams."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pandas as pd
import pytest

if TYPE_CHECKING:
    pass

from mlb_predict.mlbapi.teams import build_team_maps, get_teams_df


# ---------------------------------------------------------------------------
# build_team_maps
# ---------------------------------------------------------------------------


def test_build_team_maps_id_to_abbrev(teams_df: pd.DataFrame) -> None:
    """build_team_maps must produce a correct mlb_id → abbreviation mapping."""
    maps = build_team_maps(teams_df)
    assert maps.mlb_id_to_abbrev[111] == "BOS"
    assert maps.mlb_id_to_abbrev[147] == "NYY"
    assert maps.mlb_id_to_abbrev[133] == "OAK"


def test_build_team_maps_abbrev_to_id(teams_df: pd.DataFrame) -> None:
    """build_team_maps must produce a correct abbreviation → mlb_id mapping."""
    maps = build_team_maps(teams_df)
    assert maps.abbrev_to_mlb_id["BOS"] == 111
    assert maps.abbrev_to_mlb_id["NYY"] == 147
    assert maps.abbrev_to_mlb_id["OAK"] == 133


def test_build_team_maps_id_to_name(teams_df: pd.DataFrame) -> None:
    """build_team_maps must produce a correct mlb_id → team name mapping."""
    maps = build_team_maps(teams_df)
    assert maps.mlb_id_to_name[111] == "Boston Red Sox"
    assert maps.mlb_id_to_name[147] == "New York Yankees"


def test_build_team_maps_is_frozen(teams_df: pd.DataFrame) -> None:
    """TeamMaps must be a frozen dataclass — attribute reassignment must raise."""
    maps = build_team_maps(teams_df)
    with pytest.raises((AttributeError, TypeError)):
        maps.mlb_id_to_abbrev = {}  # type: ignore[misc]


def test_build_team_maps_all_keys_present(teams_df: pd.DataFrame) -> None:
    """All three dicts must have the same set of team IDs as keys."""
    maps = build_team_maps(teams_df)
    assert set(maps.mlb_id_to_abbrev) == set(maps.mlb_id_to_name)


def test_build_team_maps_roundtrip(teams_df: pd.DataFrame) -> None:
    """mlb_id → abbrev → mlb_id must round-trip correctly."""
    maps = build_team_maps(teams_df)
    for team_id in [111, 147, 133]:
        abbrev = maps.mlb_id_to_abbrev[team_id]
        recovered_id = maps.abbrev_to_mlb_id[abbrev]
        assert recovered_id == team_id


def test_build_team_maps_empty_df() -> None:
    """build_team_maps on an empty DataFrame must produce empty dicts."""
    df = pd.DataFrame(columns=["season", "mlb_team_id", "abbrev", "name"])
    maps = build_team_maps(df)
    assert maps.mlb_id_to_abbrev == {}
    assert maps.abbrev_to_mlb_id == {}
    assert maps.mlb_id_to_name == {}


# ---------------------------------------------------------------------------
# get_teams_df (async, mocked client)
# ---------------------------------------------------------------------------


async def test_get_teams_df_returns_dataframe(raw_teams_response: dict) -> None:
    """get_teams_df must return a DataFrame with the expected columns."""
    client = AsyncMock()
    client.get_json = AsyncMock(return_value=raw_teams_response)
    df = await get_teams_df(client, season=2024)
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"season", "mlb_team_id", "abbrev", "name"}
    assert len(df) == 3


async def test_get_teams_df_correct_values(raw_teams_response: dict) -> None:
    """get_teams_df must return rows with correct team data."""
    client = AsyncMock()
    client.get_json = AsyncMock(return_value=raw_teams_response)
    df = await get_teams_df(client, season=2024)
    row = df[df["mlb_team_id"] == 111].iloc[0]
    assert row["abbrev"] == "BOS"
    assert row["name"] == "Boston Red Sox"
    assert row["season"] == 2024


async def test_get_teams_df_drops_missing_id() -> None:
    """get_teams_df must drop teams with no id field."""
    raw = {
        "teams": [
            {"id": 111, "abbreviation": "BOS", "name": "Boston Red Sox"},
            {"abbreviation": "XX", "name": "No ID Team"},  # missing id
        ]
    }
    client = AsyncMock()
    client.get_json = AsyncMock(return_value=raw)
    df = await get_teams_df(client, season=2024)
    assert len(df) == 1
    assert df.iloc[0]["mlb_team_id"] == 111


async def test_get_teams_df_season_column(raw_teams_response: dict) -> None:
    """The season column must match the requested season."""
    client = AsyncMock()
    client.get_json = AsyncMock(return_value=raw_teams_response)
    df = await get_teams_df(client, season=2015)
    assert (df["season"] == 2015).all()


async def test_get_teams_df_empty_response() -> None:
    """An empty teams list must produce an empty DataFrame."""
    client = AsyncMock()
    client.get_json = AsyncMock(return_value={"teams": []})
    df = await get_teams_df(client, season=2024)
    assert len(df) == 0
