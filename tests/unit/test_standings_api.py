"""Tests for mlb_predict.mlbapi.standings — MLB Stats API standings and team stats."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pandas as pd
import pytest

if TYPE_CHECKING:
    pass

from mlb_predict.mlbapi.standings import (
    fetch_all_team_stats,
    fetch_standings,
    fetch_team_batting_stats,
    fetch_team_pitching_stats,
)


# ---------------------------------------------------------------------------
# Fixtures — raw API response payloads
# ---------------------------------------------------------------------------


@pytest.fixture
def raw_standings_response() -> dict:
    """Minimal raw MLB Stats API standings response with one division."""
    return {
        "records": [
            {
                "standingsType": "regularSeason",
                "league": {"id": 103},
                "division": {"id": 201},
                "teamRecords": [
                    {
                        "team": {"id": 111, "name": "Red Sox"},
                        "wins": 90,
                        "losses": 72,
                        "winningPercentage": ".556",
                        "gamesBack": "-",
                        "divisionRank": "1",
                        "leagueRank": "2",
                        "runsScored": 780,
                        "runsAllowed": 650,
                        "runDifferential": 130,
                    },
                    {
                        "team": {"id": 147, "name": "Yankees"},
                        "wins": 85,
                        "losses": 77,
                        "winningPercentage": ".525",
                        "gamesBack": "5.0",
                        "divisionRank": "2",
                        "leagueRank": "4",
                        "runsScored": 720,
                        "runsAllowed": 700,
                        "runDifferential": 20,
                    },
                ],
            }
        ]
    }


@pytest.fixture
def raw_batting_response() -> dict:
    """Raw MLB Stats API team batting stats response."""
    return {
        "stats": [
            {
                "splits": [
                    {
                        "stat": {
                            "avg": ".265",
                            "obp": ".340",
                            "slg": ".450",
                            "ops": ".790",
                            "runs": 780,
                            "hits": 1400,
                            "doubles": 280,
                            "triples": 25,
                            "homeRuns": 200,
                            "rbi": 750,
                            "stolenBases": 100,
                            "baseOnBalls": 500,
                            "strikeOuts": 1300,
                        }
                    }
                ]
            }
        ]
    }


@pytest.fixture
def raw_pitching_response() -> dict:
    """Raw MLB Stats API team pitching stats response."""
    return {
        "stats": [
            {
                "splits": [
                    {
                        "stat": {
                            "era": "3.50",
                            "wins": 90,
                            "losses": 72,
                            "saves": 45,
                            "inningsPitched": "1440.0",
                            "hits": 1250,
                            "baseOnBalls": 480,
                            "strikeOuts": 1500,
                            "whip": "1.20",
                            "homeRuns": 170,
                        }
                    }
                ]
            }
        ]
    }


# ---------------------------------------------------------------------------
# fetch_standings
# ---------------------------------------------------------------------------


async def test_fetch_standings_returns_dataframe(raw_standings_response: dict) -> None:
    """fetch_standings must return a DataFrame with the expected columns."""
    client = AsyncMock()
    client.get_json = AsyncMock(return_value=raw_standings_response)
    df = await fetch_standings(client, season=2025)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    expected_cols = {
        "team_id",
        "team_name",
        "league_id",
        "division_id",
        "wins",
        "losses",
        "pct",
        "gb",
        "division_rank",
        "runs_scored",
        "runs_allowed",
        "run_diff",
    }
    assert expected_cols.issubset(set(df.columns))


async def test_fetch_standings_correct_row_count(raw_standings_response: dict) -> None:
    """fetch_standings must return one row per team in the response."""
    client = AsyncMock()
    client.get_json = AsyncMock(return_value=raw_standings_response)
    df = await fetch_standings(client, season=2025)
    assert len(df) == 2


async def test_fetch_standings_correct_values(raw_standings_response: dict) -> None:
    """fetch_standings must extract correct wins, losses, and run differential."""
    client = AsyncMock()
    client.get_json = AsyncMock(return_value=raw_standings_response)
    df = await fetch_standings(client, season=2025)
    bos = df[df["team_id"] == 111].iloc[0]
    assert bos["wins"] == 90
    assert bos["losses"] == 72
    assert bos["run_diff"] == 130
    assert bos["division_rank"] == 1


async def test_fetch_standings_pct_parsed_as_float(raw_standings_response: dict) -> None:
    """winningPercentage string must be parsed to a numeric float."""
    client = AsyncMock()
    client.get_json = AsyncMock(return_value=raw_standings_response)
    df = await fetch_standings(client, season=2025)
    assert df["pct"].dtype == float
    bos = df[df["team_id"] == 111].iloc[0]
    assert abs(bos["pct"] - 0.556) < 0.001


async def test_fetch_standings_division_name_populated(raw_standings_response: dict) -> None:
    """Division name must be populated from the DIVISIONS constant."""
    client = AsyncMock()
    client.get_json = AsyncMock(return_value=raw_standings_response)
    df = await fetch_standings(client, season=2025)
    assert (df["division_name"] == "AL East").all()


async def test_fetch_standings_empty_response() -> None:
    """An empty records list must produce an empty DataFrame."""
    client = AsyncMock()
    client.get_json = AsyncMock(return_value={"records": []})
    df = await fetch_standings(client, season=2025)
    assert df.empty


# ---------------------------------------------------------------------------
# fetch_team_batting_stats
# ---------------------------------------------------------------------------


async def test_fetch_team_batting_stats_returns_dict(raw_batting_response: dict) -> None:
    """fetch_team_batting_stats must return a dict with batting stat keys."""
    client = AsyncMock()
    client.get_json = AsyncMock(return_value=raw_batting_response)
    result = await fetch_team_batting_stats(client, team_id=111, season=2025)
    assert isinstance(result, dict)
    assert "avg" in result
    assert "homeRuns" in result
    assert result["runs"] == 780


async def test_fetch_team_batting_stats_empty_response() -> None:
    """Empty stats response must return an empty dict."""
    client = AsyncMock()
    client.get_json = AsyncMock(return_value={"stats": []})
    result = await fetch_team_batting_stats(client, team_id=111, season=2025)
    assert result == {}


async def test_fetch_team_batting_stats_empty_splits() -> None:
    """Empty splits must return an empty dict."""
    client = AsyncMock()
    client.get_json = AsyncMock(return_value={"stats": [{"splits": []}]})
    result = await fetch_team_batting_stats(client, team_id=111, season=2025)
    assert result == {}


# ---------------------------------------------------------------------------
# fetch_team_pitching_stats
# ---------------------------------------------------------------------------


async def test_fetch_team_pitching_stats_returns_dict(raw_pitching_response: dict) -> None:
    """fetch_team_pitching_stats must return a dict with pitching stat keys."""
    client = AsyncMock()
    client.get_json = AsyncMock(return_value=raw_pitching_response)
    result = await fetch_team_pitching_stats(client, team_id=111, season=2025)
    assert isinstance(result, dict)
    assert "era" in result
    assert "whip" in result
    assert result["strikeOuts"] == 1500


async def test_fetch_team_pitching_stats_empty_response() -> None:
    """Empty stats response must return an empty dict."""
    client = AsyncMock()
    client.get_json = AsyncMock(return_value={"stats": []})
    result = await fetch_team_pitching_stats(client, team_id=111, season=2025)
    assert result == {}


# ---------------------------------------------------------------------------
# fetch_all_team_stats
# ---------------------------------------------------------------------------


async def test_fetch_all_team_stats_merges_batting_and_pitching(
    raw_batting_response: dict,
    raw_pitching_response: dict,
) -> None:
    """fetch_all_team_stats must merge batting + pitching onto the standings."""
    client = AsyncMock()

    async def mock_get_json(endpoint: str, params: dict) -> dict:
        if "hitting" in params.get("group", ""):
            return raw_batting_response
        return raw_pitching_response

    client.get_json = AsyncMock(side_effect=mock_get_json)

    standings = pd.DataFrame(
        {
            "team_id": [111],
            "team_name": ["Red Sox"],
            "wins": [90],
            "losses": [72],
            "pct": [0.556],
        }
    )
    result = await fetch_all_team_stats(client, standings_df=standings, season=2025)
    assert isinstance(result, pd.DataFrame)
    assert "bat_avg" in result.columns
    assert "pit_era" in result.columns
    assert len(result) == 1


async def test_fetch_all_team_stats_numeric_conversion(
    raw_batting_response: dict,
    raw_pitching_response: dict,
) -> None:
    """String stat values (AVG, ERA, etc.) must be converted to float."""
    client = AsyncMock()

    async def mock_get_json(endpoint: str, params: dict) -> dict:
        if "hitting" in params.get("group", ""):
            return raw_batting_response
        return raw_pitching_response

    client.get_json = AsyncMock(side_effect=mock_get_json)

    standings = pd.DataFrame(
        {
            "team_id": [111],
            "team_name": ["Red Sox"],
            "wins": [90],
            "losses": [72],
            "pct": [0.556],
        }
    )
    result = await fetch_all_team_stats(client, standings_df=standings, season=2025)
    row = result.iloc[0]
    assert isinstance(row["bat_avg"], float)
    assert abs(row["bat_avg"] - 0.265) < 0.001
    assert isinstance(row["pit_era"], float)
    assert abs(row["pit_era"] - 3.50) < 0.01
