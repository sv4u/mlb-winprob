"""MLB Stats API standings and team statistics fetcher.

Provides async functions to fetch live standings (W/L/PCT/GB by division)
and aggregate team batting/pitching statistics for a given season.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import pandas as pd

from mlb_predict.standings import DIVISIONS

from .client import MLBAPIClient

logger = logging.getLogger(__name__)

_STANDINGS_FIELDS = ",".join(
    [
        "records",
        "standingsType",
        "league,id,name",
        "division,id,name",
        "teamRecords,team,id,name,abbreviation",
        "teamRecords,wins",
        "teamRecords,losses",
        "teamRecords,winningPercentage",
        "teamRecords,gamesBack",
        "teamRecords,divisionRank",
        "teamRecords,leagueRank",
        "teamRecords,runsScored",
        "teamRecords,runsAllowed",
        "teamRecords,runDifferential",
    ]
)

# League IDs: AL = 103, NL = 104
_AL_ID = 103
_NL_ID = 104

_HITTING_FIELDS = ",".join(
    [
        "stats,splits,stat",
        "avg,obp,slg,ops",
        "runs,hits,doubles,triples,homeRuns,rbi",
        "stolenBases,baseOnBalls,strikeOuts",
    ]
)

_PITCHING_FIELDS = ",".join(
    [
        "stats,splits,stat",
        "era,wins,losses,saves",
        "inningsPitched,hits,baseOnBalls,strikeOuts",
        "whip,homeRuns",
    ]
)


async def fetch_standings(
    client: MLBAPIClient,
    *,
    season: int,
) -> pd.DataFrame:
    """Fetch MLB regular-season standings for both leagues.

    Returns a DataFrame with one row per team, including:
    team_id, team_name, abbreviation, league_id, league_name,
    division_id, division_name, wins, losses, pct, gb,
    division_rank, league_rank, runs_scored, runs_allowed, run_diff.
    """
    params: Mapping[str, Any] = {
        "leagueId": f"{_AL_ID},{_NL_ID}",
        "season": season,
        "standingsTypes": "regularSeason",
    }
    raw = await client.get_json("standings", params)

    rows: list[dict[str, Any]] = []
    for record in raw.get("records", []):
        league = record.get("league", {})
        division = record.get("division", {})
        div_id = division.get("id")
        div_meta = DIVISIONS.get(div_id, {})
        for tr in record.get("teamRecords", []):
            team = tr.get("team", {})
            rows.append(
                {
                    "team_id": team.get("id"),
                    "team_name": team.get("name", ""),
                    "league_id": league.get("id"),
                    "league_name": div_meta.get("league", league.get("name", "")),
                    "division_id": div_id,
                    "division_name": div_meta.get("name", division.get("name", "")),
                    "wins": tr.get("wins", 0),
                    "losses": tr.get("losses", 0),
                    "pct": tr.get("winningPercentage", ".000"),
                    "gb": tr.get("gamesBack", "-"),
                    "division_rank": int(tr.get("divisionRank") or 0),
                    "league_rank": int(tr.get("leagueRank") or 0),
                    "runs_scored": tr.get("runsScored", 0),
                    "runs_allowed": tr.get("runsAllowed", 0),
                    "run_diff": tr.get("runDifferential", 0),
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["pct"] = df["pct"].astype(str).str.strip().str.replace('"', "")
    df["pct"] = pd.to_numeric(df["pct"], errors="coerce").fillna(0.0)
    return df


async def fetch_team_batting_stats(
    client: MLBAPIClient,
    *,
    team_id: int,
    season: int,
) -> dict[str, Any]:
    """Fetch aggregate team batting statistics for a season."""
    params: Mapping[str, Any] = {
        "stats": "season",
        "season": season,
        "group": "hitting",
    }
    raw = await client.get_json(f"teams/{team_id}/stats", params)
    stats_list = raw.get("stats", [])
    if not stats_list:
        return {}
    splits = stats_list[0].get("splits", [])
    if not splits:
        return {}
    return splits[0].get("stat", {})


async def fetch_team_pitching_stats(
    client: MLBAPIClient,
    *,
    team_id: int,
    season: int,
) -> dict[str, Any]:
    """Fetch aggregate team pitching statistics for a season."""
    params: Mapping[str, Any] = {
        "stats": "season",
        "season": season,
        "group": "pitching",
    }
    raw = await client.get_json(f"teams/{team_id}/stats", params)
    stats_list = raw.get("stats", [])
    if not stats_list:
        return {}
    splits = stats_list[0].get("splits", [])
    if not splits:
        return {}
    return splits[0].get("stat", {})


async def fetch_all_team_stats(
    client: MLBAPIClient,
    *,
    standings_df: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    """Fetch batting + pitching stats for every team in the standings.

    Merges onto the standings DataFrame by team_id.
    """
    import asyncio

    team_ids = standings_df["team_id"].unique().tolist()

    async def _fetch_one(tid: int) -> dict[str, Any]:
        batting = await fetch_team_batting_stats(client, team_id=tid, season=season)
        pitching = await fetch_team_pitching_stats(client, team_id=tid, season=season)
        return {
            "team_id": tid,
            # Batting
            "bat_avg": batting.get("avg", ""),
            "bat_obp": batting.get("obp", ""),
            "bat_slg": batting.get("slg", ""),
            "bat_ops": batting.get("ops", ""),
            "bat_runs": batting.get("runs", 0),
            "bat_hits": batting.get("hits", 0),
            "bat_doubles": batting.get("doubles", 0),
            "bat_triples": batting.get("triples", 0),
            "bat_hr": batting.get("homeRuns", 0),
            "bat_rbi": batting.get("rbi", 0),
            "bat_sb": batting.get("stolenBases", 0),
            "bat_bb": batting.get("baseOnBalls", 0),
            "bat_so": batting.get("strikeOuts", 0),
            # Pitching
            "pit_era": pitching.get("era", ""),
            "pit_wins": pitching.get("wins", 0),
            "pit_losses": pitching.get("losses", 0),
            "pit_saves": pitching.get("saves", 0),
            "pit_ip": pitching.get("inningsPitched", ""),
            "pit_hits": pitching.get("hits", 0),
            "pit_bb": pitching.get("baseOnBalls", 0),
            "pit_so": pitching.get("strikeOuts", 0),
            "pit_whip": pitching.get("whip", ""),
            "pit_hr": pitching.get("homeRuns", 0),
        }

    results = await asyncio.gather(*[_fetch_one(tid) for tid in team_ids])
    stats_df = pd.DataFrame(results)

    for col in ["bat_avg", "bat_obp", "bat_slg", "bat_ops", "pit_era", "pit_whip"]:
        stats_df[col] = stats_df[col].astype(str).str.strip().str.replace('"', "")
        stats_df[col] = pd.to_numeric(stats_df[col], errors="coerce").fillna(0.0)

    merged = standings_df.merge(stats_df, on="team_id", how="left")
    return merged
