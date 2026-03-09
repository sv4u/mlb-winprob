"""MLB Stats API league leaders and full player stats fetcher.

Uses stats/leaders for top-N leaders by category and stats endpoint
for full batting/pitching stat tables (with pagination).
"""

from __future__ import annotations

import logging
from typing import Any

from .client import MLBAPIClient

logger = logging.getLogger(__name__)

# League IDs: AL = 103, NL = 104, both = 103,104
AL_ID = 103
NL_ID = 104

# Leader categories supported by stats/leaders (statGroup=hitting or pitching).
# See meta('leagueLeaderTypes') for full list.
HITTING_LEADER_CATEGORIES = [
    "homeRuns",
    "rbi",
    "battingAverage",
    "onBasePlusSlugging",
    "hits",
    "runs",
    "stolenBases",
    "baseOnBalls",
]
PITCHING_LEADER_CATEGORIES = [
    "earnedRunAverage",
    "strikeOuts",
    "wins",
    "saves",
    "inningsPitched",
    "whip",
    "strikeoutWalkRatio",
]


def _normalize_leader_entry(entry: dict[str, Any], rank: int, category: str) -> dict[str, Any]:
    """Build a flat leader row for one player/stat."""
    person = (entry.get("person") or {}) if isinstance(entry.get("person"), dict) else {}
    team = (entry.get("team") or {}) if isinstance(entry.get("team"), dict) else {}
    return {
        "rank": rank,
        "category": category,
        "person_id": person.get("id"),
        "name": (person.get("fullName") or person.get("name") or "").strip(),
        "team_id": team.get("id"),
        "team_name": (team.get("name") or "").strip(),
        "team_abbrev": (team.get("abbreviation") or "").strip(),
        "value": entry.get("value"),
        "stat_name": entry.get("stat", {}).get("name")
        if isinstance(entry.get("stat"), dict)
        else category,
    }


async def fetch_leaders(
    client: MLBAPIClient,
    *,
    season: int,
    league_id: int | None = None,
    limit: int = 20,
    stat_group: str = "hitting",
    categories: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Fetch league leaders for one stat group (hitting or pitching).

    Returns a list of leader tables: each element is
    { "category": str, "label": str, "leaders": [ { rank, person_id, name, team_*, value }, ... ] }.
    """
    if stat_group == "hitting":
        cat_list = categories or HITTING_LEADER_CATEGORIES
    else:
        cat_list = categories or PITCHING_LEADER_CATEGORIES

    leader_categories = ",".join(cat_list)
    params: dict[str, Any] = {
        "leaderCategories": leader_categories,
        "season": season,
        "limit": max(1, min(limit, 50)),
        "statGroup": stat_group,
    }
    if league_id is not None:
        params["leagueId"] = league_id

    raw = await client.get_json("stats/leaders", params)
    result: list[dict[str, Any]] = []

    leaders_list = raw.get("leagueLeaders", []) or []
    if isinstance(leaders_list, dict):
        leaders_list = [
            {"leaderCategory": k, "leaders": (v or {}).get("leaders", [])}
            for k, v in leaders_list.items()
        ]
    elif not isinstance(leaders_list, list):
        leaders_list = []

    for block in leaders_list:
        if not isinstance(block, dict):
            continue
        cat = block.get("leaderCategory", "") or block.get("statGroup", "")
        entries = block.get("leaders", []) or []
        label = (
            block.get("leaderCategoryDisplayName") or block.get("leaderCategory") or cat
        ).strip()
        rows = []
        for i, e in enumerate(entries[:limit], start=1):
            try:
                rows.append(_normalize_leader_entry(e, i, cat))
            except Exception as exc:
                logger.debug("Skip leader entry %s: %s", i, exc)
        result.append({"category": cat, "label": label or cat, "leaders": rows})
    return result


def _normalize_player_stat_row(
    stat: dict[str, Any],
    person: dict[str, Any],
    team: dict[str, Any],
    group: str,
) -> dict[str, Any]:
    """Build one player stat row from stats endpoint split.

    Note: MLB Stats API uses 'player' (not 'person') in stats endpoint splits.
    """
    person = person or {}
    team = team or {}
    stat = stat or {}
    row: dict[str, Any] = {
        "person_id": person.get("id"),
        "name": (person.get("fullName") or person.get("name") or "").strip(),
        "team_id": team.get("id"),
        "team_name": (team.get("name") or "").strip(),
        "team_abbrev": (team.get("abbreviation") or "").strip(),
        "group": group,
    }
    row.update({k: v for k, v in stat.items() if v is not None and v != ""})
    return row


async def fetch_player_stats(
    client: MLBAPIClient,
    *,
    season: int,
    group: str = "hitting",
    league_id: int | None = None,
    limit: int = 250,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Fetch full player stats table for a season (batting or pitching).

    Uses the stats endpoint with stats=season, group=hitting|pitching.
    Returns a list of player rows with person_id, name, team_*, and stat keys.
    """
    params: dict[str, Any] = {
        "stats": "season",
        "group": group,
        "season": season,
        "limit": max(1, min(limit, 250)),
        "offset": offset,
    }
    if league_id is not None:
        params["leagueId"] = league_id

    raw = await client.get_json("stats", params)
    result: list[dict[str, Any]] = []
    stats_list = raw.get("stats", []) or []
    if isinstance(stats_list, dict):
        stats_list = [stats_list]

    for st in stats_list:
        if not isinstance(st, dict):
            continue
        splits = st.get("splits", []) or []
        for sp in splits:
            if not isinstance(sp, dict):
                continue
            stat = sp.get("stat", {}) or {}
            # MLB Stats API uses "player" (not "person") in stats endpoint splits
            person = (
                (sp.get("player") or sp.get("person") or {})
                if isinstance(sp.get("player") or sp.get("person"), dict)
                else {}
            )
            team = (sp.get("team") or {}) if isinstance(sp.get("team"), dict) else {}
            try:
                result.append(_normalize_player_stat_row(stat, person, team, group))
            except Exception as exc:
                logger.debug("Skip player stat row: %s", exc)
    return result
