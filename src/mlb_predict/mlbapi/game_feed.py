"""MLB Stats API game feed and play-by-play fetcher.

Fetches live game feed (play-by-play) for a given gamePk. Uses the
game/{gamePk}/feed/live endpoint which includes liveData.plays.allPlays.
"""

from __future__ import annotations

import logging
from typing import Any

from .client import MLBAPIClient

logger = logging.getLogger(__name__)


def _normalize_play(play: dict[str, Any], index: int) -> dict[str, Any]:
    """Extract a minimal, UI-friendly play record from raw API play object."""
    about = play.get("about", {})
    result = play.get("result", {}) or {}
    matchup = play.get("matchup", {}) or {}
    batter = (matchup.get("batter") or {}) if isinstance(matchup.get("batter"), dict) else {}
    pitcher = (matchup.get("pitcher") or {}) if isinstance(matchup.get("pitcher"), dict) else {}
    return {
        "index": index,
        "at_bat_index": play.get("atBatIndex"),
        "inning": about.get("inning", 0),
        "half": about.get("halfInning", "").lower(),  # "top" | "bottom"
        "inning_label": f"{about.get('inning', 0)} {'Top' if (about.get('halfInning') or '').lower() == 'top' else 'Bottom'}",
        "outs": about.get("outs", 0),
        "description": (result.get("description") or play.get("description") or "").strip(),
        "event": result.get("event", ""),
        "event_type": result.get("eventType", ""),
        "runs": about.get("runs", 0),
        "batter_id": batter.get("id"),
        "batter_name": (batter.get("fullName") or batter.get("name") or "").strip(),
        "pitcher_id": pitcher.get("id"),
        "pitcher_name": (pitcher.get("fullName") or pitcher.get("name") or "").strip(),
        "home_score": about.get("homeScore")
        if isinstance(about.get("homeScore"), (int, float))
        else None,
        "away_score": about.get("awayScore")
        if isinstance(about.get("awayScore"), (int, float))
        else None,
    }


def _normalize_plays(all_plays: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize allPlays array to a list of minimal play objects."""
    out: list[dict[str, Any]] = []
    for i, p in enumerate(all_plays or []):
        try:
            out.append(_normalize_play(p, i))
        except Exception as e:
            logger.debug("Skip play %s: %s", i, e)
    return out


def _game_info_from_feed(raw: dict[str, Any]) -> dict[str, Any]:
    """Extract game metadata from feed/live response."""
    game_data = raw.get("gameData", {}) or {}
    live_data = raw.get("liveData", {}) or {}
    game = raw.get("gamePk") or game_data.get("game", {}).get("pk")
    teams = game_data.get("teams", {}) or {}
    datetime_info = game_data.get("datetime", {}) or {}
    status = (game_data.get("status") or raw.get("status") or {}) or {}
    box = live_data.get("boxscore", {}) or {}
    teams_box = box.get("teams", {}) or {}
    return {
        "game_pk": game,
        "game_date": datetime_info.get("date", ""),
        "status": status.get("detailedState", ""),
        "home_team_id": (teams.get("home") or {}).get("id")
        if isinstance(teams.get("home"), dict)
        else None,
        "home_team_name": (teams.get("home") or {}).get("name", "")
        if isinstance(teams.get("home"), dict)
        else "",
        "away_team_id": (teams.get("away") or {}).get("id")
        if isinstance(teams.get("away"), dict)
        else None,
        "away_team_name": (teams.get("away") or {}).get("name", "")
        if isinstance(teams.get("away"), dict)
        else "",
        "home_score": (teams_box.get("home") or {}).get("teamStats", {}).get("runs")
        if isinstance(teams_box.get("home"), dict)
        else None,
        "away_score": (teams_box.get("away") or {}).get("teamStats", {}).get("runs")
        if isinstance(teams_box.get("away"), dict)
        else None,
    }


async def fetch_game_feed(
    client: MLBAPIClient,
    *,
    game_pk: int,
) -> dict[str, Any]:
    """Fetch full game feed (live data including play-by-play) for a game.

    Uses the game/{gamePk}/feed/live endpoint. Returns a dict with:
    - game: normalized game info (game_pk, teams, date, status, scores)
    - plays: list of normalized play objects (inning, half, description, batter, pitcher, runs, etc.)
    - raw_plays_count: number of plays in raw allPlays (before normalization)
    """
    endpoint = f"game/{game_pk}/feed/live"
    params: dict[str, Any] = {}
    raw = await client.get_json(endpoint, params)

    live_data = raw.get("liveData", {}) or {}
    plays_data = live_data.get("plays", {}) or {}
    all_plays = plays_data.get("allPlays") or []

    game_info = _game_info_from_feed(raw)
    # Prefer linescore for final score if present
    linescore = live_data.get("linescore", {}) or {}
    teams_ls_raw = linescore.get("teams")
    teams_ls = teams_ls_raw if isinstance(teams_ls_raw, dict) else {}
    if teams_ls:
        home_raw = teams_ls.get("home")
        away_raw = teams_ls.get("away")
        home_team = home_raw if isinstance(home_raw, dict) else {}
        away_team = away_raw if isinstance(away_raw, dict) else {}
        if "runs" in home_team:
            game_info["home_score"] = home_team.get("runs")
        if "runs" in away_team:
            game_info["away_score"] = away_team.get("runs")

    plays = _normalize_plays(all_plays)
    return {
        "game": game_info,
        "plays": plays,
        "raw_plays_count": len(all_plays),
    }
