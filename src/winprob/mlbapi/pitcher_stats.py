"""Fetch individual pitcher season statistics from the MLB Stats API.

For each season we pull all pitchers' season totals (ERA, K/9, BB/9, WHIP,
innings pitched) via the ``/stats`` endpoint.  The resulting DataFrame is
keyed by (season, player_name) so it can be joined to Retrosheet gamelogs via
the starting pitcher name columns.

We also try a secondary join on (player_id_mlb) using the player-lookup
endpoint so matches survive name variations.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import pandas as pd

from winprob.mlbapi.client import MLBAPIClient

logger = logging.getLogger(__name__)

_PITCHING_STATS_ENDPOINT = "stats"

# Columns we extract from each pitching split
_STAT_KEYS = [
    "era",
    "strikeOuts",
    "baseOnBalls",
    "inningsPitched",
    "whip",
    "wins",
    "losses",
    "gamesStarted",
    "hits",
    "homeRuns",
    "earnedRuns",
]


@dataclass(frozen=True)
class PitcherSeasonStats:
    """Aggregated season pitching stats for one player."""

    player_id: int
    player_name: str
    season: int
    era: float
    k9: float        # strikeouts per 9 innings
    bb9: float       # walks per 9 innings
    fip_raw: float   # HR*13 + BB*3 - K*2 (per IP * 9), un-constant-adjusted
    whip: float
    ip: float
    games_started: int


def _ip_to_float(ip_str: str | float) -> float:
    """Convert Retrosheet/API innings-pitched string (e.g. '6.1') to float IP."""
    try:
        val = float(ip_str)
        whole = int(val)
        frac = round(val - whole, 1)
        return whole + frac / 0.3  # .1 → 1/3, .2 → 2/3
    except (TypeError, ValueError):
        return 0.0


def _parse_pitching_splits(data: dict, season: int) -> list[dict]:
    """Extract pitcher stats rows from raw API response."""
    rows = []
    for split in data.get("stats", []):
        for entry in split.get("splits", []):
            player = entry.get("player", {})
            stat = entry.get("stat", {})

            pid = player.get("id")
            name = player.get("fullName", "")
            if not pid:
                continue

            ip = _ip_to_float(stat.get("inningsPitched", 0))
            if ip < 1.0:
                continue  # skip pitchers with effectively no innings

            k = float(stat.get("strikeOuts", 0) or 0)
            bb = float(stat.get("baseOnBalls", 0) or 0)
            hr = float(stat.get("homeRuns", 0) or 0)
            er = float(stat.get("earnedRuns", 0) or 0)
            era_raw = stat.get("era", "-.--")
            whip_raw = stat.get("whip", "-.--")

            try:
                era = float(era_raw)
            except (TypeError, ValueError):
                era = (er / ip * 9) if ip > 0 else 4.50

            try:
                whip = float(whip_raw)
            except (TypeError, ValueError):
                whip = 1.35

            k9 = (k / ip * 9) if ip > 0 else 0.0
            bb9 = (bb / ip * 9) if ip > 0 else 0.0
            fip_raw = ((hr * 13 + bb * 3 - k * 2) / ip * 9) if ip > 0 else 0.0

            rows.append(
                {
                    "player_id": int(pid),
                    "player_name": name,
                    "season": season,
                    "era": era,
                    "k9": k9,
                    "bb9": bb9,
                    "fip_raw": fip_raw,
                    "whip": whip,
                    "ip": ip,
                    "games_started": int(stat.get("gamesStarted", 0) or 0),
                }
            )
    return rows


async def fetch_pitcher_season_stats(
    client: MLBAPIClient,
    season: int,
    *,
    min_ip: float = 10.0,
) -> pd.DataFrame:
    """Fetch all pitchers' season stats for one season from the MLB Stats API.

    Parameters
    ----------
    client:
        Authenticated ``MLBAPIClient`` instance.
    season:
        Season year (e.g. 2024).
    min_ip:
        Minimum innings pitched; pitchers below this threshold are excluded.
    refresh:
        If ``True``, bypass the local disk cache.

    Returns
    -------
    DataFrame
        One row per pitcher.  Columns: player_id, player_name, season, era,
        k9, bb9, fip_raw, whip, ip, games_started.
    """
    params = {
        "stats": "season",
        "season": str(season),
        "group": "pitching",
        "sportId": "1",
        "playerPool": "All",
        "limit": "2000",
    }
    raw = await client.get_json(_PITCHING_STATS_ENDPOINT, params)
    rows = _parse_pitching_splits(raw, season)
    if not rows:
        logger.warning("No pitcher stats returned for season %d", season)
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df[df["ip"] >= min_ip].reset_index(drop=True)
    return df
