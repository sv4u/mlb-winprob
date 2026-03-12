"""Per-pitcher game logs from the MLB Stats API.

Fetches individual pitcher appearance data (IP, H, ER, BB, K per game) using
the ``/people/{id}/stats?stats=gameLog&group=pitching`` endpoint.  Results are
cached as Parquet files at ``data/processed/player/pitcher_gamelogs_{season}.parquet``.

This replaces the team-level approximation in rolling.py with high-fidelity
per-pitcher box score data.  Coverage: 2000–present.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pandas as pd

from mlb_predict.mlbapi.client import MLBAPIClient

logger = logging.getLogger(__name__)

_GAMELOG_ENDPOINT_TPL = "people/{player_id}/stats"
_CACHE_PREFIX = "pitcher_gamelogs"


def _parse_pitcher_gamelog(
    raw: dict,
    mlbam_id: int,
    season: int,
) -> list[dict]:
    """Extract per-game pitcher stats from a gameLog API response."""
    rows: list[dict] = []
    for stat_group in raw.get("stats", []):
        for split in stat_group.get("splits", []):
            game_date = split.get("date", "")
            stat = split.get("stat", {})

            ip_str = stat.get("inningsPitched", "0")
            try:
                ip_val = float(ip_str)
                whole = int(ip_val)
                frac = round(ip_val - whole, 1)
                ip = whole + frac / 0.3
            except (TypeError, ValueError):
                ip = 0.0

            if ip <= 0:
                continue

            is_start = bool(stat.get("gamesStarted", 0))

            rows.append(
                {
                    "date": game_date,
                    "mlbam_id": mlbam_id,
                    "season": season,
                    "is_start": is_start,
                    "ip": round(ip, 4),
                    "hits": int(stat.get("hits", 0) or 0),
                    "earned_runs": int(stat.get("earnedRuns", 0) or 0),
                    "bb": int(stat.get("baseOnBalls", 0) or 0),
                    "k": int(stat.get("strikeOuts", 0) or 0),
                    "hr": int(stat.get("homeRuns", 0) or 0),
                    "runs": int(stat.get("runs", 0) or 0),
                    "batters_faced": int(stat.get("battersFaced", 0) or 0),
                }
            )
    return rows


async def fetch_pitcher_gamelogs_for_player(
    client: MLBAPIClient,
    mlbam_id: int,
    season: int,
) -> list[dict]:
    """Fetch one pitcher's game log for a single season."""
    endpoint = _GAMELOG_ENDPOINT_TPL.format(player_id=mlbam_id)
    params = {
        "stats": "gameLog",
        "group": "pitching",
        "season": str(season),
        "sportId": "1",
    }
    try:
        raw = await client.get_json(endpoint, params)
    except Exception as exc:
        logger.debug("Game log fetch failed for %d/%d: %s", mlbam_id, season, exc)
        return []
    return _parse_pitcher_gamelog(raw, mlbam_id, season)


async def fetch_all_pitcher_gamelogs(
    client: MLBAPIClient,
    mlbam_ids: list[int],
    season: int,
    *,
    concurrency: int = 6,
) -> pd.DataFrame:
    """Fetch game logs for all pitchers in a season.

    Returns a DataFrame with columns: date, mlbam_id, season, is_start, ip,
    hits, earned_runs, bb, k, hr, runs, batters_faced.
    """
    sem = asyncio.Semaphore(concurrency)
    all_rows: list[dict] = []

    async def _fetch_one(mid: int) -> None:
        async with sem:
            rows = await fetch_pitcher_gamelogs_for_player(client, mid, season)
            all_rows.extend(rows)

    tasks = [_fetch_one(mid) for mid in mlbam_ids]
    await asyncio.gather(*tasks)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.sort_values(["date", "mlbam_id"]).reset_index(drop=True)


def load_pitcher_gamelogs(
    cache_dir: Path,
    seasons: list[int],
) -> pd.DataFrame:
    """Load cached pitcher game log Parquet files for the given seasons."""
    frames: list[pd.DataFrame] = []
    for s in seasons:
        path = cache_dir / f"{_CACHE_PREFIX}_{s}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def save_pitcher_gamelogs(
    df: pd.DataFrame,
    cache_dir: Path,
    season: int,
) -> Path:
    """Save pitcher game logs for a season to Parquet."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{_CACHE_PREFIX}_{season}.parquet"
    df.to_parquet(path, index=False)
    logger.info("Saved %d pitcher game log entries for %d → %s", len(df), season, path)
    return path
