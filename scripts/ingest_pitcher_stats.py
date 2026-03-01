"""Ingest individual pitcher season statistics from the MLB Stats API.

Writes one Parquet file per season to data/processed/pitcher_stats/.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

import pandas as pd

from winprob.mlbapi.client import MLBAPIClient, MLBAPIConfig
from winprob.mlbapi.pitcher_stats import fetch_pitcher_season_stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def ingest_season(
    client: MLBAPIClient,
    season: int,
    out_dir: Path,
    *,
    refresh: bool,
) -> dict:
    out_path = out_dir / f"pitchers_{season}.parquet"
    try:
        df = await fetch_pitcher_season_stats(client, season)
        if df.empty:
            logger.warning("%d: no pitcher stats returned", season)
            return {"season": season, "status": "empty", "n_pitchers": 0}

        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        starters = (df["games_started"] > 0).sum()
        logger.info("%d: %d pitchers (%d starters) → %s", season, len(df), starters, out_path)
        return {"season": season, "status": "ok", "n_pitchers": len(df), "n_starters": int(starters)}
    except Exception as exc:
        logger.error("%d: FAILED — %s", season, exc)
        return {"season": season, "status": "failed", "error": str(exc)}


async def main_async(args: argparse.Namespace) -> None:
    seasons = args.seasons or list(range(2000, 2026))
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    config = MLBAPIConfig()
    sem = asyncio.Semaphore(4)

    async def bounded(season: int) -> dict:
        async with sem:
            async with MLBAPIClient(config=config) as client:
                return await ingest_season(client, season, out_dir, refresh=args.refresh)

    results = await asyncio.gather(*[bounded(s) for s in seasons])
    summary = pd.DataFrame(results)
    summary_path = out_dir / "ingest_pitcher_stats_summary.json"
    summary.to_json(summary_path, orient="records", indent=2)
    logger.info("Summary → %s", summary_path)

    failed = summary[summary["status"] != "ok"]
    if not failed.empty:
        logger.warning("Failed seasons: %s", failed["season"].tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", type=int, default=[])
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/pitcher_stats"),
    )
    ap.add_argument("--refresh", action="store_true", help="Bypass disk cache")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
