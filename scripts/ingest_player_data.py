"""Ingest player-level data: FanGraphs stats, Statcast stats, biographical
data, and per-pitcher game logs from the MLB Stats API.

Fetches per-player season stats from FanGraphs (2002+) and Statcast (2015+),
biographical data from the Chadwick register, and individual pitcher game logs
(IP, H, ER, BB, K per appearance) from the MLB Stats API.  All data is cached
as Parquet files under ``data/processed/player/``.

Usage
-----
    python scripts/ingest_player_data.py                    # all available seasons
    python scripts/ingest_player_data.py --start 2015       # from 2015 onwards
    python scripts/ingest_player_data.py --seasons 2023 2024 2025
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _collect_starting_pitcher_mlbam_ids(
    gamelogs_dir: Path,
    bio_cache_dir: Path,
    seasons: list[int],
) -> dict[int, set[int]]:
    """Collect unique starting pitcher MLBAM IDs per season from gamelogs.

    Returns {season: {mlbam_id, ...}}.
    """
    import pandas as pd

    from mlb_predict.player.biographical import build_biographical_df

    bio_df = build_biographical_df(cache_dir=bio_cache_dir)
    retro_to_mlbam: dict[str, int] = {}
    for _, row in bio_df.iterrows():
        rid = row.get("retro_id")
        mid = row.get("mlbam_id")
        if pd.notna(rid) and pd.notna(mid):
            retro_to_mlbam[str(rid).strip().lower()] = int(mid)

    result: dict[int, set[int]] = {}
    for s in seasons:
        ids: set[int] = set()
        for ext in ("parquet", "csv"):
            path = gamelogs_dir / f"gamelogs_{s}.{ext}"
            if not path.exists():
                continue
            gl = pd.read_parquet(path) if ext == "parquet" else pd.read_csv(path)
            for col in ("home_starting_pitcher_id", "visiting_starting_pitcher_id"):
                if col not in gl.columns:
                    continue
                for pid in gl[col].dropna().unique():
                    retro_key = str(pid).strip().lower()
                    mlbam = retro_to_mlbam.get(retro_key)
                    if mlbam:
                        ids.add(mlbam)
            break
        result[s] = ids
    return result


async def _ingest_pitcher_gamelogs(
    seasons: list[int],
    out: Path,
    gamelogs_dir: Path,
) -> None:
    """Fetch and cache per-pitcher game logs from the MLB Stats API."""
    from mlb_predict.mlbapi.client import MLBAPIClient
    from mlb_predict.player.pitcher_gamelogs import (
        fetch_all_pitcher_gamelogs,
        save_pitcher_gamelogs,
    )

    sp_ids = _collect_starting_pitcher_mlbam_ids(gamelogs_dir, out, seasons)

    async with MLBAPIClient() as client:
        for s in seasons:
            cache_path = out / f"pitcher_gamelogs_{s}.parquet"
            if cache_path.exists():
                print(f"  {s}: cached (skipping)")
                continue

            ids = sorted(sp_ids.get(s, set()))
            if not ids:
                print(f"  {s}: no pitcher IDs found (skipping)")
                continue

            print(f"  {s}: fetching game logs for {len(ids)} pitchers…", flush=True)
            df = await fetch_all_pitcher_gamelogs(client, ids, s)
            if not df.empty:
                save_pitcher_gamelogs(df, out, s)
            n = len(df) if not df.empty else 0
            print(f"  {s}: {n} game log entries")


def main() -> None:
    """Ingest player data for the specified seasons."""
    ap = argparse.ArgumentParser(description="Ingest player-level stats and biographical data")
    ap.add_argument(
        "--start",
        type=int,
        default=2002,
        help="First season to ingest (default: 2002)",
    )
    ap.add_argument(
        "--end",
        type=int,
        default=datetime.now(timezone.utc).year,
        help="Last season to ingest (default: current year)",
    )
    ap.add_argument(
        "--seasons",
        nargs="*",
        type=int,
        help="Explicit list of seasons (overrides --start/--end)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/player"),
        help="Output directory for cached player data",
    )
    ap.add_argument(
        "--gamelogs-dir",
        type=Path,
        default=Path("data/processed/retrosheet"),
        help="Directory containing Retrosheet gamelogs",
    )
    ap.add_argument(
        "--skip-bio",
        action="store_true",
        help="Skip biographical data ingestion",
    )
    ap.add_argument(
        "--skip-batters",
        action="store_true",
        help="Skip batter stat ingestion",
    )
    ap.add_argument(
        "--skip-pitchers",
        action="store_true",
        help="Skip pitcher stat ingestion",
    )
    ap.add_argument(
        "--skip-pitcher-gamelogs",
        action="store_true",
        help="Skip per-pitcher game log ingestion from MLB Stats API",
    )
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    seasons = args.seasons or list(range(args.start, args.end + 1))
    print(f"Ingesting player data for {len(seasons)} seasons: {seasons[0]}–{seasons[-1]}")
    print(f"Output directory: {out}")

    # --- Biographical data ---------------------------------------------------
    if not args.skip_bio:
        print("\n[1/4] Building biographical data…")
        from mlb_predict.player.biographical import build_biographical_df

        bio_df = build_biographical_df(cache_dir=out)
        print(f"  Biographical data: {len(bio_df)} players")

    # --- Batter stats --------------------------------------------------------
    if not args.skip_batters:
        print("\n[2/4] Ingesting batter stats…")
        from mlb_predict.player.ingestion import get_batter_stats_for_season

        for s in seasons:
            df = get_batter_stats_for_season(s, cache_dir=out)
            n = len(df) if not df.empty else 0
            print(f"  {s}: {n} batters")

    # --- Pitcher stats (season aggregates) -----------------------------------
    if not args.skip_pitchers:
        print("\n[3/4] Ingesting pitcher stats…")
        from mlb_predict.player.ingestion import get_pitcher_stats_for_season

        for s in seasons:
            df = get_pitcher_stats_for_season(s, cache_dir=out)
            n = len(df) if not df.empty else 0
            print(f"  {s}: {n} pitchers")

    # --- Per-pitcher game logs (MLB Stats API) --------------------------------
    if not args.skip_pitcher_gamelogs:
        print("\n[4/4] Ingesting per-pitcher game logs from MLB Stats API…")
        asyncio.run(_ingest_pitcher_gamelogs(seasons, out, args.gamelogs_dir))

    print("\nDone.")


if __name__ == "__main__":
    main()
