from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from zoneinfo import ZoneInfo

import logging

from mlb_predict.mlbapi.client import MLBAPIClient, MLBAPIConfig
from mlb_predict.mlbapi.schedule import (
    ALL_GAME_TYPES,
    GAME_TYPE_REGULAR,
    fetch_schedule_chunk,
    parse_utc_iso,
    schedule_bounds,
)
from mlb_predict.mlbapi.teams import build_team_maps, get_teams_df
from mlb_predict.util.hashing import sha256_aggregate_of_files, sha256_file

log = logging.getLogger(__name__)


def month_ranges(start: date, end: date) -> list[tuple[date, date]]:
    ranges: list[tuple[date, date]] = []
    cur = date(start.year, start.month, 1)
    while cur <= end:
        nxt = date(
            cur.year + (1 if cur.month == 12 else 0), 1 if cur.month == 12 else cur.month + 1, 1
        )
        ranges.append((max(start, cur), min(end, nxt - timedelta(days=1))))
        cur = nxt
    return ranges


async def fetch_with_adaptive_split(
    client: MLBAPIClient,
    *,
    season: int,
    start: date,
    end: date,
    game_type: str = GAME_TYPE_REGULAR,
    max_mb: int,
    max_depth: int,
    depth: int = 0,
) -> pd.DataFrame:
    span_days = (end - start).days + 1
    if span_days > 31 and depth < max_depth:
        mid = start + timedelta(days=span_days // 2)
        left = await fetch_with_adaptive_split(
            client,
            season=season,
            start=start,
            end=mid,
            game_type=game_type,
            max_mb=max_mb,
            max_depth=max_depth,
            depth=depth + 1,
        )
        right = await fetch_with_adaptive_split(
            client,
            season=season,
            start=mid + timedelta(days=1),
            end=end,
            game_type=game_type,
            max_mb=max_mb,
            max_depth=max_depth,
            depth=depth + 1,
        )
        return pd.concat([left, right], ignore_index=True)

    df = await fetch_schedule_chunk(
        client, season=season, start_date=start, end_date=end, game_type=game_type
    )
    approx_bytes = df.memory_usage(deep=True).sum()
    if approx_bytes > max_mb * 1024 * 1024 and depth < max_depth and span_days > 1:
        mid = start + timedelta(days=span_days // 2)
        left = await fetch_with_adaptive_split(
            client,
            season=season,
            start=start,
            end=mid,
            game_type=game_type,
            max_mb=max_mb,
            max_depth=max_depth,
            depth=depth + 1,
        )
        right = await fetch_with_adaptive_split(
            client,
            season=season,
            start=mid + timedelta(days=1),
            end=end,
            game_type=game_type,
            max_mb=max_mb,
            max_depth=max_depth,
            depth=depth + 1,
        )
        return pd.concat([left, right], ignore_index=True)

    return df


def add_local_times(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise UTC timestamps and ensure ``game_date_local`` is populated.

    ``normalize_schedule`` already extracts the local date from the API's
    date-group field (``dates[].date``), which is the authoritative local game
    date.  This function preserves that value and only falls back to computing
    the local datetime from ``local_timezone`` when it is absent.
    """
    out = df.copy()
    utc_dt = out["game_date_utc"].map(parse_utc_iso)
    out["game_date_utc"] = utc_dt.map(lambda x: x.isoformat())

    # Prefer the local date already set by normalize_schedule; only compute
    # from local_timezone for rows where game_date_local is still missing.
    missing_local = (
        out["game_date_local"].isna()
        if "game_date_local" in out.columns
        else pd.Series(True, index=out.index)
    )

    if missing_local.any():

        def to_local(row: Any) -> str | None:
            tz = row["local_timezone"]
            if not tz:
                return None
            try:
                return parse_utc_iso(row["game_date_utc"]).astimezone(ZoneInfo(tz)).isoformat()
            except Exception:
                return None

        computed = out[missing_local].apply(to_local, axis=1)
        if "game_date_local" not in out.columns:
            out["game_date_local"] = None
        out.loc[missing_local, "game_date_local"] = computed

    return out


async def _fetch_game_type(
    client: MLBAPIClient,
    *,
    season: int,
    game_type: str,
    max_mb: int,
    max_depth: int,
) -> pd.DataFrame:
    """Fetch all games for one (season, game_type) pair."""
    try:
        start, end = await schedule_bounds(client, season=season, game_type=game_type)
    except RuntimeError:
        log.info("No games for season=%d gameType=%s — skipping", season, game_type)
        return pd.DataFrame()
    dfs: list[pd.DataFrame] = []
    for cs, ce in month_ranges(start, end):
        dfs.append(
            await fetch_with_adaptive_split(
                client,
                season=season,
                start=cs,
                end=ce,
                game_type=game_type,
                max_mb=max_mb,
                max_depth=max_depth,
            )
        )
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


async def ingest_one_season(
    *,
    season: int,
    refresh_mlbapi: bool,
    max_mb: int,
    max_depth: int,
    game_types: tuple[str, ...] = ALL_GAME_TYPES,
) -> dict[str, Any]:
    cfg = MLBAPIConfig(rps=5.0, burst=10.0, max_concurrency=8)
    async with MLBAPIClient(config=cfg, refresh=refresh_mlbapi) as client:
        teams_df = await get_teams_df(client, season=season)
        team_maps = build_team_maps(teams_df)

        all_dfs: list[pd.DataFrame] = []
        for gt in game_types:
            gt_df = await _fetch_game_type(
                client, season=season, game_type=gt, max_mb=max_mb, max_depth=max_depth
            )
            if not gt_df.empty:
                all_dfs.append(gt_df)
        if not all_dfs:
            raise RuntimeError(f"No games found for season={season} types={game_types}")
        df = pd.concat(all_dfs, ignore_index=True)

    # When a game is rescheduled (common in 2020) the same game_pk appears in
    # multiple monthly chunks — once as "Postponed" on the original date and
    # again as "Final" on the rescheduled date.  Keep the Final entry so that
    # game_date_local reflects when the game was actually played.
    _PLAYED_STATUSES = {"Final", "Completed Early", "Game Over"}
    df["_status_rank"] = df["status"].map(lambda s: 0 if s in _PLAYED_STATUSES else 1)
    df = (
        df.sort_values(["game_pk", "_status_rank"])
        .drop_duplicates(subset=["game_pk"])
        .drop(columns=["_status_rank"])
        .sort_values("game_pk")
        .reset_index(drop=True)
    )
    df["season"] = int(season)
    df = add_local_times(df)
    df["home_abbrev"] = df["home_mlb_id"].map(team_maps.mlb_id_to_abbrev)
    df["away_abbrev"] = df["away_mlb_id"].map(team_maps.mlb_id_to_abbrev)

    out_dir = Path("data/processed/schedule")
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / f"games_{season}.parquet"
    csv_path = out_dir / f"games_{season}.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    teams_out = Path("data/processed/teams")
    teams_out.mkdir(parents=True, exist_ok=True)
    teams_df.to_parquet(teams_out / f"teams_{season}.parquet", index=False)

    raw_schedule_dir = Path("data/raw/mlb_api/schedule")
    raw_files = list(raw_schedule_dir.glob("*.json"))
    if "game_type" not in df.columns:
        df["game_type"] = GAME_TYPE_REGULAR

    checksum = {
        "season": season,
        "row_count": int(len(df)),
        "game_types": list(df["game_type"].unique()),
        "parquet_sha256": sha256_file(parquet_path),
        "csv_sha256": sha256_file(csv_path),
        "raw_payloads_sha256": sha256_aggregate_of_files(raw_files) if raw_files else None,
        "raw_file_count": int(len(raw_files)),
        "max_response_mb": max_mb,
        "max_split_depth": max_depth,
        "mlbapi_config": asdict(cfg),
    }
    (out_dir / f"games_{season}.checksum.json").write_text(
        pd.Series(checksum).to_json(), encoding="utf-8"
    )
    return checksum


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", type=int, default=[])
    ap.add_argument("--refresh-mlbapi", action="store_true")
    ap.add_argument("--max-response-mb", type=int, default=5)
    ap.add_argument("--max-split-depth", type=int, default=4)
    ap.add_argument(
        "--include-preseason",
        action="store_true",
        default=True,
        help="Ingest spring training (gameType=S) alongside regular season (default: True)",
    )
    ap.add_argument(
        "--no-preseason",
        action="store_false",
        dest="include_preseason",
        help="Skip spring training games, ingest regular season only",
    )
    args = ap.parse_args()

    game_types = ALL_GAME_TYPES if args.include_preseason else (GAME_TYPE_REGULAR,)

    seasons = args.seasons or list(range(2000, 2026))
    results = []
    for s in seasons:
        try:
            results.append(
                await ingest_one_season(
                    season=s,
                    refresh_mlbapi=args.refresh_mlbapi,
                    max_mb=args.max_response_mb,
                    max_depth=args.max_split_depth,
                    game_types=game_types,
                )
            )
        except Exception as e:
            results.append({"season": s, "status": "failed", "error": str(e)})

    out_dir = Path("data/processed/schedule")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ingest_schedule_summary.json").write_text(
        pd.DataFrame(results).to_json(orient="records", indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    asyncio.run(main())
