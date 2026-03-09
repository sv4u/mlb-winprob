"""Backfill weather cache from gamelogs using Open-Meteo historical API.

Reads Retrosheet gamelogs, collects unique (park_id, date), fetches weather
for each park-season in a single batched API call (one request per park per
season), and writes data/processed/weather/by_park_date.parquet.

Batch strategy reduces ~140 000 individual API calls to ~780 for a full
26-season reingest, staying well within Open-Meteo free-tier limits
(10 000 calls/day, 5 000/hour, 600/minute).

Progress reporting:
  - Season-level summaries printed to stdout (visible in admin dashboard logs).
  - Per-park detail logged at INFO level for troubleshooting.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from mlb_predict.external.weather import (
    PARK_LATLON,
    _NEUTRAL_HUMIDITY,
    _NEUTRAL_TEMP_F,
    _NEUTRAL_WIND_MPH,
    fetch_park_season,
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Ingest historical weather for all game (park, date) pairs."""
    ap = argparse.ArgumentParser(description="Ingest historical weather for games from Open-Meteo")
    ap.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    ap.add_argument(
        "--seasons",
        nargs="*",
        type=int,
        default=[],
        help="Seasons to process (default: all with gamelogs)",
    )
    args = ap.parse_args()

    weather_dir = args.processed_dir / "weather"
    weather_dir.mkdir(parents=True, exist_ok=True)
    retro_dir = args.processed_dir / "retrosheet"

    cache_path = weather_dir / "by_park_date.parquet"
    existing_df = None
    if cache_path.exists():
        existing_df = pd.read_parquet(cache_path)
        existing_df["game_date"] = pd.to_datetime(existing_df["game_date"]).dt.strftime("%Y-%m-%d")
    existing_keys: set[tuple[str, str]] = set()
    if existing_df is not None:
        existing_keys = set(zip(existing_df["park_id"].astype(str), existing_df["game_date"]))

    seasons = args.seasons or sorted(
        int(f.stem.split("_")[1]) for f in retro_dir.glob("gamelogs_*.parquet")
    )
    if not seasons:
        print("No gamelogs found.")
        return

    park_season_dates: dict[tuple[str, int], set[str]] = {}
    for s in seasons:
        path = retro_dir / f"gamelogs_{s}.parquet"
        if not path.exists():
            continue
        gl = pd.read_parquet(path, columns=["date", "park_id"])
        gl["date"] = pd.to_datetime(gl["date"]).dt.strftime("%Y-%m-%d")
        gl["park_id"] = gl["park_id"].astype(str)
        for _, row in gl[["date", "park_id"]].drop_duplicates().iterrows():
            park_id = str(row["park_id"])
            game_date = str(row["date"])
            if (park_id, game_date) in existing_keys:
                continue
            if park_id not in PARK_LATLON:
                continue
            park_season_dates.setdefault((park_id, s), set()).add(game_date)

    total_api_calls = len(park_season_dates)
    if total_api_calls == 0:
        print("All weather data already cached — nothing to fetch.")
        return

    total_dates = sum(len(dates) for dates in park_season_dates.values())
    print(
        f"Fetching weather: {total_api_calls} API calls "
        f"(batched from {total_dates} unique park-dates across {len(seasons)} seasons)"
    )

    rows: list[dict[str, object]] = []
    completed = 0

    seasons_in_order = sorted({s for (_, s) in park_season_dates})
    for season in seasons_in_order:
        season_parks = sorted(
            [(pk, dt) for (pk, s), dt in park_season_dates.items() if s == season]
        )
        season_dates_total = sum(len(dt) for _, dt in season_parks)
        season_api_from = 0
        season_ok = 0
        season_neutral = 0

        for park_id, dates in season_parks:
            completed += 1
            logger.info(
                "[%d/%d] %s %d (%d dates)",
                completed,
                total_api_calls,
                park_id,
                season,
                len(dates),
            )

            results = fetch_park_season(park_id, season, dates)
            season_api_from += len(results)
            for game_date in sorted(dates):
                w = results.get(game_date)
                if w is None:
                    w = {
                        "temp_f": _NEUTRAL_TEMP_F,
                        "wind_mph": _NEUTRAL_WIND_MPH,
                        "humidity": _NEUTRAL_HUMIDITY,
                    }
                    season_neutral += 1
                else:
                    season_ok += 1
                rows.append(
                    {
                        "game_date": game_date,
                        "park_id": park_id,
                        "temp_f": w["temp_f"],
                        "wind_mph": w["wind_mph"],
                        "humidity": w["humidity"],
                    }
                )

        print(
            f"  Season {season}: {len(season_parks)} parks, "
            f"{season_dates_total} dates "
            f"({season_api_from} from API, {season_neutral} neutral defaults)"
        )

    if rows:
        new_df = pd.DataFrame(rows)
        if existing_df is not None:
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["game_date", "park_id"], keep="last")
        else:
            combined = new_df.drop_duplicates(subset=["game_date", "park_id"])
        combined.to_parquet(cache_path, index=False)
        print(f"Wrote {len(combined)} rows → {cache_path}")
    else:
        print("No new weather rows fetched (or no parks in PARK_LATLON).")


if __name__ == "__main__":
    main()
