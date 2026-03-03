"""Backfill weather cache from gamelogs using Open-Meteo historical API.

Reads Retrosheet gamelogs, collects unique (park_id, date), fetches weather
for each (with rate limiting), and writes data/processed/weather/by_park_date.parquet.
Merge with existing cache if present.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from winprob.external.weather import (
    PARK_LATLON,
    _NEUTRAL_HUMIDITY,
    _NEUTRAL_TEMP_F,
    _NEUTRAL_WIND_MPH,
    fetch_weather_uncached,
)

def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest historical weather for games from Open-Meteo")
    ap.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--seasons", nargs="*", type=int, default=[], help="Seasons to process (default: all with gamelogs)")
    args = ap.parse_args()

    weather_dir = args.processed_dir / "weather"
    weather_dir.mkdir(parents=True, exist_ok=True)
    retro_dir = args.processed_dir / "retrosheet"

    cache_path = weather_dir / "by_park_date.parquet"
    existing_df = None
    if cache_path.exists():
        existing_df = pd.read_parquet(cache_path)
        existing_df["game_date"] = pd.to_datetime(existing_df["game_date"]).dt.strftime("%Y-%m-%d")
    existing_keys = set()
    if existing_df is not None:
        existing_keys = set(zip(existing_df["park_id"].astype(str), existing_df["game_date"]))

    seasons = args.seasons or sorted(
        int(f.stem.split("_")[1]) for f in retro_dir.glob("gamelogs_*.parquet")
    )
    if not seasons:
        print("No gamelogs found.")
        return

    rows = []
    for s in seasons:
        path = retro_dir / f"gamelogs_{s}.parquet"
        if not path.exists():
            continue
        gl = pd.read_parquet(path, columns=["date", "park_id"])
        gl["date"] = pd.to_datetime(gl["date"]).dt.strftime("%Y-%m-%d")
        gl["park_id"] = gl["park_id"].astype(str)
        for (game_date, park_id), _ in gl.groupby(["date", "park_id"]):
            key = (park_id, game_date)
            if key in existing_keys or park_id not in PARK_LATLON:
                continue
            existing_keys.add(key)
            w = fetch_weather_uncached(park_id, game_date)
            if w is None:
                w = {"temp_f": _NEUTRAL_TEMP_F, "wind_mph": _NEUTRAL_WIND_MPH, "humidity": _NEUTRAL_HUMIDITY}
            rows.append({
                "game_date": game_date,
                "park_id": park_id,
                "temp_f": w["temp_f"],
                "wind_mph": w["wind_mph"],
                "humidity": w["humidity"],
            })

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
