"""Historical weather at game location (temperature, wind, humidity) via Open-Meteo.

Uses the free Open-Meteo Historical Weather API (no key). Caches results under
weather_dir by (park_id, date). Park coordinates are from a static map; missing
parks get neutral defaults. Use scripts/ingest_weather.py to backfill cache from
gamelogs.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlencode

import pandas as pd

logger = logging.getLogger(__name__)

# Retrosheet park_id -> (latitude, longitude). Expand as needed; missing → neutral.
# Source: Retrosheet park codes + stadium coordinates.
PARK_LATLON: dict[str, tuple[float, float]] = {
    "ANA01": (33.8003, -117.8827),
    "ARL02": (32.7473, -97.0835),
    "ATL02": (33.8907, -84.4674),
    "BAL11": (39.2839, -76.6217),
    "BOS07": (42.3467, -71.0972),
    "CHI11": (41.8299, -87.6338),
    "CHI12": (41.9484, -87.6553),
    "CIN09": (39.0974, -84.5067),
    "CLE08": (41.4962, -81.6852),
    "DEN02": (39.7559, -104.9942),
    "DET05": (42.3390, -83.0485),
    "HOU03": (29.7570, -95.3553),
    "KAN06": (39.0517, -94.4803),
    "LOS03": (34.0739, -118.2400),
    "MIA02": (25.7781, -80.2197),
    "MIL06": (43.0280, -87.9712),
    "MIN04": (44.9817, -93.2777),
    "NYC20": (40.8296, -73.9262),
    "NYC21": (40.7571, -73.8458),
    "OAK01": (37.7516, -122.2005),
    "PHI13": (39.9061, -75.1665),
    "PHO01": (33.4453, -112.0667),
    "PIT08": (40.4469, -80.0057),
    "SAN02": (32.7076, -117.1566),
    "SEA03": (47.5914, -122.3325),
    "SFN03": (37.7786, -122.3893),
    "STL10": (38.6226, -90.1928),
    "TAM01": (27.7682, -82.6534),
    "TOR02": (43.6414, -79.3894),
    "WAS11": (38.8730, -77.0074),
}

_OPENMETEO_URL = "https://archive-api.open-meteo.com/v1/archive"
_NEUTRAL_TEMP_F: float = 72.0
_NEUTRAL_WIND_MPH: float = 8.0
_NEUTRAL_HUMIDITY: float = 0.50
_REQUEST_DELAY_S: float = 0.2


def _fetch_openmeteo(lat: float, lon: float, date: str) -> dict[str, float] | None:
    """Fetch one day hourly data; return temp (F), wind (mph), humidity (0-1) at 19:00 UTC as proxy for game time."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date,
        "end_date": date,
        "hourly": "temperature_2m,relative_humidity_2m,windspeed_10m",
    }
    try:
        url = f"{_OPENMETEO_URL}?{urlencode(params)}"
        with urlopen(Request(url), timeout=15) as r:
            data = json.loads(r.read().decode())
        h = data.get("hourly", {})
        times = h.get("time", [])
        if not times:
            return None
        # Use 19:00 (7 PM) if available, else first hour
        idx = 19 if len(times) > 19 else 0
        temp_c = h["temperature_2m"][idx]
        humidity_pct = h["relative_humidity_2m"][idx]
        wind_kmh = h["windspeed_10m"][idx]
        temp_f = (float(temp_c) * 9 / 5) + 32
        wind_mph = float(wind_kmh) * 0.621371
        return {
            "temp_f": temp_f,
            "wind_mph": wind_mph,
            "humidity": float(humidity_pct) / 100.0,
        }
    except Exception as e:
        logger.warning("Open-Meteo fetch failed for %s @ %s: %s", (lat, lon), date, e)
        return None


def fetch_weather_uncached(park_id: str, game_date: str) -> dict[str, float] | None:
    """Fetch weather for (park_id, date) from API without reading cache. Returns None if park unknown or fetch fails."""
    park_id = str(park_id).strip()
    coords = PARK_LATLON.get(park_id)
    if not coords:
        return None
    time.sleep(_REQUEST_DELAY_S)
    out = _fetch_openmeteo(coords[0], coords[1], game_date)
    return out


def get_weather_for_game(park_id: str, game_date: str, weather_dir: Path) -> dict[str, float]:
    """Return temp_f, wind_mph, humidity for (park_id, date). Uses cache or fetches."""
    park_id = str(park_id).strip()
    weather_dir = Path(weather_dir)
    weather_dir.mkdir(parents=True, exist_ok=True)
    cache_file = weather_dir / "by_park_date.parquet"

    # In-memory cache for this run (optional): we could load full cache and write back
    if cache_file.exists():
        cache_df = pd.read_parquet(cache_file)
        row = cache_df[(cache_df["park_id"] == park_id) & (cache_df["game_date"] == game_date)]
        if not row.empty:
            return {
                "temp_f": float(row["temp_f"].iloc[0]),
                "wind_mph": float(row["wind_mph"].iloc[0]),
                "humidity": float(row["humidity"].iloc[0]),
            }

    coords = PARK_LATLON.get(park_id)
    if not coords:
        return {
            "temp_f": _NEUTRAL_TEMP_F,
            "wind_mph": _NEUTRAL_WIND_MPH,
            "humidity": _NEUTRAL_HUMIDITY,
        }
    time.sleep(_REQUEST_DELAY_S)
    out = _fetch_openmeteo(coords[0], coords[1], game_date)
    if out is None:
        return {"temp_f": _NEUTRAL_TEMP_F, "wind_mph": _NEUTRAL_WIND_MPH, "humidity": _NEUTRAL_HUMIDITY}
    return out


def load_weather_season(weather_dir: Path, season: int) -> pd.DataFrame | None:
    """Load cached weather for season from weather_dir. Expects by_park_date.parquet with game_date, park_id, temp_f, wind_mph, humidity."""
    path = Path(weather_dir) / "by_park_date.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
    df["_season"] = pd.to_datetime(df["game_date"]).dt.year
    out = df[df["_season"] == season].drop(columns=["_season"])
    return out if not out.empty else None


def build_weather_features(
    gamelogs: pd.DataFrame,
    weather_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge weather (temp_f, wind_mph, humidity) onto gamelogs. Index aligned to gamelogs."""
    n = len(gamelogs)
    out = pd.DataFrame(
        {
            "game_temp_f": [_NEUTRAL_TEMP_F] * n,
            "game_wind_mph": [_NEUTRAL_WIND_MPH] * n,
            "game_humidity": [_NEUTRAL_HUMIDITY] * n,
        },
        index=gamelogs.index,
    )
    if weather_df is None or weather_df.empty:
        return out

    gl_dates = pd.to_datetime(gamelogs["date"]).dt.strftime("%Y-%m-%d")
    gl_key = pd.DataFrame(
        {"game_date": gl_dates.values, "park_id": gamelogs["park_id"].astype(str).values},
        index=gamelogs.index,
    )
    weather_df = weather_df.astype({"park_id": str})
    merged = gl_key.merge(
        weather_df,
        on=["game_date", "park_id"],
        how="left",
    )
    if "temp_f" in merged.columns:
        out["game_temp_f"] = merged["temp_f"].fillna(_NEUTRAL_TEMP_F).values
    if "wind_mph" in merged.columns:
        out["game_wind_mph"] = merged["wind_mph"].fillna(_NEUTRAL_WIND_MPH).values
    if "humidity" in merged.columns:
        out["game_humidity"] = merged["humidity"].fillna(_NEUTRAL_HUMIDITY).values
    return out
