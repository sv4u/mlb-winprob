"""Historical weather at game location (temperature, wind, humidity) via Open-Meteo.

Uses the free Open-Meteo Historical Weather API (no key). Caches results under
weather_dir by (park_id, date). Park coordinates are from a static map; missing
parks get neutral defaults. Use scripts/ingest_weather.py to backfill cache from
gamelogs.

Rate-limit strategy:
  - Open-Meteo free tier: 600 calls/min, 5 000/hour, 10 000/day.
  - Batch fetching: one API call covers an entire date range per park, reducing
    ~140 000 individual calls to ~780 for a full 26-season reingest.
  - Exponential backoff with jitter on 429 responses (up to 5 retries).
  - Respects Retry-After header when present.
"""

from __future__ import annotations

import json
import logging
import random
import time
from http.client import HTTPResponse
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

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

_REQUEST_DELAY_S: float = 0.5
_MAX_RETRIES: int = 5
_BACKOFF_BASE_S: float = 2.0
_BACKOFF_MAX_S: float = 120.0


def _game_hour_utc(lat: float, lon: float) -> int:
    """Estimate the UTC hour of a typical 7 PM local game start from longitude.

    Rough timezone offset: each 15 degrees of longitude ≈ 1 hour from UTC.
    East coast (~-75°) → UTC-5 → 7 PM local = 24:00 UTC → hour 0 (next day)
    Central (~-90°)    → UTC-6 → 7 PM local = 01:00 UTC next day → hour 1
    Mountain (~-105°)  → UTC-7 → 7 PM local = 02:00 UTC next day → hour 2
    Pacific (~-120°)   → UTC-8 → 7 PM local = 03:00 UTC next day → hour 3
    """
    local_hour = 19  # 7 PM local game time
    utc_offset = round(lon / 15.0)
    utc_hour = (local_hour - utc_offset) % 24
    return utc_hour


def _retry_delay(attempt: int, retry_after: float | None = None) -> float:
    """Compute delay before the next retry using exponential backoff with jitter."""
    if retry_after is not None and retry_after > 0:
        return min(retry_after, _BACKOFF_MAX_S)
    delay = min(_BACKOFF_BASE_S * (2**attempt), _BACKOFF_MAX_S)
    jitter = random.uniform(0, delay * 0.25)  # noqa: S311
    return delay + jitter


def _openmeteo_request(url: str, *, timeout: float = 30.0) -> dict:
    """Make an HTTP request with retry-on-429 and exponential backoff."""
    last_err: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp: HTTPResponse
            with urlopen(Request(url), timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except HTTPError as e:
            if e.code == 429:
                retry_after_hdr = e.headers.get("Retry-After") if e.headers else None
                retry_after = float(retry_after_hdr) if retry_after_hdr else None
                wait = _retry_delay(attempt, retry_after)
                logger.info(
                    "Open-Meteo 429 (attempt %d/%d), waiting %.1fs before retry",
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    wait,
                )
                time.sleep(wait)
                last_err = e
                continue
            raise
    raise last_err or RuntimeError("Open-Meteo request failed after retries")


def _fetch_openmeteo(lat: float, lon: float, date: str) -> dict[str, float] | None:
    """Fetch one day hourly data; return temp (F), wind (mph), humidity (0-1) at estimated game time."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date,
        "end_date": date,
        "hourly": "temperature_2m,relative_humidity_2m,windspeed_10m",
    }
    try:
        url = f"{_OPENMETEO_URL}?{urlencode(params)}"
        data = _openmeteo_request(url)
        h = data.get("hourly", {})
        times = h.get("time", [])
        if not times:
            return None
        target_hour = _game_hour_utc(lat, lon)
        idx = target_hour if len(times) > target_hour else 0
        temp_c = h["temperature_2m"][idx]
        humidity_pct = h["relative_humidity_2m"][idx]
        wind_kmh = h["windspeed_10m"][idx]
        if temp_c is None or humidity_pct is None or wind_kmh is None:
            return None
        temp_f = (float(temp_c) * 9 / 5) + 32
        wind_mph = float(wind_kmh) * 0.621371
        return {
            "temp_f": temp_f,
            "wind_mph": wind_mph,
            "humidity": float(humidity_pct) / 100.0,
        }
    except Exception as e:
        logger.warning("Open-Meteo fetch failed for (%s, %s) @ %s: %s", lat, lon, date, e)
        return None


def fetch_date_range(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    game_dates: set[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Fetch weather for a date range in a single API call; return {date: {temp_f, wind_mph, humidity}}.

    If game_dates is provided, only extract data for those specific dates from
    the response (the API still returns the full range, but we only parse dates
    we care about).
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,windspeed_10m",
    }
    try:
        url = f"{_OPENMETEO_URL}?{urlencode(params)}"
        data = _openmeteo_request(url, timeout=60.0)
    except Exception as e:
        logger.warning(
            "Open-Meteo batch fetch failed for (%s, %s) %s→%s: %s",
            lat,
            lon,
            start_date,
            end_date,
            e,
        )
        return {}

    h = data.get("hourly", {})
    times = h.get("time", [])
    temps = h.get("temperature_2m", [])
    humidities = h.get("relative_humidity_2m", [])
    winds = h.get("windspeed_10m", [])
    if not times:
        return {}

    target_hour = _game_hour_utc(lat, lon)
    results: dict[str, dict[str, float]] = {}

    date_to_indices: dict[str, list[int]] = {}
    for i, t in enumerate(times):
        day = t[:10]
        date_to_indices.setdefault(day, []).append(i)

    for day, indices in date_to_indices.items():
        if game_dates is not None and day not in game_dates:
            continue
        idx = None
        for i in indices:
            hour = int(times[i][11:13])
            if hour == target_hour:
                idx = i
                break
        if idx is None:
            idx = indices[0]
        tc = temps[idx] if idx < len(temps) else None
        hp = humidities[idx] if idx < len(humidities) else None
        wk = winds[idx] if idx < len(winds) else None
        if tc is None or hp is None or wk is None:
            continue
        results[day] = {
            "temp_f": (float(tc) * 9 / 5) + 32,
            "wind_mph": float(wk) * 0.621371,
            "humidity": float(hp) / 100.0,
        }

    return results


def fetch_park_season(
    park_id: str,
    season: int,
    game_dates: set[str],
) -> dict[str, dict[str, float]]:
    """Fetch weather for all game_dates at a park in one API call per season.

    Returns {date_str: {temp_f, wind_mph, humidity}}.
    """
    coords = PARK_LATLON.get(park_id.strip())
    if not coords:
        return {}
    if not game_dates:
        return {}

    sorted_dates = sorted(game_dates)
    start = sorted_dates[0]
    end = sorted_dates[-1]

    time.sleep(_REQUEST_DELAY_S)

    return fetch_date_range(
        coords[0],
        coords[1],
        start,
        end,
        game_dates=game_dates,
    )


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
        return {
            "temp_f": _NEUTRAL_TEMP_F,
            "wind_mph": _NEUTRAL_WIND_MPH,
            "humidity": _NEUTRAL_HUMIDITY,
        }
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
