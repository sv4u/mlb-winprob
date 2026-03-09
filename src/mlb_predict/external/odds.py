"""The Odds API client for MLB game and futures odds.

Uses API key from ODDS_API_KEY env or data/processed/odds/config.json.
When no key is set, the client is disabled: no requests are made and
methods return empty/not-configured results (graceful degradation).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import aiohttp

from mlb_predict.external.odds_config import get_odds_api_key

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.the-odds-api.com/v4"
_SPORT_MLB = "baseball_mlb"
_SPORT_MLB_PRESEASON = "baseball_mlb_preseason"
_REQUESTS_REMAINING_HEADER = "x-requests-remaining"
_QUOTA_WARN = 100
_QUOTA_CIRCUIT_BREAK = 20

# The Odds API full team name -> Retrosheet 3-letter code (AGENTS.md / plan)
_ODDS_API_TO_RETRO: dict[str, str] = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago White Sox": "CHA",
    "Chicago Cubs": "CHN",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KCA",
    "Los Angeles Angels": "ANA",
    "Los Angeles Dodgers": "LAN",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Yankees": "NYA",
    "New York Mets": "NYN",
    "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDN",
    "Seattle Mariners": "SEA",
    "San Francisco Giants": "SFN",
    "St. Louis Cardinals": "SLN",
    "Tampa Bay Rays": "TBA",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WAS",
}


def _to_retro(name: str) -> str:
    """Return Retrosheet code for Odds API team name, or original if unknown."""
    return _ODDS_API_TO_RETRO.get(name, name)


class OddsClient:
    """Async client for The Odds API. Disabled when no API key is configured."""

    def __init__(
        self,
        *,
        base_url: str = _BASE_URL,
        sport: str = _SPORT_MLB,
        timeout_s: float = 15.0,
        max_retries: int = 4,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 30.0,
        cache_dir: Path | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._sport = sport
        self._timeout = aiohttp.ClientTimeout(total=timeout_s)
        self._max_retries = max_retries
        self._backoff_base = backoff_base_s
        self._backoff_max = backoff_max_s
        if cache_dir is None:
            _root = Path(__file__).resolve().parent.parent.parent.parent
            cache_dir = _root / "data" / "processed" / "odds"
        self._cache_dir = Path(cache_dir)
        self._key: str | None = None

    def _resolve_key(self) -> None:
        if self._key is None:
            self._key = get_odds_api_key()

    def is_available(self) -> bool:
        """Return True if an API key is configured and the client can call the API."""
        self._resolve_key()
        return bool(self._key)

    async def get_game_odds(
        self,
        regions: str = "us",
        odds_format: str = "american",
        markets: str = "h2h",
    ) -> list[dict[str, Any]]:
        """Fetch current MLB game odds. Returns list of event objects from the API.

        When no key is set, returns [] without making a request.
        """
        self._resolve_key()
        if not self._key:
            logger.debug("Odds API key not configured; skipping game odds request")
            return []

        url = f"{self._base_url}/sports/{self._sport}/odds"
        params: dict[str, str] = {
            "apiKey": self._key,
            "regions": regions,
            "oddsFormat": odds_format,
            "markets": markets,
        }

        raw = await self._request(url, params)
        if raw is None:
            return []

        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict) and "message" in raw:
            logger.warning("Odds API response message: %s", raw.get("message"))
            return []
        return []

    async def _request(self, url: str, params: dict[str, str]) -> list | dict | None:
        last_err: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                async with aiohttp.ClientSession(timeout=self._timeout) as session:
                    async with session.get(url, params=params) as resp:
                        remaining = resp.headers.get(_REQUESTS_REMAINING_HEADER)
                        if remaining is not None:
                            try:
                                n = int(remaining)
                                if n < _QUOTA_CIRCUIT_BREAK:
                                    logger.warning(
                                        "Odds API quota low (remaining=%s); circuit break",
                                        n,
                                    )
                                    return None
                                if n < _QUOTA_WARN:
                                    logger.warning("Odds API quota below 100 (remaining=%s)", n)
                            except ValueError:
                                pass

                        if resp.status == 401:
                            body = await resp.text()
                            if "MISSING_KEY" in body or "Invalid" in body:
                                logger.warning("Odds API key missing or invalid")
                                return None
                            raise OddsAPIError(f"401 Unauthorized: {body[:200]}")

                        if resp.status == 429:
                            wait = min(
                                self._backoff_max,
                                self._backoff_base * (2 ** (attempt - 1)),
                            )
                            logger.warning("Odds API 429; retry in %.1fs", wait)
                            await asyncio.sleep(wait)
                            last_err = OddsAPIError("429 Too Many Requests")
                            continue

                        if 500 <= resp.status <= 599:
                            wait = min(
                                self._backoff_max,
                                self._backoff_base * (2 ** (attempt - 1)),
                            )
                            logger.warning("Odds API %s; retry in %.1fs", resp.status, wait)
                            await asyncio.sleep(wait)
                            last_err = OddsAPIError(f"{resp.status} server error")
                            continue

                        if resp.status != 200:
                            text = await resp.text()
                            raise OddsAPIError(f"{resp.status} {text[:200]}")

                        data = await resp.json()
                        return data
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_err = e
                wait = min(
                    self._backoff_max,
                    self._backoff_base * (2 ** (attempt - 1)),
                )
                logger.warning("Odds API request failed (attempt %s): %s", attempt, e)
                await asyncio.sleep(wait)

        if last_err:
            logger.error("Odds API failed after retries: %s", last_err)
        return None

    def get_game_odds_sync(
        self,
        regions: str = "us",
        odds_format: str = "american",
        markets: str = "h2h",
    ) -> list[dict[str, Any]]:
        """Synchronous wrapper for get_game_odds (for CLI/scripts)."""
        return asyncio.run(
            self.get_game_odds(
                regions=regions,
                odds_format=odds_format,
                markets=markets,
            )
        )

    async def get_all_mlb_odds(
        self,
        regions: str = "us",
        odds_format: str = "american",
        markets: str = "h2h",
    ) -> list[dict[str, Any]]:
        """Fetch odds from both regular-season and pre-season sport keys.

        Tags each event with ``sport_key`` so callers can distinguish game types.
        Falls back gracefully if either endpoint returns nothing.
        """
        combined: list[dict[str, Any]] = []
        for sport in (_SPORT_MLB, _SPORT_MLB_PRESEASON):
            original_sport = self._sport
            self._sport = sport
            try:
                events = await self.get_game_odds(
                    regions=regions, odds_format=odds_format, markets=markets
                )
                for ev in events:
                    ev.setdefault("sport_key", sport)
                combined.extend(events)
            except Exception:
                logger.warning("Failed to fetch odds for sport=%s", sport)
            finally:
                self._sport = original_sport
        return combined

    def events_to_retro(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add home_team_retro and away_team_retro to each event (in-place)."""
        for ev in events:
            ev["home_team_retro"] = _to_retro(ev.get("home_team") or "")
            ev["away_team_retro"] = _to_retro(ev.get("away_team") or "")
        return events

    def write_raw_game_odds(self, events: list[dict[str, Any]]) -> Path | None:
        """Write raw JSON to data/processed/odds/live/odds_YYYY-MM-DD_HHMMSS.json."""
        if not events:
            return None
        live_dir = self._cache_dir / "live"
        live_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%d_%H%M%S", time.gmtime())
        path = live_dir / f"odds_{ts}.json"
        path.write_text(json.dumps(events, indent=2))
        logger.info("Wrote raw game odds to %s", path)
        return path


class OddsAPIError(Exception):
    """The Odds API returned an error or request failed after retries."""
