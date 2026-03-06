"""In-memory TTL cache for live odds from The Odds API.

Provides a 5-minute cache to avoid burning API quota on every page load.
All public functions degrade gracefully when the API key is not configured
or the external service is unavailable.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from winprob.external.odds import OddsClient, _to_retro
from winprob.external.odds_config import get_odds_config_status

logger = logging.getLogger(__name__)

_TTL_SECONDS = 300  # 5 minutes

_cache_lock = asyncio.Lock()
_cached_events: list[dict[str, Any]] = []
_cache_ts: float = 0.0


def is_odds_configured() -> bool:
    """Return True if an Odds API key is available (env or config file)."""
    return bool(get_odds_config_status().get("configured"))


def american_to_implied(price: int | float) -> float:
    """Convert American moneyline odds to implied probability (0-1).

    Negative odds (e.g. -150): |odds| / (|odds| + 100)
    Positive odds (e.g. +130): 100 / (odds + 100)
    """
    p = float(price)
    if p < 0:
        return abs(p) / (abs(p) + 100.0)
    if p > 0:
        return 100.0 / (p + 100.0)
    return 0.5


def _pick_best_price(prices: list[int | float]) -> int | float:
    """Select the most favorable price for the bettor from a list of moneylines.

    For negative odds: least negative is best (-130 beats -150).
    For positive odds: most positive is best (+140 beats +120).
    The numeric maximum is always the best price for the bettor.
    """
    if not prices:
        return 0
    return max(prices)


async def get_cached_odds() -> list[dict[str, Any]]:
    """Return cached odds events, refreshing from the API if stale or empty.

    Thread-safe via asyncio.Lock. Returns [] if API key is not set or API fails.
    """
    global _cached_events, _cache_ts

    now = time.monotonic()
    if _cache_ts > 0 and (now - _cache_ts) < _TTL_SECONDS:
        return _cached_events

    async with _cache_lock:
        now = time.monotonic()
        if _cache_ts > 0 and (now - _cache_ts) < _TTL_SECONDS:
            return _cached_events

        if not is_odds_configured():
            return []

        stale = list(_cached_events)
        try:
            client = OddsClient()
            events = await client.get_game_odds()
            _cached_events = events
            _cache_ts = time.monotonic()
            logger.info("Odds cache refreshed: %d events", len(events))
            return _cached_events
        except Exception:
            logger.exception("Failed to refresh odds cache")
            if stale:
                logger.warning("Serving stale odds cache (%d events)", len(stale))
                return stale
            return []


def match_odds_for_game(
    events: list[dict[str, Any]],
    home_retro: str,
    away_retro: str,
) -> dict[str, Any] | None:
    """Find the matching odds event for a game and return a structured dict.

    Matches by comparing Retrosheet codes derived from The Odds API team names.
    Returns None if no match is found.
    """
    if not events or not home_retro or not away_retro:
        return None

    for ev in events:
        ev_home = _to_retro(ev.get("home_team") or "")
        ev_away = _to_retro(ev.get("away_team") or "")
        if ev_home != home_retro or ev_away != away_retro:
            continue

        bookmakers: list[dict[str, Any]] = []
        home_prices: list[int | float] = []
        away_prices: list[int | float] = []

        for bm in ev.get("bookmakers") or []:
            h2h_market = None
            for mkt in bm.get("markets") or []:
                if mkt.get("key") == "h2h":
                    h2h_market = mkt
                    break
            if not h2h_market:
                continue

            home_price: int | float | None = None
            away_price: int | float | None = None
            for outcome in h2h_market.get("outcomes") or []:
                outcome_retro = _to_retro(outcome.get("name") or "")
                if outcome_retro == home_retro:
                    home_price = outcome.get("price")
                elif outcome_retro == away_retro:
                    away_price = outcome.get("price")

            if home_price is not None and away_price is not None:
                bookmakers.append(
                    {
                        "key": bm.get("key", ""),
                        "title": bm.get("title", bm.get("key", "")),
                        "home_price": home_price,
                        "away_price": away_price,
                    }
                )
                home_prices.append(home_price)
                away_prices.append(away_price)

        best_home = _pick_best_price(home_prices)
        best_away = _pick_best_price(away_prices)

        return {
            "home_team": ev.get("home_team", ""),
            "away_team": ev.get("away_team", ""),
            "commence_time": ev.get("commence_time", ""),
            "bookmakers": bookmakers,
            "best_home_price": best_home,
            "best_away_price": best_away,
            "home_implied_prob": round(american_to_implied(best_home), 4),
            "away_implied_prob": round(american_to_implied(best_away), 4),
        }

    return None
