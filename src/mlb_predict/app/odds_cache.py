"""In-memory TTL cache for live odds from The Odds API.

Provides a 5-minute cache to avoid burning API quota on every page load.
All public functions degrade gracefully when the API key is not configured
or the external service is unavailable.

Also provides EV opportunity computation by joining model probabilities with
live market odds to identify positive-edge bets.

When MLB_PREDICT_LIVE_API=0, no outbound calls are made; cache is not refreshed.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import pandas as pd

from mlb_predict.external.odds import OddsClient, _to_retro
from mlb_predict.external.odds_config import get_odds_config_status


def _live_api_enabled() -> bool:
    """True if live external API calls (e.g. Odds API) are allowed."""
    return os.environ.get("MLB_PREDICT_LIVE_API", "1").strip() != "0"


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

        if not _live_api_enabled():
            return list(_cached_events)

        if not is_odds_configured():
            return []

        stale = list(_cached_events)
        try:
            client = OddsClient()
            events = await client.get_all_mlb_odds()
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


def american_to_decimal(price: int | float) -> float:
    """Convert American moneyline odds to decimal odds.

    Positive (e.g. +130): 130/100 + 1 = 2.30
    Negative (e.g. -150): 100/150 + 1 = 1.667
    """
    p = float(price)
    if p > 0:
        return p / 100.0 + 1.0
    if p < 0:
        return 100.0 / abs(p) + 1.0
    return 2.0


def compute_ev_opportunities(
    events: list[dict[str, Any]],
    features_df: pd.DataFrame,
    min_edge: float = 0.0,
) -> list[dict[str, Any]]:
    """Join live odds events with model probabilities to find EV+ bets.

    For each odds event that matches a game in features_df, computes EV metrics
    for both home and away moneyline sides. Returns opportunities sorted by edge
    (descending), filtered to edge > min_edge.
    """
    from mlb_predict.app.data_cache import TEAM_NAMES

    if not events or features_df.empty:
        return []

    opportunities: list[dict[str, Any]] = []

    for ev in events:
        ev_home_retro = _to_retro(ev.get("home_team") or "")
        ev_away_retro = _to_retro(ev.get("away_team") or "")
        if not ev_home_retro or not ev_away_retro:
            continue

        matched = match_odds_for_game([ev], ev_home_retro, ev_away_retro)
        if not matched or not matched["bookmakers"]:
            continue

        mask = (features_df["home_retro"] == ev_home_retro) & (
            features_df["away_retro"] == ev_away_retro
        )
        rows = features_df[mask]
        if rows.empty:
            continue

        row = rows.iloc[-1]
        prob_home = float(row["prob"]) if pd.notna(row.get("prob")) else None
        if prob_home is None:
            continue

        game_pk = int(row.get("game_pk", 0) or 0)
        date_str = str(row.get("date", ""))[:10]
        home_name = TEAM_NAMES.get(ev_home_retro, ev.get("home_team", ""))
        away_name = TEAM_NAMES.get(ev_away_retro, ev.get("away_team", ""))

        for side, price, model_prob in [
            ("home", matched["best_home_price"], prob_home),
            ("away", matched["best_away_price"], 1.0 - prob_home),
        ]:
            if price == 0:
                continue
            implied = american_to_implied(price)
            edge = model_prob - implied
            if edge <= min_edge:
                continue
            dec = american_to_decimal(price)
            ev_per_unit = model_prob * (dec - 1.0) - (1.0 - model_prob)
            b = dec - 1.0
            kelly = max(0.0, (b * model_prob - (1.0 - model_prob)) / b) if b > 0 else 0.0

            best_book = ""
            for bm in matched["bookmakers"]:
                bp = bm["home_price"] if side == "home" else bm["away_price"]
                if bp == price:
                    best_book = bm["title"]
                    break

            opportunities.append(
                {
                    "game_pk": game_pk,
                    "date": date_str,
                    "home_team": home_name,
                    "away_team": away_name,
                    "commence_time": matched["commence_time"],
                    "selection": side,
                    "team": home_name if side == "home" else away_name,
                    "odds": price,
                    "sportsbook": best_book,
                    "implied_prob": round(implied, 4),
                    "model_prob": round(model_prob, 4),
                    "edge": round(edge, 4),
                    "ev_per_unit": round(ev_per_unit, 4),
                    "kelly_pct": round(kelly * 100, 2),
                }
            )

    opportunities.sort(key=lambda x: x["edge"], reverse=True)
    return opportunities
