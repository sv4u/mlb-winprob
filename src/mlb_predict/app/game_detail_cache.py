"""In-memory cache for game detail API responses (SHAP + stats) to avoid recomputing.

SHAP attribution is expensive; caching by game_pk makes repeat visits to the same
game fast. Cache has no TTL; it is cleared whenever data or model changes:
- data_cache.switch_model() (after successful switch)
- data_cache.startup() (after successful load, including reload after ingest/update/retrain)
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)

_MAX_ENTRIES = 400
_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()


def get_game_detail_cached(game_pk: int) -> dict[str, Any] | None:
    """Return cached game detail payload (without live_odds) if present."""
    if game_pk not in _cache:
        return None
    payload = _cache[game_pk]
    _cache.move_to_end(game_pk)
    return payload


def set_game_detail_cached(game_pk: int, payload: dict[str, Any]) -> None:
    """Store game detail payload (without live_odds). Evict oldest if over max."""
    while len(_cache) >= _MAX_ENTRIES and _cache:
        _cache.popitem(last=False)
    # Store a copy so caller can add live_odds without mutating cache
    out = {k: v for k, v in payload.items() if k != "live_odds"}
    _cache[game_pk] = out


def clear_game_detail_cache() -> None:
    """Clear all cached game details (after model switch or data reload)."""
    _cache.clear()
    logger.info("Game detail cache cleared")
