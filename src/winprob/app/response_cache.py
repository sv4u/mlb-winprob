"""In-memory TTL cache for expensive GET API responses to speed up page loads.

Used for /api/games, /api/standings, /api/seasons, /api/teams, /api/cv-summary
so repeated requests (e.g. home page refresh or multiple tabs) return quickly.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from functools import wraps
from typing import Any, Callable

from starlette.requests import Request

logger = logging.getLogger(__name__)

_TTL_SECONDS = 45
_MAX_ENTRIES = 200

_CacheEntry = tuple[Any, float]  # (payload, expiry_time)
_cache: OrderedDict[str, _CacheEntry] = OrderedDict()


def _cache_key(request: Request) -> str:
    """Build a cache key from path and normalized query string."""
    path = request.url.path
    query = request.url.query
    if query:
        # Normalize query order for consistent keys
        parts = sorted(query.split("&"))
        return f"{path}?{'&'.join(parts)}"
    return path


def get_cached(key: str) -> Any | None:
    """Return cached value if present and not expired; else None."""
    now = time.monotonic()
    if key not in _cache:
        return None
    payload, expiry = _cache[key]
    if now >= expiry:
        del _cache[key]
        return None
    # Move to end for LRU
    _cache.move_to_end(key)
    return payload


def set_cached(key: str, value: Any, ttl_seconds: int = _TTL_SECONDS) -> None:
    """Store value in cache with TTL. Evict oldest if over max entries."""
    now = time.monotonic()
    while len(_cache) >= _MAX_ENTRIES and _cache:
        _cache.popitem(last=False)
    _cache[key] = (value, now + ttl_seconds)


def cache_get_response(
    ttl_seconds: int = _TTL_SECONDS,
    key_builder: Callable[[Request], str] | None = None,
):
    """Decorator that caches the return value of a GET handler by request key."""

    def decorator(f: Callable[..., Any]):
        @wraps(f)
        async def wrapped(request: Request, *args: Any, **kwargs: Any) -> Any:
            key = (key_builder or _cache_key)(request)
            cached = get_cached(key)
            if cached is not None:
                logger.debug("Response cache HIT: %s", key[:80])
                return cached
            result = await f(request, *args, **kwargs)
            # Only cache successful dict/list responses (not JSONResponse errors)
            if isinstance(result, (dict, list)) and not (
                isinstance(result, dict) and result.get("error")
            ):
                set_cached(key, result, ttl_seconds)
            return result

        return wrapped

    return decorator


def clear_response_cache() -> None:
    """Clear all cached responses (e.g. after model switch or admin refresh)."""
    _cache.clear()
    logger.info("Response cache cleared")
