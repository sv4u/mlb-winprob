"""Tests for mlb_predict.mlbapi.client.MLBAPIClient."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from aioresponses import aioresponses as aioresponses_ctx

if TYPE_CHECKING:
    pass

from mlb_predict.mlbapi.client import (
    MLBAPIClient,
    MLBAPIConfig,
    MLBAPIError,
    MLBNotFoundError,
)


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fast_config() -> MLBAPIConfig:
    """MLBAPIConfig with very fast rate limits for unit tests."""
    return MLBAPIConfig(
        max_retries=2,
        backoff_base_s=0.001,
        backoff_max_s=0.01,
        rps=10_000.0,
        burst=10_000.0,
    )


BASE_URL = "https://statsapi.mlb.com/api/v1"


# ---------------------------------------------------------------------------
# Cache key / path helpers (pure, no network)
# ---------------------------------------------------------------------------


def test_stable_params_sorted_and_none_excluded() -> None:
    """_stable_params must sort by key and drop None values."""
    result = MLBAPIClient._stable_params({"z": 3, "a": 1, "b": None})
    assert result == [("a", "1"), ("z", "3")]


def test_cache_key_same_params_different_order() -> None:
    """Cache key must be identical regardless of dict iteration order."""
    client = MLBAPIClient()
    k1 = client._cache_key("schedule", {"b": 2, "a": 1})
    k2 = client._cache_key("schedule", {"a": 1, "b": 2})
    assert k1 == k2


def test_cache_key_different_endpoints() -> None:
    """Different endpoints with identical params must produce different keys."""
    client = MLBAPIClient()
    k1 = client._cache_key("schedule", {"sportId": 1})
    k2 = client._cache_key("teams", {"sportId": 1})
    assert k1 != k2


def test_cache_key_different_params() -> None:
    """Different param values must produce different keys."""
    client = MLBAPIClient()
    k1 = client._cache_key("schedule", {"season": 2023})
    k2 = client._cache_key("schedule", {"season": 2024})
    assert k1 != k2


def test_cache_key_is_64_char_hex() -> None:
    """Cache key must be a 64-character lowercase hex string (sha256)."""
    client = MLBAPIClient()
    key = client._cache_key("schedule", {"sportId": 1})
    assert len(key) == 64
    assert all(c in "0123456789abcdef" for c in key)


def test_cache_path_structure(tmp_path: Path) -> None:
    """Cache path must use endpoint name and key as components."""
    client = MLBAPIClient(cache_dir=tmp_path)
    key = client._cache_key("schedule", {"sportId": 1})
    path = client._cache_path("schedule", key)
    assert path == tmp_path / "schedule" / f"{key}.json"


def test_cache_path_strips_leading_slash(tmp_path: Path) -> None:
    """Leading slashes in endpoint must be stripped from path components."""
    client = MLBAPIClient(cache_dir=tmp_path)
    key = client._cache_key("/schedule", {})
    path = client._cache_path("/schedule", key)
    assert "schedule" in path.parts


# ---------------------------------------------------------------------------
# Cache read/write
# ---------------------------------------------------------------------------


async def test_read_cache_returns_none_when_missing(cache_dir: Path) -> None:
    """_read_cache must return None for a nonexistent path."""
    client = MLBAPIClient(cache_dir=cache_dir)
    result = await client._read_cache(cache_dir / "nonexistent.json")
    assert result is None


async def test_write_and_read_cache_roundtrip(cache_dir: Path) -> None:
    """Writing then reading cache must yield the original payload."""
    client = MLBAPIClient(cache_dir=cache_dir)
    payload = {"dates": [{"date": "2024-04-01"}]}
    meta = {
        "ts_unix": 0.0,
        "url": "u",
        "params": {},
        "cache_key": "k",
        "endpoint": "e",
        "status": 200,
    }
    path = cache_dir / "schedule" / "abc123.json"
    await client._write_cache(path, payload, meta)
    result = await client._read_cache(path)
    assert result == payload


async def test_write_cache_creates_parent_directory(cache_dir: Path) -> None:
    """_write_cache must create missing parent directories."""
    client = MLBAPIClient(cache_dir=cache_dir)
    path = cache_dir / "deep" / "nested" / "file.json"
    meta = {
        "ts_unix": 0.0,
        "url": "u",
        "params": {},
        "cache_key": "k",
        "endpoint": "e",
        "status": 200,
    }
    await client._write_cache(path, {"ok": True}, meta)
    assert path.exists()


async def test_append_meta_writes_jsonl_line(cache_dir: Path) -> None:
    """_append_meta must append a valid JSON line to metadata.jsonl."""
    client = MLBAPIClient(cache_dir=cache_dir)
    meta = {
        "ts_unix": 1.23,
        "url": "http://example.com",
        "params": {"x": 1},
        "cache_key": "abc",
        "endpoint": "schedule",
        "status": 200,
    }
    await client._append_meta(meta)
    lines = (cache_dir / "metadata.jsonl").read_bytes().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["endpoint"] == "schedule"
    assert parsed["status"] == 200


async def test_append_meta_multiple_calls_append(cache_dir: Path) -> None:
    """Multiple _append_meta calls must append multiple lines."""
    client = MLBAPIClient(cache_dir=cache_dir)
    base = {
        "ts_unix": 0.0,
        "url": "u",
        "params": {},
        "cache_key": "k",
        "endpoint": "e",
        "status": 200,
    }
    for i in range(3):
        await client._append_meta({**base, "ts_unix": float(i)})
    lines = (cache_dir / "metadata.jsonl").read_bytes().splitlines()
    assert len(lines) == 3


# ---------------------------------------------------------------------------
# get_json — context manager enforcement
# ---------------------------------------------------------------------------


async def test_get_json_without_context_manager_raises(
    cache_dir: Path, fast_config: MLBAPIConfig
) -> None:
    """Calling get_json without entering the async context manager must raise."""
    client = MLBAPIClient(config=fast_config, cache_dir=cache_dir)
    with pytest.raises(RuntimeError, match="session not initialized"):
        await client.get_json("schedule", {"sportId": 1})


# ---------------------------------------------------------------------------
# get_json — cache hit (no network)
# ---------------------------------------------------------------------------


async def test_get_json_serves_from_cache(cache_dir: Path, fast_config: MLBAPIConfig) -> None:
    """get_json must return cached data without making an HTTP request."""
    client = MLBAPIClient(config=fast_config, cache_dir=cache_dir)
    cached_payload = {"dates": [{"date": "2024-04-01"}]}
    key = client._cache_key("schedule", {"sportId": 1})
    path = client._cache_path("schedule", key)
    meta = {
        "ts_unix": 0.0,
        "url": "u",
        "params": {},
        "cache_key": key,
        "endpoint": "schedule",
        "status": 200,
    }
    await client._write_cache(path, cached_payload, meta)

    async with MLBAPIClient(config=fast_config, cache_dir=cache_dir) as c:
        result = await c.get_json("schedule", {"sportId": 1})
    assert result == cached_payload


# ---------------------------------------------------------------------------
# get_json — cache miss → HTTP fetch
# ---------------------------------------------------------------------------


async def test_get_json_fetches_and_caches(cache_dir: Path, fast_config: MLBAPIConfig) -> None:
    """get_json must fetch from the API and write the result to cache."""
    payload = {"dates": [{"date": "2024-04-01"}]}
    # Use empty params so the mock URL needs no query string
    with aioresponses_ctx() as m:
        m.get(f"{BASE_URL}/schedule", payload=payload)
        async with MLBAPIClient(config=fast_config, cache_dir=cache_dir) as client:
            result = await client.get_json("schedule", {})

    assert result == payload
    # Verify the response was cached on disk
    key = MLBAPIClient(config=fast_config, cache_dir=cache_dir)._cache_key("schedule", {})
    cached_path = cache_dir / "schedule" / f"{key}.json"
    assert cached_path.exists()


async def test_get_json_refresh_bypasses_cache(cache_dir: Path, fast_config: MLBAPIConfig) -> None:
    """get_json with refresh=True must re-fetch even when cache exists."""
    old_payload = {"dates": [{"date": "2023-01-01"}]}
    new_payload = {"dates": [{"date": "2024-04-01"}]}
    params: dict = {}

    # Pre-populate cache with old data
    tmp_client = MLBAPIClient(config=fast_config, cache_dir=cache_dir)
    key = tmp_client._cache_key("schedule", params)
    path = tmp_client._cache_path("schedule", key)
    meta = {
        "ts_unix": 0.0,
        "url": "u",
        "params": {},
        "cache_key": key,
        "endpoint": "schedule",
        "status": 200,
    }
    await tmp_client._write_cache(path, old_payload, meta)

    with aioresponses_ctx() as m:
        m.get(f"{BASE_URL}/schedule", payload=new_payload)
        async with MLBAPIClient(config=fast_config, cache_dir=cache_dir, refresh=True) as client:
            result = await client.get_json("schedule", params)

    assert result == new_payload


async def test_get_json_cache_readonly_raises_on_miss(
    cache_dir: Path, fast_config: MLBAPIConfig
) -> None:
    """get_json in cache_readonly mode must raise MLBAPIError on cache miss."""
    async with MLBAPIClient(config=fast_config, cache_dir=cache_dir, cache_readonly=True) as client:
        with pytest.raises(MLBAPIError, match="Cache miss in readonly mode"):
            await client.get_json("schedule", {"sportId": 1})


# ---------------------------------------------------------------------------
# get_json — HTTP error handling
# ---------------------------------------------------------------------------


async def test_get_json_404_raises_not_found(cache_dir: Path, fast_config: MLBAPIConfig) -> None:
    """A 404 response must raise MLBNotFoundError immediately (no retry)."""
    with aioresponses_ctx() as m:
        m.get(f"{BASE_URL}/schedule", status=404)
        async with MLBAPIClient(config=fast_config, cache_dir=cache_dir) as client:
            with pytest.raises(MLBNotFoundError):
                await client.get_json("schedule", {})


async def test_get_json_429_retries(cache_dir: Path, fast_config: MLBAPIConfig) -> None:
    """A 429 response must be retried until success."""
    payload = {"dates": []}
    with aioresponses_ctx() as m:
        # First call: 429; second call: 200
        m.get(f"{BASE_URL}/schedule", status=429, headers={"Retry-After": "0.001"})
        m.get(f"{BASE_URL}/schedule", payload=payload)
        async with MLBAPIClient(config=fast_config, cache_dir=cache_dir) as client:
            result = await client.get_json("schedule", {})
    assert result == payload


async def test_get_json_5xx_retries(cache_dir: Path, fast_config: MLBAPIConfig) -> None:
    """A 5xx response must be retried until success."""
    payload = {"teams": []}
    with aioresponses_ctx() as m:
        m.get(f"{BASE_URL}/teams", status=503)
        m.get(f"{BASE_URL}/teams", payload=payload)
        async with MLBAPIClient(config=fast_config, cache_dir=cache_dir) as client:
            result = await client.get_json("teams", {})
    assert result == payload


async def test_get_json_exhausts_retries_raises(cache_dir: Path) -> None:
    """Exhausting all retries must raise MLBAPIError."""
    config = MLBAPIConfig(
        max_retries=2, backoff_base_s=0.001, backoff_max_s=0.01, rps=10_000.0, burst=10_000.0
    )
    with aioresponses_ctx() as m:
        for _ in range(10):  # more than max_retries
            m.get(f"{BASE_URL}/schedule", status=500)
        async with MLBAPIClient(config=config, cache_dir=cache_dir) as client:
            with pytest.raises(MLBAPIError):
                await client.get_json("schedule", {})


# ---------------------------------------------------------------------------
# Context manager lifecycle
# ---------------------------------------------------------------------------


async def test_context_manager_closes_owned_session(
    cache_dir: Path, fast_config: MLBAPIConfig
) -> None:
    """The client must close the session it owns on __aexit__."""
    async with MLBAPIClient(config=fast_config, cache_dir=cache_dir) as client:
        assert client._session is not None
    assert client._session is None


async def test_external_session_not_closed(cache_dir: Path, fast_config: MLBAPIConfig) -> None:
    """The client must not close a session it does not own."""
    import aiohttp

    async with aiohttp.ClientSession() as external_session:
        async with MLBAPIClient(
            config=fast_config, cache_dir=cache_dir, session=external_session
        ) as client:
            assert not client._owns_session
        # External session must still be open
        assert not external_session.closed
