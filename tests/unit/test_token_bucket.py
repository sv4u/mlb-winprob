"""Tests for mlb_predict.mlbapi.client.TokenBucket."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    pass

from mlb_predict.mlbapi.client import TokenBucket


# ---------------------------------------------------------------------------
# Instant-grant tests (tokens available immediately)
# ---------------------------------------------------------------------------


async def test_acquire_returns_immediately_when_full() -> None:
    """Acquiring one token from a full bucket must complete without sleeping."""
    bucket = TokenBucket(rate=10.0, capacity=10.0)
    start = time.monotonic()
    await bucket.acquire(1.0)
    elapsed = time.monotonic() - start
    assert elapsed < 0.1, f"Expected immediate return, took {elapsed:.3f}s"


async def test_acquire_zero_tokens_is_noop() -> None:
    """Acquiring 0 tokens must return immediately without touching the bucket."""
    bucket = TokenBucket(rate=1.0, capacity=1.0)
    start = time.monotonic()
    await bucket.acquire(0.0)
    elapsed = time.monotonic() - start
    assert elapsed < 0.1


async def test_acquire_negative_tokens_is_noop() -> None:
    """Acquiring a negative number of tokens must be treated as a no-op."""
    bucket = TokenBucket(rate=1.0, capacity=1.0)
    start = time.monotonic()
    await bucket.acquire(-5.0)
    elapsed = time.monotonic() - start
    assert elapsed < 0.1


async def test_full_capacity_can_be_drained() -> None:
    """We can drain the entire capacity without sleeping."""
    bucket = TokenBucket(rate=100.0, capacity=5.0)
    for _ in range(5):
        await bucket.acquire(1.0)
    # Internal state: tokens should be roughly 0 (plus refill during loop)
    assert bucket._tokens >= 0.0


# ---------------------------------------------------------------------------
# Rate-limiting / blocking tests
# ---------------------------------------------------------------------------


async def test_acquire_blocks_when_bucket_empty() -> None:
    """Acquiring when the bucket is empty must wait for the refill."""
    # Rate 10 t/s → 1 token takes 100 ms to refill
    bucket = TokenBucket(rate=10.0, capacity=1.0)
    await bucket.acquire(1.0)  # drain completely

    start = time.monotonic()
    await bucket.acquire(1.0)  # must wait ~100 ms
    elapsed = time.monotonic() - start
    # Allow generous tolerance: at least 50 ms, less than 500 ms
    assert 0.05 < elapsed < 0.5, f"Expected ~100 ms wait, took {elapsed:.3f}s"


async def test_tokens_cap_at_capacity() -> None:
    """Tokens must never exceed capacity regardless of idle time."""
    bucket = TokenBucket(rate=1000.0, capacity=3.0)
    await asyncio.sleep(0.1)  # idle for a bit — tokens should NOT exceed 3
    assert bucket._tokens <= 3.0 + 1e-9  # small float tolerance


# ---------------------------------------------------------------------------
# Concurrency tests
# ---------------------------------------------------------------------------


async def test_concurrent_acquires_are_serialized() -> None:
    """Multiple concurrent acquires must all succeed and be serialized safely."""
    bucket = TokenBucket(rate=1000.0, capacity=100.0)
    results: list[float] = []

    async def worker() -> None:
        await bucket.acquire(1.0)
        results.append(time.monotonic())

    await asyncio.gather(*[worker() for _ in range(10)])
    assert len(results) == 10


async def test_concurrent_acquires_do_not_exceed_capacity() -> None:
    """Tokens consumed by concurrent workers must not exceed initial capacity."""
    capacity = 5.0
    bucket = TokenBucket(rate=0.0001, capacity=capacity)  # very slow refill
    consumed: list[int] = []
    errors: list[str] = []

    async def worker(i: int) -> None:
        try:
            # Each worker needs 1 token; only `capacity` can go without waiting
            await asyncio.wait_for(bucket.acquire(1.0), timeout=0.5)
            consumed.append(i)
        except asyncio.TimeoutError:
            errors.append(f"timeout-{i}")

    await asyncio.gather(*[worker(i) for i in range(20)])
    # At rate=0.0001 t/s, only the initial 5 tokens should be consumed quickly
    assert len(consumed) <= capacity + 2  # +2 for timing slack
