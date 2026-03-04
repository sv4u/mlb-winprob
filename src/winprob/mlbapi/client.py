from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import aiofiles
import aiohttp

try:
    import orjson as jsonlib  # type: ignore[import-untyped,unused-ignore]
except Exception:  # pragma: no cover
    import json as jsonlib  # type: ignore[no-redef]


from winprob.errors import APIError


class MLBAPIError(APIError):
    """MLB Stats API communication failure."""


class MLBRateLimitError(MLBAPIError):
    """HTTP 429 rate-limit response from the MLB Stats API."""


class MLBNotFoundError(MLBAPIError):
    """HTTP 404 — requested resource does not exist."""


@dataclass
class TokenBucket:
    rate: float
    capacity: float

    def __post_init__(self) -> None:
        self._tokens = float(self.capacity)
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        if tokens <= 0:
            return
        if tokens > self.capacity:
            raise ValueError(f"Requested {tokens} tokens exceeds bucket capacity {self.capacity}")
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self._last
                self._last = now
                self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                needed = tokens - self._tokens
                await asyncio.sleep(max(needed / self.rate, 0.001))


@dataclass(frozen=True)
class MLBAPIConfig:
    base_url: str = "https://statsapi.mlb.com"
    api_prefix: str = "/api/v1"
    timeout_s: float = 20.0
    max_concurrency: int = 8
    max_retries: int = 6
    backoff_base_s: float = 0.5
    backoff_max_s: float = 20.0
    rps: float = 5.0
    burst: float = 10.0


class MLBAPIClient:
    def __init__(
        self,
        *,
        config: MLBAPIConfig = MLBAPIConfig(),
        cache_dir: Path = Path("data/raw/mlb_api"),
        refresh: bool = False,
        cache_readonly: bool = False,
        session: Optional[aiohttp.ClientSession] = None,
    ):
        self._cfg = config
        self._cache_dir = cache_dir
        self._refresh = refresh
        self._cache_readonly = cache_readonly
        self._sem = asyncio.Semaphore(config.max_concurrency)
        self._bucket = TokenBucket(rate=config.rps, capacity=config.burst)
        self._session = session
        self._owns_session = session is None

    async def __aenter__(self) -> "MLBAPIClient":
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self._cfg.timeout_s)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    @staticmethod
    def _stable_params(params: Mapping[str, Any]) -> list[tuple[str, str]]:
        return sorted((k, str(v)) for k, v in params.items() if v is not None)

    def _cache_key(self, endpoint: str, params: Mapping[str, Any]) -> str:
        items = self._stable_params(params)
        blob = f"{endpoint}|{items}".encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def _cache_path(self, endpoint: str, key: str) -> Path:
        safe_ep = endpoint.strip("/").replace("/", "_")
        return self._cache_dir / safe_ep / f"{key}.json"

    async def _read_cache(self, path: Path) -> Optional[dict]:
        if not path.exists():
            return None
        async with aiofiles.open(path, "rb") as f:
            raw = await f.read()
        return jsonlib.loads(raw)

    async def _append_meta(self, meta: dict) -> None:
        meta_path = self._cache_dir / "metadata.jsonl"
        dumped = jsonlib.dumps(meta)
        if isinstance(dumped, str):
            dumped = dumped.encode("utf-8")
        meta_line = dumped + b"\n"
        async with aiofiles.open(meta_path, "ab") as f:
            await f.write(meta_line)

    async def _write_cache(self, path: Path, payload: dict, meta: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = jsonlib.dumps(payload)
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        async with aiofiles.open(path, "wb") as f:
            await f.write(raw)
        await self._append_meta(meta)

    async def get_json(self, endpoint: str, params: Mapping[str, Any]) -> dict:
        if self._session is None:
            raise RuntimeError("Client session not initialized; use 'async with MLBAPIClient()'.")

        key = self._cache_key(endpoint, params)
        cpath = self._cache_path(endpoint, key)

        if not self._refresh:
            cached = await self._read_cache(cpath)
            if cached is not None:
                return cached

        if self._cache_readonly:
            raise MLBAPIError(f"Cache miss in readonly mode for endpoint={endpoint} key={key}")

        url = f"{self._cfg.base_url}{self._cfg.api_prefix}/{endpoint.lstrip('/')}"
        attempt = 0
        last_err: Optional[Exception] = None

        while attempt <= self._cfg.max_retries:
            attempt += 1
            async with self._sem:
                try:
                    await self._bucket.acquire(1.0)
                    async with self._session.get(url, params=params) as resp:
                        if resp.status == 404:
                            raise MLBNotFoundError(f"404 for {url} params={dict(params)}")

                        if resp.status == 429:
                            retry_after = resp.headers.get("Retry-After")
                            await asyncio.sleep(
                                float(retry_after)
                                if retry_after
                                else min(
                                    self._cfg.backoff_max_s,
                                    self._cfg.backoff_base_s * (2 ** (attempt - 1)),
                                )
                            )
                            last_err = MLBRateLimitError(f"429 for {url}")
                            continue

                        if 500 <= resp.status <= 599:
                            await asyncio.sleep(
                                min(
                                    self._cfg.backoff_max_s,
                                    self._cfg.backoff_base_s * (2 ** (attempt - 1)),
                                )
                            )
                            last_err = MLBAPIError(f"{resp.status} for {url}")
                            continue

                        # Non-retryable client errors (400, 401, 403, etc.)
                        if resp.status >= 400:
                            raise MLBAPIError(f"{resp.status} for {url} params={dict(params)}")
                        payload = await resp.json()
                        meta = {
                            "ts_unix": time.time(),
                            "url": url,
                            "params": dict(params),
                            "cache_key": key,
                            "endpoint": endpoint,
                            "status": resp.status,
                        }
                        await self._write_cache(cpath, payload, meta)
                        return payload

                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    last_err = e
                    await asyncio.sleep(
                        min(
                            self._cfg.backoff_max_s, self._cfg.backoff_base_s * (2 ** (attempt - 1))
                        )
                    )

        raise MLBAPIError(
            f"Failed after retries endpoint={endpoint} params={dict(params)} err={last_err}"
        )
