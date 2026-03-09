"""Request timing middleware and sub-operation instrumentation.

Provides three capabilities:

1. ``TimingMiddleware`` — ASGI middleware that adds an ``X-Process-Time-Ms``
   response header to every HTTP response and logs API request durations.
2. ``timed_operation`` — context manager (sync *and* async) that records
   named sub-operation durations within a single request via ``contextvars``.
3. ``get_request_timings`` — retrieves the list of sub-operation timings
   for the current request context.
"""

from __future__ import annotations

import logging
import time
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

_request_timings: ContextVar[list[dict[str, object]]] = ContextVar(
    "request_timings",
)


class TimingMiddleware(BaseHTTPMiddleware):
    """Measure wall-clock request duration and emit an ``X-Process-Time-Ms`` header."""

    async def dispatch(self, request: Request, call_next: object) -> Response:
        """Wrap the downstream handler with timing instrumentation."""
        token = _request_timings.set([])
        t0 = time.monotonic()

        response: Response = await call_next(request)  # type: ignore[operator]

        elapsed_ms = (time.monotonic() - t0) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"

        path = request.url.path
        if path.startswith("/api/") or path in (
            "/",
            "/standings",
            "/leaders",
            "/players",
            "/wiki",
            "/dashboard",
        ):
            sub_ops = _request_timings.get()
            if sub_ops:
                breakdown = ", ".join(f"{op['op']}={op['ms']:.1f}ms" for op in sub_ops)
                logger.info(
                    "%s %s completed in %.1fms [%s]",
                    request.method,
                    path,
                    elapsed_ms,
                    breakdown,
                )
            else:
                logger.info(
                    "%s %s completed in %.1fms",
                    request.method,
                    path,
                    elapsed_ms,
                )

        _request_timings.reset(token)
        return response


class timed_operation:
    """Context manager that records a named sub-operation duration.

    Works as both a synchronous and asynchronous context manager::

        with timed_operation("shap_computation"):
            ...

        async with timed_operation("standings_fetch"):
            ...

    Timings are collected into a per-request ``ContextVar`` and logged at
    DEBUG level.  The middleware aggregates them into the request-level log
    line.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.elapsed_ms: float = 0.0
        self._t0: float = 0.0

    def _record(self) -> None:
        self.elapsed_ms = (time.monotonic() - self._t0) * 1000
        try:
            timings = _request_timings.get()
            timings.append({"op": self.name, "ms": round(self.elapsed_ms, 2)})
        except LookupError:
            pass
        logger.debug("%s completed in %.1fms", self.name, self.elapsed_ms)

    # Sync context manager
    def __enter__(self) -> timed_operation:
        self._t0 = time.monotonic()
        return self

    def __exit__(self, *exc: object) -> None:
        self._record()

    # Async context manager
    async def __aenter__(self) -> timed_operation:
        self._t0 = time.monotonic()
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._record()


def get_request_timings() -> list[dict[str, object]]:
    """Return the sub-operation timings collected for the current request."""
    try:
        return _request_timings.get()
    except LookupError:
        return []
