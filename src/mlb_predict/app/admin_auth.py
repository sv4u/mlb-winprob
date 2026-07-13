"""Shared-secret authentication for the admin API/WebSocket surface.

The admin routes (`/api/admin/*`, `/ws/admin/shell`, `/ws/admin/repl`) can run
arbitrary shell commands and Python, so they must never be reachable without a
secret — including on a LAN-bound dev server (docker-compose publishes on
``0.0.0.0`` for LAN access).

Token precedence:
1. ``MLB_ADMIN_TOKEN`` env var, if set.
2. A random token generated once per process and logged at WARNING level so an
   operator can retrieve it from ``docker compose logs`` without pre-configuring
   anything for local/single-user use.
"""

from __future__ import annotations

import logging
import os
import secrets

from fastapi import HTTPException, Request, WebSocket

logger = logging.getLogger(__name__)

_token: str | None = None


def get_admin_token() -> str:
    """Return the admin token, generating and logging one on first use if unset."""
    global _token  # noqa: PLW0603
    if _token is not None:
        return _token

    env_val = os.environ.get("MLB_ADMIN_TOKEN", "").strip()
    if env_val:
        _token = env_val
        return _token

    _token = secrets.token_urlsafe(32)
    logger.warning(
        "MLB_ADMIN_TOKEN not set — generated a random admin token for this run: %s\n"
        "Set MLB_ADMIN_TOKEN to a fixed value to keep access across restarts. "
        "Use this token in the dashboard's admin tools (you'll be prompted for it), "
        "or pass it as the 'X-Admin-Token' header / '?token=' query param.",
        _token,
    )
    return _token


def _extract_request_token(request: Request) -> str | None:
    header = request.headers.get("x-admin-token")
    if header:
        return header
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.query_params.get("token")


def require_admin_token(request: Request) -> None:
    """FastAPI dependency: raise 403 unless a valid admin token is supplied."""
    supplied = _extract_request_token(request)
    if not supplied or not secrets.compare_digest(supplied, get_admin_token()):
        raise HTTPException(status_code=403, detail="Missing or invalid admin token.")


def check_ws_admin_token(websocket: WebSocket) -> bool:
    """Return True if the WebSocket connection carries a valid admin token."""
    supplied = websocket.query_params.get("token")
    if not supplied:
        auth = websocket.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            supplied = auth[7:].strip()
    if not supplied:
        return False
    return secrets.compare_digest(supplied, get_admin_token())
