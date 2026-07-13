"""Unit tests for the admin-token auth guard on /api/admin and /ws/admin routes."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import pytest
from fastapi import HTTPException
from starlette.datastructures import Headers, QueryParams
from starlette.requests import Request

if TYPE_CHECKING:
    pass


@pytest.fixture(autouse=True)
def _reset_admin_auth_module(monkeypatch: pytest.MonkeyPatch):
    """Reload admin_auth per test so the module-level cached token doesn't leak."""
    import mlb_predict.app.admin_auth as admin_auth

    monkeypatch.delenv("MLB_ADMIN_TOKEN", raising=False)
    importlib.reload(admin_auth)
    yield admin_auth
    monkeypatch.delenv("MLB_ADMIN_TOKEN", raising=False)
    importlib.reload(admin_auth)


def _make_request(headers: dict[str, str] | None = None, query_string: str = "") -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/admin/status",
        "headers": Headers(headers or {}).raw,
        "query_string": query_string.encode(),
    }
    return Request(scope)


class _FakeWebSocket:
    def __init__(
        self, query_params: dict[str, str] | None = None, headers: dict[str, str] | None = None
    ):
        self.query_params = QueryParams(query_params or {})
        self.headers = Headers(headers or {})


class TestGetAdminToken:
    def test_uses_env_var_when_set(self, monkeypatch, _reset_admin_auth_module) -> None:
        monkeypatch.setenv("MLB_ADMIN_TOKEN", "fixed-token")
        importlib.reload(_reset_admin_auth_module)
        assert _reset_admin_auth_module.get_admin_token() == "fixed-token"

    def test_generates_and_caches_random_token_when_unset(self, _reset_admin_auth_module) -> None:
        token1 = _reset_admin_auth_module.get_admin_token()
        token2 = _reset_admin_auth_module.get_admin_token()
        assert token1 == token2
        assert len(token1) > 20


class TestRequireAdminToken:
    def test_raises_403_with_no_token(self, _reset_admin_auth_module) -> None:
        request = _make_request()
        with pytest.raises(HTTPException) as exc_info:
            _reset_admin_auth_module.require_admin_token(request)
        assert exc_info.value.status_code == 403

    def test_raises_403_with_wrong_token(self, _reset_admin_auth_module) -> None:
        _reset_admin_auth_module.get_admin_token()
        request = _make_request(headers={"x-admin-token": "wrong"})
        with pytest.raises(HTTPException) as exc_info:
            _reset_admin_auth_module.require_admin_token(request)
        assert exc_info.value.status_code == 403

    def test_accepts_correct_token_via_header(self, _reset_admin_auth_module) -> None:
        token = _reset_admin_auth_module.get_admin_token()
        request = _make_request(headers={"x-admin-token": token})
        _reset_admin_auth_module.require_admin_token(request)  # does not raise

    def test_accepts_correct_token_via_bearer_auth_header(self, _reset_admin_auth_module) -> None:
        token = _reset_admin_auth_module.get_admin_token()
        request = _make_request(headers={"authorization": f"Bearer {token}"})
        _reset_admin_auth_module.require_admin_token(request)  # does not raise

    def test_accepts_correct_token_via_query_param(self, _reset_admin_auth_module) -> None:
        token = _reset_admin_auth_module.get_admin_token()
        request = _make_request(query_string=f"token={token}")
        _reset_admin_auth_module.require_admin_token(request)  # does not raise


class TestCheckWsAdminToken:
    def test_false_with_no_token(self, _reset_admin_auth_module) -> None:
        ws = _FakeWebSocket()
        assert _reset_admin_auth_module.check_ws_admin_token(ws) is False

    def test_false_with_wrong_token(self, _reset_admin_auth_module) -> None:
        _reset_admin_auth_module.get_admin_token()
        ws = _FakeWebSocket(query_params={"token": "wrong"})
        assert _reset_admin_auth_module.check_ws_admin_token(ws) is False

    def test_true_with_correct_token_via_query_param(self, _reset_admin_auth_module) -> None:
        token = _reset_admin_auth_module.get_admin_token()
        ws = _FakeWebSocket(query_params={"token": token})
        assert _reset_admin_auth_module.check_ws_admin_token(ws) is True

    def test_true_with_correct_token_via_bearer_header(self, _reset_admin_auth_module) -> None:
        token = _reset_admin_auth_module.get_admin_token()
        ws = _FakeWebSocket(headers={"authorization": f"Bearer {token}"})
        assert _reset_admin_auth_module.check_ws_admin_token(ws) is True
