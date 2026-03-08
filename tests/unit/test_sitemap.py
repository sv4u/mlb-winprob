"""Tests for the sitemap page template and route registration."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    pass


_TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "src" / "winprob" / "app" / "templates"


# ---------------------------------------------------------------------------
# Template content tests
# ---------------------------------------------------------------------------


@pytest.fixture
def sitemap_html() -> str:
    """Read the sitemap.html template as a string."""
    return (_TEMPLATE_DIR / "sitemap.html").read_text()


_EXPECTED_PAGE_PATHS = [
    "/",
    "/season/2026",
    "/standings",
    "/odds",
    "/wiki",
    "/chat",
    "/dashboard",
    "/sitemap",
]

_EXPECTED_API_PATHS = [
    "/api/version",
    "/api/seasons",
    "/api/teams",
    "/api/games",
    "/api/upsets",
    "/api/cv-summary",
    "/api/standings",
    "/api/team-stats",
    "/api/admin/status",
    "/api/admin/ingest",
    "/api/admin/update",
    "/api/admin/retrain",
]


def test_sitemap_template_exists() -> None:
    """sitemap.html must exist in the templates directory."""
    assert (_TEMPLATE_DIR / "sitemap.html").exists()


@pytest.mark.parametrize("path", _EXPECTED_PAGE_PATHS)
def test_sitemap_contains_page_path(sitemap_html: str, path: str) -> None:
    """The sitemap template must reference every HTML page path."""
    assert path in sitemap_html


@pytest.mark.parametrize("path", _EXPECTED_API_PATHS)
def test_sitemap_contains_api_path(sitemap_html: str, path: str) -> None:
    """The sitemap template must reference every API endpoint path."""
    assert path in sitemap_html


def test_sitemap_contains_game_detail_path(sitemap_html: str) -> None:
    """The sitemap must reference the dynamic game detail path pattern."""
    assert "/game/{game_pk}" in sitemap_html or "/game/" in sitemap_html


def test_sitemap_has_method_badges(sitemap_html: str) -> None:
    """The sitemap must contain GET and POST method badges."""
    assert "method-get" in sitemap_html
    assert "method-post" in sitemap_html


# ---------------------------------------------------------------------------
# Navigation link tests — every page must link to the sitemap
# ---------------------------------------------------------------------------


_TEMPLATES_WITH_SITEMAP_NAV = [
    "index.html",
    "game.html",
    "season_2026.html",
    "standings.html",
    "ev_calculator.html",
    "odds_hub.html",
    "wiki.html",
    "chat.html",
    "dashboard.html",
    "sitemap.html",
]


@pytest.mark.parametrize("template_name", _TEMPLATES_WITH_SITEMAP_NAV)
def test_template_has_sitemap_nav_link(template_name: str) -> None:
    """Each page template must include a navigation link to /sitemap."""
    content = (_TEMPLATE_DIR / template_name).read_text()
    assert 'href="/sitemap"' in content


# ---------------------------------------------------------------------------
# Route registration test
# ---------------------------------------------------------------------------


def test_sitemap_route_registered() -> None:
    """The /sitemap GET route must be registered on the FastAPI app."""
    from winprob.app.main import app

    paths = {route.path for route in app.routes if hasattr(route, "path")}  # type: ignore[union-attr]
    assert "/sitemap" in paths


def test_xml_sitemap_route_registered() -> None:
    """The /sitemap.xml GET route must be registered on the FastAPI app."""
    from winprob.app.main import app

    paths = {route.path for route in app.routes if hasattr(route, "path")}  # type: ignore[union-attr]
    assert "/sitemap.xml" in paths


# ---------------------------------------------------------------------------
# XML sitemap content tests
# ---------------------------------------------------------------------------


def test_html_sitemap_references_xml_version(sitemap_html: str) -> None:
    """The HTML sitemap must link to the XML sitemap."""
    assert "/sitemap.xml" in sitemap_html


async def test_xml_sitemap_contains_expected_paths() -> None:
    """The xml_sitemap handler must emit all static page paths."""
    from starlette.requests import Request

    from winprob.app.main import xml_sitemap

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/sitemap.xml",
        "root_path": "",
        "headers": [(b"host", b"localhost")],
        "query_string": b"",
        "server": ("localhost", 30087),
    }
    req = Request(scope)
    resp = await xml_sitemap(req)
    body = resp.body.decode()  # type: ignore[union-attr]

    assert '<?xml version="1.0"' in body
    assert "http://www.sitemaps.org/schemas/sitemap/0.9" in body
    for path in [
        "/",
        "/season/2026",
        "/standings",
        "/odds",
        "/wiki",
        "/chat",
        "/dashboard",
        "/sitemap",
    ]:
        assert f"{path}</loc>" in body
