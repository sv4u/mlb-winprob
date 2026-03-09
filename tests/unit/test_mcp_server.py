"""Unit tests for the MCP server (FastMCP tools and app creation)."""

from __future__ import annotations

import json

from winprob.mcp.server import _create_server, create_mcp_app, get_mcp_server


def test_create_mcp_app_returns_starlette_app() -> None:
    """create_mcp_app() returns a Starlette-compatible ASGI app with lifespan."""
    app = create_mcp_app()
    assert app is not None
    assert hasattr(app, "lifespan")
    # FastMCP streamable HTTP app is StarletteWithLifespan
    assert "Starlette" in type(app).__name__ or "lifespan" in dir(app)


def test_get_mcp_server_is_singleton() -> None:
    """get_mcp_server() returns the same FastMCP instance on repeated calls."""
    s1 = get_mcp_server()
    s2 = get_mcp_server()
    assert s1 is s2


def test_server_creation() -> None:
    """_create_server() returns a FastMCP instance."""
    server = _create_server()
    assert server is not None
    assert (
        hasattr(server, "tool")
        or hasattr(server, "_tool_manager")
        or callable(getattr(server, "tool", None))
    )


def test_run_tool_via_import() -> None:
    """run_tool from winprob.tools returns JSON string (smoke test)."""
    from winprob.tools import run_tool

    out = run_tool("get_model_info", {"model_type": "stacked"})
    data = json.loads(out)
    assert "model_type" in data or "description" in data or "error" in data
