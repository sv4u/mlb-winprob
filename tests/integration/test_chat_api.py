"""Integration tests for MCP server mount (chat UI/API removed)."""

from __future__ import annotations

from winprob.app.main import app


def test_mcp_mount_registered_in_app() -> None:
    """The FastAPI app has the /mcp mount (MCP server is registered)."""
    from starlette.routing import Mount

    mounts = [r for r in app.routes if isinstance(r, Mount)]
    mcp_mounts = [m for m in mounts if m.path == "/mcp"]
    assert len(mcp_mounts) == 1, "Expected exactly one /mcp mount for MCP Streamable HTTP"
