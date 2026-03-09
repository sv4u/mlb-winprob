"""MCP server for MLB Win Probability — tools, predictions, odds, models.

Exposes tools over the Model Context Protocol (Streamable HTTP) so Cursor and
other MCP clients can query predictions, standings, SHAP explanations, drift,
odds, and model info.

The MCP server is mounted at /mcp when the FastAPI app runs; use that URL when
configuring Cursor or other MCP clients (e.g. on your home network).
"""

from winprob.mcp.server import create_mcp_app

__all__ = ["create_mcp_app"]
