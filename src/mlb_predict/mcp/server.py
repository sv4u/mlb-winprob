"""MCP server: FastMCP instance and Streamable HTTP app for mounting in FastAPI.

All tools delegate to mlb_predict.tools.run_tool so they use the same in-memory
data and model as the web app (no separate process). The app must be mounted
after the FastAPI lifespan has loaded data (combined lifespan in main.py).
"""

from __future__ import annotations

import logging
from typing import Any

from fastmcp import FastMCP
from fastmcp.server.http import create_streamable_http_app

from mlb_predict.tools import run_tool

logger = logging.getLogger(__name__)

MCP_SERVER_NAME = "MLB Win Probability"
_STREAMABLE_HTTP_PATH = "/"


def _call(tool_name: str, **kwargs: Any) -> str:
    """Run a tool by name and return its JSON string result."""
    # Drop None values so run_tool sees only provided params
    params = {k: v for k, v in kwargs.items() if v is not None}
    return run_tool(tool_name, params)


def _create_server() -> FastMCP:
    """Build FastMCP server and register all tools (same surface as chat)."""
    mcp = FastMCP(MCP_SERVER_NAME)

    @mcp.tool()
    def query_predictions(
        team: str | None = None,
        season: int | None = None,
        date: str | None = None,
    ) -> str:
        """Find game predictions by team, season, or date. Team: Retrosheet code (e.g. LAD, NYA)."""
        return _call("query_predictions", team=team, season=season, date=date)

    @mcp.tool()
    def explain_prediction(game_pk: int) -> str:
        """Get SHAP breakdown and key stats for a single game by MLB game ID (game_pk)."""
        return _call("explain_prediction", game_pk=game_pk)

    @mcp.tool()
    def compare_models() -> str:
        """Get cross-validation summary for all models."""
        return _call("compare_models")

    @mcp.tool()
    def get_team_stats(season: int | None = None) -> str:
        """Get batting and pitching stats for all teams in a season (e.g. 2026)."""
        return _call("get_team_stats", season=season)

    @mcp.tool()
    def get_standings(season: int | None = None) -> str:
        """Get predicted and actual division standings for a season."""
        return _call("get_standings", season=season)

    @mcp.tool()
    def find_upsets(
        season: int | None = None,
        min_prob: float | None = None,
        limit: int | None = None,
    ) -> str:
        """Find biggest upsets (favorites that lost). min_prob: e.g. 0.65, limit: e.g. 20."""
        return _call("find_upsets", season=season, min_prob=min_prob, limit=limit)

    @mcp.tool()
    def get_drift_metrics() -> str:
        """Get prediction drift summary vs previous snapshot."""
        return _call("get_drift_metrics")

    @mcp.tool()
    def get_model_info(model_type: str) -> str:
        """Get short description of a model type (e.g. stacked, lightgbm, logistic)."""
        return _call("get_model_info", model_type=model_type)

    @mcp.tool()
    def describe_feature(name: str) -> str:
        """Get description of a feature or sabermetric term (e.g. home_elo, woba)."""
        return _call("describe_feature", name=name)

    @mcp.tool()
    def get_season_summary() -> str:
        """List available seasons and the active model."""
        return _call("get_season_summary")

    @mcp.tool()
    def find_ev_bets(season: int | None = None, min_edge: float | None = None) -> str:
        """Find positive-EV moneyline bets vs the model (optional season, min edge)."""
        return _call("find_ev_bets", season=season, min_edge=min_edge)

    @mcp.tool()
    def get_live_odds() -> str:
        """Get live moneyline odds for today's games (when ODDS_API_KEY is set)."""
        return _call("get_live_odds")

    return mcp


_server: FastMCP | None = None


def get_mcp_server() -> FastMCP:
    """Return the singleton FastMCP server instance."""
    global _server
    if _server is None:
        _server = _create_server()
    return _server


def create_mcp_app():
    """Create the Streamable HTTP ASGI app for the MCP server (mount at /mcp)."""
    server = get_mcp_server()
    return create_streamable_http_app(
        server,
        streamable_http_path=_STREAMABLE_HTTP_PATH,
    )
