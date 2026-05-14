"""Unit tests for mlb_predict.tools (run_tool and TOOL_SCHEMAS)."""

from __future__ import annotations

import json

from mlb_predict.tools import TOOL_SCHEMAS, run_tool


def test_tool_schemas_count() -> None:
    """There are 12 tools registered."""
    assert len(TOOL_SCHEMAS) == 12
    names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert "query_predictions" in names
    assert "find_ev_bets" in names
    assert "get_live_odds" in names


def test_tool_schemas_flat() -> None:
    """Each schema has type function and flat parameters."""
    for s in TOOL_SCHEMAS:
        assert s["type"] == "function"
        fn = s["function"]
        assert "name" in fn
        assert "parameters" in fn
        params = fn["parameters"]
        assert params["type"] == "object"
        assert "properties" in params


def test_run_tool_unknown() -> None:
    """Unknown tool returns error JSON."""
    out = run_tool("no_such_tool", {})
    data = json.loads(out)
    assert "error" in data
    assert "no_such_tool" in data["error"]


def test_run_tool_get_model_info() -> None:
    """get_model_info returns description."""
    out = run_tool("get_model_info", {"model_type": "stacked"})
    data = json.loads(out)
    assert "model_type" in data
    assert "description" in data
    assert data["model_type"] == "stacked"


def test_run_tool_describe_feature() -> None:
    """describe_feature returns description and optional glossary."""
    out = run_tool("describe_feature", {"name": "home_elo"})
    data = json.loads(out)
    assert data["feature_or_term"] == "home_elo"
    assert "Elo" in data["description"]


def test_run_tool_get_season_summary_returns_valid_structure() -> None:
    """get_season_summary returns JSON with seasons and/or active_model or error."""
    out = run_tool("get_season_summary", {})
    data = json.loads(out)
    assert isinstance(data, dict)
    assert "seasons" in data or "error" in data
    if "error" not in data:
        assert "active_model" in data or "available_models" in data


def test_run_tool_find_ev_bets_not_ready() -> None:
    """find_ev_bets returns error when data is not loaded."""
    out = run_tool("find_ev_bets", {})
    data = json.loads(out)
    assert "error" in data or "message" in data


def test_run_tool_get_live_odds_not_configured() -> None:
    """get_live_odds returns not-configured message when no API key is set."""
    out = run_tool("get_live_odds", {})
    data = json.loads(out)
    assert "message" in data
    msg = data["message"].lower()
    assert "not configured" in msg or "not available" in msg


def test_run_tool_compare_models_returns_list() -> None:
    """compare_models returns JSON array (possibly empty)."""
    out = run_tool("compare_models", {})
    data = json.loads(out)
    assert isinstance(data, list) or (isinstance(data, dict) and "error" not in data)


def test_run_tool_get_drift_metrics_returns_dict() -> None:
    """get_drift_metrics returns dict with message or metrics."""
    out = run_tool("get_drift_metrics", {})
    data = json.loads(out)
    assert isinstance(data, dict)
    assert "message" in data or "mean_abs_delta" in data or "error" in data
