"""Unit tests for winprob.llm.engine."""

from __future__ import annotations

import pytest

from winprob.llm.engine import ChatEngine, _fallback_tool_for_message


def test_fallback_tool_keywords() -> None:
    """Keyword fallback maps user text to tool names."""
    assert _fallback_tool_for_message("show me the standings") == "get_standings"
    assert _fallback_tool_for_message("why did they win? explain") == "explain_prediction"
    assert _fallback_tool_for_message("any EV bets?") == "find_ev_bets"
    assert _fallback_tool_for_message("drift metrics") == "get_drift_metrics"
    assert _fallback_tool_for_message("random query") is None


def test_engine_session_cap() -> None:
    """Sessions are capped at max_sessions (LRU eviction)."""
    engine = ChatEngine(max_sessions=3, max_session_messages=5)
    engine.append_user("s1", "hi")
    engine.append_user("s2", "hi")
    engine.append_user("s3", "hi")
    assert engine.session_count() == 3
    engine.append_user("s4", "hi")
    assert engine.session_count() == 3
    assert "s1" not in engine._sessions
    assert "s4" in engine._sessions


def test_engine_message_cap() -> None:
    """Messages per session are capped at max_session_messages."""
    engine = ChatEngine(max_sessions=10, max_session_messages=3)
    engine.append_user("s1", "m1")
    engine.append_user("s1", "m2")
    engine.append_user("s1", "m3")
    msgs = engine._get_or_create_messages("s1")
    assert len(msgs) == 3
    engine.append_user("s1", "m4")
    assert len(msgs) == 3
    assert msgs[0]["content"] == "m2"


def test_engine_append_assistant_only() -> None:
    """append_assistant_only adds one assistant message."""
    engine = ChatEngine(max_sessions=10, max_session_messages=10)
    engine.append_user("s1", "hi")
    engine.append_assistant_only("s1", "hello")
    msgs = engine._get_or_create_messages("s1")
    assert len(msgs) == 2
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "hello"


@pytest.mark.asyncio
async def test_engine_ollama_available_unreachable() -> None:
    """When Ollama is not running, ollama_available returns False (or True if it is)."""
    engine = ChatEngine()
    # We cannot assert False always because Ollama might be running in CI
    result = await engine.ollama_available()
    assert isinstance(result, bool)
