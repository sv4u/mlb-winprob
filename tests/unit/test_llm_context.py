"""Unit tests for winprob.llm.context."""

from __future__ import annotations

from winprob.llm.context import build_system_prompt


def test_build_system_prompt_non_empty() -> None:
    """System prompt is non-empty and mentions key concepts."""
    prompt = build_system_prompt()
    assert len(prompt) > 100
    assert "MLB" in prompt or "Win Probability" in prompt
    assert "query_predictions" in prompt or "tools" in prompt.lower()
    assert "stacked" in prompt or "model" in prompt.lower()


def test_build_system_prompt_deterministic() -> None:
    """Same prompt on repeated calls."""
    assert build_system_prompt() == build_system_prompt()
