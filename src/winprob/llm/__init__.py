"""LLM chat and tool-calling for the MLB Win Probability system.

This module provides:
- knowledge: feature descriptions, model docs, sabermetric glossary (for describe_feature and context).
- context: system prompt builder for the chat model.
- tools: tool functions and flat JSON schemas for the 3B model (query_predictions, explain_prediction, etc.).
- engine: ChatEngine (Ollama client, history, tool-calling loop, streaming).

Public API:
- get_feature_description
- get_glossary_term
- get_model_docs
- build_system_prompt
- TOOL_SCHEMAS
- run_tool
"""

from __future__ import annotations

from winprob.llm.knowledge import (
    get_feature_description,
    get_glossary_term,
    get_model_docs,
)
from winprob.llm.context import build_system_prompt
from winprob.llm.tools import TOOL_SCHEMAS, run_tool

__all__ = [
    "build_system_prompt",
    "get_feature_description",
    "get_glossary_term",
    "get_model_docs",
    "run_tool",
    "TOOL_SCHEMAS",
]
