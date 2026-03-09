"""Backend tools for MCP and other consumers: predictions, standings, SHAP, drift, odds.

Tool implementations live in run.py; knowledge (feature/model descriptions) in knowledge.py.
"""

from mlb_predict.tools.knowledge import (
    FEATURE_LABELS,
    GLOSSARY,
    MODEL_DOCS,
    get_feature_description,
    get_glossary_term,
    get_model_docs,
)
from mlb_predict.tools.run import TOOL_SCHEMAS, run_tool

__all__ = [
    "FEATURE_LABELS",
    "GLOSSARY",
    "MODEL_DOCS",
    "TOOL_SCHEMAS",
    "get_feature_description",
    "get_glossary_term",
    "get_model_docs",
    "run_tool",
]
