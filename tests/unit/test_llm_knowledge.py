"""Unit tests for winprob.llm.knowledge."""

from __future__ import annotations

from winprob.llm.knowledge import (
    FEATURE_LABELS,
    GLOSSARY,
    MODEL_DOCS,
    get_feature_description,
    get_glossary_term,
    get_model_docs,
)


def test_get_feature_description_known() -> None:
    """Known feature returns its label."""
    assert get_feature_description("home_elo") == "Home team Elo rating"
    assert get_feature_description("park_run_factor") == "Park run factor (1.0 = neutral)"


def test_get_feature_description_unknown() -> None:
    """Unknown feature returns the name."""
    assert get_feature_description("unknown_feat") == "unknown_feat"
    assert get_feature_description("") == "unknown feature"


def test_get_glossary_term_known() -> None:
    """Known term returns definition."""
    out = get_glossary_term("elo")
    assert out is not None
    assert "1500" in out
    out = get_glossary_term("wOBA")
    assert get_glossary_term("woba") is not None


def test_get_glossary_term_unknown() -> None:
    """Unknown term returns None."""
    assert get_glossary_term("xyzz") is None
    assert get_glossary_term("") is None


def test_get_model_docs_known() -> None:
    """Known model returns description."""
    assert (
        "stacked" in get_model_docs("stacked").lower()
        or "meta" in get_model_docs("stacked").lower()
    )
    assert "logistic" in get_model_docs("logistic").lower()


def test_get_model_docs_unknown() -> None:
    """Unknown model returns generic message."""
    out = get_model_docs("unknown_model")
    assert "unknown_model" in out


def test_feature_labels_non_empty() -> None:
    """FEATURE_LABELS has expected core keys."""
    assert "home_elo" in FEATURE_LABELS
    assert "away_elo" in FEATURE_LABELS
    assert "elo_diff" in FEATURE_LABELS
    assert len(FEATURE_LABELS) >= 20


def test_glossary_non_empty() -> None:
    """GLOSSARY has expected terms."""
    assert "elo" in GLOSSARY
    assert "woba" in GLOSSARY
    assert "fip" in GLOSSARY


def test_model_docs_has_all_types() -> None:
    """MODEL_DOCS has stacked and base models."""
    assert "stacked" in MODEL_DOCS
    assert "logistic" in MODEL_DOCS
    assert "lightgbm" in MODEL_DOCS
