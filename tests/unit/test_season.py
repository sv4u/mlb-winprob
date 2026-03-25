"""Tests for mlb_predict.season — calendar-based default season resolution."""

from __future__ import annotations

from datetime import date

from mlb_predict.season import infer_target_mlb_season, resolve_api_season


def test_infer_target_mlb_season_regular_months() -> None:
    """Jan–Oct map to the same calendar year."""
    assert infer_target_mlb_season(date(2026, 3, 15)) == 2026
    assert infer_target_mlb_season(date(2025, 8, 1)) == 2025


def test_infer_target_mlb_season_off_season() -> None:
    """Nov–Dec point at the upcoming season year."""
    assert infer_target_mlb_season(date(2025, 11, 1)) == 2026
    assert infer_target_mlb_season(date(2025, 12, 31)) == 2026


def test_resolve_api_season_explicit() -> None:
    """Explicit positive season bypasses inference."""
    assert resolve_api_season(2024, available_seasons=[2023, 2024, 2025]) == 2024


def test_resolve_api_season_falls_back_to_loaded_data() -> None:
    """When inferred year is missing, pick the best season at or below it."""
    assert (
        resolve_api_season(None, available_seasons=[2024, 2025], reference=date(2026, 3, 1)) == 2025
    )


def test_resolve_api_season_empty_available() -> None:
    """No data → pure inference."""
    assert resolve_api_season(None, available_seasons=[], reference=date(2026, 1, 10)) == 2026
