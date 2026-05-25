"""Unit tests for MLB-aware season resolution."""

from __future__ import annotations

from datetime import date

import pytest

from mlb_predict.season import current_mlb_season, resolve_season


@pytest.mark.parametrize(
    ("as_of", "expected"),
    [
        (date(2026, 5, 25), 2026),
        (date(2026, 2, 1), 2026),
        (date(2026, 1, 15), 2025),
        (date(2025, 11, 10), 2025),
        (date(2025, 12, 31), 2025),
        (date(2026, 10, 31), 2026),
    ],
)
def test_current_mlb_season(as_of: date, expected: int) -> None:
    """current_mlb_season follows Feb–Oct/Nov–Dec calendar year and Jan previous year."""
    assert current_mlb_season(as_of=as_of) == expected


def test_resolve_season_explicit() -> None:
    """Explicit requested season is returned unchanged."""
    assert resolve_season(2019, available=[2000, 2019, 2025]) == 2019


def test_resolve_season_default_current() -> None:
    """Missing request uses MLB-aware current season even when not in available list."""
    as_of = date(2026, 5, 25)
    assert resolve_season(None, available=[2000, 2025], as_of=as_of) == 2026
