"""MLB calendar season helpers shared by API defaults and the web UI."""

from __future__ import annotations

from datetime import date


def current_mlb_season(as_of: date | None = None) -> int:
    """Return the active MLB season year for a given calendar date.

    MLB-aware rule (UI + API default):
    - February through October: calendar year (spring training + regular season).
    - November and December: calendar year (recently completed season still "current").
    - January: previous calendar year (offseason before spring training opens).

    Args:
        as_of: Date to evaluate. Defaults to today in the local timezone.

    Returns:
        Four-digit MLB season year.
    """
    today = as_of or date.today()
    if today.month == 1:
        return today.year - 1
    return today.year


def resolve_season(
    requested: int | None,
    available: list[int] | None = None,
    *,
    as_of: date | None = None,
) -> int:
    """Resolve an explicit season query param or fall back to the MLB-aware current season.

    When the current season is not yet present in ``available`` (e.g. before ingest),
    still returns the current season so callers can show an empty state rather than
    silently switching to historical data.

    Args:
        requested: Season from a query parameter, or ``None`` to use the default.
        available: Optional sorted list of seasons present in feature data.
        as_of: Date used for ``current_mlb_season`` when ``requested`` is ``None``.

    Returns:
        Resolved season year.
    """
    if requested is not None:
        return requested
    return current_mlb_season(as_of=as_of)
