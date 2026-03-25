"""MLB calendar helpers: infer the prediction / UI season from the wall clock.

The dashboard and APIs should not hard-code a single year.  During the
off-season (Nov–Dec) the *next* calendar year's spring training and schedule
are usually the relevant prediction target; during Jan–Oct the dominant MLB
season label is the calendar year (spring training and the regular season
that follows).
"""

from __future__ import annotations

from datetime import date
from typing import Iterable


def infer_target_mlb_season(reference: date | None = None) -> int:
    """Return the MLB *season year* used for default predictions and dashboards.

    Args:
        reference: Date to interpret; defaults to today (local).

    Returns:
        Season label: ``reference.year + 1`` in November and December (upcoming
        season focus), otherwise ``reference.year``.
    """
    d = reference or date.today()
    if d.month in (11, 12):
        return d.year + 1
    return d.year


def resolve_api_season(
    requested: int | None,
    *,
    available_seasons: Iterable[int] | None = None,
    reference: date | None = None,
) -> int:
    """Resolve an API or tool season: explicit request wins, else best match to data.

    If the caller omits ``requested`` (or passes 0), uses
    :func:`infer_target_mlb_season` and then picks the best season present in
    ``available_seasons`` (so a deployment that only has 2025 data still returns
    useful defaults when the calendar says 2026).

    Args:
        requested: Explicit season from query/proto, or None/0 for automatic.
        available_seasons: Seasons found in loaded feature rows (may be empty).
        reference: Optional calendar date for inference.

    Returns:
        Resolved season year.
    """
    if requested is not None and int(requested) > 0:
        return int(requested)
    inferred = infer_target_mlb_season(reference)
    seasons = sorted({int(s) for s in (available_seasons or [])}, reverse=True)
    if not seasons:
        return inferred
    if inferred in seasons:
        return inferred
    not_above = [s for s in seasons if s <= inferred]
    if not_above:
        return max(not_above)
    return max(seasons)
