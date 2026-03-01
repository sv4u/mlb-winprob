"""Park run-factor computation from historical Retrosheet game logs.

A park factor > 1 indicates a hitter-friendly park (more runs per game than
average), < 1 indicates a pitcher-friendly park.  We compute the factor from
all seasons of available data to stabilize estimates for small-sample parks.
"""

from __future__ import annotations

import pandas as pd

_NEUTRAL_FACTOR: float = 1.0
_MIN_GAMES: int = 30  # minimum games to trust the estimate; else use 1.0


def compute_park_factors(gamelogs_all: pd.DataFrame) -> dict[str, float]:
    """Build a park_id → run-factor mapping from all available game logs.

    Parameters
    ----------
    gamelogs_all:
        Concatenation of all Retrosheet game logs (multiple seasons).
        Required columns: ``park_id``, ``home_score``, ``visiting_score``.

    Returns
    -------
    dict
        Mapping from Retrosheet park_id string to float run factor.
        Missing parks default to 1.0 at lookup time.
    """
    gl = gamelogs_all.copy()
    gl["total_runs"] = (
        pd.to_numeric(gl["home_score"], errors="coerce").fillna(0)
        + pd.to_numeric(gl["visiting_score"], errors="coerce").fillna(0)
    )

    park_stats = (
        gl.groupby("park_id")
        .agg(total_runs=("total_runs", "sum"), games=("total_runs", "count"))
        .reset_index()
    )
    park_stats["runs_per_game"] = park_stats["total_runs"] / park_stats["games"]

    league_avg = gl["total_runs"].sum() / max(len(gl), 1)

    factors: dict[str, float] = {}
    for _, row in park_stats.iterrows():
        if row["games"] >= _MIN_GAMES and league_avg > 0:
            factors[str(row["park_id"])] = float(row["runs_per_game"] / league_avg)
        else:
            factors[str(row["park_id"])] = _NEUTRAL_FACTOR

    return factors
