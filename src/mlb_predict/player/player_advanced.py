"""Player advanced stats (FanGraphs + Statcast) for the sabermetrics dashboard API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

_BAT_METRICS: tuple[tuple[str, bool], ...] = (
    ("woba", True),
    ("wrc_plus", True),
    ("ops", True),
    ("iso", True),
    ("babip", True),
    ("k_pct", False),
    ("bb_pct", True),
    ("xwoba", True),
    ("xba", True),
    ("xslg", True),
    ("barrel_pct", True),
    ("hard_hit_pct", True),
)

_PIT_METRICS: tuple[tuple[str, bool], ...] = (
    ("fip", False),
    ("xfip", False),
    ("era", False),
    ("whip", False),
    ("k_pct", True),
    ("bb_pct", False),
    ("est_woba", False),
    ("whiff_rate", True),
)

_PCT_DISPLAY_COLS = frozenset(
    {"k_pct", "bb_pct", "barrel_pct", "hard_hit_pct", "whiff_rate"},
)


def default_player_dir() -> Path:
    """Return the processed player stats directory (env override supported)."""
    env = os.environ.get("PLAYER_STATS_DIR", "").strip()
    if env:
        return Path(env)
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    return repo_root / "data" / "processed" / "player"


def _percentile_series(series: pd.Series, *, higher_is_better: bool) -> pd.Series:
    """League percentile (0–100) for each value in *series*."""
    valid = series.dropna()
    if valid.empty:
        return pd.Series([None] * len(series), index=series.index, dtype=object)
    ranked = series.rank(method="average", ascending=higher_is_better, pct=True)
    return (ranked * 100).round(0).astype("Int64")


def _display_rate(value: float | None, col: str) -> float | None:
    """Format a stat for JSON output (percentages as 0–100 scale)."""
    if value is None or pd.isna(value):
        return None
    val = float(value)
    if col in _PCT_DISPLAY_COLS:
        if abs(val) <= 1.0:
            return round(val * 100, 1)
        return round(val, 1)
    if col in ("woba", "xwoba", "xba", "xslg", "iso", "babip", "ops", "est_woba"):
        return round(val, 3)
    if col in ("fip", "xfip", "era", "whip"):
        return round(val, 2)
    if col == "wrc_plus":
        return round(val, 0)
    return round(val, 2)


def _parquet_path(season: int, group: str, player_dir: Path) -> Path:
    """Resolve the Parquet path for *group* (hitting or pitching)."""
    if group == "pitching":
        return player_dir / f"pitcher_stats_{season}.parquet"
    return player_dir / f"batter_stats_{season}.parquet"


def load_player_advanced_stats(
    season: int,
    group: str = "hitting",
    *,
    limit: int = 100,
    offset: int = 0,
    search: str | None = None,
    player_dir: Path | None = None,
) -> dict[str, Any]:
    """Load combined FanGraphs + Statcast player stats from on-disk Parquet.

    Returns a payload suitable for ``GET /api/player-advanced-stats``.
    """
    directory = player_dir or default_player_dir()
    path = _parquet_path(season, group, directory)
    if not path.exists():
        return {
            "season": season,
            "group": group,
            "source": "player_parquet",
            "players": [],
            "count": 0,
            "offset": offset,
            "total": 0,
            "message": (
                f"No advanced player data for {season}. Run: "
                f"python scripts/ingest_player_data.py --seasons {season}"
            ),
        }

    df = pd.read_parquet(path)
    if df.empty or "player_id" not in df.columns:
        return {
            "season": season,
            "group": group,
            "source": "player_parquet",
            "players": [],
            "count": 0,
            "offset": offset,
            "total": 0,
            "message": f"Player stats file for {season} is empty.",
        }

    work = df.copy()
    if "team" in work.columns and "team_abbrev" not in work.columns:
        work = work.rename(columns={"team": "team_abbrev"})

    metrics = _BAT_METRICS if group != "pitching" else _PIT_METRICS
    for col, higher in metrics:
        if col in work.columns:
            work[f"{col}_pctile"] = _percentile_series(work[col], higher_is_better=higher)

    if search:
        needle = search.strip().lower()
        if needle and "name" in work.columns:
            work = work[work["name"].astype(str).str.lower().str.contains(needle, na=False)]

    work = work.sort_values(
        by="wrc_plus" if group != "pitching" and "wrc_plus" in work.columns else "player_id",
        ascending=False,
        na_position="last",
    )
    total = len(work)
    page = work.iloc[offset : offset + limit]

    players: list[dict[str, Any]] = []
    for _, row in page.iterrows():
        stats: dict[str, Any] = {}
        for col, _higher in metrics:
            if col not in work.columns:
                continue
            raw = row.get(col)
            if pd.isna(raw):
                continue
            stats[col] = _display_rate(float(raw), col)
            pct_col = f"{col}_pctile"
            if pct_col in work.columns and pd.notna(row.get(pct_col)):
                stats[f"{col}_pctile"] = int(row[pct_col])

        players.append(
            {
                "player_id": int(row["player_id"]),
                "name": str(row["name"])
                if "name" in work.columns and pd.notna(row.get("name"))
                else None,
                "team_abbrev": (
                    str(row["team_abbrev"])
                    if "team_abbrev" in work.columns and pd.notna(row.get("team_abbrev"))
                    else None
                ),
                "stats": stats,
            }
        )

    return {
        "season": season,
        "group": group,
        "source": "player_parquet",
        "offset": offset,
        "count": len(players),
        "total": total,
        "players": players,
    }
