"""FanGraphs team advanced stats for the sabermetrics dashboard API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from mlb_predict.app.data_cache import TEAM_NAMES
from mlb_predict.statcast.fangraphs import fg_to_retro_code

_BAT_METRICS: tuple[tuple[str, bool], ...] = (
    ("bat_woba", True),
    ("bat_xwoba", True),
    ("bat_iso", True),
    ("bat_babip", True),
    ("bat_barrel_pct", True),
    ("bat_hard_pct", True),
)

_PIT_METRICS: tuple[tuple[str, bool], ...] = (
    ("pit_fip", False),
    ("pit_xfip", False),
    ("pit_k_pct", True),
    ("pit_bb_pct", False),
    ("pit_hr_fb", False),
    ("pit_whip", False),
)

_BAT_API_KEYS = {
    "bat_woba": "woba",
    "bat_xwoba": "xwoba",
    "bat_iso": "iso",
    "bat_babip": "babip",
    "bat_barrel_pct": "barrel_pct",
    "bat_hard_pct": "hard_pct",
}

_PIT_API_KEYS = {
    "pit_fip": "fip",
    "pit_xfip": "xfip",
    "pit_k_pct": "k_pct",
    "pit_bb_pct": "bb_pct",
    "pit_hr_fb": "hr_fb",
    "pit_whip": "whip",
}

_PCT_DISPLAY_COLS = frozenset(
    {"bat_barrel_pct", "bat_hard_pct", "pit_k_pct", "pit_bb_pct", "pit_hr_fb"}
)


def default_fangraphs_dir() -> Path:
    """Return the processed FanGraphs directory (env override supported)."""
    env = os.environ.get("FANGRAPHS_DIR", "").strip()
    if env:
        return Path(env)
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    return repo_root / "data" / "processed" / "fangraphs"


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
    if col in ("bat_woba", "bat_xwoba", "bat_iso", "bat_babip"):
        return round(val, 3)
    return round(val, 2)


def load_team_advanced_stats(
    season: int,
    fg_dir: Path | None = None,
) -> dict[str, Any]:
    """Load FanGraphs advanced team stats for *season* from on-disk Parquet.

    Returns a payload suitable for ``GET /api/team-advanced-stats``.
    """
    directory = fg_dir or default_fangraphs_dir()
    path = directory / f"fangraphs_{season}.parquet"
    if not path.exists():
        return {
            "season": season,
            "source": "fangraphs_parquet",
            "teams": [],
            "message": (
                f"No FanGraphs data for {season}. Run: "
                f"python scripts/ingest_fangraphs.py --seasons {season}"
            ),
        }

    df = pd.read_parquet(path)
    if df.empty or "team_fg" not in df.columns:
        return {
            "season": season,
            "source": "fangraphs_parquet",
            "teams": [],
            "message": f"FanGraphs file for {season} is empty.",
        }

    work = df.copy()
    work["retro_code"] = work["team_fg"].astype(str).map(fg_to_retro_code)
    work["team_name"] = work["retro_code"].map(lambda r: TEAM_NAMES.get(str(r), str(r)))

    for col, higher in _BAT_METRICS + _PIT_METRICS:
        if col in work.columns:
            work[f"{col}_pctile"] = _percentile_series(work[col], higher_is_better=higher)

    teams: list[dict[str, Any]] = []
    for _, row in work.iterrows():
        batting: dict[str, Any] = {}
        for src, api_key in _BAT_API_KEYS.items():
            if src not in work.columns:
                continue
            raw = row.get(src)
            if pd.isna(raw):
                continue
            batting[api_key] = _display_rate(float(raw), src)
            pct_col = f"{src}_pctile"
            if pct_col in work.columns and pd.notna(row.get(pct_col)):
                batting[f"{api_key}_pctile"] = int(row[pct_col])

        pitching: dict[str, Any] = {}
        for src, api_key in _PIT_API_KEYS.items():
            if src not in work.columns:
                continue
            raw = row.get(src)
            if pd.isna(raw):
                continue
            pitching[api_key] = _display_rate(float(raw), src)
            pct_col = f"{src}_pctile"
            if pct_col in work.columns and pd.notna(row.get(pct_col)):
                pitching[f"{api_key}_pctile"] = int(row[pct_col])

        teams.append(
            {
                "retro_code": str(row.get("retro_code", "")),
                "team_fg": str(row.get("team_fg", "")),
                "team_name": str(row.get("team_name", "")),
                "batting": batting,
                "pitching": pitching,
            }
        )

    teams.sort(key=lambda t: t.get("team_name", ""))
    return {
        "season": season,
        "source": "fangraphs_parquet",
        "teams": teams,
    }
