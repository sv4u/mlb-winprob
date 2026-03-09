"""Vegas opening (and optional closing) lines as features.

Expects parquet files under vegas_dir: odds_YYYY.parquet with columns
game_date, home_team, away_team, vegas_implied_home_win [, vegas_line_movement].
Team codes should match Retrosheet (e.g. NYA, BOS). Use scripts/ingest_vegas.py
to convert a CSV of money lines into this format.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def money_line_to_implied_prob(ml_american: float) -> float:
    """Convert American money line to implied probability (0-1).

    Positive odds (e.g. 150): P = 100 / (100 + odds)
    Negative odds (e.g. -170): P = |odds| / (|odds| + 100)
    """
    if pd.isna(ml_american):
        return 0.5
    o = float(ml_american)
    if o > 0:
        return 100.0 / (100.0 + o)
    return abs(o) / (abs(o) + 100.0)


def load_vegas_season(vegas_dir: Path, season: int) -> pd.DataFrame | None:
    """Load Vegas odds for one season from vegas_dir/odds_YYYY.parquet.

    Returns DataFrame with game_date, home_team, away_team, vegas_implied_home_win
    and optionally vegas_line_movement. None if file missing.
    """
    path = Path(vegas_dir) / f"odds_{season}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    return df


def build_vegas_features(
    gamelogs: pd.DataFrame,
    vegas_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge Vegas implied home win prob (and line movement) onto gamelogs.

    gamelogs must have date, home_team, visiting_team. vegas_df must have
    game_date, home_team, away_team, vegas_implied_home_win. Merge on
    (date, home_team, away_team). Missing matches get 0.5 (no edge).
    """
    out = pd.DataFrame(index=gamelogs.index)
    out["vegas_implied_home_win"] = 0.5
    out["vegas_line_movement"] = 0.0

    if vegas_df is None or vegas_df.empty:
        return out

    gl_dates = pd.to_datetime(gamelogs["date"]).dt.date
    gl_key = pd.DataFrame(
        {
            "game_date": gl_dates.values,
            "home_team": gamelogs["home_team"].values,
            "away_team": gamelogs["visiting_team"].values,
        },
        index=gamelogs.index,
    )
    vegas_df = vegas_df.astype({"home_team": str, "away_team": str})
    gl_key = gl_key.assign(home_team=gl_key["home_team"].astype(str))
    gl_key = gl_key.assign(away_team=gl_key["away_team"].astype(str))
    merged = gl_key.merge(
        vegas_df,
        on=["game_date", "home_team", "away_team"],
        how="left",
        suffixes=("", "_v"),
    )
    if "vegas_implied_home_win" in merged.columns:
        out["vegas_implied_home_win"] = merged["vegas_implied_home_win"].fillna(0.5).values
    if "vegas_line_movement" in merged.columns:
        out["vegas_line_movement"] = merged["vegas_line_movement"].fillna(0.0).values
    return out
