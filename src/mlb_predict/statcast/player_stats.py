"""Statcast individual player data for lineup-weighted and pitcher features.

Uses pybaseball to fetch batter xwOBA, barrel%, hard hit% and pitcher expected wOBA
by season. Data is cached under cache_dir. Retrosheet lineup/starting pitcher IDs
are mapped to MLBAM via Chadwick register for lookups.

Statcast data is available from 2015 onward; earlier seasons use fallbacks (team
FanGraphs or league average) in the feature builder.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_STATCAST_FIRST_SEASON: int = 2015
_LEAGUE_AVG_XWOBA: float = 0.320
_LEAGUE_AVG_BARREL_PCT: float = 0.08  # decimal to match FanGraphs bat_barrel_pct
_LEAGUE_AVG_PIT_EST_WOBA: float = 0.320

_HOME_LINEUP_ID_COLS: list[str] = [f"home_{i}_id" for i in range(1, 10)]
_AWAY_LINEUP_ID_COLS: list[str] = [f"visiting_{i}_id" for i in range(1, 10)]


def _load_chadwick_register() -> pd.DataFrame:
    """Load Chadwick register (retro_id -> mlbam) once. Cached by pybaseball."""
    try:
        from pybaseball import chadwick_register

        reg = chadwick_register()
        return reg[["key_retro", "key_mlbam"]].dropna().astype({"key_mlbam": int})
    except Exception as e:
        logger.warning("Chadwick register load failed: %s", e)
        return pd.DataFrame(columns=["key_retro", "key_mlbam"])


def _retro_to_mlbam_map(register: pd.DataFrame) -> dict[str, int]:
    """Build retro_id (str) -> mlbam (int) lookup. Normalize retro to lowercase."""
    out: dict[str, int] = {}
    for _, row in register.iterrows():
        k = str(row["key_retro"]).strip().lower()
        if k and pd.notna(row["key_mlbam"]):
            out[k] = int(row["key_mlbam"])
    return out


def fetch_batter_statcast(season: int) -> pd.DataFrame:
    """Fetch Statcast batter expected stats and exitvelo/barrels for one season.

    Returns DataFrame with columns: player_id (mlbam), xwoba, barrel_pct, hard_hit_pct.
    """
    try:
        from pybaseball import (
            statcast_batter_exitvelo_barrels,
            statcast_batter_expected_stats,
        )
    except ImportError:
        return pd.DataFrame(columns=["player_id", "xwoba", "barrel_pct", "hard_hit_pct"])

    exp = statcast_batter_expected_stats(season, minPA=1)
    bar = statcast_batter_exitvelo_barrels(season, minBBE=1)
    if exp.empty or bar.empty:
        return pd.DataFrame(columns=["player_id", "xwoba", "barrel_pct", "hard_hit_pct"])

    exp = exp.rename(columns={"est_woba": "xwoba"})[["player_id", "xwoba"]]
    bar = bar.copy()
    bar["barrel_pct"] = bar["brl_percent"] / 100.0
    bar["hard_hit_pct"] = bar["ev95percent"] / 100.0
    bar = bar[["player_id", "barrel_pct", "hard_hit_pct"]]
    merged = exp.merge(bar, on="player_id", how="outer")
    merged["player_id"] = merged["player_id"].astype(int)
    return merged


def fetch_pitcher_statcast(season: int) -> pd.DataFrame:
    """Fetch Statcast pitcher expected stats (xwOBA allowed) for one season."""
    try:
        from pybaseball import statcast_pitcher_expected_stats
    except ImportError:
        return pd.DataFrame(columns=["player_id", "est_woba"])

    df = statcast_pitcher_expected_stats(season, minPA=1)
    if df.empty:
        return pd.DataFrame(columns=["player_id", "est_woba"])
    df = df[["player_id", "est_woba"]].copy()
    df["player_id"] = df["player_id"].astype(int)
    return df


def get_batter_statcast_for_season(season: int, cache_dir: Path) -> pd.DataFrame:
    """Load or fetch and cache batter Statcast for season. Returns empty if season < 2015."""
    if season < _STATCAST_FIRST_SEASON:
        return pd.DataFrame(columns=["player_id", "xwoba", "barrel_pct", "hard_hit_pct"])
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"batter_{season}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = fetch_batter_statcast(season)
    if not df.empty:
        df.to_parquet(path, index=False)
    return df


def get_pitcher_statcast_for_season(season: int, cache_dir: Path) -> pd.DataFrame:
    """Load or fetch and cache pitcher Statcast for season."""
    if season < _STATCAST_FIRST_SEASON:
        return pd.DataFrame(columns=["player_id", "est_woba"])
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"pitcher_{season}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = fetch_pitcher_statcast(season)
    if not df.empty:
        df.to_parquet(path, index=False)
    return df


def build_lineup_statcast_features(
    gamelogs: pd.DataFrame,
    prior_season: int,
    batter_stats: pd.DataFrame,
    retro_to_mlbam: dict[str, int],
) -> pd.DataFrame:
    """Compute lineup-weighted xwOBA and barrel% for home/away from prior-season Statcast.

    prior_season is the year of Statcast data to use (e.g. season-1 for games in season).
    Returns DataFrame with columns: home_lineup_xwoba, away_lineup_xwoba,
    home_lineup_barrel_pct, away_lineup_barrel_pct. Index aligned to gamelogs.
    """
    if batter_stats.empty:
        n = len(gamelogs)
        return pd.DataFrame(
            {
                "home_lineup_xwoba": [_LEAGUE_AVG_XWOBA] * n,
                "away_lineup_xwoba": [_LEAGUE_AVG_XWOBA] * n,
                "home_lineup_barrel_pct": [_LEAGUE_AVG_BARREL_PCT] * n,
                "away_lineup_barrel_pct": [_LEAGUE_AVG_BARREL_PCT] * n,
            },
            index=gamelogs.index,
        )

    bat = batter_stats.set_index("player_id")
    home_xwoba = []
    home_barrel = []
    away_xwoba = []
    away_barrel = []

    for idx, row in gamelogs.iterrows():

        def lineup_avg(id_cols: list[str], stat_col: str, default: float) -> float:
            vals = []
            for c in id_cols:
                rid = row.get(c)
                if pd.isna(rid) or rid == "":
                    continue
                rid = str(rid).strip().lower()
                mlbam = retro_to_mlbam.get(rid)
                if mlbam is None:
                    continue
                v = bat.loc[bat.index == mlbam, stat_col]
                if not v.empty and pd.notna(v.iloc[0]):
                    vals.append(float(v.iloc[0]))
            return float(sum(vals)) / len(vals) if vals else default

        home_xwoba.append(lineup_avg(_HOME_LINEUP_ID_COLS, "xwoba", _LEAGUE_AVG_XWOBA))
        home_barrel.append(lineup_avg(_HOME_LINEUP_ID_COLS, "barrel_pct", _LEAGUE_AVG_BARREL_PCT))
        away_xwoba.append(lineup_avg(_AWAY_LINEUP_ID_COLS, "xwoba", _LEAGUE_AVG_XWOBA))
        away_barrel.append(lineup_avg(_AWAY_LINEUP_ID_COLS, "barrel_pct", _LEAGUE_AVG_BARREL_PCT))

    return pd.DataFrame(
        {
            "home_lineup_xwoba": home_xwoba,
            "away_lineup_xwoba": away_xwoba,
            "home_lineup_barrel_pct": home_barrel,
            "away_lineup_barrel_pct": away_barrel,
        },
        index=gamelogs.index,
    )


def build_pitcher_statcast_features(
    gamelogs: pd.DataFrame,
    prior_season: int,
    pitcher_stats: pd.DataFrame,
    retro_to_mlbam: dict[str, int],
) -> pd.DataFrame:
    """Look up prior-season Statcast est_woba (xwOBA allowed) for each game's starters.

    Returns DataFrame with home_sp_est_woba, away_sp_est_woba. Index aligned to gamelogs.
    """
    if pitcher_stats.empty:
        n = len(gamelogs)
        return pd.DataFrame(
            {
                "home_sp_est_woba": [_LEAGUE_AVG_PIT_EST_WOBA] * n,
                "away_sp_est_woba": [_LEAGUE_AVG_PIT_EST_WOBA] * n,
            },
            index=gamelogs.index,
        )

    pit = pitcher_stats.set_index("player_id")
    home_est = []
    away_est = []

    for idx, row in gamelogs.iterrows():

        def lookup(col: str) -> float:
            rid = row.get(col)
            if pd.isna(rid) or rid == "":
                return _LEAGUE_AVG_PIT_EST_WOBA
            rid = str(rid).strip().lower()
            mlbam = retro_to_mlbam.get(rid)
            if mlbam is None:
                return _LEAGUE_AVG_PIT_EST_WOBA
            v = pit.loc[pit.index == mlbam, "est_woba"]
            return float(v.iloc[0]) if not v.empty else _LEAGUE_AVG_PIT_EST_WOBA

        home_est.append(lookup("home_starting_pitcher_id"))
        away_est.append(lookup("visiting_starting_pitcher_id"))

    return pd.DataFrame(
        {"home_sp_est_woba": home_est, "away_sp_est_woba": away_est},
        index=gamelogs.index,
    )
