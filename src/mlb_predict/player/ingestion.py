"""Player-level stat ingestion from FanGraphs and expanded Statcast.

Fetches per-player season stats from two sources:
- FanGraphs (via pybaseball): wRC+, OPS, K%, BB%, ISO, BABIP for batters;
  FIP, xFIP, swinging strike % for pitchers. Coverage: 2002+.
- Statcast (via pybaseball): xwOBA, xBA, xSLG, barrel%, hard hit%,
  sprint speed for batters; xwOBA allowed, whiff rate for pitchers.
  Coverage: 2015+.

Results are cached under ``data/processed/player/`` as Parquet files.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_FANGRAPHS_FIRST_SEASON: int = 2002
_STATCAST_FIRST_SEASON: int = 2015


# ---------------------------------------------------------------------------
# FanGraphs player-level
# ---------------------------------------------------------------------------

_FG_BAT_RENAME: dict[str, str] = {
    "IDfg": "fg_id",
    "Name": "name",
    "Team": "team",
    "wRC+": "wrc_plus",
    "OPS": "ops",
    "K%": "k_pct",
    "BB%": "bb_pct",
    "ISO": "iso",
    "BABIP": "babip",
    "wOBA": "woba",
    "HR": "hr",
    "PA": "pa",
}

_FG_PIT_RENAME: dict[str, str] = {
    "IDfg": "fg_id",
    "Name": "name",
    "Team": "team",
    "FIP": "fip",
    "xFIP": "xfip",
    "K%": "k_pct",
    "BB%": "bb_pct",
    "WHIP": "whip",
    "ERA": "era",
    "IP": "ip",
    "GS": "gs",
    "HR": "hr",
    "SO": "so",
    "BB": "bb",
}


def fetch_fg_batters(season: int) -> pd.DataFrame:
    """Fetch FanGraphs per-batter stats for one season.

    Returns DataFrame with columns: fg_id, name, team, wrc_plus, ops, k_pct,
    bb_pct, iso, babip, woba, hr, pa.
    """
    if season < _FANGRAPHS_FIRST_SEASON:
        return pd.DataFrame()
    try:
        from pybaseball import batting_stats

        df = batting_stats(season, season, qual=1)
        available = {k: v for k, v in _FG_BAT_RENAME.items() if k in df.columns}
        out = df[list(available.keys())].rename(columns=available).copy()
        out["season"] = season
        for pct_col in ("k_pct", "bb_pct"):
            if pct_col in out.columns:
                vals = out[pct_col]
                if vals.dtype == object:
                    out[pct_col] = vals.str.rstrip(" %").astype(float) / 100.0
                elif vals.max() > 1.0:
                    out[pct_col] = vals / 100.0
        return out.reset_index(drop=True)
    except Exception as exc:
        logger.warning("FanGraphs batter fetch failed for %d: %s", season, exc)
        return pd.DataFrame()


def fetch_fg_pitchers(season: int) -> pd.DataFrame:
    """Fetch FanGraphs per-pitcher stats for one season.

    Returns DataFrame with columns: fg_id, name, team, fip, xfip, k_pct,
    bb_pct, whip, era, ip, gs, hr, so, bb.
    """
    if season < _FANGRAPHS_FIRST_SEASON:
        return pd.DataFrame()
    try:
        from pybaseball import pitching_stats

        df = pitching_stats(season, season, qual=1)
        available = {k: v for k, v in _FG_PIT_RENAME.items() if k in df.columns}
        out = df[list(available.keys())].rename(columns=available).copy()
        out["season"] = season
        for pct_col in ("k_pct", "bb_pct"):
            if pct_col in out.columns:
                vals = out[pct_col]
                if vals.dtype == object:
                    out[pct_col] = vals.str.rstrip(" %").astype(float) / 100.0
                elif vals.max() > 1.0:
                    out[pct_col] = vals / 100.0
        return out.reset_index(drop=True)
    except Exception as exc:
        logger.warning("FanGraphs pitcher fetch failed for %d: %s", season, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Expanded Statcast
# ---------------------------------------------------------------------------


def fetch_statcast_batters(season: int) -> pd.DataFrame:
    """Fetch expanded Statcast batter stats: xwOBA, xBA, xSLG, barrel%, hard hit%, sprint speed.

    Returns DataFrame with columns: player_id (mlbam), xwoba, xba, xslg,
    barrel_pct, hard_hit_pct, sprint_speed.
    """
    if season < _STATCAST_FIRST_SEASON:
        return pd.DataFrame()
    try:
        from pybaseball import (
            statcast_batter_exitvelo_barrels,
            statcast_batter_expected_stats,
        )

        exp = statcast_batter_expected_stats(season, minPA=1)
        bar = statcast_batter_exitvelo_barrels(season, minBBE=1)

        parts: list[pd.DataFrame] = []
        if not exp.empty:
            exp_cols: dict[str, str] = {"player_id": "player_id"}
            for src, dst in [("est_woba", "xwoba"), ("est_ba", "xba"), ("est_slg", "xslg")]:
                if src in exp.columns:
                    exp_cols[src] = dst
            parts.append(exp[list(exp_cols.keys())].rename(columns=exp_cols))

        if not bar.empty:
            bar_out = bar[["player_id"]].copy()
            if "brl_percent" in bar.columns:
                bar_out["barrel_pct"] = bar["brl_percent"] / 100.0
            if "ev95percent" in bar.columns:
                bar_out["hard_hit_pct"] = bar["ev95percent"] / 100.0
            parts.append(bar_out)

        if not parts:
            return pd.DataFrame()

        merged = parts[0]
        for p in parts[1:]:
            merged = merged.merge(p, on="player_id", how="outer")
        merged["player_id"] = merged["player_id"].astype(int)
        return merged.reset_index(drop=True)
    except Exception as exc:
        logger.warning("Statcast batter fetch failed for %d: %s", season, exc)
        return pd.DataFrame()


def fetch_statcast_pitchers(season: int) -> pd.DataFrame:
    """Fetch expanded Statcast pitcher stats: xwOBA allowed, whiff rate.

    Returns DataFrame with columns: player_id (mlbam), est_woba, whiff_rate.
    """
    if season < _STATCAST_FIRST_SEASON:
        return pd.DataFrame()
    try:
        from pybaseball import statcast_pitcher_expected_stats

        df = statcast_pitcher_expected_stats(season, minPA=1)
        if df.empty:
            return pd.DataFrame()

        out_cols = {"player_id": "player_id"}
        if "est_woba" in df.columns:
            out_cols["est_woba"] = "est_woba"
        if "whiff_percent" in df.columns:
            out_cols["whiff_percent"] = "whiff_rate"

        out = df[list(out_cols.keys())].rename(columns=out_cols).copy()
        if "whiff_rate" in out.columns and out["whiff_rate"].max() > 1.0:
            out["whiff_rate"] = out["whiff_rate"] / 100.0
        out["player_id"] = out["player_id"].astype(int)
        return out.reset_index(drop=True)
    except Exception as exc:
        logger.warning("Statcast pitcher fetch failed for %d: %s", season, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Unified cache load/store
# ---------------------------------------------------------------------------


def get_batter_stats_for_season(season: int, cache_dir: Path) -> pd.DataFrame:
    """Load or fetch+cache combined batter stats (FanGraphs + Statcast) for a season.

    Returns DataFrame with mlbam_id as the player key. FanGraphs players are
    joined via the Chadwick register (fg_id → mlbam_id).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"batter_stats_{season}.parquet"
    if path.exists():
        return pd.read_parquet(path)

    sc = fetch_statcast_batters(season)
    fg = fetch_fg_batters(season)

    if sc.empty and fg.empty:
        return pd.DataFrame()

    if not fg.empty:
        fg = _attach_mlbam_to_fg(fg, "batter")

    merged = _merge_sc_fg_frames(sc, fg)
    if not merged.empty and "team" in merged.columns:
        merged = merged.rename(columns={"team": "team_abbrev"})

    merged = merged.drop_duplicates(subset=["player_id"], keep="first").reset_index(drop=True)
    merged.to_parquet(path, index=False)
    logger.info("Saved batter stats: %d players for %d → %s", len(merged), season, path)
    return merged


def get_pitcher_stats_for_season(season: int, cache_dir: Path) -> pd.DataFrame:
    """Load or fetch+cache combined pitcher stats (FanGraphs + Statcast) for a season."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"pitcher_stats_{season}.parquet"
    if path.exists():
        return pd.read_parquet(path)

    sc = fetch_statcast_pitchers(season)
    fg = fetch_fg_pitchers(season)

    if sc.empty and fg.empty:
        return pd.DataFrame()

    if not fg.empty:
        fg = _attach_mlbam_to_fg(fg, "pitcher")

    merged = _merge_sc_fg_frames(sc, fg)
    if not merged.empty and "team" in merged.columns:
        merged = merged.rename(columns={"team": "team_abbrev"})

    merged = merged.drop_duplicates(subset=["player_id"], keep="first").reset_index(drop=True)
    merged.to_parquet(path, index=False)
    logger.info("Saved pitcher stats: %d players for %d → %s", len(merged), season, path)
    return merged


def _merge_sc_fg_frames(sc: pd.DataFrame, fg: pd.DataFrame) -> pd.DataFrame:
    """Outer-join Statcast and FanGraphs frames on MLBAM player id."""
    if sc.empty and fg.empty:
        return pd.DataFrame()

    if not sc.empty and not fg.empty:
        fg_merge = fg.drop(columns=["season", "fg_id"], errors="ignore")
        merged = sc.merge(
            fg_merge,
            left_on="player_id",
            right_on="mlbam_id",
            how="outer",
        )
        merged["player_id"] = merged["player_id"].fillna(merged["mlbam_id"]).astype(int)
        return merged.drop(columns=["mlbam_id"], errors="ignore")

    if not sc.empty:
        return sc.copy()

    out = fg.rename(columns={"mlbam_id": "player_id"})
    return out.drop(columns=["fg_id"], errors="ignore")


def _attach_mlbam_to_fg(fg_df: pd.DataFrame, player_type: str) -> pd.DataFrame:
    """Map FanGraphs fg_id → mlbam_id via Chadwick register.

    Falls back to name-based matching if fg_id mapping is unavailable.
    """
    try:
        from pybaseball import chadwick_register

        reg = chadwick_register()
        if "key_fangraphs" in reg.columns and "key_mlbam" in reg.columns:
            id_map = (
                reg[["key_fangraphs", "key_mlbam"]]
                .dropna()
                .drop_duplicates(subset=["key_fangraphs"], keep="first")
            )
            id_map["key_fangraphs"] = id_map["key_fangraphs"].astype(int)
            id_map["key_mlbam"] = id_map["key_mlbam"].astype(int)
            fg_out = fg_df.merge(id_map, left_on="fg_id", right_on="key_fangraphs", how="left")
            fg_out = fg_out.rename(columns={"key_mlbam": "mlbam_id"})
            fg_out = fg_out.drop(columns=["key_fangraphs"], errors="ignore")
            fg_out = fg_out.dropna(subset=["mlbam_id"])
            fg_out["mlbam_id"] = fg_out["mlbam_id"].astype(int)
            return fg_out
    except Exception as exc:
        logger.warning("FanGraphs → MLBAM ID mapping failed: %s", exc)

    fg_df["mlbam_id"] = pd.NA
    return fg_df
