"""Bullpen usage and fatigue features from Retrosheet gamelogs.

Uses pitchers_used (and team ER) to compute:
- Bullpen usage: rolling mean of (pitchers_used - 1) over last 15/30 games,
  as a proxy for recent bullpen workload/fatigue.
- Bullpen ERA proxy: rolling mean of team ER allowed over last 15/30 games
  (blend of starter + bullpen; true relief ERA would require play-by-play).
"""

from __future__ import annotations

import pandas as pd

_WINDOWS = [15, 30]
_NEUTRAL_USAGE = 2.0  # typical relievers per game
_NEUTRAL_ERA_PROXY = 2.5  # ~4.5 ERA × 5 IP / 9 ≈ 2.5 ER per game


def build_bullpen_features(gamelogs: pd.DataFrame) -> pd.DataFrame:
    """Compute bullpen usage and ERA proxy per team per game (pre-game, shifted).

    Returns
    -------
    DataFrame
        Aligned on gamelogs.index. Columns: home_bullpen_usage_15/30,
        away_bullpen_usage_15/30, home_bullpen_era_proxy_15/30, away_bullpen_era_proxy_15/30.
    """
    orig_index = gamelogs.index
    gl = gamelogs.copy()
    gl["date"] = pd.to_datetime(gl["date"])
    gl["game_num"] = pd.to_numeric(gl["game_num"], errors="coerce").fillna(0).astype(int)
    gl = gl.sort_values(["date", "game_num"])

    home_pu = pd.to_numeric(gl["home_pitchers_used"], errors="coerce").fillna(2).clip(lower=1)
    away_pu = pd.to_numeric(gl["visiting_pitchers_used"], errors="coerce").fillna(2).clip(lower=1)
    gl["home_relievers"] = (home_pu - 1).clip(lower=0)
    gl["away_relievers"] = (away_pu - 1).clip(lower=0)
    gl["home_er"] = pd.to_numeric(gl["home_er"], errors="coerce").fillna(_NEUTRAL_ERA_PROXY)
    gl["visiting_er"] = pd.to_numeric(gl["visiting_er"], errors="coerce").fillna(_NEUTRAL_ERA_PROXY)

    def _roll_team(grp: pd.DataFrame, val_col: str, windows: list[int]) -> pd.DataFrame:
        out = pd.DataFrame(index=grp.index)
        for w in windows:
            out[f"w{w}"] = grp[val_col].rolling(w, min_periods=1).mean().shift(1)
        return out

    # Build a unified team-level view so bullpen fatigue accumulates across
    # home and away games (a road-trip workload carries into home games).
    home_view = pd.DataFrame(
        {
            "gl_idx": gl.index,
            "team": gl["home_team"].values,
            "relievers": gl["home_relievers"].values,
            "er": gl["home_er"].values,
            "side": "home",
        }
    )
    away_view = pd.DataFrame(
        {
            "gl_idx": gl.index,
            "team": gl["visiting_team"].values,
            "relievers": gl["away_relievers"].values,
            "er": gl["visiting_er"].values,
            "side": "away",
        }
    )
    combined = pd.concat([home_view, away_view], ignore_index=True)
    combined = combined.sort_values("gl_idx")

    home_usage_15 = pd.Series(_NEUTRAL_USAGE, index=gl.index)
    home_usage_30 = pd.Series(_NEUTRAL_USAGE, index=gl.index)
    away_usage_15 = pd.Series(_NEUTRAL_USAGE, index=gl.index)
    away_usage_30 = pd.Series(_NEUTRAL_USAGE, index=gl.index)
    home_era_15 = pd.Series(_NEUTRAL_ERA_PROXY, index=gl.index)
    home_era_30 = pd.Series(_NEUTRAL_ERA_PROXY, index=gl.index)
    away_era_15 = pd.Series(_NEUTRAL_ERA_PROXY, index=gl.index)
    away_era_30 = pd.Series(_NEUTRAL_ERA_PROXY, index=gl.index)

    for _team, grp in combined.groupby("team", sort=False):
        r = _roll_team(grp, "relievers", _WINDOWS)
        e = _roll_team(grp, "er", _WINDOWS)
        home_rows = grp[grp["side"] == "home"]
        away_rows = grp[grp["side"] == "away"]
        if not home_rows.empty:
            home_usage_15.loc[home_rows["gl_idx"].values] = r.loc[home_rows.index, "w15"].values
            home_usage_30.loc[home_rows["gl_idx"].values] = r.loc[home_rows.index, "w30"].values
            home_era_15.loc[home_rows["gl_idx"].values] = e.loc[home_rows.index, "w15"].values
            home_era_30.loc[home_rows["gl_idx"].values] = e.loc[home_rows.index, "w30"].values
        if not away_rows.empty:
            away_usage_15.loc[away_rows["gl_idx"].values] = r.loc[away_rows.index, "w15"].values
            away_usage_30.loc[away_rows["gl_idx"].values] = r.loc[away_rows.index, "w30"].values
            away_era_15.loc[away_rows["gl_idx"].values] = e.loc[away_rows.index, "w15"].values
            away_era_30.loc[away_rows["gl_idx"].values] = e.loc[away_rows.index, "w30"].values

    out = pd.DataFrame(
        {
            "home_bullpen_usage_15": home_usage_15,
            "home_bullpen_usage_30": home_usage_30,
            "away_bullpen_usage_15": away_usage_15,
            "away_bullpen_usage_30": away_usage_30,
            "home_bullpen_era_proxy_15": home_era_15,
            "home_bullpen_era_proxy_30": home_era_30,
            "away_bullpen_era_proxy_15": away_era_15,
            "away_bullpen_era_proxy_30": away_era_30,
        },
        index=gl.index,
    )
    out = out.reindex(orig_index)
    out = out.fillna(
        {
            "home_bullpen_usage_15": _NEUTRAL_USAGE,
            "home_bullpen_usage_30": _NEUTRAL_USAGE,
            "away_bullpen_usage_15": _NEUTRAL_USAGE,
            "away_bullpen_usage_30": _NEUTRAL_USAGE,
            "home_bullpen_era_proxy_15": _NEUTRAL_ERA_PROXY,
            "home_bullpen_era_proxy_30": _NEUTRAL_ERA_PROXY,
            "away_bullpen_era_proxy_15": _NEUTRAL_ERA_PROXY,
            "away_bullpen_era_proxy_30": _NEUTRAL_ERA_PROXY,
        }
    )
    return out
