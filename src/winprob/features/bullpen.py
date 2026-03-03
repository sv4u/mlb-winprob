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
_NEUTRAL_ERA_PROXY = 4.5


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

    home_usage_15 = pd.Series(_NEUTRAL_USAGE, index=gl.index)
    home_usage_30 = pd.Series(_NEUTRAL_USAGE, index=gl.index)
    away_usage_15 = pd.Series(_NEUTRAL_USAGE, index=gl.index)
    away_usage_30 = pd.Series(_NEUTRAL_USAGE, index=gl.index)
    home_era_15 = pd.Series(_NEUTRAL_ERA_PROXY, index=gl.index)
    home_era_30 = pd.Series(_NEUTRAL_ERA_PROXY, index=gl.index)
    away_era_15 = pd.Series(_NEUTRAL_ERA_PROXY, index=gl.index)
    away_era_30 = pd.Series(_NEUTRAL_ERA_PROXY, index=gl.index)

    for _team, grp in gl.groupby("home_team", sort=False):
        r = _roll_team(grp, "home_relievers", _WINDOWS)
        e = _roll_team(grp, "home_er", _WINDOWS)
        home_usage_15.loc[grp.index] = r["w15"]
        home_usage_30.loc[grp.index] = r["w30"]
        home_era_15.loc[grp.index] = e["w15"]
        home_era_30.loc[grp.index] = e["w30"]

    for _team, grp in gl.groupby("visiting_team", sort=False):
        r = _roll_team(grp, "away_relievers", _WINDOWS)
        e = _roll_team(grp, "visiting_er", _WINDOWS)
        away_usage_15.loc[grp.index] = r["w15"]
        away_usage_30.loc[grp.index] = r["w30"]
        away_era_15.loc[grp.index] = e["w15"]
        away_era_30.loc[grp.index] = e["w30"]

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