"""Season-to-date starting pitcher ERA with Bayesian shrinkage.

Because Retrosheet gamelogs record team-level earned runs (not per-pitcher),
we use the team's total earned runs in each game as a proxy for the starter's
contribution.  A fixed 5-inning estimate is applied to convert ER to ERA
(ERA ≈ ER * 1.8).  Bayesian shrinkage blends this toward the league mean so
first-time starters are not treated as blank slates.
"""

from __future__ import annotations

import inspect
import pandas as pd
import numpy as np

# League-average ERA used as the Bayesian prior.
_LEAGUE_AVG_ERA: float = 4.50
# How many prior "ghost starts" to apply for shrinkage.
_SHRINKAGE_STARTS: int = 10
# Assumed average innings per start (used to convert ER → ERA).
_AVG_IP_PER_START: float = 5.0


def build_pitcher_stats(gamelogs: pd.DataFrame) -> pd.DataFrame:
    """Compute pre-game starter ERA estimates for every game.

    Parameters
    ----------
    gamelogs:
        Parsed Retrosheet game log.  Required columns:
        ``date``, ``game_num``, ``home_starting_pitcher_id``,
        ``visiting_starting_pitcher_id``, ``home_er``, ``visiting_er``.

    Returns
    -------
    DataFrame
        Aligned on ``gamelogs.index``.  Columns:
        ``home_sp_era``, ``home_sp_n_starts``,
        ``away_sp_era``, ``away_sp_n_starts``.
    """
    gl = gamelogs.reset_index(drop=True)
    game_num = pd.to_numeric(gl["game_num"], errors="coerce").fillna(0).astype(int)

    def _era_with_shrinkage(cum_er: pd.Series, n_starts: pd.Series) -> pd.Series:
        """Blend observed (cum ER → ERA) with league-average prior."""
        # ERA from observed starts (5-inning assumption)
        obs_era = cum_er * (9.0 / _AVG_IP_PER_START) / n_starts.clip(lower=1)
        # Shrinkage weight: more starts → trust observed more
        w_obs = n_starts / (n_starts + _SHRINKAGE_STARTS)
        return w_obs * obs_era + (1 - w_obs) * _LEAGUE_AVG_ERA

    def _make_view(pitcher_col: str, er_col: str, side: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "gl_idx": gl.index,
                "pitcher_id": gl[pitcher_col].astype(str).fillna("unknown"),
                "er": pd.to_numeric(gl[er_col], errors="coerce").fillna(0),
                "date": gl["date"],
                "game_num": game_num,
                "side": side,
            }
        )

    combined = pd.concat(
        [
            _make_view("home_starting_pitcher_id", "home_er", "home"),
            _make_view("visiting_starting_pitcher_id", "visiting_er", "away"),
        ],
        ignore_index=True,
    ).sort_values(["pitcher_id", "date", "game_num"])

    def _pitcher_group(grp: pd.DataFrame) -> pd.DataFrame:
        """Cumulative ER and start count shifted back one game."""
        er_vals = grp["er"].values.astype(float)
        cum_er = pd.Series(er_vals).cumsum().shift(1).fillna(0)
        n_starts = pd.Series(np.arange(len(grp), dtype=float))  # 0, 1, 2, …

        sp_era = _era_with_shrinkage(cum_er, n_starts)

        return pd.DataFrame(
            {
                "gl_idx": grp["gl_idx"].values,
                "side": grp["side"].values,
                "sp_era": sp_era.values,
                "sp_n_starts": n_starts.values,
            },
            index=grp.index,
        )

    _gb = combined.groupby("pitcher_id", group_keys=False)
    _kw = {"include_groups": True} if "include_groups" in inspect.signature(_gb.apply).parameters else {}
    stats = _gb.apply(_pitcher_group, **_kw)

    home_s = (
        stats[stats["side"] == "home"]
        .set_index("gl_idx")[["sp_era", "sp_n_starts"]]
        .add_prefix("home_")
    )
    away_s = (
        stats[stats["side"] == "away"]
        .set_index("gl_idx")[["sp_era", "sp_n_starts"]]
        .add_prefix("away_")
    )
    return home_s.join(away_s)
