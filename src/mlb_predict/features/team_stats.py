"""Rolling team performance statistics — v3.

Improvements over v2
---------------------
* **EWMA** (exponentially weighted moving average, span=20): weights recent games
  more heavily than a uniform rolling window.
* **Home/away performance splits**: separate rolling stats for each team's
  home-game subsequence and away-game subsequence.  A 20–5 team at home that
  is 10–15 on the road looks identical under plain win% but very different
  under home/away splits.

All stats use shift(1) after chronological sorting to prevent leakage.
Cross-season continuity is supported by accepting gamelogs spanning all seasons.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

_WINDOWS: list[int] = [7, 14, 15, 30, 60]  # 7/14 for in-season hot/cold (Phase 5b)
_EWMA_SPAN: int = 20
_SPLIT_WINDOW: int = 20  # rolling window within the home-only / away-only subsequence

_NEUTRAL_WIN_PCT: float = 0.500
_NEUTRAL_RUN_DIFF: float = 0.0
_NEUTRAL_PYTHAG: float = 0.500


def _pythag(rs: np.ndarray | pd.Series, ra: np.ndarray | pd.Series) -> np.ndarray:
    rs = np.asarray(rs, dtype=float)
    ra = np.asarray(ra, dtype=float)
    denom = rs**2 + ra**2
    # np.where evaluates both branches before selecting, so rs**2 / denom fires
    # a RuntimeWarning for every row where denom == 0.  Suppress it here — the
    # result is always replaced by _NEUTRAL_PYTHAG in those cases anyway.
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, rs**2 / denom, _NEUTRAL_PYTHAG)


def _compute_streak(win_arr: np.ndarray) -> np.ndarray:
    """Pre-game win/loss streak.  Positive = win streak, negative = loss streak."""
    n = len(win_arr)
    out = np.zeros(n, dtype=float)
    for i in range(1, n):
        prev = win_arr[i - 1]
        if np.isnan(prev):
            out[i] = 0.0
        elif prev == 1.0:
            out[i] = max(out[i - 1], 0.0) + 1.0
        else:
            out[i] = min(out[i - 1], 0.0) - 1.0
    return out


def _split_rolling(
    grp: pd.DataFrame,
    mask: np.ndarray,
    window: int = _SPLIT_WINDOW,
) -> tuple[pd.Series, pd.Series]:
    """Compute venue-specific rolling win% and Pythagorean for every row in *grp*.

    Parameters
    ----------
    grp : DataFrame
        All games for one team (both home and away), sorted chronologically.
    mask : ndarray of bool
        Boolean mask selecting the subset of games to compute rolling stats
        on (e.g. home-only or away-only).

    Strategy
    --------
    1. Extract the subsequence matching *mask* for this team.
    2. Compute rolling stats on that subsequence, shifted by 1 (no leakage).
    3. Forward-fill the result back into the full game sequence so that each
       game carries the most recent rolling stat from the selected venue.

    Returns
    -------
    win_pct_series, pythag_series  — aligned on grp.index
    """
    venue_mask = np.asarray(mask, dtype=bool)

    win_pct_full = pd.Series(np.nan, index=grp.index)
    pythag_full = pd.Series(np.nan, index=grp.index)

    venue_idx = grp.index[venue_mask]
    if len(venue_idx) == 0:
        return win_pct_full.fillna(_NEUTRAL_WIN_PCT), pythag_full.fillna(_NEUTRAL_PYTHAG)

    venue_grp = grp.loc[venue_idx]
    roll_win = venue_grp["win"].rolling(window, min_periods=1).sum().shift(1)
    roll_rs = venue_grp["rs"].rolling(window, min_periods=1).sum().shift(1)
    roll_ra = venue_grp["ra"].rolling(window, min_periods=1).sum().shift(1)
    n = venue_grp["win"].rolling(window, min_periods=1).count().shift(1)

    wp = (roll_win / n).fillna(_NEUTRAL_WIN_PCT)
    py = pd.Series(_pythag(roll_rs.fillna(0), roll_ra.fillna(0)), index=venue_idx).where(
        n.notna() & (n > 0), other=_NEUTRAL_PYTHAG
    )

    win_pct_full.loc[venue_idx] = wp.values
    pythag_full.loc[venue_idx] = py.values

    win_pct_full = win_pct_full.ffill().fillna(_NEUTRAL_WIN_PCT)
    pythag_full = pythag_full.ffill().fillna(_NEUTRAL_PYTHAG)
    return win_pct_full, pythag_full


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_team_rolling_stats(
    gamelogs: pd.DataFrame,
    *,
    windows: list[int] = _WINDOWS,
    ewma_span: int = _EWMA_SPAN,
) -> pd.DataFrame:
    """Compute pre-game rolling, EWMA, and home/away-split stats.

    Returns
    -------
    DataFrame
        Indexed on ``gamelogs.index``.  Key column groups:
        - Multi-window: ``{side}_win_pct_{W}``, ``_run_diff_{W}``, ``_pythag_{W}``
        - EWMA:         ``{side}_win_pct_ewm``, ``_run_diff_ewm``, ``_pythag_ewm``
        - Streak:       ``{side}_streak``
        - Rest days:    ``{side}_rest_days``
        - Home splits:  ``home_win_pct_home_only``, ``home_pythag_home_only``
        - Away splits:  ``away_win_pct_away_only``, ``away_pythag_away_only``
    """
    gl = gamelogs.reset_index(drop=True)
    gl["date"] = pd.to_datetime(gl["date"])
    game_num = pd.to_numeric(gl["game_num"], errors="coerce").fillna(0).astype(int)

    def _make_view(team_col: str, rs_col: str, ra_col: str, side: str) -> pd.DataFrame:
        rs = pd.to_numeric(gl[rs_col], errors="coerce").fillna(0)
        ra = pd.to_numeric(gl[ra_col], errors="coerce").fillna(0)
        return pd.DataFrame(
            {
                "gl_idx": gl.index,
                "team": gl[team_col],
                "date": gl["date"],
                "game_num": game_num,
                "rs": rs,
                "ra": ra,
                "win": (rs > ra).astype(float),
                "side": side,
                "is_home": (side == "home"),
            }
        )

    combined = pd.concat(
        [
            _make_view("home_team", "home_score", "visiting_score", "home"),
            _make_view("visiting_team", "visiting_score", "home_score", "away"),
        ],
        ignore_index=True,
    ).sort_values(["team", "date", "game_num"])

    def _rolling_group(grp: pd.DataFrame) -> pd.DataFrame:
        idx = grp.index
        rows: dict[str, list] = {
            "gl_idx": grp["gl_idx"].values.tolist(),
            "side": grp["side"].values.tolist(),
        }

        # Multi-window rolling
        for w in windows:
            roll_win = grp["win"].rolling(w, min_periods=1).sum().shift(1)
            roll_rs = grp["rs"].rolling(w, min_periods=1).sum().shift(1)
            roll_ra = grp["ra"].rolling(w, min_periods=1).sum().shift(1)
            n = grp["win"].rolling(w, min_periods=1).count().shift(1)
            rows[f"win_pct_{w}"] = (roll_win / n).fillna(_NEUTRAL_WIN_PCT).values.tolist()
            rows[f"run_diff_{w}"] = (
                ((roll_rs - roll_ra) / n).fillna(_NEUTRAL_RUN_DIFF).values.tolist()
            )
            py = pd.Series(_pythag(roll_rs.fillna(0), roll_ra.fillna(0)), index=idx)
            rows[f"pythag_{w}"] = py.where(n.notna() & (n > 0), _NEUTRAL_PYTHAG).values.tolist()
            rows[f"n_games_{w}"] = n.fillna(0).values.tolist()

        # EWMA
        ewm_win = grp["win"].ewm(span=ewma_span, adjust=False).mean().shift(1)
        ewm_rs = grp["rs"].ewm(span=ewma_span, adjust=False).mean().shift(1)
        ewm_ra = grp["ra"].ewm(span=ewma_span, adjust=False).mean().shift(1)
        rows["win_pct_ewm"] = ewm_win.fillna(_NEUTRAL_WIN_PCT).values.tolist()
        rows["run_diff_ewm"] = (ewm_rs - ewm_ra).fillna(_NEUTRAL_RUN_DIFF).values.tolist()
        rows["pythag_ewm"] = (
            pd.Series(_pythag(ewm_rs.fillna(0), ewm_ra.fillna(0)), index=idx)
            .fillna(_NEUTRAL_PYTHAG)
            .values.tolist()
        )

        # Streak
        rows["streak"] = _compute_streak(grp["win"].values).tolist()

        # Rest days
        dates = grp["date"].values
        rest = np.empty(len(dates))
        rest[0] = 3.0
        for k in range(1, len(dates)):
            delta = (dates[k] - dates[k - 1]).astype("timedelta64[D]").astype(int)
            rest[k] = min(float(delta), 10.0)
        rows["rest_days"] = rest.tolist()

        # Home/away splits: compute both venue-specific rolling stats per team,
        # then select the one matching each row's venue context so the home side
        # gets home-only splits and the away side gets away-only splits.
        is_home = grp["is_home"].values.astype(bool)
        wp_home_split, py_home_split = _split_rolling(grp, is_home)
        wp_away_split, py_away_split = _split_rolling(grp, ~is_home)
        rows["win_pct_split"] = np.where(
            is_home, wp_home_split.values, wp_away_split.values
        ).tolist()
        rows["pythag_split"] = np.where(
            is_home, py_home_split.values, py_away_split.values
        ).tolist()

        # Run distribution: scoring variance (std of RS) and 1-run game win %
        w30 = 30
        roll_rs = grp["rs"].rolling(w30, min_periods=5).std().shift(1)
        rows["run_std_30"] = roll_rs.fillna(2.0).values.tolist()  # neutral ~2 runs std
        one_run = (grp["rs"] - grp["ra"]).abs() == 1
        one_run_win = (
            (one_run.astype(float) * grp["win"]).rolling(w30, min_periods=1).sum().shift(1)
        )
        one_run_n = one_run.astype(float).rolling(w30, min_periods=1).sum().shift(1)
        rows["one_run_win_pct_30"] = (
            (one_run_win / one_run_n.replace(0, np.nan)).fillna(_NEUTRAL_WIN_PCT).values.tolist()
        )

        return pd.DataFrame(rows, index=idx)

    _gb = combined.groupby("team", group_keys=False)
    _kw = (
        {"include_groups": False}
        if "include_groups" in inspect.signature(_gb.apply).parameters
        else {}
    )
    stats = _gb.apply(_rolling_group, **_kw)

    stat_cols = (
        [f"win_pct_{w}" for w in windows]
        + [f"run_diff_{w}" for w in windows]
        + [f"pythag_{w}" for w in windows]
        + [f"n_games_{w}" for w in windows]
        + ["win_pct_ewm", "run_diff_ewm", "pythag_ewm"]
        + ["streak", "rest_days"]
        + ["win_pct_split", "pythag_split"]
        + ["run_std_30", "one_run_win_pct_30"]
    )

    home_s = stats[stats["side"] == "home"].set_index("gl_idx")[stat_cols].add_prefix("home_")
    away_s = stats[stats["side"] == "away"].set_index("gl_idx")[stat_cols].add_prefix("away_")

    result = home_s.join(away_s)
    return result.rename(
        columns={
            "home_win_pct_split": "home_win_pct_home_only",
            "home_pythag_split": "home_pythag_home_only",
            "away_win_pct_split": "away_win_pct_away_only",
            "away_pythag_split": "away_pythag_away_only",
        }
    )
