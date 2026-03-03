"""Lineup quality and continuity features from Retrosheet batting order data.

Uses home_1_id..home_9_id and visiting_1_id..visiting_9_id from gamelogs to compute:
- Lineup continuity: number of same starters as previous game (0-9).
- Lineup runs proxy: average of each starter's team runs-per-game when he started
  (season-to-date, shifted), as a proxy for lineup strength.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_HOME_ID_COLS = [f"home_{i}_id" for i in range(1, 10)]
_AWAY_ID_COLS = [f"visiting_{i}_id" for i in range(1, 10)]
_NEUTRAL_CONTINUITY = 4.5  # middle of 0-9
_NEUTRAL_RUNS_PROXY = 4.5  # league-avg runs per game


def _starter_set(row: pd.Series, id_cols: list[str]) -> set:
    """Set of non-null starter IDs for one game."""
    out = set()
    for c in id_cols:
        v = row.get(c)
        if pd.notna(v) and str(v).strip():
            out.add(str(v).strip())
    return out


def build_lineup_continuity(gamelogs: pd.DataFrame) -> pd.DataFrame:
    """Compute lineup continuity (number of same starters as previous game) per team per game.

    Returns
    -------
    DataFrame
        Aligned on gamelogs.index. Columns: home_lineup_continuity, away_lineup_continuity.
    """
    orig_index = gamelogs.index
    gl = gamelogs.copy()
    gl["date"] = pd.to_datetime(gl["date"])
    gl["game_num"] = pd.to_numeric(gl["game_num"], errors="coerce").fillna(0).astype(int)
    gl = gl.sort_values(["date", "game_num"])

    home_sets = gl[_HOME_ID_COLS].apply(
        lambda row: _starter_set(row, _HOME_ID_COLS),
        axis=1,
    )
    away_sets = gl[_AWAY_ID_COLS].apply(
        lambda row: _starter_set(row, _AWAY_ID_COLS),
        axis=1,
    )

    home_continuity = np.full(len(gl), _NEUTRAL_CONTINUITY, dtype=float)
    away_continuity = np.full(len(gl), _NEUTRAL_CONTINUITY, dtype=float)

    # Iterate chronologically; for each team track previous game's starters
    home_team_prev: dict[str, set] = {}
    away_team_prev: dict[str, set] = {}

    for i in range(len(gl)):
        h_team = str(gl.iloc[i]["home_team"])
        a_team = str(gl.iloc[i]["visiting_team"])
        h_set = home_sets.iloc[i]
        a_set = away_sets.iloc[i]

        if h_team in home_team_prev:
            overlap = len(h_set & home_team_prev[h_team])
            home_continuity[i] = float(overlap)
        if a_team in away_team_prev:
            overlap = len(a_set & away_team_prev[a_team])
            away_continuity[i] = float(overlap)

        home_team_prev[h_team] = h_set
        away_team_prev[a_team] = a_set

    out = pd.DataFrame(
        {
            "home_lineup_continuity": home_continuity,
            "away_lineup_continuity": away_continuity,
        },
        index=gl.index,
    )
    return out.reindex(orig_index)


def build_lineup_features(gamelogs: pd.DataFrame) -> pd.DataFrame:
    """Build lineup continuity features (and placeholder for strength proxy).

    Returns
    -------
    DataFrame
        Columns: home_lineup_continuity, away_lineup_continuity.
    """
    return build_lineup_continuity(gamelogs)
