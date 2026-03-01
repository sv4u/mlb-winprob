"""Team Elo rating system computed sequentially across all seasons.

Elo ratings carry forward between seasons with a regression-to-mean step at
the start of each new season.  For each game the **pre-game** Elo is recorded
so there is no data leakage.

References
----------
FiveThirtyEight MLB Elo model (now archived):
  https://github.com/fivethirtyeight/data/tree/master/mlb-elo

Key parameters
--------------
INITIAL_ELO : 1500  (all teams start equal)
K_FACTOR    : 20    (sensitivity to each result; standard for ~162-game seasons)
HOME_FIELD  : 24    (Elo-point home-field advantage in expected-score formula)
REGRESSION  : 0.33  (fraction pulled back to 1500 at start of each season)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

_INITIAL_ELO: float = 1500.0
_K_FACTOR: float = 20.0
_HOME_FIELD_ADV: float = 24.0  # Elo points added to home team expected score
_REGRESSION: float = 0.33       # fraction of gap to 1500 closed at season start


@dataclass
class EloState:
    """Mutable Elo rating store for all teams."""

    ratings: dict[str, float] = field(default_factory=dict)

    def get(self, team: str) -> float:
        """Return current rating, defaulting to INITIAL_ELO for new teams."""
        return self.ratings.get(team, _INITIAL_ELO)

    def update(self, team: str, new_rating: float) -> None:
        self.ratings[team] = new_rating

    def apply_season_regression(self) -> None:
        """Pull every team's rating toward 1500 at the start of a new season."""
        for team in list(self.ratings):
            self.ratings[team] = (
                self.ratings[team] + _REGRESSION * (_INITIAL_ELO - self.ratings[team])
            )


def _expected_score(elo_home: float, elo_away: float) -> float:
    """Expected win probability for home team given Elo ratings."""
    return 1.0 / (1.0 + 10.0 ** ((elo_away - elo_home - _HOME_FIELD_ADV) / 400.0))


def _new_rating(old: float, actual: float, expected: float) -> float:
    return old + _K_FACTOR * (actual - expected)


def compute_elo_ratings(gamelogs_all: pd.DataFrame) -> pd.DataFrame:
    """Compute sequential pre-game Elo ratings for every game in *gamelogs_all*.

    Games must span multiple seasons and be sorted chronologically.
    A regression-to-mean step is applied whenever the season year changes.

    Parameters
    ----------
    gamelogs_all:
        Concatenation of Retrosheet gamelogs across seasons.  Required columns:
        ``date``, ``game_num``, ``home_team``, ``visiting_team``,
        ``home_score``, ``visiting_score``.

    Returns
    -------
    DataFrame
        Aligned on ``gamelogs_all.index``.  Columns:
        ``home_elo``, ``away_elo``, ``elo_diff`` (home − away, pre-HFA).
    """
    gl = gamelogs_all.reset_index(drop=True).copy()
    gl["date"] = pd.to_datetime(gl["date"])
    gl["season"] = gl["date"].dt.year
    gl["game_num"] = pd.to_numeric(gl["game_num"], errors="coerce").fillna(0).astype(int)
    gl = gl.sort_values(["date", "game_num"]).reset_index(drop=True)

    state = EloState()
    current_season: int | None = None

    home_elos: list[float] = []
    away_elos: list[float] = []

    for _, row in gl.iterrows():
        season = int(row["season"])

        # Season boundary: apply regression to mean once per new season
        if current_season is not None and season != current_season:
            state.apply_season_regression()
        current_season = season

        home_team = str(row["home_team"])
        away_team = str(row["visiting_team"])

        elo_h = state.get(home_team)
        elo_a = state.get(away_team)

        home_elos.append(elo_h)
        away_elos.append(elo_a)

        # Update ratings based on outcome
        home_score = pd.to_numeric(row["home_score"], errors="coerce")
        away_score = pd.to_numeric(row["visiting_score"], errors="coerce")

        if pd.notna(home_score) and pd.notna(away_score):
            actual_h = 1.0 if home_score > away_score else 0.0
            exp_h = _expected_score(elo_h, elo_a)
            state.update(home_team, _new_rating(elo_h, actual_h, exp_h))
            state.update(away_team, _new_rating(elo_a, 1.0 - actual_h, 1.0 - exp_h))
        # If score unknown (future game), ratings don't change

    elo_df = pd.DataFrame(
        {
            "home_elo": home_elos,
            "away_elo": away_elos,
        },
        index=gl.index,
    )
    elo_df["elo_diff"] = elo_df["home_elo"] - elo_df["away_elo"]

    # Restore original index alignment
    elo_df.index = gamelogs_all.index[gl.index]
    return elo_df.reindex(gamelogs_all.index)
