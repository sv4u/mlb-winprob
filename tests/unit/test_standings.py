"""Tests for mlb_predict.standings — predicted standings computation and merging."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    pass

from mlb_predict.standings import (
    DIVISION_DISPLAY_ORDER,
    DIVISIONS,
    RETRO_TO_DIVISION,
    RETRO_TO_MLB_ID,
    TEAM_DIVISION_MAP,
    compute_league_leaders,
    compute_predicted_standings,
    merge_predicted_actual,
)


# ---------------------------------------------------------------------------
# Division / team mapping constants
# ---------------------------------------------------------------------------


def test_team_division_map_has_30_teams() -> None:
    """TEAM_DIVISION_MAP must contain exactly 30 MLB teams."""
    assert len(TEAM_DIVISION_MAP) == 30


def test_retro_to_mlb_id_has_30_entries() -> None:
    """RETRO_TO_MLB_ID reverse map must have 30 unique Retrosheet codes."""
    assert len(RETRO_TO_MLB_ID) == 30


def test_retro_to_division_has_30_entries() -> None:
    """RETRO_TO_DIVISION must map 30 Retrosheet codes to division IDs."""
    assert len(RETRO_TO_DIVISION) == 30


def test_divisions_has_six_entries() -> None:
    """DIVISIONS must define exactly 6 divisions (3 AL + 3 NL)."""
    assert len(DIVISIONS) == 6


def test_division_display_order_covers_all_divisions() -> None:
    """DIVISION_DISPLAY_ORDER must include all 6 division IDs."""
    assert set(DIVISION_DISPLAY_ORDER) == set(DIVISIONS.keys())


def test_each_division_has_five_teams() -> None:
    """Every division must contain exactly 5 teams."""
    div_counts: dict[int, int] = {}
    for _, (div_id, _) in TEAM_DIVISION_MAP.items():
        div_counts[div_id] = div_counts.get(div_id, 0) + 1
    for div_id, count in div_counts.items():
        assert count == 5, f"Division {div_id} has {count} teams, expected 5"


def test_retro_to_mlb_id_roundtrip() -> None:
    """MLB ID → retro code → MLB ID must round-trip correctly."""
    for mlb_id, (_, retro_code) in TEAM_DIVISION_MAP.items():
        assert RETRO_TO_MLB_ID[retro_code] == mlb_id


def test_al_and_nl_each_have_three_divisions() -> None:
    """AL and NL must each contain 3 divisions."""
    al_divs = [d for d, info in DIVISIONS.items() if info["league"] == "AL"]
    nl_divs = [d for d, info in DIVISIONS.items() if info["league"] == "NL"]
    assert len(al_divs) == 3
    assert len(nl_divs) == 3


# ---------------------------------------------------------------------------
# Fixtures — synthetic feature DataFrames for standings tests
# ---------------------------------------------------------------------------


@pytest.fixture
def mini_features_df() -> pd.DataFrame:
    """Small 4-team, 4-game features DataFrame for testing predicted standings.

    Division 201 (AL East): BOS (111) and NYA (147)
    Division 200 (AL West): SEA (136) and HOU (117)
    """
    return pd.DataFrame(
        {
            "game_pk": [1, 2, 3, 4],
            "season": [2026, 2026, 2026, 2026],
            "date": pd.to_datetime(["2026-04-01"] * 4).date,
            "home_retro": ["BOS", "NYA", "SEA", "HOU"],
            "away_retro": ["NYA", "BOS", "HOU", "SEA"],
            "home_mlb_id": [111, 147, 136, 117],
            "away_mlb_id": [147, 111, 117, 136],
            "home_win": [np.nan] * 4,
            "prob": [0.6, 0.55, 0.7, 0.45],
        }
    )


@pytest.fixture
def predicted_df(mini_features_df: pd.DataFrame) -> pd.DataFrame:
    """Pre-computed predicted standings from the mini features fixture."""
    return compute_predicted_standings(mini_features_df, season=2026)


@pytest.fixture
def actual_standings_df() -> pd.DataFrame:
    """Simulated actual standings from the MLB Stats API for 4 teams."""
    return pd.DataFrame(
        {
            "team_id": [111, 147, 136, 117],
            "team_name": ["Red Sox", "Yankees", "Mariners", "Astros"],
            "wins": [50, 45, 55, 40],
            "losses": [30, 35, 25, 40],
            "pct": [0.625, 0.563, 0.688, 0.500],
            "gb": ["-", "5.0", "-", "15.0"],
            "division_rank": [1, 2, 1, 2],
            "league_rank": [2, 3, 1, 5],
            "runs_scored": [400, 380, 420, 350],
            "runs_allowed": [320, 340, 280, 360],
            "run_diff": [80, 40, 140, -10],
        }
    )


# ---------------------------------------------------------------------------
# compute_predicted_standings
# ---------------------------------------------------------------------------


def test_compute_predicted_standings_returns_dataframe(mini_features_df: pd.DataFrame) -> None:
    """compute_predicted_standings must return a non-empty DataFrame."""
    result = compute_predicted_standings(mini_features_df, season=2026)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_compute_predicted_standings_team_count(mini_features_df: pd.DataFrame) -> None:
    """Result must contain exactly as many teams as appear in the feature data."""
    result = compute_predicted_standings(mini_features_df, season=2026)
    assert len(result) == 4


def test_compute_predicted_standings_has_required_columns(
    mini_features_df: pd.DataFrame,
) -> None:
    """Result must include all expected columns."""
    result = compute_predicted_standings(mini_features_df, season=2026)
    required = {
        "retro_code",
        "mlb_id",
        "division_id",
        "pred_wins",
        "pred_losses",
        "pred_total_games",
        "pred_win_pct",
        "division_name",
        "league",
        "pred_division_rank",
        "pred_gb",
        "pred_gb_str",
    }
    assert required.issubset(set(result.columns))


def test_compute_predicted_standings_wins_plus_losses_equals_games(
    mini_features_df: pd.DataFrame,
) -> None:
    """Predicted wins + losses must equal total games for every team."""
    result = compute_predicted_standings(mini_features_df, season=2026)
    for _, row in result.iterrows():
        assert abs(row["pred_wins"] + row["pred_losses"] - row["pred_total_games"]) < 0.1


def test_compute_predicted_standings_win_pct_range(mini_features_df: pd.DataFrame) -> None:
    """Predicted win percentage must be between 0 and 1."""
    result = compute_predicted_standings(mini_features_df, season=2026)
    assert (result["pred_win_pct"] >= 0).all()
    assert (result["pred_win_pct"] <= 1).all()


def test_compute_predicted_standings_division_rank_assigned(
    mini_features_df: pd.DataFrame,
) -> None:
    """Teams within a division must have unique, sequential rank values."""
    result = compute_predicted_standings(mini_features_df, season=2026)
    for _, grp in result.groupby("division_id"):
        ranks = sorted(grp["pred_division_rank"].tolist())
        assert ranks == list(range(1, len(grp) + 1))


def test_compute_predicted_standings_leader_has_zero_gb(mini_features_df: pd.DataFrame) -> None:
    """Division leaders (rank 1) must have GB of '-'."""
    result = compute_predicted_standings(mini_features_df, season=2026)
    leaders = result[result["pred_division_rank"] == 1]
    assert (leaders["pred_gb_str"] == "-").all()


def test_compute_predicted_standings_empty_season() -> None:
    """An empty features DataFrame must return an empty DataFrame."""
    df = pd.DataFrame(columns=["season", "home_retro", "away_retro", "prob"])
    result = compute_predicted_standings(df, season=2026)
    assert result.empty


def test_compute_predicted_standings_no_probs() -> None:
    """Features with all NaN probabilities must return an empty DataFrame."""
    df = pd.DataFrame(
        {
            "game_pk": [1],
            "season": [2026],
            "date": pd.to_datetime(["2026-04-01"]).date,
            "home_retro": ["BOS"],
            "away_retro": ["NYA"],
            "home_mlb_id": [111],
            "away_mlb_id": [147],
            "home_win": [np.nan],
            "prob": [np.nan],
        }
    )
    result = compute_predicted_standings(df, season=2026)
    assert result.empty


def test_compute_predicted_standings_wrong_season_returns_empty(
    mini_features_df: pd.DataFrame,
) -> None:
    """Querying a season not present in the data must return empty."""
    result = compute_predicted_standings(mini_features_df, season=2025)
    assert result.empty


def test_compute_predicted_standings_game_type_s_returns_empty_without_spring_data(
    mini_features_df: pd.DataFrame,
) -> None:
    """When game_type='S' and features have no game_type column, returns empty."""
    result = compute_predicted_standings(mini_features_df, season=2026, game_type="S")
    assert result.empty


def test_compute_predicted_standings_game_type_r_filters_to_regular_only() -> None:
    """When game_type='R', only rows with game_type R are included."""
    df = pd.DataFrame(
        {
            "game_pk": [1, 2],
            "season": [2026, 2026],
            "date": pd.to_datetime(["2026-04-01", "2026-03-15"]).date,
            "home_retro": ["BOS", "BOS"],
            "away_retro": ["NYA", "NYA"],
            "home_mlb_id": [111, 111],
            "away_mlb_id": [147, 147],
            "home_win": [np.nan, np.nan],
            "prob": [0.6, 0.5],
            "game_type": ["R", "S"],
        }
    )
    result = compute_predicted_standings(df, season=2026, game_type="R")
    assert not result.empty
    assert len(result) == 2
    bos = result[result["retro_code"] == "BOS"].iloc[0]
    assert bos["pred_total_games"] == 1


def test_compute_predicted_standings_prob_aggregation(mini_features_df: pd.DataFrame) -> None:
    """Verify specific probability aggregation for one team.

    BOS is home in game 1 (prob=0.6) and away in game 2 (prob=0.55).
    Expected wins = 0.6 (home) + (1 - 0.55) (away) = 0.6 + 0.45 = 1.05
    """
    result = compute_predicted_standings(mini_features_df, season=2026)
    bos = result[result["retro_code"] == "BOS"].iloc[0]
    assert abs(bos["pred_wins"] - 1.0) < 0.15  # ~1.05


# ---------------------------------------------------------------------------
# merge_predicted_actual
# ---------------------------------------------------------------------------


def test_merge_predicted_actual_adds_actual_columns(
    predicted_df: pd.DataFrame,
    actual_standings_df: pd.DataFrame,
) -> None:
    """Merging must add actual_wins, actual_losses, and delta columns."""
    merged = merge_predicted_actual(predicted_df, actual_standings_df)
    assert "actual_wins" in merged.columns
    assert "actual_losses" in merged.columns
    assert "wins_delta" in merged.columns
    assert "pct_delta" in merged.columns
    assert "rank_delta" in merged.columns


def test_merge_predicted_actual_preserves_row_count(
    predicted_df: pd.DataFrame,
    actual_standings_df: pd.DataFrame,
) -> None:
    """Merge must not drop or duplicate rows from the predicted side."""
    merged = merge_predicted_actual(predicted_df, actual_standings_df)
    assert len(merged) == len(predicted_df)


def test_merge_predicted_actual_computes_correct_delta(
    predicted_df: pd.DataFrame,
    actual_standings_df: pd.DataFrame,
) -> None:
    """wins_delta must equal actual_wins - pred_wins."""
    merged = merge_predicted_actual(predicted_df, actual_standings_df)
    for _, row in merged.iterrows():
        expected_delta = row["actual_wins"] - row["pred_wins"]
        assert abs(row["wins_delta"] - expected_delta) < 0.01


def test_merge_predicted_actual_empty_pred_returns_pred() -> None:
    """Merging with an empty predicted DataFrame must return it unchanged."""
    pred = pd.DataFrame()
    actual = pd.DataFrame({"team_id": [111], "wins": [50]})
    result = merge_predicted_actual(pred, actual)
    assert result.empty


def test_merge_predicted_actual_empty_actual_returns_pred(predicted_df: pd.DataFrame) -> None:
    """Merging with empty actual data must return predicted data unchanged."""
    actual = pd.DataFrame()
    result = merge_predicted_actual(predicted_df, actual)
    assert len(result) == len(predicted_df)
    assert "actual_wins" not in result.columns


# ---------------------------------------------------------------------------
# compute_league_leaders
# ---------------------------------------------------------------------------


def test_compute_league_leaders_returns_both_leagues(predicted_df: pd.DataFrame) -> None:
    """compute_league_leaders must return entries for AL (the only league in mini data)."""
    leaders = compute_league_leaders(predicted_df)
    assert "AL" in leaders
    assert "predicted_leader" in leaders["AL"]


def test_compute_league_leaders_predicted_has_correct_fields(
    predicted_df: pd.DataFrame,
) -> None:
    """Each predicted_leader must include team_name, pred_wins, pred_losses, pred_win_pct."""
    leaders = compute_league_leaders(predicted_df)
    for league_data in leaders.values():
        pl = league_data["predicted_leader"]
        assert "retro_code" in pl
        assert "pred_wins" in pl
        assert "pred_losses" in pl
        assert "pred_win_pct" in pl


def test_compute_league_leaders_selects_highest_pct(predicted_df: pd.DataFrame) -> None:
    """The predicted leader must be the team with the highest pred_win_pct."""
    leaders = compute_league_leaders(predicted_df)
    al_leader = leaders["AL"]["predicted_leader"]
    al_teams = predicted_df[predicted_df["league"] == "AL"]
    best_pct = al_teams["pred_win_pct"].max()
    assert al_leader["pred_win_pct"] == best_pct


def test_compute_league_leaders_with_actual_data(
    predicted_df: pd.DataFrame,
    actual_standings_df: pd.DataFrame,
) -> None:
    """When actual data is merged, compute_league_leaders must include actual_leader."""
    merged = merge_predicted_actual(predicted_df, actual_standings_df)
    leaders = compute_league_leaders(merged)
    assert "actual_leader" in leaders["AL"]
    al = leaders["AL"]["actual_leader"]
    assert "actual_wins" in al
    assert "actual_losses" in al
    assert "actual_win_pct" in al


def test_compute_league_leaders_no_actual_omits_actual_leader(
    predicted_df: pd.DataFrame,
) -> None:
    """Without actual data, compute_league_leaders must not include actual_leader."""
    leaders = compute_league_leaders(predicted_df)
    assert "actual_leader" not in leaders.get("AL", {})
