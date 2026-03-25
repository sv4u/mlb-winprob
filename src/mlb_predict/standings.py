"""Standings computation: predicted vs actual, division/league structure.

Computes expected team records from per-game win probabilities and compares
them against live MLB standings fetched from the Stats API.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# MLB team → division/league mapping (2026 alignment structure)
# Uses MLB Stats API team IDs.
# Division IDs: ALE=201, ALC=202, ALW=200, NLE=204, NLC=205, NLW=203
# ---------------------------------------------------------------------------

DIVISIONS: dict[int, dict[str, Any]] = {
    201: {"name": "AL East", "league": "AL", "league_id": 103},
    202: {"name": "AL Central", "league": "AL", "league_id": 103},
    200: {"name": "AL West", "league": "AL", "league_id": 103},
    204: {"name": "NL East", "league": "NL", "league_id": 104},
    205: {"name": "NL Central", "league": "NL", "league_id": 104},
    203: {"name": "NL West", "league": "NL", "league_id": 104},
}

# MLB Stats API team_id → (division_id, retrosheet_code)
TEAM_DIVISION_MAP: dict[int, tuple[int, str]] = {
    # AL East
    110: (201, "BAL"),  # Orioles
    111: (201, "BOS"),  # Red Sox
    147: (201, "NYA"),  # Yankees
    139: (201, "TBA"),  # Rays
    141: (201, "TOR"),  # Blue Jays
    # AL Central
    145: (202, "CHA"),  # White Sox
    114: (202, "CLE"),  # Guardians
    116: (202, "DET"),  # Tigers
    118: (202, "KCA"),  # Royals
    142: (202, "MIN"),  # Twins
    # AL West
    108: (200, "ANA"),  # Angels
    117: (200, "HOU"),  # Astros
    133: (200, "ATH"),  # Athletics
    136: (200, "SEA"),  # Mariners
    140: (200, "TEX"),  # Rangers
    # NL East
    144: (204, "ATL"),  # Braves
    146: (204, "MIA"),  # Marlins
    121: (204, "NYN"),  # Mets
    143: (204, "PHI"),  # Phillies
    120: (204, "WAS"),  # Nationals
    # NL Central
    112: (205, "CHN"),  # Cubs
    113: (205, "CIN"),  # Reds
    158: (205, "MIL"),  # Brewers
    134: (205, "PIT"),  # Pirates
    138: (205, "SLN"),  # Cardinals
    # NL West
    109: (203, "ARI"),  # D-backs
    115: (203, "COL"),  # Rockies
    119: (203, "LAN"),  # Dodgers
    135: (203, "SDN"),  # Padres
    137: (203, "SFN"),  # Giants
}

# Retrosheet code → MLB Stats API team_id (reverse mapping)
RETRO_TO_MLB_ID: dict[str, int] = {v[1]: k for k, v in TEAM_DIVISION_MAP.items()}

# Retrosheet code → division_id
RETRO_TO_DIVISION: dict[str, int] = {v[1]: v[0] for v in TEAM_DIVISION_MAP.values()}

DIVISION_DISPLAY_ORDER: list[int] = [201, 202, 200, 204, 205, 203]


def compute_predicted_standings(
    features_df: pd.DataFrame,
    season: int | None = None,
    *,
    game_type: str = "R",
) -> pd.DataFrame:
    """Aggregate per-game win probabilities into expected team records.

    For each team, sums the predicted P(win) across all games to get
    expected wins.  A team's P(win) in a game is:
      - prob      when the team is the home team
      - 1 - prob  when the team is the away team

    game_type: "R" = regular season only, "S" = spring training only.
    Uses the "game_type" column in features_df when present; default "R".

    When ``season`` is omitted, :func:`mlb_predict.season.infer_target_mlb_season`
    supplies the default year (API callers should normally pass an explicit year).
    """
    if season is None:
        from mlb_predict.season import infer_target_mlb_season

        season = infer_target_mlb_season()
    df = features_df[features_df["season"] == season].copy()
    if df.empty:
        return pd.DataFrame()

    if "game_type" in df.columns:
        df = df[df["game_type"] == game_type].copy()
    elif game_type != "R":
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()

    has_prob = df["prob"].notna()
    df_pred = df[has_prob].copy()
    if df_pred.empty:
        return pd.DataFrame()

    all_teams: set[str] = set(df_pred["home_retro"].unique()) | set(df_pred["away_retro"].unique())

    rows: list[dict[str, Any]] = []
    for team in sorted(all_teams):
        if team not in RETRO_TO_MLB_ID:
            continue

        home_games = df_pred[df_pred["home_retro"] == team]
        away_games = df_pred[df_pred["away_retro"] == team]

        home_exp_wins = home_games["prob"].sum()
        away_exp_wins = (1.0 - away_games["prob"]).sum()

        total_games = len(home_games) + len(away_games)
        exp_wins = home_exp_wins + away_exp_wins
        exp_losses = total_games - exp_wins

        rows.append(
            {
                "retro_code": team,
                "mlb_id": RETRO_TO_MLB_ID[team],
                "division_id": RETRO_TO_DIVISION[team],
                "pred_wins": round(exp_wins, 1),
                "pred_losses": round(exp_losses, 1),
                "pred_total_games": total_games,
                "pred_win_pct": round(exp_wins / total_games, 3) if total_games else 0.0,
                "games_as_home": len(home_games),
                "games_as_away": len(away_games),
            }
        )

    pred_df = pd.DataFrame(rows)
    pred_df["division_name"] = pred_df["division_id"].map(
        lambda d: DIVISIONS.get(d, {}).get("name", "")
    )
    pred_df["league"] = pred_df["division_id"].map(lambda d: DIVISIONS.get(d, {}).get("league", ""))

    # Compute predicted division rank within each division
    pred_df = pred_df.sort_values(["division_id", "pred_win_pct"], ascending=[True, False])
    pred_df["pred_division_rank"] = pred_df.groupby("division_id").cumcount() + 1

    # Compute predicted GB within each division
    gb_values = []
    for _, grp in pred_df.groupby("division_id"):
        leader_wins = grp["pred_wins"].iloc[0]
        leader_losses = grp["pred_losses"].iloc[0]
        gb = ((leader_wins - grp["pred_wins"]) + (grp["pred_losses"] - leader_losses)) / 2
        gb_values.append(gb.round(1))
    pred_df["pred_gb"] = pd.concat(gb_values)
    pred_df.loc[pred_df["pred_gb"] == 0.0, "pred_gb_str"] = "-"
    pred_df.loc[pred_df["pred_gb"] != 0.0, "pred_gb_str"] = pred_df["pred_gb"].astype(str)

    return pred_df


def merge_predicted_actual(
    pred_df: pd.DataFrame,
    actual_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge predicted standings with actual live standings.

    Joins on MLB team ID and computes deltas for wins, losses, and win%.
    """
    if pred_df.empty or actual_df.empty:
        return pred_df

    actual_slim = actual_df[
        [
            "team_id",
            "team_name",
            "wins",
            "losses",
            "pct",
            "gb",
            "division_rank",
            "league_rank",
            "runs_scored",
            "runs_allowed",
            "run_diff",
        ]
    ].rename(
        columns={
            "team_id": "mlb_id",
            "wins": "actual_wins",
            "losses": "actual_losses",
            "pct": "actual_win_pct",
            "gb": "actual_gb",
            "division_rank": "actual_division_rank",
            "league_rank": "actual_league_rank",
        }
    )

    merged = pred_df.merge(actual_slim, on="mlb_id", how="left")

    # Compute deltas
    merged["wins_delta"] = merged["actual_wins"] - merged["pred_wins"]
    merged["losses_delta"] = merged["actual_losses"] - merged["pred_losses"]
    merged["pct_delta"] = merged["actual_win_pct"] - merged["pred_win_pct"]
    merged["rank_delta"] = merged["pred_division_rank"] - merged["actual_division_rank"]

    return merged


def compute_league_leaders(standings_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Identify predicted and actual league leaders for AL and NL.

    Returns a dict keyed by league ("AL", "NL") with predicted_leader and
    actual_leader sub-dicts.
    """
    result: dict[str, dict[str, Any]] = {}

    for league in ["AL", "NL"]:
        league_df = standings_df[standings_df["league"] == league]
        if league_df.empty:
            continue

        # Predicted leader: highest pred_win_pct
        pred_leader_idx = league_df["pred_win_pct"].idxmax()
        pred_leader = league_df.loc[pred_leader_idx]

        entry: dict[str, Any] = {
            "predicted_leader": {
                "team_name": pred_leader.get("team_name", pred_leader.get("retro_code", "")),
                "retro_code": pred_leader["retro_code"],
                "pred_wins": pred_leader["pred_wins"],
                "pred_losses": pred_leader["pred_losses"],
                "pred_win_pct": pred_leader["pred_win_pct"],
            },
        }

        # Actual leader (if available)
        if "actual_win_pct" in league_df.columns and league_df["actual_win_pct"].notna().any():
            act_league = league_df[league_df["actual_win_pct"].notna()]
            if not act_league.empty:
                act_leader_idx = act_league["actual_win_pct"].idxmax()
                act_leader = act_league.loc[act_leader_idx]
                entry["actual_leader"] = {
                    "team_name": act_leader.get("team_name", act_leader.get("retro_code", "")),
                    "retro_code": act_leader["retro_code"],
                    "actual_wins": int(act_leader["actual_wins"]),
                    "actual_losses": int(act_leader["actual_losses"]),
                    "actual_win_pct": float(act_leader["actual_win_pct"]),
                }

        result[league] = entry

    return result
