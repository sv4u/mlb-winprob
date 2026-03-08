"""Build spring training feature matrices from schedule scores + prior-season team state.

Spring training games have no Retrosheet gamelogs, so features are built from
each team's end-of-prior-season state (Elo, rolling stats, pitcher stats, etc.)
exactly like ``build_features_2026.py`` does for pre-season predictions — but
with actual game outcomes (home_win) derived from the schedule's final scores.

Output
------
``data/processed/features/features_spring_<season>.parquet`` for each season
that has completed spring training games in the schedule.

The ``is_spring`` column is set to 1.0 for all rows so the model can learn
the distinction between spring training and regular season games.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_PROCESSED = Path("data/processed")
_FEATURES = _PROCESSED / "features"
_TEAM_MAP = _PROCESSED / "team_id_map_retro_to_mlb.csv"

_PLAYED_STATUSES = {"Final", "Completed Early", "Game Over"}

_DEF: dict[str, float] = {
    "elo": 1500.0,
    "win_pct_15": 0.500,
    "win_pct_30": 0.500,
    "win_pct_60": 0.500,
    "run_diff_15": 0.0,
    "run_diff_30": 0.0,
    "run_diff_60": 0.0,
    "pythag_15": 0.500,
    "pythag_30": 0.500,
    "pythag_60": 0.500,
    "win_pct_ewm": 0.500,
    "run_diff_ewm": 0.0,
    "pythag_ewm": 0.500,
    "streak": 0.0,
    "win_pct_home_only": 0.500,
    "pythag_home_only": 0.500,
    "win_pct_away_only": 0.500,
    "pythag_away_only": 0.500,
    "run_std_30": 3.5,
    "one_run_win_pct_30": 0.500,
    "sp_era": 4.50,
    "sp_k9": 8.50,
    "sp_bb9": 3.00,
    "sp_whip": 1.30,
    "sp_est_woba": 0.320,
    "bat_woba": 0.320,
    "bat_barrel_pct": 0.080,
    "bat_hard_pct": 0.370,
    "bat_iso": 0.170,
    "bat_babip": 0.300,
    "bat_xwoba": 0.320,
    "pit_fip": 4.20,
    "pit_xfip": 4.20,
    "pit_k_pct": 0.220,
    "pit_bb_pct": 0.085,
    "pit_hr_fb": 0.110,
    "pit_whip": 1.30,
}

_OFF_SEASON_REST: float = 5.0

_HOME_COL_MAP: dict[str, str] = {
    "home_elo": "elo",
    "home_win_pct_15": "win_pct_15",
    "home_win_pct_30": "win_pct_30",
    "home_win_pct_60": "win_pct_60",
    "home_run_diff_15": "run_diff_15",
    "home_run_diff_30": "run_diff_30",
    "home_run_diff_60": "run_diff_60",
    "home_pythag_15": "pythag_15",
    "home_pythag_30": "pythag_30",
    "home_pythag_60": "pythag_60",
    "home_win_pct_ewm": "win_pct_ewm",
    "home_run_diff_ewm": "run_diff_ewm",
    "home_pythag_ewm": "pythag_ewm",
    "home_streak": "streak",
    "home_run_std_30": "run_std_30",
    "home_one_run_win_pct_30": "one_run_win_pct_30",
    "home_sp_era": "sp_era",
    "home_sp_k9": "sp_k9",
    "home_sp_bb9": "sp_bb9",
    "home_sp_whip": "sp_whip",
    "home_sp_est_woba": "sp_est_woba",
    "home_bat_woba": "bat_woba",
    "home_bat_barrel_pct": "bat_barrel_pct",
    "home_bat_hard_pct": "bat_hard_pct",
    "home_bat_iso": "bat_iso",
    "home_bat_babip": "bat_babip",
    "home_bat_xwoba": "bat_xwoba",
    "home_pit_fip": "pit_fip",
    "home_pit_xfip": "pit_xfip",
    "home_pit_k_pct": "pit_k_pct",
    "home_pit_bb_pct": "pit_bb_pct",
    "home_pit_hr_fb": "pit_hr_fb",
    "home_pit_whip": "pit_whip",
    "home_win_pct_home_only": "win_pct_home_only",
    "home_pythag_home_only": "pythag_home_only",
}

_AWAY_COL_MAP: dict[str, str] = {
    "away_elo": "elo",
    "away_win_pct_15": "win_pct_15",
    "away_win_pct_30": "win_pct_30",
    "away_win_pct_60": "win_pct_60",
    "away_run_diff_15": "run_diff_15",
    "away_run_diff_30": "run_diff_30",
    "away_run_diff_60": "run_diff_60",
    "away_pythag_15": "pythag_15",
    "away_pythag_30": "pythag_30",
    "away_pythag_60": "pythag_60",
    "away_win_pct_ewm": "win_pct_ewm",
    "away_run_diff_ewm": "run_diff_ewm",
    "away_pythag_ewm": "pythag_ewm",
    "away_streak": "streak",
    "away_run_std_30": "run_std_30",
    "away_one_run_win_pct_30": "one_run_win_pct_30",
    "away_sp_era": "sp_era",
    "away_sp_k9": "sp_k9",
    "away_sp_bb9": "sp_bb9",
    "away_sp_whip": "sp_whip",
    "away_sp_est_woba": "sp_est_woba",
    "away_bat_woba": "bat_woba",
    "away_bat_barrel_pct": "bat_barrel_pct",
    "away_bat_hard_pct": "bat_hard_pct",
    "away_bat_iso": "bat_iso",
    "away_bat_babip": "bat_babip",
    "away_bat_xwoba": "bat_xwoba",
    "away_pit_fip": "pit_fip",
    "away_pit_xfip": "pit_xfip",
    "away_pit_k_pct": "pit_k_pct",
    "away_pit_bb_pct": "pit_bb_pct",
    "away_pit_hr_fb": "pit_hr_fb",
    "away_pit_whip": "pit_whip",
    "away_win_pct_away_only": "win_pct_away_only",
    "away_pythag_away_only": "pythag_away_only",
}


def _build_mlb_to_retro(processed_dir: Path | None = None) -> dict[int, str]:
    """MLB team ID -> Retrosheet code, preferring the most recent valid entry."""
    team_map = (processed_dir or _PROCESSED) / "team_id_map_retro_to_mlb.csv"
    if not team_map.exists():
        return {}
    tm = pd.read_csv(team_map)
    tm = tm.sort_values("valid_to_season").drop_duplicates("mlb_team_id", keep="last")
    return dict(zip(tm["mlb_team_id"].astype(int), tm["retro_team_code"]))


def _extract_state_from_role(
    features: pd.DataFrame,
    team_col: str,
    col_map: dict[str, str],
) -> dict[str, dict[str, float]]:
    """Extract the last-known feature state for each team from their home or away rows."""
    available = {c: n for c, n in col_map.items() if c in features.columns}
    result: dict[str, dict[str, float]] = {}
    features = features.copy()
    features["date"] = pd.to_datetime(features["date"], errors="coerce").dt.date
    for team, grp in features.sort_values("date").groupby(team_col):
        last = grp.iloc[-1]
        result[str(team)] = {neutral: float(last[col]) for col, neutral in available.items()}
    return result


def _last_game_dates(
    features_all: pd.DataFrame,
    team_col: str,
) -> dict[str, object]:
    """Return the date of each team's chronologically last game in a given role."""
    tmp = features_all.copy()
    tmp["_date"] = pd.to_datetime(tmp["date"], errors="coerce").dt.date
    return (
        tmp.dropna(subset=["_date"])
        .sort_values("_date")
        .groupby(team_col)["_date"]
        .last()
        .to_dict()
    )


def _build_team_state(features_all: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Merge home- and away-role states into a single per-team dict.

    For role-agnostic stats (Elo, rolling win pct, pitcher metrics, etc.),
    the value from whichever role had the more recent game is preferred so
    that prior-season carry-forward uses the freshest team state.
    Role-specific splits (home-only, away-only) always come from their
    respective roles.
    """
    home_states = _extract_state_from_role(features_all, "home_retro", _HOME_COL_MAP)
    away_states = _extract_state_from_role(features_all, "away_retro", _AWAY_COL_MAP)
    home_dates = _last_game_dates(features_all, "home_retro")
    away_dates = _last_game_dates(features_all, "away_retro")
    teams = set(home_states) | set(away_states)
    merged: dict[str, dict[str, float]] = {}
    for team in teams:
        h = home_states.get(team, {})
        a = away_states.get(team, {})
        h_date = home_dates.get(team)
        a_date = away_dates.get(team)
        away_more_recent = a_date is not None and (h_date is None or a_date > h_date)
        state: dict[str, float] = {}
        for key in set(h) | set(a):
            h_val = h.get(key, np.nan)
            a_val = a.get(key, np.nan)
            h_ok = not np.isnan(h_val)
            a_ok = not np.isnan(a_val)
            if h_ok and a_ok:
                state[key] = a_val if away_more_recent else h_val
            elif h_ok:
                state[key] = h_val
            else:
                state[key] = a_val
        state["win_pct_home_only"] = h.get("win_pct_home_only", np.nan)
        state["pythag_home_only"] = h.get("pythag_home_only", np.nan)
        state["win_pct_away_only"] = a.get("win_pct_away_only", np.nan)
        state["pythag_away_only"] = a.get("pythag_away_only", np.nan)
        merged[team] = state
    return merged


def _resolve(state: dict[str, float], key: str) -> float:
    """Return the state value or the league-average default if missing/NaN."""
    v = state.get(key, np.nan)
    if not np.isnan(v):
        return float(v)
    return _DEF.get(key, 0.0)


def _park_factor_by_home_team(features_all: pd.DataFrame) -> dict[str, float]:
    """Median park_run_factor per home_retro code (stable estimate)."""
    if "park_run_factor" not in features_all.columns:
        return {}
    return (
        features_all[features_all["season"] == features_all["season"].max()]
        .groupby("home_retro")["park_run_factor"]
        .median()
        .to_dict()
    )


def _feature_hash(row: dict) -> str:
    numeric = {k: round(v, 6) for k, v in row.items() if isinstance(v, float) and not np.isnan(v)}
    payload = json.dumps(numeric, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _build_game_row(
    g: pd.Series,
    *,
    season: int,
    idx: int,
    n_games: int,
    team_state: dict[str, dict[str, float]],
    park_factors: dict[str, float],
    mlb_to_retro: dict[int, str],
) -> dict | None:
    """Build a single feature row for a completed spring training game."""
    home_score = g.get("home_score")
    away_score = g.get("away_score")
    if pd.isna(home_score) or pd.isna(away_score):
        return None
    if int(home_score) == int(away_score):
        return None

    h_code = mlb_to_retro.get(int(g["home_mlb_id"]), "")
    a_code = mlb_to_retro.get(int(g["away_mlb_id"]), "")
    h = team_state.get(h_code, {})
    a = team_state.get(a_code, {})

    h_elo = _resolve(h, "elo")
    a_elo = _resolve(a, "elo")
    h_p30 = _resolve(h, "pythag_30")
    a_p30 = _resolve(a, "pythag_30")
    h_pewm = _resolve(h, "pythag_ewm")
    a_pewm = _resolve(a, "pythag_ewm")
    h_home = _resolve(h, "win_pct_home_only")
    a_away = _resolve(a, "win_pct_away_only")
    h_sp_whip = _resolve(h, "sp_whip")
    a_sp_whip = _resolve(a, "sp_whip")
    h_bat_iso = _resolve(h, "bat_iso")
    a_bat_iso = _resolve(a, "bat_iso")
    h_bat_xwoba = _resolve(h, "bat_xwoba")
    a_bat_xwoba = _resolve(a, "bat_xwoba")
    h_pit_whip = _resolve(h, "pit_whip")
    a_pit_whip = _resolve(a, "pit_whip")

    game_date = pd.to_datetime(g.get("game_date_local") or g.get("game_date_utc"))

    row: dict = {
        "game_pk": int(g["game_pk"]),
        "date": game_date.date() if not pd.isna(game_date) else None,
        "season": season,
        "home_mlb_id": int(g["home_mlb_id"]),
        "away_mlb_id": int(g["away_mlb_id"]),
        "home_retro": h_code,
        "away_retro": a_code,
        "game_type": "S",
        "home_win": float(int(home_score) > int(away_score)),
        "is_spring": 1.0,
        # Elo
        "home_elo": h_elo,
        "away_elo": a_elo,
        "elo_diff": h_elo - a_elo,
        # Multi-window rolling — home
        "home_win_pct_7": _resolve(h, "win_pct_15"),
        "home_win_pct_14": _resolve(h, "win_pct_15"),
        "home_win_pct_15": _resolve(h, "win_pct_15"),
        "home_win_pct_30": _resolve(h, "win_pct_30"),
        "home_win_pct_60": _resolve(h, "win_pct_60"),
        "home_run_diff_7": _resolve(h, "run_diff_15"),
        "home_run_diff_14": _resolve(h, "run_diff_15"),
        "home_run_diff_15": _resolve(h, "run_diff_15"),
        "home_run_diff_30": _resolve(h, "run_diff_30"),
        "home_run_diff_60": _resolve(h, "run_diff_60"),
        "home_pythag_7": _resolve(h, "pythag_15"),
        "home_pythag_14": _resolve(h, "pythag_15"),
        "home_pythag_15": _resolve(h, "pythag_15"),
        "home_pythag_30": h_p30,
        "home_pythag_60": _resolve(h, "pythag_60"),
        # Multi-window rolling — away
        "away_win_pct_7": _resolve(a, "win_pct_15"),
        "away_win_pct_14": _resolve(a, "win_pct_15"),
        "away_win_pct_15": _resolve(a, "win_pct_15"),
        "away_win_pct_30": _resolve(a, "win_pct_30"),
        "away_win_pct_60": _resolve(a, "win_pct_60"),
        "away_run_diff_7": _resolve(a, "run_diff_15"),
        "away_run_diff_14": _resolve(a, "run_diff_15"),
        "away_run_diff_15": _resolve(a, "run_diff_15"),
        "away_run_diff_30": _resolve(a, "run_diff_30"),
        "away_run_diff_60": _resolve(a, "run_diff_60"),
        "away_pythag_7": _resolve(a, "pythag_15"),
        "away_pythag_14": _resolve(a, "pythag_15"),
        "away_pythag_15": _resolve(a, "pythag_15"),
        "away_pythag_30": a_p30,
        "away_pythag_60": _resolve(a, "pythag_60"),
        # EWMA
        "home_win_pct_ewm": _resolve(h, "win_pct_ewm"),
        "away_win_pct_ewm": _resolve(a, "win_pct_ewm"),
        "home_run_diff_ewm": _resolve(h, "run_diff_ewm"),
        "away_run_diff_ewm": _resolve(a, "run_diff_ewm"),
        "home_pythag_ewm": h_pewm,
        "away_pythag_ewm": a_pewm,
        # Home/away splits
        "home_win_pct_home_only": h_home,
        "home_pythag_home_only": _resolve(h, "pythag_home_only"),
        "away_win_pct_away_only": a_away,
        "away_pythag_away_only": _resolve(a, "pythag_away_only"),
        # Run distribution
        "home_run_std_30": _resolve(h, "run_std_30"),
        "away_run_std_30": _resolve(a, "run_std_30"),
        "home_one_run_win_pct_30": _resolve(h, "one_run_win_pct_30"),
        "away_one_run_win_pct_30": _resolve(a, "one_run_win_pct_30"),
        # Streak and rest
        "home_streak": _resolve(h, "streak"),
        "away_streak": _resolve(a, "streak"),
        "home_rest_days": _OFF_SEASON_REST,
        "away_rest_days": _OFF_SEASON_REST,
        # Pitcher stats
        "home_sp_era": _resolve(h, "sp_era"),
        "away_sp_era": _resolve(a, "sp_era"),
        "home_sp_k9": _resolve(h, "sp_k9"),
        "away_sp_k9": _resolve(a, "sp_k9"),
        "home_sp_bb9": _resolve(h, "sp_bb9"),
        "away_sp_bb9": _resolve(a, "sp_bb9"),
        "home_sp_whip": h_sp_whip,
        "away_sp_whip": a_sp_whip,
        # FanGraphs batting
        "home_bat_woba": _resolve(h, "bat_woba"),
        "away_bat_woba": _resolve(a, "bat_woba"),
        "home_bat_barrel_pct": _resolve(h, "bat_barrel_pct"),
        "away_bat_barrel_pct": _resolve(a, "bat_barrel_pct"),
        "home_bat_hard_pct": _resolve(h, "bat_hard_pct"),
        "away_bat_hard_pct": _resolve(a, "bat_hard_pct"),
        "home_bat_iso": h_bat_iso,
        "away_bat_iso": a_bat_iso,
        "home_bat_babip": _resolve(h, "bat_babip"),
        "away_bat_babip": _resolve(a, "bat_babip"),
        "home_bat_xwoba": h_bat_xwoba,
        "away_bat_xwoba": a_bat_xwoba,
        # FanGraphs pitching
        "home_pit_fip": _resolve(h, "pit_fip"),
        "away_pit_fip": _resolve(a, "pit_fip"),
        "home_pit_xfip": _resolve(h, "pit_xfip"),
        "away_pit_xfip": _resolve(a, "pit_xfip"),
        "home_pit_k_pct": _resolve(h, "pit_k_pct"),
        "away_pit_k_pct": _resolve(a, "pit_k_pct"),
        "home_pit_bb_pct": _resolve(h, "pit_bb_pct"),
        "away_pit_bb_pct": _resolve(a, "pit_bb_pct"),
        "home_pit_hr_fb": _resolve(h, "pit_hr_fb"),
        "away_pit_hr_fb": _resolve(a, "pit_hr_fb"),
        "home_pit_whip": h_pit_whip,
        "away_pit_whip": a_pit_whip,
        # Differentials
        "pythag_diff_30": h_p30 - a_p30,
        "pythag_diff_ewm": h_pewm - a_pewm,
        "home_away_split_diff": h_home - a_away,
        "sp_era_diff": _resolve(a, "sp_era") - _resolve(h, "sp_era"),
        "woba_diff": _resolve(h, "bat_woba") - _resolve(a, "bat_woba"),
        "fip_diff": _resolve(a, "pit_fip") - _resolve(h, "pit_fip"),
        "xwoba_diff": h_bat_xwoba - a_bat_xwoba,
        "whip_diff": a_pit_whip - h_pit_whip,
        "iso_diff": h_bat_iso - a_bat_iso,
        # Lineup
        "home_lineup_continuity": 4.5,
        "away_lineup_continuity": 4.5,
        "home_lineup_xwoba": h_bat_xwoba,
        "away_lineup_xwoba": a_bat_xwoba,
        "home_lineup_barrel_pct": _resolve(h, "bat_barrel_pct"),
        "away_lineup_barrel_pct": _resolve(a, "bat_barrel_pct"),
        # Statcast pitcher
        "home_sp_est_woba": _resolve(h, "sp_est_woba"),
        "away_sp_est_woba": _resolve(a, "sp_est_woba"),
        # Bullpen (off-season defaults)
        "home_bullpen_usage_15": 2.0,
        "home_bullpen_usage_30": 2.0,
        "away_bullpen_usage_15": 2.0,
        "away_bullpen_usage_30": 2.0,
        "home_bullpen_era_proxy_15": 4.5,
        "home_bullpen_era_proxy_30": 4.5,
        "away_bullpen_era_proxy_15": 4.5,
        "away_bullpen_era_proxy_30": 4.5,
        # Park and context
        "park_run_factor": park_factors.get(h_code, 1.0),
        "season_progress": idx / max(n_games - 1, 1),
        "day_night": 1.0,
        "interleague": 0.0,
        "day_of_week": game_date.weekday() / 6.0 if not pd.isna(game_date) else 0.5,
        # Vegas / weather (generally unavailable for spring training)
        "vegas_implied_home_win": 0.5,
        "vegas_line_movement": 0.0,
        "game_temp_f": 72.0,
        "game_wind_mph": 5.0,
        "game_humidity": 50.0,
    }
    row["feature_hash"] = _feature_hash(row)
    return row


def build_spring_features_for_season(
    season: int,
    features_all: pd.DataFrame,
    team_state: dict[str, dict[str, float]],
    park_factors: dict[str, float],
    mlb_to_retro: dict[int, str],
    processed_dir: Path | None = None,
) -> pd.DataFrame | None:
    """Build spring training features for a single season.

    Returns None if no completed spring training games are found.
    """
    base = processed_dir or _PROCESSED
    sched_path = base / "schedule" / f"games_{season}.parquet"
    if not sched_path.exists():
        log.info("  %d: schedule not found — skipping", season)
        return None

    sched = pd.read_parquet(sched_path)
    if "game_type" not in sched.columns:
        log.info("  %d: no game_type column in schedule — skipping", season)
        return None

    spring = sched[sched["game_type"] == "S"].copy()
    if spring.empty:
        log.info("  %d: no spring training games in schedule", season)
        return None

    completed = spring[spring["status"].isin(_PLAYED_STATUSES)].copy()
    if completed.empty:
        log.info("  %d: no completed spring training games", season)
        return None

    has_scores = completed.dropna(subset=["home_score", "away_score"])
    if has_scores.empty:
        log.info("  %d: completed spring games but no scores in schedule data", season)
        return None

    rows: list[dict] = []
    n_games = len(has_scores)
    for idx, (_, g) in enumerate(has_scores.iterrows()):
        row = _build_game_row(
            g,
            season=season,
            idx=idx,
            n_games=n_games,
            team_state=team_state,
            park_factors=park_factors,
            mlb_to_retro=mlb_to_retro,
        )
        if row is not None:
            rows.append(row)

    if not rows:
        return None

    return pd.DataFrame(rows)


def main() -> None:
    """Build spring training features for all requested seasons."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", type=int, default=[])
    ap.add_argument(
        "--processed-dir",
        type=Path,
        default=_PROCESSED,
        help="Root of the processed data directory",
    )
    args = ap.parse_args()

    current_year = datetime.now(timezone.utc).year
    seasons = args.seasons or list(range(2000, current_year + 1))
    out_dir = args.processed_dir / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_files = sorted(
        f
        for f in (args.processed_dir / "features").glob("features_*.parquet")
        if "spring" not in f.stem
    )
    if not feature_files:
        log.warning("No historical feature files found — spring features need regular-season data")
        return

    log.info("Loading %d historical feature files for team state…", len(feature_files))
    features_all = pd.concat([pd.read_parquet(f) for f in feature_files], ignore_index=True)

    team_state_by_season: dict[int, dict[str, dict[str, float]]] = {}
    park_factors_by_season: dict[int, dict[str, float]] = {}

    mlb_to_retro = _build_mlb_to_retro(processed_dir=args.processed_dir)

    results = []
    for season in seasons:
        prior_season = season - 1
        if prior_season not in team_state_by_season:
            prior_data = features_all[features_all["season"] <= prior_season]
            if prior_data.empty:
                log.info("  %d: no prior-season features — skipping", season)
                results.append({"season": season, "status": "skipped", "n_games": 0})
                continue
            team_state_by_season[prior_season] = _build_team_state(prior_data)
            park_factors_by_season[prior_season] = _park_factor_by_home_team(prior_data)

        team_state = team_state_by_season[prior_season]
        park_factors = park_factors_by_season[prior_season]

        df = build_spring_features_for_season(
            season,
            features_all,
            team_state,
            park_factors,
            mlb_to_retro,
            processed_dir=args.processed_dir,
        )

        if df is None or df.empty:
            results.append({"season": season, "status": "no_spring_games", "n_games": 0})
            continue

        out_path = out_dir / f"features_spring_{season}.parquet"
        df.to_parquet(out_path, index=False)
        n_with_outcome = int(df["home_win"].notna().sum())
        log.info("  %d: %d spring games with outcomes → %s", season, n_with_outcome, out_path)
        results.append(
            {"season": season, "status": "ok", "n_games": len(df), "n_with_outcome": n_with_outcome}
        )

    summary_path = out_dir / "build_spring_features_summary.json"
    pd.DataFrame(results).to_json(summary_path, orient="records", indent=2)
    log.info("\nSummary → %s", summary_path)


if __name__ == "__main__":
    main()
