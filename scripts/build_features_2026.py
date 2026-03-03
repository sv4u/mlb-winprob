"""Build pre-season 2026 feature matrix from 2025 end-of-season team state.

Because the 2026 regular season has not yet started there are no Retrosheet
gamelogs to drive the normal pipeline.  Instead this script:

1. Loads all historical feature files (2000–2025) to read each team's
   end-of-season Elo rating, rolling stats, pitcher stats, and FanGraphs metrics.
2. Reads the already-ingested 2026 MLB schedule.
3. Maps MLB team IDs → Retrosheet codes using the team_id_map crosswalk.
4. Attaches each team's end-of-2025 state to the scheduled game rows.
5. Computes differential features and season_progress.
6. Saves data/processed/features/features_2026.parquet.

Re-run this script as the 2026 season progresses and Retrosheet-linked features
become available from build_features.py, which will overwrite this file.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_PROCESSED = Path("data/processed")
_SCHEDULE = _PROCESSED / "schedule" / "games_2026.parquet"
_TEAM_MAP = _PROCESSED / "team_id_map_retro_to_mlb.csv"
_FEATURES = _PROCESSED / "features"
_OUT = _FEATURES / "features_2026.parquet"

# League-average fall-backs for teams with missing data
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

# How many rest days to assign for the off-season gap
_OFF_SEASON_REST: float = 5.0

# Map of feature column names → neutral key (strips home_/away_ prefix)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_mlb_to_retro() -> dict[int, str]:
    """MLB team ID → Retrosheet code, preferring the most recent valid entry."""
    tm = pd.read_csv(_TEAM_MAP)
    # Keep the row with the highest valid_to_season per team
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
    # Normalize date to datetime.date so mixed-type columns (str vs date) sort correctly
    features = features.copy()
    features["date"] = pd.to_datetime(features["date"], errors="coerce").dt.date
    for team, grp in features.sort_values("date").groupby(team_col):
        last = grp.iloc[-1]
        result[str(team)] = {neutral: float(last[col]) for col, neutral in available.items()}
    return result


def _build_team_state(features_all: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Merge home- and away-role states into a single per-team dict."""
    home_states = _extract_state_from_role(features_all, "home_retro", _HOME_COL_MAP)
    away_states = _extract_state_from_role(features_all, "away_retro", _AWAY_COL_MAP)

    teams = set(home_states) | set(away_states)
    merged: dict[str, dict[str, float]] = {}
    for team in teams:
        h = home_states.get(team, {})
        a = away_states.get(team, {})
        state: dict[str, float] = {}
        # Prefer home-role for shared keys; fill gaps from away-role
        for key in set(h) | set(a):
            h_val = h.get(key, np.nan)
            a_val = a.get(key, np.nan)
            state[key] = h_val if not np.isnan(h_val) else a_val
        # Role-specific split stats — always from the correct perspective
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
    if key not in _DEF:
        raise KeyError(f"No league-average default defined for feature '{key}' in _DEF")
    return _DEF[key]


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


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_2026_features(out_path: Path = _OUT) -> pd.DataFrame:
    """Build and save features_2026.parquet. Returns the resulting DataFrame."""
    if not _SCHEDULE.exists():
        raise FileNotFoundError(
            f"2026 schedule not found at {_SCHEDULE}. Run ingest_schedule.py first."
        )

    # Explicitly exclude the 2026 file itself so team state is always derived
    # from historical (2000–2025) data — not from a previous pre-season estimate.
    feature_files = sorted(f for f in _FEATURES.glob("features_*.parquet") if "2026" not in f.stem)
    if not feature_files:
        raise RuntimeError("No historical feature files found. Run build_features.py first.")

    log.info("Loading %d historical feature files…", len(feature_files))
    features_all = pd.concat([pd.read_parquet(f) for f in feature_files], ignore_index=True)

    team_state = _build_team_state(features_all)
    park_factors = _park_factor_by_home_team(features_all)
    log.info("Built end-of-season state for %d teams", len(team_state))

    sched = pd.read_parquet(_SCHEDULE).sort_values("game_date_local").reset_index(drop=True)
    n_games = len(sched)
    log.info("Processing %d 2026 scheduled games…", n_games)

    mlb_to_retro = _build_mlb_to_retro()
    sched["home_retro"] = sched["home_mlb_id"].map(mlb_to_retro)
    sched["away_retro"] = sched["away_mlb_id"].map(mlb_to_retro)

    missing = sched[sched["home_retro"].isna() | sched["away_retro"].isna()]
    if not missing.empty:
        log.warning(
            "%d games have unmapped team IDs — they will use league-average features", len(missing)
        )

    rows: list[dict] = []
    for idx, g in sched.iterrows():
        h_code = g.get("home_retro") or ""
        a_code = g.get("away_retro") or ""
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

        row: dict = {
            "game_pk": int(g["game_pk"]),
            "date": pd.to_datetime(g["game_date_local"]).date(),
            "season": 2026,
            "home_mlb_id": int(g["home_mlb_id"]),
            "away_mlb_id": int(g["away_mlb_id"]),
            "home_retro": h_code,
            "away_retro": a_code,
            "home_win": np.nan,
            # Elo
            "home_elo": h_elo,
            "away_elo": a_elo,
            "elo_diff": h_elo - a_elo,
            # Multi-window rolling — home (7/14 share the 15-game value pre-season)
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
            # Home/away performance splits
            "home_win_pct_home_only": h_home,
            "home_pythag_home_only": _resolve(h, "pythag_home_only"),
            "away_win_pct_away_only": a_away,
            "away_pythag_away_only": _resolve(a, "pythag_away_only"),
            # Run distribution / 1-run games
            "home_run_std_30": _resolve(h, "run_std_30"),
            "away_run_std_30": _resolve(a, "run_std_30"),
            "home_one_run_win_pct_30": _resolve(h, "one_run_win_pct_30"),
            "away_one_run_win_pct_30": _resolve(a, "one_run_win_pct_30"),
            # Streak and rest
            "home_streak": _resolve(h, "streak"),
            "away_streak": _resolve(a, "streak"),
            "home_rest_days": _OFF_SEASON_REST,
            "away_rest_days": _OFF_SEASON_REST,
            # Pitcher stats (2025 season averages as pre-season prior)
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
            # Differential features
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
            # Statcast lineup-weighted (use team-level FG xwOBA as proxy)
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
            "season_progress": int(idx) / max(n_games - 1, 1),
            # Contextual (defaults for scheduled games)
            "day_night": 1.0,
            "interleague": 0.0,
            "day_of_week": pd.to_datetime(g["game_date_local"]).weekday() / 6.0,
            # Vegas / weather (unavailable pre-season)
            "vegas_implied_home_win": 0.5,
            "vegas_line_movement": 0.0,
            "game_temp_f": 72.0,
            "game_wind_mph": 5.0,
            "game_humidity": 50.0,
        }
        row["feature_hash"] = _feature_hash(row)
        rows.append(row)

    df = pd.DataFrame(rows)
    _FEATURES.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info("✓ Saved %d 2026 game features → %s", len(df), out_path)
    return df


if __name__ == "__main__":
    build_2026_features()
