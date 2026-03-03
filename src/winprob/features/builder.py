"""Feature matrix builder (v2) for the MLB win-probability model.

Feature set
-----------
* Elo ratings (home, away, diff)  — sequential, cross-season with regression-to-mean
* Multi-window rolling team stats (15 / 30 / 60 games, cross-season warm-start)
  win%, run-diff, Pythagorean expectation for home and away teams
* Streak and rest-days for each team
* Starting pitcher ERA/K9/BB9 — blend of prior-season MLB API stats and
  in-gamelog cumulative stats for the current season
* Park run factor (historical)
* Differential features — Elo diff, Pythagorean diff, SP ERA diff
* Season progress (0 = opener, 1 = last game)

Output columns (feature_hash covers all numeric features)
---------
game_pk, date, season, home_mlb_id, away_mlb_id, home_retro, away_retro,
home_win (target), <numeric features>, feature_hash
"""

from __future__ import annotations

import hashlib
import json
import logging
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from winprob.features.elo import compute_elo_ratings
from winprob.features.team_stats import build_team_rolling_stats
from winprob.features.park_factors import compute_park_factors, _NEUTRAL_FACTOR
from winprob.features.pitcher_stats import build_pitcher_stats
from winprob.features.lineup import build_lineup_features
from winprob.features.bullpen import build_bullpen_features
from winprob.external.vegas import build_vegas_features, load_vegas_season
from winprob.external.weather import build_weather_features, load_weather_season
from winprob.statcast.player_stats import (
    _LEAGUE_AVG_BARREL_PCT,
    _LEAGUE_AVG_PIT_EST_WOBA,
    _LEAGUE_AVG_XWOBA,
    _load_chadwick_register,
    _retro_to_mlbam_map,
    build_lineup_statcast_features,
    build_pitcher_statcast_features,
    get_batter_statcast_for_season,
    get_pitcher_statcast_for_season,
)

_LEAGUE_AVG_ERA: float = 4.50
_LEAGUE_AVG_K9: float = 8.5
_LEAGUE_AVG_BB9: float = 3.0
_LEAGUE_AVG_WHIP: float = 1.30
_SHRINKAGE_PRIOR_W: float = 0.30  # weight given to prior-season API stats (vs gamelog)
_MIN_STARTS_FOR_BLEND: int = 15  # in-season ERA weight ramps up over this many starts

FEATURE_COLS: list[str] = [
    # --- Elo ------------------------------------------------------------------
    "home_elo",
    "away_elo",
    "elo_diff",
    # --- Multi-window rolling -------------------------------------------------
    "home_win_pct_7",
    "home_win_pct_14",
    "home_win_pct_15",
    "home_win_pct_30",
    "home_win_pct_60",
    "away_win_pct_7",
    "away_win_pct_14",
    "away_win_pct_15",
    "away_win_pct_30",
    "away_win_pct_60",
    "home_run_diff_7",
    "home_run_diff_14",
    "home_run_diff_15",
    "home_run_diff_30",
    "home_run_diff_60",
    "away_run_diff_7",
    "away_run_diff_14",
    "away_run_diff_15",
    "away_run_diff_30",
    "away_run_diff_60",
    "home_pythag_7",
    "home_pythag_14",
    "home_pythag_15",
    "home_pythag_30",
    "home_pythag_60",
    "away_pythag_7",
    "away_pythag_14",
    "away_pythag_15",
    "away_pythag_30",
    "away_pythag_60",
    # --- EWMA rolling ---------------------------------------------------------
    "home_win_pct_ewm",
    "away_win_pct_ewm",
    "home_run_diff_ewm",
    "away_run_diff_ewm",
    "home_pythag_ewm",
    "away_pythag_ewm",
    # --- Home/away performance splits ----------------------------------------
    "home_win_pct_home_only",
    "home_pythag_home_only",
    "away_win_pct_away_only",
    "away_pythag_away_only",
    # --- Run distribution / 1-run games ---------------------------------------
    "home_run_std_30",
    "away_run_std_30",
    "home_one_run_win_pct_30",
    "away_one_run_win_pct_30",
    # --- Streak and rest ------------------------------------------------------
    "home_streak",
    "away_streak",
    "home_rest_days",
    "away_rest_days",
    # --- Pitcher stats --------------------------------------------------------
    "home_sp_era",
    "away_sp_era",
    "home_sp_k9",
    "away_sp_k9",
    "home_sp_bb9",
    "away_sp_bb9",
    "home_sp_whip",
    "away_sp_whip",
    # --- FanGraphs advanced metrics (prior season) ---------------------------
    "home_bat_woba",
    "away_bat_woba",
    "home_bat_barrel_pct",
    "away_bat_barrel_pct",
    "home_bat_hard_pct",
    "away_bat_hard_pct",
    "home_bat_iso",
    "away_bat_iso",
    "home_bat_babip",
    "away_bat_babip",
    "home_bat_xwoba",
    "away_bat_xwoba",
    "home_pit_fip",
    "away_pit_fip",
    "home_pit_xfip",
    "away_pit_xfip",
    "home_pit_k_pct",
    "away_pit_k_pct",
    "home_pit_bb_pct",
    "away_pit_bb_pct",
    "home_pit_hr_fb",
    "away_pit_hr_fb",
    "home_pit_whip",
    "away_pit_whip",
    # --- Differentials --------------------------------------------------------
    "pythag_diff_30",  # home_pythag_30 - away_pythag_30
    "pythag_diff_ewm",  # home_pythag_ewm - away_pythag_ewm
    "home_away_split_diff",  # home_win_pct_home_only - away_win_pct_away_only
    "sp_era_diff",  # away_sp_era - home_sp_era
    "woba_diff",  # home_bat_woba - away_bat_woba
    "fip_diff",  # away_pit_fip - home_pit_fip
    "xwoba_diff",  # home_bat_xwoba - away_bat_xwoba
    "whip_diff",  # away_pit_whip - home_pit_whip (lower home whip = home advantage)
    "iso_diff",  # home_bat_iso - away_bat_iso
    # --- Lineup ---------------------------------------------------------------
    "home_lineup_continuity",
    "away_lineup_continuity",
    # --- Statcast lineup-weighted (prior-season player xwOBA/barrel%) ----------
    "home_lineup_xwoba",
    "away_lineup_xwoba",
    "home_lineup_barrel_pct",
    "away_lineup_barrel_pct",
    # --- Statcast pitcher (prior-season xwOBA allowed) ------------------------
    "home_sp_est_woba",
    "away_sp_est_woba",
    # --- Bullpen --------------------------------------------------------------
    "home_bullpen_usage_15",
    "home_bullpen_usage_30",
    "away_bullpen_usage_15",
    "away_bullpen_usage_30",
    "home_bullpen_era_proxy_15",
    "home_bullpen_era_proxy_30",
    "away_bullpen_era_proxy_15",
    "away_bullpen_era_proxy_30",
    # --- Park / context -------------------------------------------------------
    "park_run_factor",
    "season_progress",
    # --- Contextual -----------------------------------------------------------
    "day_night",
    "interleague",
    "day_of_week",
    # --- Vegas (opening implied prob, line movement) --------------------------
    "vegas_implied_home_win",
    "vegas_line_movement",
    # --- Weather (game location) ----------------------------------------------
    "game_temp_f",
    "game_wind_mph",
    "game_humidity",
]

FEATURE_COLS = list(dict.fromkeys(FEATURE_COLS))


# ---------------------------------------------------------------------------
# Pitcher feature helpers
# ---------------------------------------------------------------------------


def _normalize_name(name: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_only = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(ascii_only.lower().split())


def _load_api_pitcher_map(pitcher_stats_dir: Path, season: int) -> dict[str, dict[str, float]]:
    """Load MLB API pitcher stats for *season* into a name→stats lookup dict.

    Returns
    -------
    dict
        Mapping normalized pitcher name → {era, k9, bb9, whip}.
    """
    path = pitcher_stats_dir / f"pitchers_{season}.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    result: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        key = _normalize_name(str(row["player_name"]))
        result[key] = {
            "era": float(row.get("era", _LEAGUE_AVG_ERA)),
            "k9": float(row.get("k9", _LEAGUE_AVG_K9)),
            "bb9": float(row.get("bb9", _LEAGUE_AVG_BB9)),
            "whip": float(row.get("whip", _LEAGUE_AVG_WHIP)),
        }
    return result


def _pitcher_api_features(
    gamelogs: pd.DataFrame,
    prior_api_map: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Look up prior-season API stats for each game's starting pitchers.

    The gamelog's ``home_starting_pitcher_name`` and
    ``visiting_starting_pitcher_name`` are matched against the normalized-name
    lookup.  Missing pitchers default to league-average values.
    """
    home_era, home_k9, home_bb9, home_whip = [], [], [], []
    away_era, away_k9, away_bb9, away_whip = [], [], [], []

    for _, row in gamelogs.iterrows():

        def _lookup(name_col: str) -> dict[str, float]:
            name = row.get(name_col, "")
            if not name or pd.isna(name):
                return {}
            return prior_api_map.get(_normalize_name(str(name)), {})

        h = _lookup("home_starting_pitcher_name")
        a = _lookup("visiting_starting_pitcher_name")

        home_era.append(h.get("era", _LEAGUE_AVG_ERA))
        home_k9.append(h.get("k9", _LEAGUE_AVG_K9))
        home_bb9.append(h.get("bb9", _LEAGUE_AVG_BB9))
        home_whip.append(h.get("whip", _LEAGUE_AVG_WHIP))
        away_era.append(a.get("era", _LEAGUE_AVG_ERA))
        away_k9.append(a.get("k9", _LEAGUE_AVG_K9))
        away_bb9.append(a.get("bb9", _LEAGUE_AVG_BB9))
        away_whip.append(a.get("whip", _LEAGUE_AVG_WHIP))

    return pd.DataFrame(
        {
            "home_sp_era": home_era,
            "home_sp_k9": home_k9,
            "home_sp_bb9": home_bb9,
            "home_sp_whip": home_whip,
            "away_sp_era": away_era,
            "away_sp_k9": away_k9,
            "away_sp_bb9": away_bb9,
            "away_sp_whip": away_whip,
        },
        index=gamelogs.index,
    )


# ---------------------------------------------------------------------------
# Feature hash
# ---------------------------------------------------------------------------


def _hash_feature_row(row: pd.Series) -> str:
    vals = {c: (float(row[c]) if pd.notna(row.get(c)) else None) for c in FEATURE_COLS}
    return hashlib.sha256(json.dumps(vals, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def _fangraphs_features(
    gamelogs: pd.DataFrame,
    fg_home_map: dict[str, dict[str, float]],
    fg_away_map: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Add FanGraphs team advanced metrics for home and away teams."""
    _FG_KEYS = [
        "bat_woba",
        "bat_barrel_pct",
        "bat_hard_pct",
        "bat_iso",
        "bat_babip",
        "bat_xwoba",
        "pit_fip",
        "pit_xfip",
        "pit_k_pct",
        "pit_bb_pct",
        "pit_hr_fb",
        "pit_whip",
    ]
    _DEFAULTS = {
        "bat_woba": 0.320,
        "bat_barrel_pct": 0.08,
        "bat_hard_pct": 0.38,
        "bat_iso": 0.170,
        "bat_babip": 0.300,
        "bat_xwoba": 0.320,
        "pit_fip": 4.20,
        "pit_xfip": 4.20,
        "pit_k_pct": 0.22,
        "pit_bb_pct": 0.085,
        "pit_hr_fb": 0.11,
        "pit_whip": 1.30,
    }

    rows: dict[str, list] = {f"home_{k}": [] for k in _FG_KEYS}
    rows.update({f"away_{k}": [] for k in _FG_KEYS})

    for _, row in gamelogs.iterrows():
        hstats = fg_home_map.get(str(row["home_team"]), {})
        astats = fg_away_map.get(str(row["visiting_team"]), {})
        for k in _FG_KEYS:
            rows[f"home_{k}"].append(hstats.get(k, _DEFAULTS[k]))
            rows[f"away_{k}"].append(astats.get(k, _DEFAULTS[k]))

    return pd.DataFrame(rows, index=gamelogs.index)


def build_feature_matrix(
    *,
    season: int,
    gamelogs_season: pd.DataFrame,
    gamelogs_all: pd.DataFrame,
    crosswalk: pd.DataFrame,
    park_factors: dict[str, float],
    prior_api_map: dict[str, dict[str, float]],
    fg_home_map: dict[str, dict[str, float]] | None = None,
    fg_away_map: dict[str, dict[str, float]] | None = None,
    statcast_cache_dir: Path | None = None,
    vegas_dir: Path | None = None,
    weather_dir: Path | None = None,
) -> pd.DataFrame:
    """Assemble the v2 feature matrix for one season.

    Parameters
    ----------
    season:
        Target season year.
    gamelogs_season:
        Retrosheet gamelogs for *season* only (for within-season indexing).
    gamelogs_all:
        All gamelogs across all seasons (for cross-season Elo and rolling stats).
    crosswalk:
        Crosswalk parquet for *season*.
    park_factors:
        Pre-computed park run factors.
    prior_api_map:
        MLB API pitcher stats from the previous season
        (name → {era, k9, bb9}).
    """
    gl = gamelogs_season.reset_index(drop=True)

    # --- Identify season rows within gamelogs_all ---------------------------
    # gamelogs_all is already sorted and 0-indexed from load_all_gamelogs().
    gl_dates = pd.to_datetime(gamelogs_all["date"])
    season_mask = (gl_dates.dt.year == season).values  # numpy bool array

    # --- Elo ratings (computed from all seasons) ----------------------------
    elo_all = compute_elo_ratings(gamelogs_all).sort_index()
    elo_season = elo_all.iloc[season_mask].reset_index(drop=True)

    # --- Multi-window rolling stats (cross-season) --------------------------
    team_feats = build_team_rolling_stats(gamelogs_all).sort_index()
    team_feats_season = team_feats.iloc[season_mask].reset_index(drop=True)

    # --- Lineup continuity (optional: requires batting order columns) --------
    lineup_cols = [f"home_{i}_id" for i in range(1, 10)] + [
        f"visiting_{i}_id" for i in range(1, 10)
    ]
    if all(c in gamelogs_all.columns for c in lineup_cols):
        try:
            lineup_all = build_lineup_features(gamelogs_all).sort_index()
            lineup_season = lineup_all.iloc[season_mask].reset_index(drop=True)
        except Exception as exc:
            logger.warning("Lineup feature computation failed for season %d: %s", season, exc)
            lineup_season = None
    else:
        lineup_season = None

    # --- Prior-season API pitcher stats -------------------------------------
    sp_feats = _pitcher_api_features(gl, prior_api_map)

    # --- In-season gamelog pitcher ERA (Bayesian shrinkage) + blend with API -
    required_er_cols = [
        "home_er",
        "visiting_er",
        "home_starting_pitcher_id",
        "visiting_starting_pitcher_id",
    ]
    if all(c in gamelogs_all.columns for c in required_er_cols):
        try:
            pitcher_gamelog_all = build_pitcher_stats(gamelogs_all).sort_index()
            pitcher_season = pitcher_gamelog_all.iloc[season_mask].reset_index(drop=True)
            n_starts_h = np.minimum(
                pitcher_season["home_sp_n_starts"].values.astype(float), _MIN_STARTS_FOR_BLEND
            )
            n_starts_a = np.minimum(
                pitcher_season["away_sp_n_starts"].values.astype(float), _MIN_STARTS_FOR_BLEND
            )
            w_home = (1.0 - _SHRINKAGE_PRIOR_W) * (n_starts_h / _MIN_STARTS_FOR_BLEND)
            w_away = (1.0 - _SHRINKAGE_PRIOR_W) * (n_starts_a / _MIN_STARTS_FOR_BLEND)
            sp_feats["home_sp_era"] = (1.0 - w_home) * sp_feats[
                "home_sp_era"
            ] + w_home * pitcher_season["home_sp_era"].values
            sp_feats["away_sp_era"] = (1.0 - w_away) * sp_feats[
                "away_sp_era"
            ] + w_away * pitcher_season["away_sp_era"].values
        except Exception as exc:
            logger.warning(
                "In-season pitcher ERA blend failed for season %d, "
                "using API-only stats: %s", season, exc
            )

    # --- FanGraphs advanced metrics (prior season) --------------------------
    fg_feats = _fangraphs_features(gl, fg_home_map or {}, fg_away_map or {})

    # --- Outcome (home_win) -------------------------------------------------
    home_score = pd.to_numeric(gl["home_score"], errors="coerce")
    away_score = pd.to_numeric(gl["visiting_score"], errors="coerce")
    home_win = np.where(
        home_score.notna() & away_score.notna(),
        (home_score > away_score).astype(float),
        np.nan,
    )

    # --- Park factor lookup -------------------------------------------------
    park_run_factor = gl["park_id"].astype(str).map(park_factors).fillna(_NEUTRAL_FACTOR)

    # --- Season progress (0 → 1 over the season) ----------------------------
    dates = pd.to_datetime(gl["date"])
    if len(dates) > 1:
        d_min, d_max = dates.min(), dates.max()
        span = (d_max - d_min).days or 1
        season_progress = (dates - d_min).dt.days / span
    else:
        season_progress = pd.Series([0.5] * len(gl), index=gl.index)

    # --- Combine everything -------------------------------------------------
    combined = pd.DataFrame(index=gl.index)
    combined["date"] = pd.to_datetime(gl["date"]).dt.date
    combined["home_team"] = gl["home_team"].values
    combined["visiting_team"] = gl["visiting_team"].values
    combined["game_num"] = pd.to_numeric(gl["game_num"], errors="coerce").fillna(0).astype(int)
    combined["home_win"] = home_win
    combined["park_run_factor"] = park_run_factor.values
    combined["season_progress"] = season_progress.values

    # --- Contextual: day/night, interleague, day of week ---------------------
    day_night_raw = gl.get("day_night", pd.Series("D", index=gl.index))
    combined["day_night"] = (day_night_raw.astype(str).str.upper() == "N").astype(float)
    if "home_team_league" in gl.columns and "visiting_team_league" in gl.columns:
        combined["interleague"] = (
            gl["home_team_league"].astype(str) != gl["visiting_team_league"].astype(str)
        ).astype(float)
    else:
        combined["interleague"] = 0.0
    combined["day_of_week"] = pd.to_datetime(gl["date"]).dt.dayofweek.values.astype(float) / 6.0

    for col in elo_season.columns:
        combined[col] = elo_season[col].values

    for col in team_feats_season.columns:
        combined[col] = team_feats_season[col].values

    if lineup_season is not None:
        for col in lineup_season.columns:
            combined[col] = lineup_season[col].values
    else:
        combined["home_lineup_continuity"] = 4.5
        combined["away_lineup_continuity"] = 4.5

    # --- Bullpen usage / ERA proxy (optional) ---------------------------------
    if (
        "home_pitchers_used" in gamelogs_all.columns
        and "visiting_pitchers_used" in gamelogs_all.columns
    ):
        try:
            bullpen_all = build_bullpen_features(gamelogs_all).sort_index()
            bullpen_season = bullpen_all.iloc[season_mask].reset_index(drop=True)
            for col in bullpen_season.columns:
                combined[col] = bullpen_season[col].values
        except Exception as exc:
            logger.warning("Bullpen feature computation failed for season %d: %s", season, exc)
            for c in [
                "home_bullpen_usage_15",
                "home_bullpen_usage_30",
                "away_bullpen_usage_15",
                "away_bullpen_usage_30",
                "home_bullpen_era_proxy_15",
                "home_bullpen_era_proxy_30",
                "away_bullpen_era_proxy_15",
                "away_bullpen_era_proxy_30",
            ]:
                combined[c] = 2.0 if "usage" in c else 4.5
    else:
        for c in [
            "home_bullpen_usage_15",
            "home_bullpen_usage_30",
            "away_bullpen_usage_15",
            "away_bullpen_usage_30",
        ]:
            combined[c] = 2.0
        for c in [
            "home_bullpen_era_proxy_15",
            "home_bullpen_era_proxy_30",
            "away_bullpen_era_proxy_15",
            "away_bullpen_era_proxy_30",
        ]:
            combined[c] = 4.5

    for col in sp_feats.columns:
        combined[col] = sp_feats[col].values

    for col in fg_feats.columns:
        combined[col] = fg_feats[col].values

    # --- Statcast lineup-weighted and pitcher (prior-season player-level) ----
    lineup_cols = [f"home_{i}_id" for i in range(1, 10)] + [
        f"visiting_{i}_id" for i in range(1, 10)
    ]
    prior_season = season - 1
    cache_dir = statcast_cache_dir or Path("data/processed/statcast_player")
    if all(c in gl.columns for c in lineup_cols) and all(
        c in gl.columns for c in ["home_starting_pitcher_id", "visiting_starting_pitcher_id"]
    ):
        try:
            reg = _load_chadwick_register()
            retro_to_mlbam = _retro_to_mlbam_map(reg)
            batter_stats = get_batter_statcast_for_season(prior_season, cache_dir)
            pitcher_stats = get_pitcher_statcast_for_season(prior_season, cache_dir)
            lineup_sc = build_lineup_statcast_features(
                gl, prior_season, batter_stats, retro_to_mlbam
            )
            pit_sc = build_pitcher_statcast_features(
                gl, prior_season, pitcher_stats, retro_to_mlbam
            )
            for col in lineup_sc.columns:
                combined[col] = lineup_sc[col].values
            for col in pit_sc.columns:
                combined[col] = pit_sc[col].values
        except Exception as exc:
            logger.warning("Statcast feature computation failed for season %d: %s", season, exc)
            combined["home_lineup_xwoba"] = combined.get("home_bat_xwoba", _LEAGUE_AVG_XWOBA)
            combined["away_lineup_xwoba"] = combined.get("away_bat_xwoba", _LEAGUE_AVG_XWOBA)
            combined["home_lineup_barrel_pct"] = combined.get(
                "home_bat_barrel_pct", _LEAGUE_AVG_BARREL_PCT
            )
            combined["away_lineup_barrel_pct"] = combined.get(
                "away_bat_barrel_pct", _LEAGUE_AVG_BARREL_PCT
            )
            combined["home_sp_est_woba"] = _LEAGUE_AVG_PIT_EST_WOBA
            combined["away_sp_est_woba"] = _LEAGUE_AVG_PIT_EST_WOBA
    else:
        combined["home_lineup_xwoba"] = combined.get("home_bat_xwoba", _LEAGUE_AVG_XWOBA)
        combined["away_lineup_xwoba"] = combined.get("away_bat_xwoba", _LEAGUE_AVG_XWOBA)
        combined["home_lineup_barrel_pct"] = combined.get(
            "home_bat_barrel_pct", _LEAGUE_AVG_BARREL_PCT
        )
        combined["away_lineup_barrel_pct"] = combined.get(
            "away_bat_barrel_pct", _LEAGUE_AVG_BARREL_PCT
        )
        combined["home_sp_est_woba"] = _LEAGUE_AVG_PIT_EST_WOBA
        combined["away_sp_est_woba"] = _LEAGUE_AVG_PIT_EST_WOBA

    # --- Vegas opening line (and line movement) ----------------------------
    v_dir = vegas_dir or Path("data/processed/vegas")
    vegas_df = load_vegas_season(v_dir, season)
    vegas_feats = build_vegas_features(gl, vegas_df)
    for col in vegas_feats.columns:
        combined[col] = vegas_feats[col].values

    # --- Weather (temp, wind, humidity at game location) ----------------------
    w_dir = weather_dir or Path("data/processed/weather")
    weather_df = load_weather_season(w_dir, season)
    weather_feats = build_weather_features(gl, weather_df)
    for col in weather_feats.columns:
        combined[col] = weather_feats[col].values

    # --- Differential features ----------------------------------------------
    combined["pythag_diff_30"] = combined["home_pythag_30"] - combined["away_pythag_30"]
    combined["pythag_diff_ewm"] = combined.get("home_pythag_ewm", 0.5) - combined.get(
        "away_pythag_ewm", 0.5
    )
    combined["home_away_split_diff"] = combined.get("home_win_pct_home_only", 0.5) - combined.get(
        "away_win_pct_away_only", 0.5
    )
    combined["sp_era_diff"] = combined["away_sp_era"] - combined["home_sp_era"]
    combined["woba_diff"] = combined.get("home_bat_woba", 0.320) - combined.get(
        "away_bat_woba", 0.320
    )
    combined["fip_diff"] = combined.get("away_pit_fip", 4.20) - combined.get("home_pit_fip", 4.20)
    combined["xwoba_diff"] = combined.get("home_bat_xwoba", 0.320) - combined.get(
        "away_bat_xwoba", 0.320
    )
    combined["whip_diff"] = combined.get("away_pit_whip", 1.30) - combined.get(
        "home_pit_whip", 1.30
    )
    combined["iso_diff"] = combined.get("home_bat_iso", 0.170) - combined.get("away_bat_iso", 0.170)

    # --- Join crosswalk to get game_pk and MLB team IDs ---------------------
    cw_matched = crosswalk[crosswalk["status"] == "matched"][
        [
            "date",
            "home_retro",
            "away_retro",
            "dh_game_num",
            "home_mlb_id",
            "away_mlb_id",
            "mlb_game_pk",
        ]
    ].copy()
    cw_matched["date"] = pd.to_datetime(cw_matched["date"]).dt.date
    cw_matched["dh_game_num"] = (
        pd.to_numeric(cw_matched["dh_game_num"], errors="coerce").fillna(0).astype(int)
    )

    merged = combined.merge(
        cw_matched,
        left_on=["date", "home_team", "visiting_team", "game_num"],
        right_on=["date", "home_retro", "away_retro", "dh_game_num"],
        how="inner",
    )
    merged["season"] = season
    merged = merged.rename(columns={"mlb_game_pk": "game_pk"})

    # --- Feature hash -------------------------------------------------------
    merged["feature_hash"] = merged.apply(_hash_feature_row, axis=1)

    output_cols = (
        [
            "game_pk",
            "date",
            "season",
            "home_mlb_id",
            "away_mlb_id",
            "home_retro",
            "away_retro",
            "home_win",
        ]
        + FEATURE_COLS
        + ["feature_hash"]
    )
    for c in output_cols:
        if c not in merged.columns:
            merged[c] = np.nan

    return merged[output_cols].sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Utility loaders
# ---------------------------------------------------------------------------


def load_or_build_park_factors(
    processed_dir: Path = Path("data/processed"),
    exclude_season: int | None = None,
) -> dict[str, float]:
    """Load all gamelogs and return park run factors."""
    retro_dir = processed_dir / "retrosheet"
    frames: list[pd.DataFrame] = []
    for f in sorted(retro_dir.glob("gamelogs_*.parquet")):
        s = int(f.stem.split("_")[1])
        if exclude_season is not None and s >= exclude_season:
            continue
        frames.append(pd.read_parquet(f, columns=["park_id", "home_score", "visiting_score"]))
    if not frames:
        return {}
    return compute_park_factors(pd.concat(frames, ignore_index=True))


def load_all_gamelogs(processed_dir: Path = Path("data/processed")) -> pd.DataFrame:
    """Concatenate all available Retrosheet gamelogs, sorted chronologically."""
    frames = []
    for f in sorted((processed_dir / "retrosheet").glob("gamelogs_*.parquet")):
        frames.append(pd.read_parquet(f))
    if not frames:
        raise RuntimeError("No gamelog parquets found")
    gl = pd.concat(frames, ignore_index=True)
    gl["date"] = pd.to_datetime(gl["date"])
    gl["game_num"] = pd.to_numeric(gl["game_num"], errors="coerce").fillna(0).astype(int)
    return gl.sort_values(["date", "game_num"]).reset_index(drop=True)
