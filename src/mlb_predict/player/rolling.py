"""Per-player EWMA rolling stats computed from Retrosheet gamelogs.

Batter stats (EWMA span=20 games):
  OPS, ISO, K%, BB% are derived from gamelog columns (H, 2B, 3B, HR, SO, BB, AB).
  Statcast/FanGraphs stats (xwOBA, barrel%, hard_hit%, wRC+, sprint_speed) are
  looked up from prior-season cached player data and used as static priors that
  blend with rolling gamelog stats.

Pitcher stats (EWMA span=5 starts):
  ERA is derived from gamelog ER/IP columns.  K/9, BB/9, WHIP from gamelogs.
  FIP, xwOBA allowed, swinging strike% from prior-season cached data.

Cross-season warm-start: end-of-previous-season EWMA state seeds the first
window of the next season (same approach as team-level rolling in team_stats.py).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_BATTER_EWMA_SPAN: int = 20
_PITCHER_EWMA_SPAN: int = 5


# ---------------------------------------------------------------------------
# League-average fallback constants
# ---------------------------------------------------------------------------

_AVG_OPS: float = 0.728
_AVG_ISO: float = 0.150
_AVG_K_PCT: float = 0.220
_AVG_BB_PCT: float = 0.085
_AVG_XWOBA: float = 0.320
_AVG_BARREL_PCT: float = 0.080
_AVG_HARD_HIT_PCT: float = 0.380
_AVG_WRC_PLUS: float = 100.0
_AVG_SPRINT_SPEED: float = 27.0

_AVG_ERA: float = 4.50
_AVG_FIP: float = 4.20
_AVG_K9: float = 8.5
_AVG_BB9: float = 3.0
_AVG_WHIP: float = 1.30
_AVG_PIT_XWOBA: float = 0.320
_AVG_SWSTR_PCT: float = 0.110


# ---------------------------------------------------------------------------
# Batter rolling stats from gamelogs
# ---------------------------------------------------------------------------

_GL_HOME_BAT_COLS = {
    "home_{i}_id": "player_id",
}

_HOME_LINEUP_COLS = [f"home_{i}_id" for i in range(1, 10)]
_AWAY_LINEUP_COLS = [f"visiting_{i}_id" for i in range(1, 10)]


def _compute_batter_game_stats(gamelogs: pd.DataFrame) -> pd.DataFrame:
    """Extract per-batter per-game performance from Retrosheet gamelogs.

    Retrosheet gamelogs provide team-level box-score totals, not individual
    player lines. We approximate per-batter stats by distributing team totals
    evenly across the 9 starters and tracking which player IDs appeared.

    For accurate per-batter rolling stats, we rely on the player appearing in
    the lineup repeatedly and EWMA smoothing out the approximation noise.

    Returns DataFrame with columns: date, player_id, team, is_home,
    batting_order, approx_pa, approx_h, approx_2b, approx_3b, approx_hr,
    approx_bb, approx_so, approx_ab.
    """
    rows: list[dict] = []
    gl = gamelogs.copy()
    if gl.empty or "date" not in gl.columns:
        return pd.DataFrame()
    gl["date"] = pd.to_datetime(gl["date"])

    for idx, row in gl.iterrows():
        game_date = row["date"]

        for side, id_cols, team_col, score_prefix in [
            ("home", _HOME_LINEUP_COLS, "home_team", "home_"),
            ("away", _AWAY_LINEUP_COLS, "visiting_team", "visiting_"),
        ]:
            team_ab = _safe_float(
                row.get(
                    f"{score_prefix}abs",
                    row.get(f"{score_prefix}at_bats", row.get(f"{score_prefix}ab")),
                )
            )
            team_h = _safe_float(row.get(f"{score_prefix}hits", row.get(f"{score_prefix}h")))
            team_2b = _safe_float(row.get(f"{score_prefix}doubles", row.get(f"{score_prefix}2b")))
            team_3b = _safe_float(row.get(f"{score_prefix}triples", row.get(f"{score_prefix}3b")))
            team_hr = _safe_float(row.get(f"{score_prefix}homeruns", row.get(f"{score_prefix}hr")))
            team_bb = _safe_float(row.get(f"{score_prefix}bb", row.get(f"{score_prefix}walks")))
            team_so = _safe_float(
                row.get(
                    f"{score_prefix}k",
                    row.get(f"{score_prefix}strikeouts", row.get(f"{score_prefix}so")),
                )
            )

            if np.isnan(team_ab) or team_ab == 0:
                continue

            starters = []
            for i, col in enumerate(id_cols):
                pid = row.get(col)
                if pd.notna(pid) and str(pid).strip():
                    starters.append((str(pid).strip(), i + 1))

            n = len(starters) or 9
            for pid, order in starters:
                rows.append(
                    {
                        "date": game_date,
                        "player_id": pid,
                        "team": str(row.get(team_col, "")),
                        "is_home": side == "home",
                        "batting_order": order,
                        "approx_ab": team_ab / n,
                        "approx_h": team_h / n,
                        "approx_2b": team_2b / n,
                        "approx_3b": team_3b / n,
                        "approx_hr": team_hr / n,
                        "approx_bb": team_bb / n,
                        "approx_so": team_so / n,
                    }
                )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["approx_pa"] = df["approx_ab"] + df["approx_bb"]

    slg_num = (
        (df["approx_h"] - df["approx_2b"] - df["approx_3b"] - df["approx_hr"])
        + 2 * df["approx_2b"]
        + 3 * df["approx_3b"]
        + 4 * df["approx_hr"]
    )
    df["approx_obp"] = (df["approx_h"] + df["approx_bb"]) / df["approx_pa"].clip(lower=1)
    df["approx_slg"] = slg_num / df["approx_ab"].clip(lower=1)
    df["approx_ops"] = df["approx_obp"] + df["approx_slg"]
    df["approx_iso"] = df["approx_slg"] - (df["approx_h"] / df["approx_ab"].clip(lower=1))
    df["approx_k_pct"] = df["approx_so"] / df["approx_pa"].clip(lower=1)
    df["approx_bb_pct"] = df["approx_bb"] / df["approx_pa"].clip(lower=1)

    return df.sort_values("date").reset_index(drop=True)


def build_batter_rolling(
    gamelogs: pd.DataFrame,
    prior_batter_stats: pd.DataFrame | None = None,
    retro_to_mlbam: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Compute per-batter EWMA rolling stats from gamelogs.

    Returns DataFrame indexed by (player_id, game_index) with EWMA columns:
    ops_ewm, iso_ewm, k_pct_ewm, bb_pct_ewm.

    If prior_batter_stats is provided, xwoba_ewm, barrel_pct_ewm,
    hard_hit_pct_ewm, wrc_plus_ewm are also included (static priors from
    prior season, not rolled).
    """
    game_stats = _compute_batter_game_stats(gamelogs)
    if game_stats.empty:
        return pd.DataFrame()

    ewma_cols = {
        "approx_ops": "ops_ewm",
        "approx_iso": "iso_ewm",
        "approx_k_pct": "k_pct_ewm",
        "approx_bb_pct": "bb_pct_ewm",
    }

    results: list[pd.DataFrame] = []
    for pid, grp in game_stats.groupby("player_id"):
        grp = grp.sort_values("date")
        ewma_row = {}
        for src, dst in ewma_cols.items():
            ewma_row[dst] = grp[src].ewm(span=_BATTER_EWMA_SPAN, min_periods=1).mean().values

        player_df = pd.DataFrame(ewma_row, index=grp.index)
        player_df["player_id"] = pid
        player_df["date"] = grp["date"].values
        player_df["batting_order"] = grp["batting_order"].values
        results.append(player_df)

    if not results:
        return pd.DataFrame()

    rolling_df = pd.concat(results, ignore_index=True)

    if prior_batter_stats is not None and not prior_batter_stats.empty and retro_to_mlbam:
        rolling_df = _attach_prior_batter_stats(rolling_df, prior_batter_stats, retro_to_mlbam)
    else:
        rolling_df["xwoba_ewm"] = _AVG_XWOBA
        rolling_df["barrel_pct_ewm"] = _AVG_BARREL_PCT
        rolling_df["hard_hit_pct_ewm"] = _AVG_HARD_HIT_PCT
        rolling_df["wrc_plus_ewm"] = _AVG_WRC_PLUS
        rolling_df["sprint_speed"] = _AVG_SPRINT_SPEED

    return rolling_df


def _attach_prior_batter_stats(
    rolling_df: pd.DataFrame,
    prior_stats: pd.DataFrame,
    retro_to_mlbam: dict[str, int],
) -> pd.DataFrame:
    """Join prior-season static stats to the rolling DataFrame.

    Uses retro_id → mlbam mapping to link gamelog player IDs to Statcast/FG data.
    """
    stat_lookup: dict[int, dict[str, float]] = {}
    for _, row in prior_stats.iterrows():
        mid = int(row.get("player_id", 0))
        if mid == 0:
            continue
        stat_lookup[mid] = {
            "xwoba_ewm": float(row.get("xwoba", _AVG_XWOBA)),
            "barrel_pct_ewm": float(row.get("barrel_pct", _AVG_BARREL_PCT)),
            "hard_hit_pct_ewm": float(row.get("hard_hit_pct", _AVG_HARD_HIT_PCT)),
            "wrc_plus_ewm": float(row.get("wrc_plus", _AVG_WRC_PLUS)),
            "sprint_speed": float(row.get("sprint_speed", _AVG_SPRINT_SPEED)),
        }

    xwoba, barrel, hard_hit, wrc, speed = [], [], [], [], []
    for _, row in rolling_df.iterrows():
        retro_id = str(row["player_id"]).strip().lower()
        mlbam = retro_to_mlbam.get(retro_id)
        stats = stat_lookup.get(mlbam, {}) if mlbam else {}
        xwoba.append(stats.get("xwoba_ewm", _AVG_XWOBA))
        barrel.append(stats.get("barrel_pct_ewm", _AVG_BARREL_PCT))
        hard_hit.append(stats.get("hard_hit_pct_ewm", _AVG_HARD_HIT_PCT))
        wrc.append(stats.get("wrc_plus_ewm", _AVG_WRC_PLUS))
        speed.append(stats.get("sprint_speed", _AVG_SPRINT_SPEED))

    rolling_df["xwoba_ewm"] = xwoba
    rolling_df["barrel_pct_ewm"] = barrel
    rolling_df["hard_hit_pct_ewm"] = hard_hit
    rolling_df["wrc_plus_ewm"] = wrc
    rolling_df["sprint_speed"] = speed

    return rolling_df


# ---------------------------------------------------------------------------
# Pitcher rolling stats from gamelogs
# ---------------------------------------------------------------------------


def build_pitcher_rolling(
    gamelogs: pd.DataFrame,
    prior_pitcher_stats: pd.DataFrame | None = None,
    retro_to_mlbam: dict[str, int] | None = None,
    pitcher_game_logs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute per-pitcher EWMA rolling stats.

    When *pitcher_game_logs* (from the MLB Stats API) is provided, individual
    pitcher box scores (exact IP, H, ER, BB, K per appearance) are used.
    Otherwise falls back to a team-level approximation from Retrosheet gamelogs.

    Returns DataFrame with columns: player_id, date, era_ewm, k9_ewm, bb9_ewm,
    whip_ewm, plus static priors (fip, xwoba_allowed, swstr_pct) from prior season.
    """
    if pitcher_game_logs is not None and not pitcher_game_logs.empty:
        df = _pitcher_game_stats_from_api(pitcher_game_logs, retro_to_mlbam)
    else:
        df = _pitcher_game_stats_from_gamelogs(gamelogs)

    if df.empty:
        return pd.DataFrame()

    ewma_cols = {
        "era_game": "era_ewm",
        "k9_game": "k9_ewm",
        "bb9_game": "bb9_ewm",
        "whip_game": "whip_ewm",
    }

    results: list[pd.DataFrame] = []
    for pid, grp in df.groupby("player_id"):
        grp = grp.sort_values("date")
        ewma_row = {}
        for src, dst in ewma_cols.items():
            ewma_row[dst] = grp[src].ewm(span=_PITCHER_EWMA_SPAN, min_periods=1).mean().values

        player_df = pd.DataFrame(ewma_row, index=grp.index)
        player_df["player_id"] = pid
        player_df["date"] = grp["date"].values
        results.append(player_df)

    if not results:
        return pd.DataFrame()

    rolling_df = pd.concat(results, ignore_index=True)

    if prior_pitcher_stats is not None and not prior_pitcher_stats.empty and retro_to_mlbam:
        rolling_df = _attach_prior_pitcher_stats(rolling_df, prior_pitcher_stats, retro_to_mlbam)
    else:
        rolling_df["fip_ewm"] = _AVG_FIP
        rolling_df["xwoba_allowed_ewm"] = _AVG_PIT_XWOBA
        rolling_df["swstr_pct_ewm"] = _AVG_SWSTR_PCT

    return rolling_df


def _pitcher_game_stats_from_api(
    pitcher_game_logs: pd.DataFrame,
    retro_to_mlbam: dict[str, int] | None,
) -> pd.DataFrame:
    """Build per-game pitcher stats from MLB Stats API game logs (high fidelity)."""
    pgl = pitcher_game_logs.copy()
    pgl["date"] = pd.to_datetime(pgl["date"], errors="coerce")
    pgl = pgl.dropna(subset=["date"])
    if pgl.empty:
        return pd.DataFrame()

    mlbam_to_retro: dict[int, str] = {}
    if retro_to_mlbam:
        mlbam_to_retro = {v: k for k, v in retro_to_mlbam.items()}

    rows: list[dict] = []
    for _, row in pgl.iterrows():
        ip = float(row.get("ip", 0))
        if ip <= 0:
            continue

        er = float(row.get("earned_runs", 0))
        h_val = float(row.get("hits", 0))
        bb_val = float(row.get("bb", 0))
        so_val = float(row.get("k", 0))

        mlbam = int(row.get("mlbam_id", 0))
        retro_id = mlbam_to_retro.get(mlbam, f"mlbam_{mlbam}")

        rows.append(
            {
                "date": row["date"],
                "player_id": retro_id,
                "era_game": (er / ip * 9.0) if ip > 0 else _AVG_ERA,
                "k9_game": (so_val / ip * 9.0) if ip > 0 else _AVG_K9,
                "bb9_game": (bb_val / ip * 9.0) if ip > 0 else _AVG_BB9,
                "whip_game": ((h_val + bb_val) / ip) if ip > 0 else _AVG_WHIP,
                "ip": ip,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _pitcher_game_stats_from_gamelogs(gamelogs: pd.DataFrame) -> pd.DataFrame:
    """Approximate per-game pitcher stats from team-level Retrosheet gamelogs."""
    gl = gamelogs.copy()
    if gl.empty or "date" not in gl.columns:
        return pd.DataFrame()
    gl["date"] = pd.to_datetime(gl["date"])

    rows: list[dict] = []
    for _, row in gl.iterrows():
        total_outs = _safe_float(row.get("num_outs"))

        for side, pid_col, er_col, prefix in [
            ("home", "home_starting_pitcher_id", "home_er", "visiting_"),
            ("away", "visiting_starting_pitcher_id", "visiting_er", "home_"),
        ]:
            pid = row.get(pid_col)
            if pd.isna(pid) or str(pid).strip() == "":
                continue

            er = _safe_float(row.get(er_col))

            ip_full = total_outs / 6.0 if not np.isnan(total_outs) and total_outs > 0 else 0.0
            if ip_full == 0:
                continue

            h_val = _safe_float(row.get(f"{prefix}hits", row.get(f"{prefix}h", 0)))
            bb_val = _safe_float(row.get(f"{prefix}bb", row.get(f"{prefix}walks", 0)))
            so_val = _safe_float(
                row.get(
                    f"{prefix}k",
                    row.get(f"{prefix}strikeouts", row.get(f"{prefix}so", 0)),
                )
            )

            era_game = (er / ip_full * 9.0) if ip_full > 0 and not np.isnan(er) else _AVG_ERA
            k9 = (so_val / ip_full * 9.0) if ip_full > 0 else _AVG_K9
            bb9 = (bb_val / ip_full * 9.0) if ip_full > 0 else _AVG_BB9
            whip = ((h_val + bb_val) / ip_full) if ip_full > 0 else _AVG_WHIP

            rows.append(
                {
                    "date": row["date"],
                    "player_id": str(pid).strip(),
                    "era_game": era_game,
                    "k9_game": k9,
                    "bb9_game": bb9,
                    "whip_game": whip,
                    "ip": ip_full,
                }
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _attach_prior_pitcher_stats(
    rolling_df: pd.DataFrame,
    prior_stats: pd.DataFrame,
    retro_to_mlbam: dict[str, int],
) -> pd.DataFrame:
    """Join prior-season static pitcher stats."""
    stat_lookup: dict[int, dict[str, float]] = {}
    for _, row in prior_stats.iterrows():
        mid = int(row.get("player_id", 0))
        if mid == 0:
            continue
        stat_lookup[mid] = {
            "fip_ewm": float(row.get("fip", _AVG_FIP)),
            "xwoba_allowed_ewm": float(row.get("est_woba", _AVG_PIT_XWOBA)),
            "swstr_pct_ewm": float(row.get("whiff_rate", row.get("swstr_pct", _AVG_SWSTR_PCT))),
        }

    fip, xwoba, swstr = [], [], []
    for _, row in rolling_df.iterrows():
        retro_id = str(row["player_id"]).strip().lower()
        mlbam = retro_to_mlbam.get(retro_id)
        stats = stat_lookup.get(mlbam, {}) if mlbam else {}
        fip.append(stats.get("fip_ewm", _AVG_FIP))
        xwoba.append(stats.get("xwoba_allowed_ewm", _AVG_PIT_XWOBA))
        swstr.append(stats.get("swstr_pct_ewm", _AVG_SWSTR_PCT))

    rolling_df["fip_ewm"] = fip
    rolling_df["xwoba_allowed_ewm"] = xwoba
    rolling_df["swstr_pct_ewm"] = swstr

    return rolling_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(val: Any) -> float:
    """Convert a value to float, returning NaN on failure."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return float("nan")
    try:
        return float(val)
    except (ValueError, TypeError):
        return float("nan")


def get_latest_batter_rolling_for_game(
    batter_rolling: pd.DataFrame,
    player_ids: list[str],
    game_date: pd.Timestamp,
) -> dict[str, dict[str, float]]:
    """For each player_id, get the most recent EWMA stats before game_date.

    Returns {player_id: {ops_ewm, iso_ewm, k_pct_ewm, bb_pct_ewm, ...}}.
    """
    result: dict[str, dict[str, float]] = {}
    if batter_rolling.empty:
        return result

    for pid in player_ids:
        mask = (batter_rolling["player_id"] == pid) & (batter_rolling["date"] < game_date)
        player_rows = batter_rolling.loc[mask]
        if player_rows.empty:
            continue
        latest = player_rows.iloc[-1]
        stat_cols = [c for c in latest.index if c not in ("player_id", "date", "batting_order")]
        result[pid] = {c: float(latest[c]) for c in stat_cols}

    return result


def get_latest_pitcher_rolling_for_game(
    pitcher_rolling: pd.DataFrame,
    player_id: str,
    game_date: pd.Timestamp,
) -> dict[str, float]:
    """Get the most recent EWMA pitcher stats before game_date."""
    if pitcher_rolling.empty:
        return {}

    mask = (pitcher_rolling["player_id"] == player_id) & (pitcher_rolling["date"] < game_date)
    player_rows = pitcher_rolling.loc[mask]
    if player_rows.empty:
        return {}

    latest = player_rows.iloc[-1]
    stat_cols = [c for c in latest.index if c not in ("player_id", "date")]
    return {c: float(latest[c]) for c in stat_cols}
