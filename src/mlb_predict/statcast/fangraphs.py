"""FanGraphs team-level advanced metrics via pybaseball.

We pull season-aggregate batting and pitching metrics for each team per season.
These are used as *prior-season priors* in the feature builder (the model sees
what each team's Statcast profile looked like the previous year going into each
game).

Batting features (per team per season)
---------------------------------------
wOBA, ISO, BABIP, Hard%, Barrel%, xwOBA

Pitching features (per team per season)
-----------------------------------------
FIP, xFIP, K%, BB%, HR/FB%, WHIP
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Retrosheet 3-letter codes → FanGraphs team abbreviations.
# Retrosheet uses NL/AL city codes; FanGraphs uses BBREF-style codes.
RETRO_TO_FG: dict[str, str] = {
    # Identical codes
    "ARI": "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BOS": "BOS",
    "CIN": "CIN",
    "CLE": "CLE",
    "COL": "COL",
    "DET": "DET",
    "HOU": "HOU",
    "MIA": "MIA",
    "MIL": "MIL",
    "MIN": "MIN",
    "OAK": "OAK",
    "PHI": "PHI",
    "PIT": "PIT",
    "SEA": "SEA",
    "TEX": "TEX",
    "TOR": "TOR",
    # Differences
    "ANA": "LAA",  # Angels (2000-2004)
    "ATH": "OAK",  # Athletics (2025+ Las Vegas move in Retrosheet)
    "CHA": "CHW",  # White Sox
    "CHN": "CHC",  # Cubs
    "FLO": "MIA",  # Marlins (pre-2012)
    "KCA": "KCR",  # Royals
    "LAN": "LAD",  # Dodgers
    "MON": "WSN",  # Expos (pre-2005) → Nationals
    "NYA": "NYY",  # Yankees
    "NYN": "NYM",  # Mets
    "SDN": "SDP",  # Padres
    "SFN": "SFG",  # Giants
    "SLN": "STL",  # Cardinals
    "TBA": "TBR",  # Rays
    "WAS": "WSN",  # Nationals (post-2005)
}

_BAT_COLS = {
    "Team": "team_fg",
    "wOBA": "bat_woba",
    "ISO": "bat_iso",
    "BABIP": "bat_babip",
    "Hard%": "bat_hard_pct",
    "Barrel%": "bat_barrel_pct",
    "xwOBA": "bat_xwoba",
}

_PIT_COLS = {
    "Team": "team_fg",
    "FIP": "pit_fip",
    "xFIP": "pit_xfip",
    "K%": "pit_k_pct",
    "BB%": "pit_bb_pct",
    "HR/FB": "pit_hr_fb",
    "WHIP": "pit_whip",
}


def _safe_fetch_batting(season: int) -> pd.DataFrame:
    try:
        from pybaseball import team_batting  # type: ignore[import-untyped,unused-ignore]

        df = team_batting(season, season)
        return df[[c for c in _BAT_COLS if c in df.columns]].rename(columns=_BAT_COLS)
    except Exception as exc:
        logger.warning("FanGraphs batting fetch failed for %d: %s", season, exc)
        return pd.DataFrame()


def _safe_fetch_pitching(season: int) -> pd.DataFrame:
    try:
        from pybaseball import team_pitching  # type: ignore[import-untyped,unused-ignore]

        df = team_pitching(season, season)
        return df[[c for c in _PIT_COLS if c in df.columns]].rename(columns=_PIT_COLS)
    except Exception as exc:
        logger.warning("FanGraphs pitching fetch failed for %d: %s", season, exc)
        return pd.DataFrame()


def load_fg_team_map(
    fg_dir: Path,
    season: int,
) -> dict[str, dict[str, float]]:
    """Load FanGraphs data for *season* as a Retrosheet-code → stats dict.

    Parameters
    ----------
    fg_dir:
        Directory containing ``fangraphs_YYYY.parquet`` files.
    season:
        Season to load.

    Returns
    -------
    dict
        Mapping Retrosheet team code → {bat_woba, bat_iso, bat_barrel_pct,
        bat_hard_pct, bat_xwoba, pit_fip, pit_xfip, pit_k_pct, pit_bb_pct,
        pit_whip}.
    """
    path = Path(fg_dir) / f"fangraphs_{season}.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)

    # Build fg_code → stats dict
    fg_map: dict[str, dict[str, float]] = {}
    stat_cols = [c for c in df.columns if c not in ("team_fg", "season")]
    for _, row in df.iterrows():
        fg_code = str(row.get("team_fg", ""))
        if not fg_code:
            continue
        fg_map[fg_code] = {c: float(row[c]) for c in stat_cols if pd.notna(row.get(c))}

    # Re-key by Retrosheet code
    retro_map: dict[str, dict[str, float]] = {}
    for fg_code, stats in fg_map.items():
        retro_codes = [k for k, v in RETRO_TO_FG.items() if v == fg_code]
        for rc in retro_codes:
            retro_map[rc] = stats
    return retro_map


def fetch_team_advanced_stats(season: int) -> pd.DataFrame:
    """Fetch FanGraphs batting + pitching for all 30 teams in *season*.

    Returns
    -------
    DataFrame
        One row per team.  Columns include bat_woba, bat_iso, bat_babip,
        bat_hard_pct, bat_barrel_pct, bat_xwoba, pit_fip, pit_xfip,
        pit_k_pct, pit_bb_pct, pit_hr_fb, pit_whip.
        The ``retro_code`` column maps to Retrosheet team codes.
    """
    bat = _safe_fetch_batting(season)
    pit = _safe_fetch_pitching(season)

    if bat.empty and pit.empty:
        return pd.DataFrame()

    if not bat.empty and not pit.empty:
        merged = bat.merge(pit, on="team_fg", how="outer")
    elif not bat.empty:
        merged = bat
    else:
        merged = pit

    merged["season"] = season
    return merged.reset_index(drop=True)
