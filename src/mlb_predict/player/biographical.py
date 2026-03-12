"""Player biographical data: batting side, throwing side, birth date, position.

Sources
-------
- Chadwick register (primary): provides key_retro, key_mlbam, birth_year,
  birth_month, birth_day, bats, throws.
- MLB Stats API (fallback for current-season players missing from Chadwick).

The biographical DataFrame is built once per training run and cached at
``data/processed/player/biographical.parquet``.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_PEAK_AGE: float = 27.5
_CACHE_FILENAME = "biographical.parquet"

_POSITION_CATEGORIES: dict[str, int] = {
    "C": 0,
    "1B": 1,
    "2B": 2,
    "3B": 3,
    "SS": 4,
    "LF": 5,
    "CF": 6,
    "RF": 7,
    "OF": 5,
    "DH": 8,
    "P": 9,
}

BAT_SIDE_MAP: dict[str, float] = {"L": -1.0, "R": 1.0, "B": 0.0}
THROW_SIDE_MAP: dict[str, float] = {"L": -1.0, "R": 1.0}


def _load_chadwick_bio() -> pd.DataFrame:
    """Load biographical columns from the Chadwick register.

    Returns a DataFrame with columns: mlbam_id, retro_id, birth_year,
    birth_month, birth_day, bats, throws.
    """
    try:
        from pybaseball import chadwick_register

        reg = chadwick_register()
        cols = [
            "key_retro",
            "key_mlbam",
            "birth_year",
            "birth_month",
            "birth_day",
            "bats",
            "throws",
        ]
        available = [c for c in cols if c in reg.columns]
        df = reg[available].dropna(subset=["key_mlbam"]).copy()
        df["key_mlbam"] = df["key_mlbam"].astype(int)
        df = df.rename(columns={"key_retro": "retro_id", "key_mlbam": "mlbam_id"})
        return df
    except Exception as exc:
        logger.warning("Chadwick register bio load failed: %s", exc)
        return pd.DataFrame(
            columns=[
                "retro_id",
                "mlbam_id",
                "birth_year",
                "birth_month",
                "birth_day",
                "bats",
                "throws",
            ]
        )


def build_biographical_df(cache_dir: Path | None = None) -> pd.DataFrame:
    """Build or load the biographical DataFrame.

    Returns
    -------
    DataFrame
        Columns: mlbam_id, retro_id, bat_side (float), throw_side (float),
        birth_date (date), peak-normalised fields computed at query time.
    """
    cache_path = (cache_dir or Path("data/processed/player")) / _CACHE_FILENAME
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    raw = _load_chadwick_bio()
    if raw.empty:
        return raw

    df = raw.copy()
    if "bats" in df.columns:
        df["bat_side"] = df["bats"].map(BAT_SIDE_MAP).fillna(0.0)
    else:
        df["bat_side"] = 0.0
    if "throws" in df.columns:
        df["throw_side"] = df["throws"].map(THROW_SIDE_MAP).fillna(1.0)
    else:
        df["throw_side"] = 1.0

    birth_cols = ["birth_year", "birth_month", "birth_day"]
    if all(c in df.columns for c in birth_cols):
        df["birth_year"] = pd.to_numeric(df["birth_year"], errors="coerce")
        df["birth_month"] = pd.to_numeric(df["birth_month"], errors="coerce").fillna(1).astype(int)
        df["birth_day"] = pd.to_numeric(df["birth_day"], errors="coerce").fillna(1).astype(int)
        valid = df["birth_year"].notna()
        dates = pd.to_datetime(
            df.loc[valid, ["birth_year", "birth_month", "birth_day"]].rename(
                columns={"birth_year": "year", "birth_month": "month", "birth_day": "day"}
            ),
            errors="coerce",
        )
        df.loc[valid, "birth_date"] = dates.dt.date
    else:
        df["birth_date"] = pd.NaT

    out = df[["mlbam_id", "retro_id", "bat_side", "throw_side", "birth_date"]].copy()
    out = out.drop_duplicates(subset=["mlbam_id"], keep="first").reset_index(drop=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache_path, index=False)
    logger.info("Saved biographical data: %d players → %s", len(out), cache_path)
    return out


def player_age_at_date(birth_date: date | None, game_date: date) -> float:
    """Return normalised age (relative to peak) for a single player on a given date.

    Returns 0.0 if birth_date is unknown.
    """
    if birth_date is None or pd.isna(birth_date):
        return 0.0
    age_years = (game_date - birth_date).days / 365.25
    return float(age_years - _PEAK_AGE) / 10.0  # scale so ±1 covers ~10 years from peak


def encode_position(pos_str: str | None) -> float:
    """Encode position string to a normalised float in [0, 1]."""
    if pos_str is None or pd.isna(pos_str):
        return 0.5
    pos = str(pos_str).strip().upper()
    idx = _POSITION_CATEGORIES.get(pos, 5)
    return float(idx) / 9.0


def build_bio_lookup(bio_df: pd.DataFrame) -> dict[int, dict[str, float]]:
    """Create mlbam_id → {bat_side, throw_side, birth_date_ordinal} lookup.

    birth_date is stored as ordinal for fast age computation.
    """
    lookup: dict[int, dict[str, float]] = {}
    for _, row in bio_df.iterrows():
        mid = int(row["mlbam_id"])
        bd = row.get("birth_date")
        bd_ordinal = float(bd.toordinal()) if bd is not None and not pd.isna(bd) else float("nan")
        lookup[mid] = {
            "bat_side": float(row.get("bat_side", 0.0)),
            "throw_side": float(row.get("throw_side", 1.0)),
            "birth_date_ordinal": bd_ordinal,
        }
    return lookup
