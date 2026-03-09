from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from mlb_predict.mlbapi.client import TokenBucket


GAMELOG_COLUMNS = [
    "date",
    "game_num",
    "day_of_week",
    "visiting_team",
    "visiting_team_league",
    "visiting_team_game_num",
    "home_team",
    "home_team_league",
    "home_team_game_num",
    "visiting_score",
    "home_score",
    "num_outs",
    "day_night",
    "completion_info",
    "forfeit_info",
    "protest_info",
    "park_id",
    "attendance",
    "time_of_game_minutes",
    "visiting_line_score",
    "home_line_score",
    "visiting_abs",
    "visiting_hits",
    "visiting_doubles",
    "visiting_triples",
    "visiting_homeruns",
    "visiting_rbi",
    "visiting_sac_hits",
    "visiting_sac_flies",
    "visiting_hbp",
    "visiting_bb",
    "visiting_iw",
    "visiting_k",
    "visiting_sb",
    "visiting_cs",
    "visiting_gdp",
    "visiting_ci",
    "visiting_lob",
    "visiting_pitchers_used",
    "visiting_individual_er",
    "visiting_er",
    "visiting_wp",
    "visiting_balks",
    "visiting_po",
    "visiting_assists",
    "visiting_errors",
    "visiting_pb",
    "visiting_dp",
    "visiting_tp",
    "home_abs",
    "home_hits",
    "home_doubles",
    "home_triples",
    "home_homeruns",
    "home_rbi",
    "home_sac_hits",
    "home_sac_flies",
    "home_hbp",
    "home_bb",
    "home_iw",
    "home_k",
    "home_sb",
    "home_cs",
    "home_gdp",
    "home_ci",
    "home_lob",
    "home_pitchers_used",
    "home_individual_er",
    "home_er",
    "home_wp",
    "home_balks",
    "home_po",
    "home_assists",
    "home_errors",
    "home_pb",
    "home_dp",
    "home_tp",
    "ump_home_id",
    "ump_home_name",
    "ump_first_id",
    "ump_first_name",
    "ump_second_id",
    "ump_second_name",
    "ump_third_id",
    "ump_third_name",
    "ump_lf_id",
    "ump_lf_name",
    "ump_rf_id",
    "ump_rf_name",
    "visiting_manager_id",
    "visiting_manager_name",
    "home_manager_id",
    "home_manager_name",
    "winning_pitcher_id",
    "winning_pitcher_name",
    "losing_pitcher_id",
    "losing_pitcher_name",
    "save_pitcher_id",
    "save_pitcher_name",
    "game_winning_rbi_id",
    "game_winning_rbi_name",
    "visiting_starting_pitcher_id",
    "visiting_starting_pitcher_name",
    "home_starting_pitcher_id",
    "home_starting_pitcher_name",
    "visiting_1_id",
    "visiting_1_name",
    "visiting_1_pos",
    "visiting_2_id",
    "visiting_2_name",
    "visiting_2_pos",
    "visiting_3_id",
    "visiting_3_name",
    "visiting_3_pos",
    "visiting_4_id",
    "visiting_4_name",
    "visiting_4_pos",
    "visiting_5_id",
    "visiting_5_name",
    "visiting_5_pos",
    "visiting_6_id",
    "visiting_6_name",
    "visiting_6_pos",
    "visiting_7_id",
    "visiting_7_name",
    "visiting_7_pos",
    "visiting_8_id",
    "visiting_8_name",
    "visiting_8_pos",
    "visiting_9_id",
    "visiting_9_name",
    "visiting_9_pos",
    "home_1_id",
    "home_1_name",
    "home_1_pos",
    "home_2_id",
    "home_2_name",
    "home_2_pos",
    "home_3_id",
    "home_3_name",
    "home_3_pos",
    "home_4_id",
    "home_4_name",
    "home_4_pos",
    "home_5_id",
    "home_5_name",
    "home_5_pos",
    "home_6_id",
    "home_6_name",
    "home_6_pos",
    "home_7_id",
    "home_7_name",
    "home_7_pos",
    "home_8_id",
    "home_8_name",
    "home_8_pos",
    "home_9_id",
    "home_9_name",
    "home_9_pos",
    "misc",
    "acquisition_info",
]


@dataclass(frozen=True)
class RetrosheetGLSource:
    primary: str = "chadwick"  # "chadwick" or "retrosheet"
    enable_fallback: bool = True

    def url_for(self, season: int, kind: str) -> str:
        if kind == "chadwick":
            return f"https://raw.githubusercontent.com/chadwickbureau/retrosheet/master/seasons/{season}/GL{season}.TXT"
        if kind == "retrosheet":
            return f"https://www.retrosheet.org/gamelogs/gl{season}.zip"
        raise ValueError(f"Unknown kind: {kind}")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


async def _http_get_bytes(url: str, *, bucket: TokenBucket, timeout_s: float) -> bytes:
    import aiohttp

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_s)) as sess:
        await bucket.acquire(1.0)
        async with sess.get(url) as resp:
            resp.raise_for_status()
            return await resp.read()


def _extract_gl_txt_from_zip(zip_bytes: bytes, season: int) -> bytes:
    want_upper = f"GL{season}.TXT"
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for n in zf.namelist():
            if n.upper() == want_upper:
                return zf.read(n)
    raise FileNotFoundError(f"Could not find {want_upper} inside zip")


async def download_gamelog_txt(
    *,
    season: int,
    out_path: Path,
    source: RetrosheetGLSource = RetrosheetGLSource(),
    refresh: bool = False,
    bucket_chadwick: Optional[TokenBucket] = None,
    bucket_retrosheet: Optional[TokenBucket] = None,
) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not refresh:
        data = out_path.read_bytes()
        return {
            "season": season,
            "path": str(out_path),
            "sha256": sha256_bytes(data),
            "cached": True,
            "source_used": "cache",
            "url_used": None,
            "fallback_reason": None,
        }

    bucket_chadwick = bucket_chadwick or TokenBucket(rate=2.0, capacity=4.0)
    bucket_retrosheet = bucket_retrosheet or TokenBucket(rate=1.0, capacity=2.0)

    primary = source.primary
    fallback = "retrosheet" if primary == "chadwick" else "chadwick"

    async def attempt(kind: str) -> tuple[bytes, str]:
        url = source.url_for(season, kind)
        bucket = bucket_chadwick if kind == "chadwick" else bucket_retrosheet
        if kind == "chadwick":
            data = await _http_get_bytes(url, bucket=bucket, timeout_s=60.0)
            return data, url
        z = await _http_get_bytes(url, bucket=bucket, timeout_s=60.0)
        data = _extract_gl_txt_from_zip(z, season)
        return data, url

    try:
        data, url_used = await attempt(primary)
        out_path.write_bytes(data)
        return {
            "season": season,
            "path": str(out_path),
            "sha256": sha256_bytes(data),
            "cached": False,
            "source_used": primary,
            "url_used": url_used,
            "fallback_reason": None,
        }
    except Exception as e_primary:
        if not source.enable_fallback:
            raise
        data, url_used = await attempt(fallback)
        out_path.write_bytes(data)
        return {
            "season": season,
            "path": str(out_path),
            "sha256": sha256_bytes(data),
            "cached": False,
            "source_used": fallback,
            "url_used": url_used,
            "fallback_reason": f"{type(e_primary).__name__}: {e_primary}",
        }


def parse_gamelog_txt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, sep=",", quotechar='"', dtype=str)
    if df.shape[1] != len(GAMELOG_COLUMNS):
        raise ValueError(
            f"Unexpected column count: got {df.shape[1]} expected {len(GAMELOG_COLUMNS)}"
        )
    df.columns = GAMELOG_COLUMNS
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce").dt.date
    for c in ["game_num", "visiting_score", "home_score"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
