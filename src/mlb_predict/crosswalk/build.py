from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from mlb_predict.ingest.id_map import RetroTeamMap


@dataclass(frozen=True)
class CrosswalkResult:
    df: pd.DataFrame
    coverage_pct: float
    matched: int
    missing: int
    ambiguous: int


def _prep_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    """Normalise a schedule DataFrame for crosswalk matching.

    Crucially, ``date`` is derived from ``game_date_local`` (the API's local
    calendar date grouping) rather than the UTC timestamp.  West-Coast evening
    games cross UTC midnight and would land on the wrong date if we used the
    UTC value, causing systematic mismatches against Retrosheet gamelogs, which
    always record the local game date.
    """
    s = schedule.copy()
    if "game_date_local" in s.columns and s["game_date_local"].notna().any():
        s["date"] = pd.to_datetime(s["game_date_local"], errors="coerce").dt.date
    else:
        # Fallback for schedules ingested before this fix: parse UTC timestamp.
        s["date"] = pd.to_datetime(
            s["game_date_utc"].str.replace("Z", "+00:00", regex=False), errors="coerce"
        ).dt.date
    s["game_number"] = pd.to_numeric(s["game_number"], errors="coerce").astype("Int64")
    return s


_CROSSWALK_COLS = [
    "date",
    "home_mlb_id",
    "away_mlb_id",
    "home_retro",
    "away_retro",
    "dh_game_num",
    "status",
    "mlb_game_pk",
    "match_confidence",
    "notes",
]


def build_crosswalk(
    *, season: int, schedule: pd.DataFrame, gamelogs: pd.DataFrame, retro_team_map: RetroTeamMap
) -> CrosswalkResult:
    sched = _prep_schedule(schedule)

    gl = gamelogs.copy().dropna(subset=["date", "home_team", "visiting_team"])
    if gl.empty:
        return CrosswalkResult(
            df=pd.DataFrame(columns=_CROSSWALK_COLS),
            coverage_pct=0.0,
            matched=0,
            missing=0,
            ambiguous=0,
        )
    gl["home_mlb_id"] = gl["home_team"].map(
        lambda x: retro_team_map.retro_to_mlb_id(str(x), season)
    )
    gl["away_mlb_id"] = gl["visiting_team"].map(
        lambda x: retro_team_map.retro_to_mlb_id(str(x), season)
    )
    gl["dh_game_num"] = pd.to_numeric(gl["game_num"], errors="coerce").astype("Int64")

    merged = gl.merge(
        sched[["game_pk", "date", "home_mlb_id", "away_mlb_id", "game_number", "venue_id"]],
        on=["date", "home_mlb_id", "away_mlb_id"],
        how="left",
    )

    def resolve_group(g: pd.DataFrame) -> pd.DataFrame:
        if g["game_pk"].isna().all():
            return g.head(1).assign(
                status="missing", mlb_game_pk=pd.NA, match_confidence=0.0, notes="no_schedule_match"
            )
        cands = g.dropna(subset=["game_pk"])
        if cands["game_pk"].nunique() == 1:
            pk = int(cands["game_pk"].iloc[0])
            return g.head(1).assign(
                status="matched", mlb_game_pk=pk, match_confidence=1.0, notes="unique"
            )
        if pd.notna(g["dh_game_num"].iloc[0]):
            m = cands[cands["game_number"] == g["dh_game_num"].iloc[0]]
            if m["game_pk"].nunique() == 1:
                pk = int(m["game_pk"].iloc[0])
                return g.head(1).assign(
                    status="matched",
                    mlb_game_pk=pk,
                    match_confidence=0.9,
                    notes="matched_on_game_number",
                )
        return g.head(1).assign(
            status="ambiguous", mlb_game_pk=pd.NA, match_confidence=0.0, notes="multiple_candidates"
        )

    key_cols = ["date", "home_mlb_id", "away_mlb_id", "home_team", "visiting_team", "dh_game_num"]
    # Iterate groups explicitly so key columns remain available inside
    # resolve_group without relying on the deprecated include_groups API
    # (removed in pandas 2.3).
    resolved_rows = [
        resolve_group(g) for _, g in merged.groupby(key_cols, dropna=False, sort=False)
    ]
    resolved = (
        pd.concat(resolved_rows, ignore_index=True)
        if resolved_rows
        else pd.DataFrame(columns=merged.columns)
    )

    out = resolved[
        [
            "date",
            "home_mlb_id",
            "away_mlb_id",
            "home_team",
            "visiting_team",
            "dh_game_num",
            "status",
            "mlb_game_pk",
            "match_confidence",
            "notes",
        ]
    ].rename(columns={"home_team": "home_retro", "visiting_team": "away_retro"})

    # --- Fallback pass: try swapped home/away for any still-missing games ------
    # In unusual seasons (e.g. 2020 COVID relocations) a game may be played at
    # a neutral site so that the "home" team in Retrosheet differs from the
    # "home" team in the MLB API schedule.  For these rows we attempt a second
    # lookup with home_mlb_id / away_mlb_id reversed.
    still_missing_mask = out["status"] == "missing"
    if still_missing_mask.any():
        # Build a date × mlb_ids lookup into the schedule (take first game_pk if dups).
        sched_pk_map: dict[tuple, int] = {}
        for _, srow in sched[["date", "home_mlb_id", "away_mlb_id", "game_pk"]].iterrows():
            key = (srow["date"], int(srow["home_mlb_id"]), int(srow["away_mlb_id"]))
            if key not in sched_pk_map:
                sched_pk_map[key] = int(srow["game_pk"])

        for idx in out[still_missing_mask].index:
            r = out.loc[idx]
            if pd.isna(r["home_mlb_id"]) or pd.isna(r["away_mlb_id"]):
                continue
            swapped_key = (r["date"], int(r["away_mlb_id"]), int(r["home_mlb_id"]))
            if swapped_key in sched_pk_map:
                out.loc[idx, "status"] = "matched"
                out.loc[idx, "mlb_game_pk"] = sched_pk_map[swapped_key]
                out.loc[idx, "match_confidence"] = 0.6
                out.loc[idx, "notes"] = "matched_swapped_home_away"

    matched = int((out["status"] == "matched").sum())
    ambiguous = int((out["status"] == "ambiguous").sum())
    missing = int((out["status"] == "missing").sum())
    coverage_pct = 100.0 * matched / max(len(out), 1)

    return CrosswalkResult(
        df=out, coverage_pct=coverage_pct, matched=matched, missing=missing, ambiguous=ambiguous
    )
