from __future__ import annotations

from datetime import date, datetime
from typing import Any, Mapping

import pandas as pd

from .client import MLBAPIClient


SCHEDULE_FIELDS_MIN = ",".join(
    [
        "dates,date",
        "dates,games,gamePk",
        "dates,games,gameDate",
        "dates,games,gameType",
        "dates,games,status,detailedState",
        "dates,games,teams,home,team,id",
        "dates,games,teams,away,team,id",
        "dates,games,venue,id",
        "dates,games,venue,timeZone,id",
        "dates,games,doubleHeader",
        "dates,games,gameNumber",
    ]
)

GAME_TYPE_REGULAR = "R"
GAME_TYPE_SPRING = "S"
GAME_TYPE_POSTSEASON = "P"
GAME_TYPE_EXHIBITION = "E"
ALL_GAME_TYPES = (GAME_TYPE_REGULAR, GAME_TYPE_SPRING)

# Columns guaranteed to be present in a normalized schedule DataFrame.
_SCHEDULE_COLS = [
    "game_pk",
    "game_date_utc",
    "game_date_local",
    "home_mlb_id",
    "away_mlb_id",
    "venue_id",
    "local_timezone",
    "double_header",
    "game_number",
    "status",
    "game_type",
]


async def schedule_bounds(
    client: MLBAPIClient,
    *,
    season: int,
    game_type: str = GAME_TYPE_REGULAR,
    sport_id: int = 1,
) -> tuple[date, date]:
    """Return (first_date, last_date) for the given season and game type."""
    params: Mapping[str, Any] = {
        "sportId": sport_id,
        "season": season,
        "gameType": game_type,
        "fields": "dates,date",
    }
    raw = await client.get_json("schedule", params)
    ds = [d.get("date") for d in raw.get("dates", []) if d.get("date")]
    if not ds:
        raise RuntimeError(
            f"No dates returned for season={season} gameType={game_type}"
        )
    ds_sorted = sorted(ds)
    return (date.fromisoformat(ds_sorted[0]), date.fromisoformat(ds_sorted[-1]))


async def schedule_bounds_regular_season(
    client: MLBAPIClient, *, season: int, sport_id: int = 1
) -> tuple[date, date]:
    """Return (first_date, last_date) for the regular season."""
    return await schedule_bounds(
        client, season=season, game_type=GAME_TYPE_REGULAR, sport_id=sport_id
    )


def normalize_schedule(
    raw: dict[str, Any],
    *,
    game_type_override: str | None = None,
) -> pd.DataFrame:
    """Parse raw MLB Stats API schedule JSON into a normalized DataFrame.

    The `game_date_local` column is populated from the API's ``dates[].date``
    field, which reflects the **local** game date at the venue.  This is
    critical for matching against Retrosheet gamelogs, which also use local
    dates.  ``game_date_utc`` is preserved for reference but should *not* be
    used for date-based joins because West-Coast games (7 PM PDT → 2 AM UTC
    next day) would land on the wrong calendar date.

    If *game_type_override* is given it is used for all rows; otherwise
    the value is read from ``games[].gameType`` in the API response.
    """
    rows: list[dict[str, Any]] = []
    for d in raw.get("dates", []):
        local_date_str: str | None = d.get("date")
        for g in d.get("games", []):
            teams = g.get("teams", {})
            venue = g.get("venue") or {}
            tz = (venue.get("timeZone") or {}).get("id") or None
            rows.append(
                {
                    "game_pk": g.get("gamePk"),
                    "game_date_utc": g.get("gameDate"),
                    "game_date_local": local_date_str,
                    "home_mlb_id": ((teams.get("home") or {}).get("team") or {}).get("id"),
                    "away_mlb_id": ((teams.get("away") or {}).get("team") or {}).get("id"),
                    "venue_id": venue.get("id"),
                    "local_timezone": tz,
                    "double_header": g.get("doubleHeader"),
                    "game_number": g.get("gameNumber"),
                    "status": ((g.get("status") or {}).get("detailedState")),
                    "game_type": game_type_override or g.get("gameType", GAME_TYPE_REGULAR),
                }
            )
    if not rows:
        return pd.DataFrame(columns=_SCHEDULE_COLS)
    df = pd.DataFrame(rows).dropna(subset=["game_pk", "home_mlb_id", "away_mlb_id"])
    if df.empty:
        return pd.DataFrame(columns=_SCHEDULE_COLS)
    df["game_pk"] = df["game_pk"].astype(int)
    df["home_mlb_id"] = df["home_mlb_id"].astype(int)
    df["away_mlb_id"] = df["away_mlb_id"].astype(int)
    return df


async def fetch_schedule_chunk(
    client: MLBAPIClient,
    *,
    season: int,
    start_date: date,
    end_date: date,
    game_type: str = GAME_TYPE_REGULAR,
    sport_id: int = 1,
    fields: str = SCHEDULE_FIELDS_MIN,
) -> pd.DataFrame:
    """Fetch one date-range chunk of the schedule for a given game type."""
    params: Mapping[str, Any] = {
        "sportId": sport_id,
        "season": season,
        "gameType": game_type,
        "startDate": start_date.isoformat(),
        "endDate": end_date.isoformat(),
        "fields": fields,
    }
    raw = await client.get_json("schedule", params)
    return normalize_schedule(raw, game_type_override=game_type)


def parse_utc_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))
