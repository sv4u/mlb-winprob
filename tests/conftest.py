"""Shared pytest fixtures for the mlb-winprob test suite."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pytest

if TYPE_CHECKING:
    pass

from winprob.retrosheet.gamelogs import GAMELOG_COLUMNS


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """Temporary cache directory for MLBAPIClient tests."""
    d = tmp_path / "mlb_api"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Schedule DataFrame fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def schedule_df() -> pd.DataFrame:
    """Minimal valid schedule DataFrame for crosswalk and schedule tests."""
    return pd.DataFrame(
        {
            "game_pk": [745803, 745804, 745805],
            "game_date_utc": [
                "2024-04-01T17:10:00Z",
                "2024-04-01T19:40:00Z",
                "2024-04-02T18:10:00Z",
            ],
            "home_mlb_id": [111, 147, 116],
            "away_mlb_id": [133, 139, 145],
            "venue_id": [3, 3313, 2394],
            "local_timezone": [
                "America/New_York",
                "America/New_York",
                "America/Detroit",
            ],
            "double_header": ["N", "N", "N"],
            "game_number": [1, 1, 1],
            "status": ["Final", "Final", "Scheduled"],
            "game_type": ["R", "R", "R"],
            "season": [2024, 2024, 2024],
        }
    )


@pytest.fixture
def raw_schedule_response() -> dict:
    """Raw MLB Stats API schedule response with two games."""
    return {
        "dates": [
            {
                "date": "2024-04-01",
                "games": [
                    {
                        "gamePk": 745803,
                        "gameDate": "2024-04-01T17:10:00Z",
                        "gameType": "R",
                        "teams": {
                            "home": {"team": {"id": 111}},
                            "away": {"team": {"id": 133}},
                        },
                        "venue": {"id": 3, "timeZone": {"id": "America/New_York"}},
                        "doubleHeader": "N",
                        "gameNumber": 1,
                        "status": {"detailedState": "Final"},
                    },
                    {
                        "gamePk": 745804,
                        "gameDate": "2024-04-01T19:40:00Z",
                        "gameType": "R",
                        "teams": {
                            "home": {"team": {"id": 147}},
                            "away": {"team": {"id": 139}},
                        },
                        "venue": {"id": 3313, "timeZone": {"id": "America/New_York"}},
                        "doubleHeader": "N",
                        "gameNumber": 1,
                        "status": {"detailedState": "Final"},
                    },
                ],
            }
        ]
    }


# ---------------------------------------------------------------------------
# Teams fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def raw_teams_response() -> dict:
    """Raw MLB Stats API teams response."""
    return {
        "teams": [
            {"id": 111, "abbreviation": "BOS", "name": "Boston Red Sox"},
            {"id": 147, "abbreviation": "NYY", "name": "New York Yankees"},
            {"id": 133, "abbreviation": "OAK", "name": "Oakland Athletics"},
        ]
    }


@pytest.fixture
def teams_df() -> pd.DataFrame:
    """Sample teams DataFrame."""
    return pd.DataFrame(
        {
            "season": [2024, 2024, 2024],
            "mlb_team_id": [111, 147, 133],
            "abbrev": ["BOS", "NYY", "OAK"],
            "name": ["Boston Red Sox", "New York Yankees", "Oakland Athletics"],
        }
    )


# ---------------------------------------------------------------------------
# Retrosheet gamelog fixtures
# ---------------------------------------------------------------------------


def _make_gamelog_row(
    date: str = "20240401",
    game_num: str = "0",
    visiting_team: str = "OAK",
    home_team: str = "BOS",
    visiting_score: str = "3",
    home_score: str = "5",
) -> list[str]:
    """Build a 161-field gamelog row matching GAMELOG_COLUMNS."""
    row = [""] * len(GAMELOG_COLUMNS)
    col_idx = {name: i for i, name in enumerate(GAMELOG_COLUMNS)}
    row[col_idx["date"]] = date
    row[col_idx["game_num"]] = game_num
    row[col_idx["visiting_team"]] = visiting_team
    row[col_idx["home_team"]] = home_team
    row[col_idx["visiting_score"]] = visiting_score
    row[col_idx["home_score"]] = home_score
    row[col_idx["day_of_week"]] = "Mon"
    return row


@pytest.fixture
def gamelog_txt_path(tmp_path: Path) -> Path:
    """Write a minimal two-game GL TXT file and return its path."""
    rows = [
        _make_gamelog_row("20240401", "0", "OAK", "BOS", "3", "5"),
        _make_gamelog_row("20240401", "0", "CWS", "DET", "1", "4"),
    ]
    lines = [",".join(f'"{v}"' for v in row) for row in rows]
    p = tmp_path / "GL2024.TXT"
    p.write_text("\n".join(lines))
    return p


@pytest.fixture
def gamelog_zip_bytes(gamelog_txt_path: Path) -> bytes:
    """In-memory ZIP archive containing GL2024.TXT."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(gamelog_txt_path, arcname="GL2024.TXT")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Crosswalk / team-map fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def retro_team_map_csv(tmp_path: Path) -> Path:
    """Write a minimal team ID map CSV and return its path."""
    content = (
        "retro_team_code,mlb_team_id,valid_from_season,valid_to_season\n"
        "BOS,111,1998,2099\n"
        "OAK,133,1968,2099\n"
        "DET,116,1901,2099\n"
        "CWS,145,1901,2099\n"
        "NYY,147,1903,2099\n"
    )
    p = tmp_path / "team_id_map_retro_to_mlb.csv"
    p.write_text(content)
    return p


@pytest.fixture
def gamelogs_df() -> pd.DataFrame:
    """Sample gamelogs DataFrame for crosswalk tests."""
    import datetime

    return pd.DataFrame(
        {
            "date": [datetime.date(2024, 4, 1), datetime.date(2024, 4, 1)],
            "game_num": [0, 0],
            "visiting_team": ["OAK", "CWS"],
            "home_team": ["BOS", "DET"],
            "visiting_score": [3, 1],
            "home_score": [5, 4],
        }
    )
