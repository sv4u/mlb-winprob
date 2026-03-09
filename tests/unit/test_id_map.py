"""Tests for mlb_predict.ingest.id_map."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pytest

if TYPE_CHECKING:
    pass

from mlb_predict.ingest.id_map import RetroTeamMap, load_retro_team_map


# ---------------------------------------------------------------------------
# RetroTeamMap.retro_to_mlb_id
# ---------------------------------------------------------------------------


@pytest.fixture
def team_map_df() -> pd.DataFrame:
    """Minimal DataFrame for constructing a RetroTeamMap."""
    return pd.DataFrame(
        {
            "retro_team_code": ["BOS", "NYY", "OAK", "MON"],
            "mlb_team_id": [111, 147, 133, None],
            "valid_from_season": [1998, 1903, 1968, 1969],
            "valid_to_season": [2099, 2099, 2099, 2004],
        }
    )


@pytest.fixture
def retro_map(team_map_df: pd.DataFrame) -> RetroTeamMap:
    """RetroTeamMap wrapping the minimal fixture DataFrame."""
    return RetroTeamMap(df=team_map_df.dropna(subset=["mlb_team_id"]))


def test_retro_to_mlb_id_found(retro_map: RetroTeamMap) -> None:
    """retro_to_mlb_id must return the correct integer MLB team ID."""
    assert retro_map.retro_to_mlb_id("BOS", 2024) == 111
    assert retro_map.retro_to_mlb_id("NYY", 2024) == 147
    assert retro_map.retro_to_mlb_id("OAK", 2024) == 133


def test_retro_to_mlb_id_valid_at_boundary_start(retro_map: RetroTeamMap) -> None:
    """A team at the exact valid_from_season boundary must be found."""
    assert retro_map.retro_to_mlb_id("BOS", 1998) == 111


def test_retro_to_mlb_id_valid_at_boundary_end(retro_map: RetroTeamMap) -> None:
    """A team at the exact valid_to_season boundary must be found."""
    assert retro_map.retro_to_mlb_id("BOS", 2099) == 111


def test_retro_to_mlb_id_before_valid_from_raises(retro_map: RetroTeamMap) -> None:
    """A season before valid_from_season must raise KeyError."""
    with pytest.raises(KeyError, match="BOS"):
        retro_map.retro_to_mlb_id("BOS", 1997)


def test_retro_to_mlb_id_after_valid_to_raises(team_map_df: pd.DataFrame) -> None:
    """A season after valid_to_season must raise KeyError (team no longer valid)."""
    df = team_map_df.dropna(subset=["mlb_team_id"])
    m = RetroTeamMap(df=df)
    with pytest.raises(KeyError, match="MON"):
        m.retro_to_mlb_id("MON", 2005)


def test_retro_to_mlb_id_unknown_team_raises(retro_map: RetroTeamMap) -> None:
    """An unknown retro_team_code must raise KeyError."""
    with pytest.raises(KeyError, match="UNKNOWN"):
        retro_map.retro_to_mlb_id("UNKNOWN", 2024)


def test_retro_to_mlb_id_ambiguous_raises() -> None:
    """Multiple matching rows for the same team/season must raise KeyError."""
    df = pd.DataFrame(
        {
            "retro_team_code": ["BOS", "BOS"],
            "mlb_team_id": [111, 112],
            "valid_from_season": [2000, 2000],
            "valid_to_season": [2099, 2099],
        }
    )
    m = RetroTeamMap(df=df)
    with pytest.raises(KeyError, match="BOS"):
        m.retro_to_mlb_id("BOS", 2024)


def test_retro_to_mlb_id_return_type_is_int(retro_map: RetroTeamMap) -> None:
    """retro_to_mlb_id must always return a Python int (not numpy int)."""
    result = retro_map.retro_to_mlb_id("BOS", 2024)
    assert type(result) is int


# ---------------------------------------------------------------------------
# load_retro_team_map
# ---------------------------------------------------------------------------


def test_load_retro_team_map_success(retro_team_map_csv: Path) -> None:
    """load_retro_team_map must successfully load a valid CSV file."""
    m = load_retro_team_map(retro_team_map_csv)
    assert isinstance(m, RetroTeamMap)
    assert len(m.df) >= 1


def test_load_retro_team_map_values(retro_team_map_csv: Path) -> None:
    """load_retro_team_map must correctly parse team IDs from the CSV."""
    m = load_retro_team_map(retro_team_map_csv)
    assert m.retro_to_mlb_id("BOS", 2024) == 111
    assert m.retro_to_mlb_id("OAK", 2024) == 133


def test_load_retro_team_map_missing_column_raises(tmp_path: Path) -> None:
    """load_retro_team_map must raise ValueError when a required column is absent."""
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("retro_team_code,mlb_team_id\nBOS,111\n")
    with pytest.raises(ValueError, match="Missing required columns"):
        load_retro_team_map(bad_csv)


def test_load_retro_team_map_all_required_columns(retro_team_map_csv: Path) -> None:
    """Loaded DataFrame must contain all four required columns."""
    m = load_retro_team_map(retro_team_map_csv)
    required = {"retro_team_code", "mlb_team_id", "valid_from_season", "valid_to_season"}
    assert required.issubset(set(m.df.columns))


def test_load_retro_team_map_file_not_found() -> None:
    """load_retro_team_map must raise an error when the file does not exist."""
    with pytest.raises(Exception):  # FileNotFoundError from pd.read_csv
        load_retro_team_map(Path("/nonexistent/path/team_map.csv"))
