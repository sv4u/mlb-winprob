"""
Property-based tests using Hypothesis.

These tests verify system-level invariants that must hold for all valid inputs,
complementing the example-based unit tests.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

if TYPE_CHECKING:
    pass

from winprob.mlbapi.client import MLBAPIClient
from winprob.mlbapi.schedule import normalize_schedule
from winprob.mlbapi.teams import build_team_maps
from winprob.retrosheet.gamelogs import sha256_bytes
from winprob.util.hashing import sha256_aggregate_of_files, sha256_file


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_team_id_st = st.integers(min_value=1, max_value=9999)
_game_pk_st = st.integers(min_value=1, max_value=999_999)
_abbrev_st = st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=2, max_size=4)
_name_st = st.text(min_size=3, max_size=40)


def _game_dict_st() -> st.SearchStrategy[dict]:
    """Strategy producing a single game dict compatible with normalize_schedule."""
    return st.fixed_dictionaries(
        {
            "gamePk": _game_pk_st,
            "gameDate": st.just("2024-04-01T17:10:00Z"),
            "gameType": st.sampled_from(["R", "S"]),
            "teams": st.fixed_dictionaries(
                {
                    "home": st.fixed_dictionaries(
                        {"team": st.fixed_dictionaries({"id": _team_id_st})}
                    ),
                    "away": st.fixed_dictionaries(
                        {"team": st.fixed_dictionaries({"id": _team_id_st})}
                    ),
                }
            ),
            "venue": st.just({"id": 1, "timeZone": {"id": "America/New_York"}}),
            "doubleHeader": st.sampled_from(["N", "Y"]),
            "gameNumber": st.integers(min_value=1, max_value=2),
            "status": st.fixed_dictionaries({"detailedState": st.just("Final")}),
        }
    )


# ---------------------------------------------------------------------------
# sha256_bytes — determinism & sensitivity
# ---------------------------------------------------------------------------


@given(data=st.binary(min_size=0, max_size=1024))
def test_sha256_bytes_deterministic(data: bytes) -> None:
    """sha256_bytes must always return the same digest for the same input."""
    assert sha256_bytes(data) == sha256_bytes(data)


@given(a=st.binary(min_size=1, max_size=512), b=st.binary(min_size=1, max_size=512))
def test_sha256_bytes_different_inputs_usually_differ(a: bytes, b: bytes) -> None:
    """Different inputs almost always produce different digests (collision resistance)."""
    assume(a != b)
    assert sha256_bytes(a) != sha256_bytes(b)


@given(data=st.binary(min_size=0, max_size=1024))
def test_sha256_bytes_matches_hashlib(data: bytes) -> None:
    """sha256_bytes must match the hashlib reference for all inputs."""
    assert sha256_bytes(data) == hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# sha256_file — determinism & multi-chunk consistency
# ---------------------------------------------------------------------------


@given(content=st.binary(min_size=0, max_size=4096))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_sha256_file_deterministic(content: bytes) -> None:
    """sha256_file must always return the same digest for the same file content."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        f.write(content)
        tmp = Path(f.name)
    try:
        assert sha256_file(tmp) == sha256_file(tmp)
    finally:
        tmp.unlink(missing_ok=True)


@given(content=st.binary(min_size=0, max_size=4096))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_sha256_file_matches_hashlib(content: bytes) -> None:
    """sha256_file must match hashlib's sha256 for arbitrary file contents."""
    expected = hashlib.sha256(content).hexdigest()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        f.write(content)
        tmp = Path(f.name)
    try:
        assert sha256_file(tmp) == expected
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# sha256_aggregate_of_files — order independence
# ---------------------------------------------------------------------------


@given(files=st.lists(st.binary(min_size=1, max_size=256), min_size=1, max_size=5, unique=True))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_sha256_aggregate_order_independent(files: list[bytes]) -> None:
    """Aggregate hash must be identical regardless of how files are listed."""
    import random

    with tempfile.TemporaryDirectory() as d:
        paths = []
        for i, content in enumerate(files):
            p = Path(d) / f"file_{i}.bin"
            p.write_bytes(content)
            paths.append(p)

        forward = sha256_aggregate_of_files(paths)
        shuffled = paths[:]
        random.shuffle(shuffled)
        backward = sha256_aggregate_of_files(shuffled)

        assert forward == backward


# ---------------------------------------------------------------------------
# MLBAPIClient — cache key invariants
# ---------------------------------------------------------------------------


@given(
    endpoint=st.text(
        min_size=1,
        max_size=50,
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="/_"),
    ),
    params=st.dictionaries(
        keys=st.text(
            min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll"))
        ),
        values=st.integers(min_value=1, max_value=9999),
        min_size=0,
        max_size=5,
    ),
)
def test_cache_key_deterministic(endpoint: str, params: dict) -> None:
    """Cache key must be identical for the same endpoint and params on every call."""
    client = MLBAPIClient()
    k1 = client._cache_key(endpoint, params)
    k2 = client._cache_key(endpoint, params)
    assert k1 == k2


@given(
    endpoint=st.text(
        min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=("Lu", "Ll"))
    ),
    params=st.dictionaries(
        keys=st.text(
            min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("Lu", "Ll"))
        ),
        values=st.integers(min_value=1, max_value=9999),
        min_size=0,
        max_size=4,
    ),
)
def test_cache_key_is_hex_sha256(endpoint: str, params: dict) -> None:
    """Cache key must always be a 64-character lowercase hex string."""
    client = MLBAPIClient()
    key = client._cache_key(endpoint, params)
    assert len(key) == 64
    assert all(c in "0123456789abcdef" for c in key)


@given(
    params_a=st.dictionaries(
        keys=st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz"),
        values=st.integers(min_value=1, max_value=100),
        min_size=1,
        max_size=4,
    )
)
def test_cache_key_same_regardless_of_dict_order(params_a: dict) -> None:
    """Cache key must be order-independent: dict iteration order must not matter."""
    client = MLBAPIClient()
    # Reconstruct in reversed insertion order (Python 3.7+ preserves insertion order)
    params_b = dict(reversed(list(params_a.items())))
    assert client._cache_key("endpoint", params_a) == client._cache_key("endpoint", params_b)


# ---------------------------------------------------------------------------
# normalize_schedule — output invariants
# ---------------------------------------------------------------------------


@given(games=st.lists(_game_dict_st(), min_size=0, max_size=20))
def test_normalize_schedule_rows_never_exceed_input(games: list[dict]) -> None:
    """normalize_schedule must never produce more rows than games in the input."""
    raw = {"dates": [{"date": "2024-04-01", "games": games}]}
    df = normalize_schedule(raw)
    assert len(df) <= len(games)


@given(games=st.lists(_game_dict_st(), min_size=1, max_size=20))
def test_normalize_schedule_game_pk_always_int(games: list[dict]) -> None:
    """All game_pk values in the output must be plain Python/numpy integers."""
    raw = {"dates": [{"date": "2024-04-01", "games": games}]}
    df = normalize_schedule(raw)
    if len(df) > 0:
        assert pd.api.types.is_integer_dtype(df["game_pk"])


@given(games=st.lists(_game_dict_st(), min_size=1, max_size=20))
def test_normalize_schedule_no_null_game_pk(games: list[dict]) -> None:
    """normalize_schedule must drop games with null game_pk, so no nulls remain."""
    raw = {"dates": [{"date": "2024-04-01", "games": games}]}
    df = normalize_schedule(raw)
    assert df["game_pk"].notna().all()


# ---------------------------------------------------------------------------
# build_team_maps — structural invariants
# ---------------------------------------------------------------------------


@given(
    teams=st.lists(
        st.fixed_dictionaries(
            {
                "mlb_team_id": _team_id_st,
                "abbrev": _abbrev_st,
                "name": _name_st,
            }
        ),
        min_size=0,
        max_size=30,
        unique_by=lambda t: t["mlb_team_id"],
    )
)
def test_build_team_maps_id_coverage(teams: list[dict]) -> None:
    """mlb_id_to_abbrev and mlb_id_to_name must have identical key sets."""
    df = pd.DataFrame(
        [
            {
                "season": 2024,
                "mlb_team_id": t["mlb_team_id"],
                "abbrev": t["abbrev"],
                "name": t["name"],
            }
            for t in teams
        ]
    )
    maps = build_team_maps(df)
    assert set(maps.mlb_id_to_abbrev.keys()) == set(maps.mlb_id_to_name.keys())


@given(
    teams=st.lists(
        st.fixed_dictionaries(
            {
                "mlb_team_id": _team_id_st,
                "abbrev": _abbrev_st,
                "name": _name_st,
            }
        ),
        min_size=1,
        max_size=30,
        unique_by=lambda t: (t["mlb_team_id"], t["abbrev"]),
    )
)
def test_build_team_maps_abbrev_roundtrip(teams: list[dict]) -> None:
    """mlb_id → abbrev → mlb_id must round-trip for every team when abbrevs are unique."""
    assume(len({t["abbrev"] for t in teams}) == len(teams))
    df = pd.DataFrame(
        [
            {
                "season": 2024,
                "mlb_team_id": t["mlb_team_id"],
                "abbrev": t["abbrev"],
                "name": t["name"],
            }
            for t in teams
        ]
    )
    maps = build_team_maps(df)
    for team in teams:
        tid = team["mlb_team_id"]
        abbrev = maps.mlb_id_to_abbrev[tid]
        assert maps.abbrev_to_mlb_id[abbrev] == tid
