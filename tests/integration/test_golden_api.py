"""Verify HTTP API responses match golden files (snake_case, nulls, float precision).

Run after gateway refactor to ensure the gRPC gateway returns identical JSON.
Golden files are captured with scripts/capture_golden_api.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastapi.testclient import TestClient

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "golden"

# (golden_stem, method, path, params, json_body)
# params/body None means default query/body.
GOLDEN_ENDPOINTS = [
    ("api_health", "GET", "/api/health", None, None),
    ("api_version", "GET", "/api/version", None, None),
    ("api_seasons", "GET", "/api/seasons", None, None),
    ("api_teams", "GET", "/api/teams", None, None),
    ("api_games", "GET", "/api/games", {"limit": 5, "offset": 0}, None),
    ("api_upsets", "GET", "/api/upsets", {"limit": 5}, None),
    ("api_cv_summary", "GET", "/api/cv-summary", None, None),
    ("api_standings", "GET", "/api/standings", {"season": 2026}, None),
    ("api_team_stats", "GET", "/api/team-stats", {"season": 2026}, None),
    ("api_active_model", "GET", "/api/active-model", None, None),
    ("api_admin_status", "GET", "/api/admin/status", None, None),
    ("api_switch_model", "POST", "/api/admin/switch-model", None, {"model_type": "stacked"}),
]


def _normalize_for_compare(golden: dict | list, actual: dict | list) -> bool:
    """Return True if actual matches golden, with allowed variance for variable fields."""
    if type(golden) is not type(actual):
        return False
    if isinstance(golden, list):
        if len(golden) != len(actual):
            return False
        return all(_normalize_for_compare(g, a) for g, a in zip(golden, actual))
    # dict
    if set(golden.keys()) != set(actual.keys()):
        return False
    for k in golden:
        gv, av = golden[k], actual[k]
        if k == "git_commit":
            if not isinstance(av, str):
                return False
            continue
        if k == "ready":
            if not isinstance(av, bool):
                return False
            continue
        if isinstance(gv, (dict, list)):
            if not _normalize_for_compare(gv, av):
                return False
        elif gv != av:
            return False
    return True


@pytest.fixture
def api_client() -> TestClient:
    """TestClient for the FastAPI app (current or gateway)."""
    from mlb_predict.app.main import app

    return TestClient(app)


@pytest.mark.parametrize("stem,method,path,params,body", GOLDEN_ENDPOINTS)
def test_api_matches_golden(
    api_client: TestClient,
    stem: str,
    method: str,
    path: str,
    params: dict | None,
    body: dict | None,
) -> None:
    """Each endpoint response matches the corresponding golden file."""
    golden_path = GOLDEN_DIR / f"{stem}.json"
    if not golden_path.exists():
        pytest.skip(f"Golden file not found: {golden_path}")
    golden_data = json.loads(golden_path.read_text())

    if method == "GET":
        r = api_client.get(path, params=params or {})
    else:
        r = api_client.post(path, json=body or {})

    if r.status_code != 200:
        # If golden was captured with an error, compare status and body
        if isinstance(golden_data, dict) and golden_data.get("_status") == r.status_code:
            assert golden_data.get("body") == r.json()
        else:
            pytest.fail(f"{path} returned {r.status_code}: {r.text}")
        return

    actual = r.json()
    # Variable fields (e.g. git_commit) are accepted by _normalize_for_compare
    assert _normalize_for_compare(golden_data, actual), (
        f"{stem}: response does not match golden file"
    )


def test_game_detail_matches_golden_if_present(api_client: TestClient) -> None:
    """If a api_game_detail_<pk>.json golden exists, GET /api/games/<pk> matches it."""
    for p in GOLDEN_DIR.glob("api_game_detail_*.json"):
        if "placeholder" in p.stem:
            continue
        # stem is api_game_detail_745444
        pk = p.stem.replace("api_game_detail_", "")
        try:
            game_pk = int(pk)
        except ValueError:
            continue
        golden_data = json.loads(p.read_text())
        if isinstance(golden_data, dict) and golden_data.get("_skip"):
            continue
        r = api_client.get(f"/api/games/{game_pk}")
        if r.status_code != 200:
            pytest.skip(f"game_pk {game_pk} not available (status {r.status_code})")
        actual = r.json()
        assert _normalize_for_compare(golden_data, actual), (
            f"game detail {game_pk} does not match golden"
        )
        return  # one is enough
    pytest.skip("No api_game_detail_<pk>.json golden file (except placeholder)")
