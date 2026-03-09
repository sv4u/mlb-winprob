"""Capture current HTTP API responses to tests/golden/*.json for gateway regression tests.

Run from repo root with the app's data loaded (e.g. after running the app once, or with
data/processed and data/models present). Uses FastAPI TestClient so no server is required.

  uv run python scripts/capture_golden_api.py

Writes tests/golden/api_<name>.json. For deterministic CI, set GIT_COMMIT=testabc before
capturing so admin/status and version responses are stable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GOLDEN_DIR = REPO_ROOT / "tests" / "golden"
GOLDEN_DIR.mkdir(parents=True, exist_ok=True)


def capture() -> None:
    from fastapi.testclient import TestClient

    from mlb_predict.app.main import app

    client = TestClient(app)
    # Trigger lifespan so startup runs
    client.get("/api/health")

    def get(path: str, params: dict | None = None) -> dict | list:
        r = client.get(path, params=params or {})
        r.raise_for_status()
        return r.json()

    def post(path: str, json_body: dict | None = None) -> dict:
        r = client.post(path, json=json_body or {})
        r.raise_for_status()
        return r.json()

    captures: list[tuple[str, dict | list]] = []

    # 1. health
    captures.append(("api_health", get("/api/health")))
    # 2. version
    captures.append(("api_version", get("/api/version")))
    # 3. seasons (may 503 if not ready)
    r = client.get("/api/seasons")
    captures.append(
        (
            "api_seasons",
            r.json() if r.status_code == 200 else {"_status": r.status_code, "body": r.json()},
        )
    )
    # 4. teams
    r = client.get("/api/teams")
    captures.append(
        (
            "api_teams",
            r.json() if r.status_code == 200 else {"_status": r.status_code, "body": r.json()},
        )
    )
    # 5. games
    r = client.get("/api/games", params={"limit": 5, "offset": 0})
    captures.append(
        (
            "api_games",
            r.json() if r.status_code == 200 else {"_status": r.status_code, "body": r.json()},
        )
    )

    game_pk: int | None = None
    if r.status_code == 200 and isinstance(r.json(), dict) and r.json().get("games"):
        game_pk = r.json()["games"][0].get("game_pk")

    # 6. game detail (use first game_pk from games if available)
    if game_pk is not None:
        r = client.get(f"/api/games/{game_pk}")
        captures.append(
            (
                f"api_game_detail_{game_pk}",
                r.json() if r.status_code == 200 else {"_status": r.status_code, "body": r.json()},
            )
        )
    else:
        captures.append(("api_game_detail_placeholder", {"_skip": "no game_pk from /api/games"}))

    # 7. upsets
    r = client.get("/api/upsets", params={"limit": 5})
    captures.append(
        (
            "api_upsets",
            r.json() if r.status_code == 200 else {"_status": r.status_code, "body": r.json()},
        )
    )
    # 8. cv-summary
    captures.append(("api_cv_summary", get("/api/cv-summary")))
    # 9. standings
    r = client.get("/api/standings", params={"season": 2026})
    captures.append(
        (
            "api_standings",
            r.json() if r.status_code == 200 else {"_status": r.status_code, "body": r.json()},
        )
    )
    # 10. team-stats
    r = client.get("/api/team-stats", params={"season": 2026})
    captures.append(
        (
            "api_team_stats",
            r.json() if r.status_code == 200 else {"_status": r.status_code, "body": r.json()},
        )
    )
    # 11. active-model
    captures.append(("api_active_model", get("/api/active-model")))
    # 12. admin status
    captures.append(("api_admin_status", get("/api/admin/status")))
    # 13. switch-model (no body = validation error; use stacked to stay consistent)
    r = client.post("/api/admin/switch-model", json={"model_type": "stacked"})
    captures.append(
        (
            "api_switch_model",
            r.json() if r.status_code == 200 else {"_status": r.status_code, "body": r.json()},
        )
    )

    for name, data in captures:
        out = GOLDEN_DIR / f"{name}.json"
        with open(out, "w") as f:
            json.dump(data, f, indent=2, sort_keys=False)
        print(f"Wrote {out}")


if __name__ == "__main__":
    capture()
