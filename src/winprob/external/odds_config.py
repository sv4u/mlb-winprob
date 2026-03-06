"""Odds API key configuration: env var and config file under data/processed/odds/.

The API key is read from (1) ODDS_API_KEY env var, then (2) data/processed/odds/config.json.
Never log or expose the key in API responses. Used by OddsClient and admin dashboard.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ODDS_CONFIG_PATH = _REPO_ROOT / "data" / "processed" / "odds" / "config.json"

CONFIG_KEY = "api_key"


def get_odds_api_key() -> str | None:
    """Return the Odds API key from env or config file, or None if not set.

    Precedence: ODDS_API_KEY env var, then config file. Never returns the key
    in logs or to callers that might expose it; this function only returns the
    value for use in server-side requests.
    """
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if key:
        return key
    if not ODDS_CONFIG_PATH.exists():
        return None
    try:
        data = json.loads(ODDS_CONFIG_PATH.read_text())
        key = (data.get(CONFIG_KEY) or "").strip()
        return key if key else None
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def set_odds_api_key(api_key: str) -> None:
    """Write the API key to data/processed/odds/config.json.

    Creates parent directories if needed. Overwrites existing file.
    Does not log the key.
    """
    api_key = (api_key or "").strip()
    ODDS_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ODDS_CONFIG_PATH.write_text(json.dumps({CONFIG_KEY: api_key}, indent=2))


def get_odds_config_status() -> dict[str, str | bool | None]:
    """Return status for admin UI: configured, source (env | file | null).

    Never includes the actual key.
    """
    if os.environ.get("ODDS_API_KEY", "").strip():
        return {"configured": True, "source": "env"}
    if ODDS_CONFIG_PATH.exists():
        try:
            data = json.loads(ODDS_CONFIG_PATH.read_text())
            if (data.get(CONFIG_KEY) or "").strip():
                return {"configured": True, "source": "file"}
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    return {"configured": False, "source": None}
