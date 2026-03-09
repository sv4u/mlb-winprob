#!/usr/bin/env -S uv run python
"""Ingest live MLB game odds from The Odds API.

Writes raw JSON to data/processed/odds/live/ and optionally updates
latest_game_odds.parquet. When no API key is set (env ODDS_API_KEY or
data/processed/odds/config.json), prints a friendly message and exits
without calling the API.

Usage:
  uv run scripts/ingest_odds.py              # use cache if fresh
  uv run scripts/ingest_odds.py --refresh    # always fetch from API
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Repo root on path for mlb_predict
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


async def _run(refresh: bool) -> int:
    from mlb_predict.external.odds import OddsClient
    from mlb_predict.external.odds_config import get_odds_api_key

    if not get_odds_api_key():
        print(
            "Live Odds API key is not configured. Set ODDS_API_KEY or add the key "
            "via Admin Dashboard (Dashboard → Live Odds API Key) or in "
            "data/processed/odds/config.json.",
            file=sys.stderr,
        )
        return 1

    client = OddsClient()
    if not client.is_available():
        print("Odds client is not available (no API key).", file=sys.stderr)
        return 1

    events = await client.get_game_odds()
    if not events:
        print("No game odds returned (API may be empty or key invalid).")
        return 0

    client.events_to_retro(events)
    path = client.write_raw_game_odds(events)
    if path:
        print(f"Wrote {len(events)} events to {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest MLB game odds from The Odds API")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Always fetch from API (ignores cache)",
    )
    args = parser.parse_args()
    # --refresh is accepted for future cache use; OddsClient currently always fetches
    return asyncio.run(_run(refresh=args.refresh))


if __name__ == "__main__":
    raise SystemExit(main())
