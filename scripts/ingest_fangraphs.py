"""Ingest FanGraphs team-level advanced metrics (via pybaseball).

Writes one Parquet file per season to data/processed/fangraphs/.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from mlb_predict.statcast.fangraphs import fetch_team_advanced_stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", type=int, default=[])
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed/fangraphs"))
    ap.add_argument("--delay", type=float, default=1.5, help="Seconds between API calls")
    args = ap.parse_args()

    seasons = args.seasons or list(range(2002, 2026))  # FanGraphs Statcast from ~2002
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for s in seasons:
        out_path = args.out_dir / f"fangraphs_{s}.parquet"
        try:
            df = fetch_team_advanced_stats(s)
            if df.empty:
                logger.warning("%d: empty response", s)
                results.append({"season": s, "status": "empty"})
                continue
            df.to_parquet(out_path, index=False)
            logger.info("%d: %d teams → %s", s, len(df), out_path)
            results.append({"season": s, "status": "ok", "n_teams": len(df)})
        except Exception as exc:
            logger.error("%d: FAILED — %s", s, exc)
            results.append({"season": s, "status": "failed", "error": str(exc)})
        time.sleep(args.delay)

    summary_path = args.out_dir / "summary.json"
    pd.DataFrame(results).to_json(summary_path, orient="records", indent=2)
    logger.info("Summary → %s", summary_path)


if __name__ == "__main__":
    main()
