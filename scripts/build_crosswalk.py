from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from winprob.crosswalk.build import build_crosswalk
from winprob.ingest.id_map import load_retro_team_map

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-5s  %(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", type=int, default=[])
    ap.add_argument("--min-coverage", type=float, default=99.0)
    args = ap.parse_args()

    seasons = args.seasons or list(range(2000, 2026))
    out_dir = Path("data/processed/crosswalk")
    out_dir.mkdir(parents=True, exist_ok=True)

    retro_map = load_retro_team_map()

    rows = []
    for s in seasons:
        schedule_path = Path("data/processed/schedule") / f"games_{s}.parquet"
        gamelogs_path = Path("data/processed/retrosheet") / f"gamelogs_{s}.parquet"

        if not gamelogs_path.exists():
            logger.info("Season %d: no gamelogs found at %s — skipping crosswalk", s, gamelogs_path)
            continue

        if not schedule_path.exists():
            logger.info("Season %d: no schedule found at %s — skipping crosswalk", s, schedule_path)
            continue

        try:
            schedule = pd.read_parquet(schedule_path)
            gamelogs = pd.read_parquet(gamelogs_path)
            res = build_crosswalk(
                season=s, schedule=schedule, gamelogs=gamelogs, retro_team_map=retro_map
            )

            res.df.to_parquet(out_dir / f"game_id_map_{s}.parquet", index=False)
            res.df[res.df["status"] != "matched"].to_parquet(
                out_dir / f"unresolved_{s}.parquet", index=False
            )

            rows.append(
                {
                    "season": s,
                    "coverage_pct": res.coverage_pct,
                    "matched": res.matched,
                    "missing": res.missing,
                    "ambiguous": res.ambiguous,
                }
            )
        except Exception as e:
            logger.error("Season %d crosswalk failed: %s", s, e)
            (out_dir / f"crosswalk_failed_{s}.txt").write_text(str(e), encoding="utf-8")
            rows.append(
                {
                    "season": s,
                    "coverage_pct": 0.0,
                    "matched": 0,
                    "missing": 0,
                    "ambiguous": 0,
                    "error": str(e),
                }
            )

    cov = pd.DataFrame(rows).sort_values("season")
    cov.to_parquet(out_dir / "crosswalk_coverage_report.parquet", index=False)
    cov.to_csv(out_dir / "crosswalk_coverage_report.csv", index=False)

    bad = cov[pd.to_numeric(cov["coverage_pct"], errors="coerce").fillna(0.0) < args.min_coverage]
    bad.to_csv(out_dir / "crosswalk_seasons_below_threshold.csv", index=False)


if __name__ == "__main__":
    main()
