"""Compute drift between successive prediction snapshots for a given season."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from winprob.drift.compute import compute_drift


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--snapshot-dir", type=Path, default=Path("data/processed/predictions"))
    ap.add_argument("--drift-dir", type=Path, default=Path("data/processed/drift"))
    args = ap.parse_args()

    result = compute_drift(
        season=args.season,
        snapshot_dir=args.snapshot_dir,
        drift_dir=args.drift_dir,
    )

    if not result:
        print(f"Season {args.season}: fewer than 2 snapshots — no drift to compute.")
        return

    for kind, metrics in result.items():
        print(f"\n{kind.upper()} drift (season {args.season}):")
        for k, v in asdict(metrics).items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
