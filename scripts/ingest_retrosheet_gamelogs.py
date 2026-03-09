from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import pandas as pd

from mlb_predict.retrosheet.gamelogs import (
    RetrosheetGLSource,
    download_gamelog_txt,
    parse_gamelog_txt,
)
from mlb_predict.util.hashing import sha256_file


async def ingest_one(
    season: int, *, refresh: bool, primary_source: str, enable_fallback: bool
) -> dict:
    raw_path = Path("data/raw/retrosheet/gamelogs") / f"GL{season}.TXT"
    meta = await download_gamelog_txt(
        season=season,
        out_path=raw_path,
        refresh=refresh,
        source=RetrosheetGLSource(primary=primary_source, enable_fallback=enable_fallback),
    )

    df = parse_gamelog_txt(raw_path)
    out_dir = Path("data/processed/retrosheet")
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / f"gamelogs_{season}.parquet"
    csv_path = out_dir / f"gamelogs_{season}.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    checksum = {
        "season": season,
        "row_count": int(len(df)),
        "raw_sha256": meta["sha256"],
        "parquet_sha256": sha256_file(parquet_path),
        "csv_sha256": sha256_file(csv_path),
        "raw_path": str(raw_path),
        "source_used": meta.get("source_used"),
        "url_used": meta.get("url_used"),
        "fallback_reason": meta.get("fallback_reason"),
    }
    (out_dir / f"gamelogs_{season}.checksum.json").write_text(
        pd.Series(checksum).to_json(), encoding="utf-8"
    )
    return checksum


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", type=int, default=[])
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--primary-source", choices=["chadwick", "retrosheet"], default="chadwick")
    ap.add_argument("--no-fallback", action="store_true")
    args = ap.parse_args()

    seasons = args.seasons or list(range(2000, 2026))
    results = []
    for s in seasons:
        try:
            results.append(
                await ingest_one(
                    s,
                    refresh=args.refresh,
                    primary_source=args.primary_source,
                    enable_fallback=not args.no_fallback,
                )
            )
        except Exception as e:
            results.append({"season": s, "status": "failed", "error": str(e)})

    out = Path("data/processed/retrosheet")
    out.mkdir(parents=True, exist_ok=True)
    (out / "ingest_gamelogs_summary.json").write_text(
        pd.DataFrame(results).to_json(orient="records", indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    asyncio.run(main())
