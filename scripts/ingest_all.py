from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


@dataclass
class TaskResult:
    stage: str
    season: int
    status: str
    error_type: str | None = None
    error_message: str | None = None
    duration_seconds: float | None = None


async def _run(cmd: str) -> int:
    proc = await asyncio.create_subprocess_shell(cmd)
    await proc.wait()
    return int(proc.returncode)


async def run_stage(stage: str, season: int, cmd: str) -> TaskResult:
    loop = asyncio.get_running_loop()
    start = loop.time()
    try:
        rc = await _run(cmd)
        dur = loop.time() - start
        if rc != 0:
            return TaskResult(
                stage=stage,
                season=season,
                status="failed",
                error_type="NonZeroExit",
                error_message=str(rc),
                duration_seconds=dur,
            )
        return TaskResult(stage=stage, season=season, status="success", duration_seconds=dur)
    except Exception as e:
        dur = loop.time() - start
        return TaskResult(
            stage=stage,
            season=season,
            status="failed",
            error_type=type(e).__name__,
            error_message=str(e),
            duration_seconds=dur,
        )


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", type=int, default=[])
    ap.add_argument("--refresh-mlbapi", action="store_true")
    ap.add_argument("--refresh-retro", action="store_true")
    args = ap.parse_args()

    current_year = datetime.now(timezone.utc).year
    seasons = args.seasons or list(range(2000, current_year + 1))
    # Include the current year only when explicitly requested or when it is
    # within the historical default range; always include it so ingestion can
    # pick up a season that has already started, but downstream scripts will
    # gracefully handle a missing-data year via try/except.
    seasons = sorted(set(seasons))

    results: list[TaskResult] = []
    mlb_sem = asyncio.Semaphore(4)
    retro_sem = asyncio.Semaphore(4)

    async def schedule_task(season: int) -> TaskResult:
        async with mlb_sem:
            cmd = f"{sys.executable} scripts/ingest_schedule.py --seasons {season}" + (
                " --refresh-mlbapi" if args.refresh_mlbapi else ""
            )
            return await run_stage("schedule", season, cmd)

    async def retro_task(season: int) -> TaskResult:
        async with retro_sem:
            cmd = f"{sys.executable} scripts/ingest_retrosheet_gamelogs.py --seasons {season}" + (
                " --refresh" if args.refresh_retro else ""
            )
            return await run_stage("retrosheet_gamelogs", season, cmd)

    results.extend(await asyncio.gather(*[asyncio.create_task(schedule_task(s)) for s in seasons]))
    results.extend(await asyncio.gather(*[asyncio.create_task(retro_task(s)) for s in seasons]))

    results.append(
        await run_stage(
            "crosswalk",
            -1,
            f"{sys.executable} scripts/build_crosswalk.py --seasons " + " ".join(map(str, seasons)),
        )
    )

    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.__dict__ for r in results])
    df.to_json(out / "ingest_summary.json", orient="records", indent=2)
    df.to_parquet(out / "ingest_failures.parquet", index=False)

    if (df["status"] == "failed").any():
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
