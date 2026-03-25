"""Optional background refresh: re-run the UPDATE pipeline on an interval.

Controlled by environment variables:

* ``MLB_AUTO_UPDATE_INTERVAL_HOURS`` — if > 0, sleep for that many hours between
  non-destructive UPDATE pipeline runs (schedule, gamelogs, features rebuild).
* ``MLB_AUTO_RETRAIN_QUICK`` — if ``1``, after each successful UPDATE also run
  a **quick** retrain (lighter than full CV; see admin retrain tier).

UPDATE reloads Parquet → DuckDB → in-memory scores.  Retraining is expensive;
keep it opt-in.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


async def periodic_data_and_model_refresh(
    interval_hours: float,
    reload_fn: Callable[[], None],
) -> None:
    """Loop: wait, then run UPDATE (and optionally quick RETRAIN) when idle."""
    from mlb_predict.app.admin import PipelineKind, conflicting_pipeline, run_pipeline

    min_sleep_sec = 300.0
    sleep_sec = max(min_sleep_sec, float(interval_hours) * 3600.0)
    retrain_quick = os.environ.get("MLB_AUTO_RETRAIN_QUICK", "").strip() == "1"

    while True:
        await asyncio.sleep(sleep_sec)
        if conflicting_pipeline() is not None:
            logger.info("Auto-update skipped — another pipeline is running")
            continue
        logger.info("Auto-update: starting UPDATE pipeline")
        await run_pipeline(PipelineKind.UPDATE, on_success=reload_fn)
        if retrain_quick and conflicting_pipeline() is None:
            logger.info("Auto-update: starting quick RETRAIN")
            await run_pipeline(
                PipelineKind.RETRAIN,
                on_success=reload_fn,
                training_tier="quick",
            )


def spawn_auto_update_task(
    loop: asyncio.AbstractEventLoop,
    reload_fn: Callable[[], None],
) -> asyncio.Task[Any] | None:
    """Start the periodic refresh task if ``MLB_AUTO_UPDATE_INTERVAL_HOURS`` > 0."""
    raw = os.environ.get("MLB_AUTO_UPDATE_INTERVAL_HOURS", "").strip()
    if not raw:
        return None
    try:
        hours = float(raw)
    except ValueError:
        logger.warning("Invalid MLB_AUTO_UPDATE_INTERVAL_HOURS=%r — ignoring", raw)
        return None
    if hours <= 0:
        return None
    logger.info("Scheduling auto-update every %.2f hours", hours)
    return loop.create_task(periodic_data_and_model_refresh(hours, reload_fn))
