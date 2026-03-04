"""Admin pipeline runner — async background task management for ingest and retrain.

Tracks pipeline state, captures log output, and triggers model reload on success.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
_MODEL_DIR = _REPO_ROOT / "data" / "models"
_LOG_DIR = _REPO_ROOT / "logs"
_MAX_LOG_LINES = 500


class PipelineKind(str, Enum):
    INGEST = "ingest"
    UPDATE = "update"
    RETRAIN = "retrain"


class PipelineStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class PipelineState:
    """Mutable state container for a single pipeline run."""

    kind: PipelineKind
    status: PipelineStatus = PipelineStatus.IDLE
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_seconds: float | None = None
    log_lines: list[str] = field(default_factory=list)
    error: str | None = None

    def reset(self) -> None:
        self.status = PipelineStatus.RUNNING
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.finished_at = None
        self.elapsed_seconds = None
        self.log_lines = []
        self.error = None

    def finish(self, ok: bool, error: str | None = None) -> None:
        self.status = PipelineStatus.SUCCESS if ok else PipelineStatus.FAILED
        self.finished_at = datetime.now(timezone.utc).isoformat()
        self.error = error
        if self.started_at:
            start = datetime.fromisoformat(self.started_at)
            end = datetime.fromisoformat(self.finished_at)
            self.elapsed_seconds = (end - start).total_seconds()
        else:
            self.elapsed_seconds = None

    def append_log(self, line: str) -> None:
        if len(self.log_lines) < _MAX_LOG_LINES:
            self.log_lines.append(line.rstrip("\n"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "status": self.status.value,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
            "log_tail": self.log_lines[-80:],
            "log_line_count": len(self.log_lines),
        }


_states: dict[PipelineKind, PipelineState] = {
    PipelineKind.INGEST: PipelineState(kind=PipelineKind.INGEST),
    PipelineKind.UPDATE: PipelineState(kind=PipelineKind.UPDATE),
    PipelineKind.RETRAIN: PipelineState(kind=PipelineKind.RETRAIN),
}


def get_state(kind: PipelineKind) -> PipelineState:
    return _states[kind]


def conflicting_pipeline() -> PipelineKind | None:
    """Return the first pipeline that is currently RUNNING, or None."""
    for k, s in _states.items():
        if s.status == PipelineStatus.RUNNING:
            return k
    return None


def _clean_processed_data(state: PipelineState) -> None:
    """Remove all processed data except immutable/protected artifacts."""
    import shutil

    preserve = {
        "team_id_map_retro_to_mlb.csv",
        "predictions",
        "drift",
        "vegas",
        "statcast_player",
    }
    if not _PROCESSED_DIR.exists():
        state.append_log("  No processed data directory — nothing to clean.")
        return
    for item in sorted(_PROCESSED_DIR.iterdir()):
        if item.name in preserve:
            state.append_log(f"  Preserved: {item.name}")
            continue
        if item.is_dir():
            shutil.rmtree(item)
            state.append_log(f"  Removed directory: {item.name}/")
        elif item.is_file():
            item.unlink()
            state.append_log(f"  Removed file: {item.name}")


def _clean_models(state: PipelineState) -> None:
    """Remove all model artifacts under data/models/."""
    import shutil

    if not _MODEL_DIR.exists():
        state.append_log("  No models directory — nothing to clean.")
        return
    for item in sorted(_MODEL_DIR.iterdir()):
        if item.is_dir():
            shutil.rmtree(item)
            state.append_log(f"  Removed model: {item.name}/")
        elif item.is_file():
            item.unlink()
            state.append_log(f"  Removed file: {item.name}")


def _python_bin() -> str:
    import shutil

    return shutil.which("python") or "python"


def _ingest_commands() -> list[tuple[str, str]]:
    """Full re-ingestion of all seasons (2000–current year)."""
    from datetime import datetime as dt

    python = _python_bin()
    year = dt.now(timezone.utc).year
    seasons = " ".join(str(s) for s in range(2000, year + 1))

    return [
        (
            f"Ingest schedules & gamelogs (2000–{year})",
            f"{python} scripts/ingest_all.py --seasons {seasons} --refresh-mlbapi --refresh-retro",
        ),
        (
            f"Ingest pitcher stats (2000–{year})",
            f"{python} scripts/ingest_pitcher_stats.py --seasons {seasons} --refresh",
        ),
        (
            f"Ingest FanGraphs metrics (2000–{year})",
            f"{python} scripts/ingest_fangraphs.py --seasons {seasons}",
        ),
        (
            f"Ingest weather data (2000–{year})",
            f"{python} scripts/ingest_weather.py --seasons {seasons}",
        ),
        (
            f"Build feature matrices (2000–{year})",
            f"{python} scripts/build_features.py --seasons {seasons}",
        ),
        ("Build 2026 pre-season features (if needed)", f"{python} scripts/build_features_2026.py"),
    ]


def _update_commands() -> list[tuple[str, str]]:
    """Update current season only (non-destructive)."""
    from datetime import datetime as dt

    python = _python_bin()
    year = str(dt.now(timezone.utc).year)

    return [
        (
            "Refresh schedule (MLB Stats API)",
            f"{python} scripts/ingest_schedule.py --seasons {year} --refresh-mlbapi",
        ),
        (
            "Refresh Retrosheet gamelogs",
            f"{python} scripts/ingest_retrosheet_gamelogs.py --seasons {year} --refresh",
        ),
        ("Rebuild crosswalk", f"{python} scripts/build_crosswalk.py --seasons {year}"),
        (
            "Refresh pitcher stats",
            f"{python} scripts/ingest_pitcher_stats.py --seasons {year} --refresh",
        ),
        (
            "Refresh FanGraphs metrics",
            f"{python} scripts/ingest_fangraphs.py --seasons {year}",
        ),
        (
            "Refresh weather data",
            f"{python} scripts/ingest_weather.py --seasons {year}",
        ),
        ("Rebuild feature matrix", f"{python} scripts/build_features.py --seasons {year}"),
        ("Build 2026 pre-season features (if needed)", f"{python} scripts/build_features_2026.py"),
    ]


def _retrain_commands() -> list[tuple[str, str]]:
    """Retrain all production models from scratch."""
    python = _python_bin()
    models = "logistic lightgbm xgboost catboost mlp stacked"
    return [
        ("Train all production models", f"{python} scripts/train_model.py --models {models}"),
    ]


async def _stream_process(
    cmd: str,
    state: PipelineState,
) -> int:
    """Run a shell command, streaming stdout/stderr into state.log_lines."""
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(_REPO_ROOT),
    )
    assert proc.stdout is not None
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        decoded = line.decode("utf-8", errors="replace")
        state.append_log(decoded)
    await proc.wait()
    return proc.returncode or 0


async def run_pipeline(
    kind: PipelineKind,
    on_success: Callable[[], Any] | None = None,
) -> None:
    """Execute a pipeline (ingest or retrain) as a background task.

    After successful completion, calls ``on_success`` (typically a model reload)
    and writes a timestamp marker.
    """
    state = get_state(kind)

    blocker = conflicting_pipeline()
    if blocker is not None:
        logger.warning("Pipeline %s blocked — %s is already running", kind.value, blocker.value)
        return

    state.reset()

    try:
        if kind == PipelineKind.INGEST:
            state.append_log(">>> Clearing all processed data…")
            _clean_processed_data(state)
        elif kind == PipelineKind.RETRAIN:
            state.append_log(">>> Clearing all model artifacts…")
            _clean_models(state)

        if kind == PipelineKind.INGEST:
            commands = _ingest_commands()
        elif kind == PipelineKind.UPDATE:
            commands = _update_commands()
        else:
            commands = _retrain_commands()
        for desc, cmd in commands:
            state.append_log(f">>> {desc}")
            logger.info("[%s] %s", kind.value, desc)
            step_t0 = time.monotonic()
            rc = await _stream_process(cmd, state)
            step_ms = (time.monotonic() - step_t0) * 1000
            state.append_log(f"    [{step_ms / 1000:.1f}s]")
            logger.info("[%s] %s completed in %.1fs", kind.value, desc, step_ms / 1000)
            if rc != 0:
                state.finish(ok=False, error=f"Step '{desc}' exited with code {rc}")
                return

        if kind == PipelineKind.RETRAIN:
            marker = _MODEL_DIR / ".last_retrain"
        else:
            marker = _PROCESSED_DIR / ".last_ingest"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

        state.finish(ok=True)
        logger.info("[%s] Pipeline completed successfully", kind.value)

        if on_success:
            state.append_log(">>> Reloading model and data cache…")
            try:
                on_success()
                state.append_log(">>> Reload complete.")
            except Exception as exc:
                state.append_log(f">>> Reload warning: {exc}")
                logger.warning("Post-pipeline reload failed: %s", exc)

    except Exception as exc:
        state.finish(ok=False, error=str(exc))
        logger.exception("[%s] Pipeline failed with exception", kind.value)


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------


def gather_data_status() -> dict[str, Any]:
    """Collect information about ingested data for the dashboard."""
    features_dir = _PROCESSED_DIR / "features"
    feature_files = sorted(features_dir.glob("features_*.parquet")) if features_dir.exists() else []
    seasons_available: list[int] = []
    total_games = 0
    for f in feature_files:
        try:
            import pandas as pd

            df = pd.read_parquet(f, columns=["game_pk"])
            season = int(f.stem.split("_")[1])
            seasons_available.append(season)
            total_games += len(df)
        except Exception:
            pass

    last_ingest: str | None = None
    marker = _PROCESSED_DIR / ".last_ingest"
    if marker.exists():
        last_ingest = marker.read_text().strip()

    crosswalk_report: dict[str, Any] | None = None
    cw_path = _PROCESSED_DIR / "crosswalk" / "crosswalk_coverage_report.csv"
    if cw_path.exists():
        try:
            import pandas as pd

            cw = pd.read_csv(cw_path)
            crosswalk_report = cw.to_dict(orient="records")
        except Exception:
            pass

    return {
        "seasons_available": sorted(seasons_available),
        "total_games": total_games,
        "feature_files_count": len(feature_files),
        "last_ingest": last_ingest,
        "crosswalk_report": crosswalk_report,
    }


def gather_model_status() -> dict[str, Any]:
    """Collect information about trained models for the dashboard."""
    last_retrain: str | None = None
    marker = _MODEL_DIR / ".last_retrain"
    if marker.exists():
        last_retrain = marker.read_text().strip()

    models_found: list[dict[str, Any]] = []
    if _MODEL_DIR.exists():
        for d in sorted(_MODEL_DIR.iterdir()):
            if d.is_dir() and (d / "model.joblib").exists():
                meta_path = d / "metadata.json"
                meta: dict[str, Any] = {}
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text())
                    except Exception:
                        pass
                models_found.append(
                    {
                        "name": d.name,
                        "model_type": meta.get("model_type", d.name.split("_")[0]),
                        "version": meta.get("model_version", meta.get("version", "unknown")),
                        "training_seasons": meta.get("training_seasons"),
                        "trained_at": meta.get("trained_at"),
                    }
                )

    cv_summary: list[dict[str, Any]] = []
    for name in ("cv_summary_v3.json", "cv_summary_v2.json", "cv_summary.json"):
        p = _MODEL_DIR / name
        if p.exists():
            try:
                cv_summary = json.loads(p.read_text())
            except Exception:
                pass
            break

    latest_per_type: dict[str, dict[str, Any]] = {}
    for m in models_found:
        mt = m["model_type"]
        if mt not in latest_per_type or m["name"] > latest_per_type[mt]["name"]:
            latest_per_type[mt] = m

    return {
        "last_retrain": last_retrain,
        "models_found": models_found,
        "production_models": list(latest_per_type.values()),
        "total_artifacts": len(models_found),
        "cv_summary": cv_summary,
    }
