"""Admin pipeline runner — async background task management for ingest and retrain.

Tracks pipeline state, captures log output, and triggers model reload on success.
Also provides WebSocket-based shell and Python REPL sessions.
"""

from __future__ import annotations

import asyncio
import code
import io
import json
import logging
import time
from contextlib import redirect_stderr, redirect_stdout
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


@dataclass
class PipelineOptions:
    """User-configurable options for pipeline runs."""

    include_preseason: bool = True
    seasons: list[int] | None = None
    refresh_mlbapi: bool = True
    refresh_retro: bool = True
    skip_cv: bool = False
    skip_stage1: bool = False


class PipelineStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class StepInfo:
    """Progress tracker for a single pipeline step."""

    description: str
    status: str = "pending"
    elapsed_seconds: float | None = None


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
    steps: list[StepInfo] = field(default_factory=list)
    current_step_index: int = -1

    def reset(self) -> None:
        self.status = PipelineStatus.RUNNING
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.finished_at = None
        self.elapsed_seconds = None
        self.log_lines = []
        self.error = None
        self.steps = []
        self.current_step_index = -1

    def init_steps(self, descriptions: list[str]) -> None:
        """Pre-populate the step list so the UI can show all steps upfront."""
        self.steps = [StepInfo(description=d) for d in descriptions]

    def begin_step(self, index: int) -> None:
        """Mark a step as running."""
        self.current_step_index = index
        if 0 <= index < len(self.steps):
            self.steps[index].status = "running"

    def complete_step(self, index: int, elapsed: float) -> None:
        """Mark a step as complete with its duration."""
        if 0 <= index < len(self.steps):
            self.steps[index].status = "complete"
            self.steps[index].elapsed_seconds = elapsed

    def fail_step(self, index: int) -> None:
        """Mark a step as failed."""
        if 0 <= index < len(self.steps):
            self.steps[index].status = "failed"

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
            "steps": [
                {
                    "description": s.description,
                    "status": s.status,
                    "elapsed_seconds": s.elapsed_seconds,
                }
                for s in self.steps
            ],
            "current_step_index": self.current_step_index,
            "total_steps": len(self.steps),
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
        "weather",
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


def _archive_models(state: PipelineState, tier: str | None = None) -> None:
    """Archive model artifacts to data/models/archive/ for drift analysis.

    When *tier* is ``"quick"`` or ``"full"``, only archives models in that
    tier's subdirectory.  When ``None``, archives legacy top-level models.
    """
    from mlb_predict.model.artifacts import TrainingTier, archive_models

    if tier is not None:
        training_tier = TrainingTier(tier)
        count = archive_models(_MODEL_DIR, tier=training_tier)
        state.append_log(f"  Archived {count} {tier} model artifact(s) to archive/")
    else:
        count = archive_models(_MODEL_DIR, tier=None)
        state.append_log(f"  Archived {count} legacy model artifact(s) to archive/")


def has_processed_data() -> bool:
    """Return True if processed feature files exist."""
    features_dir = _PROCESSED_DIR / "features"
    if not features_dir.exists():
        return False
    return any(features_dir.glob("features_*.parquet"))


def has_trained_models() -> bool:
    """Return True if any trained model artifacts exist (any tier)."""
    from mlb_predict.model.artifacts import has_trained_models as _has

    return _has(_MODEL_DIR)


def _python_bin() -> str:
    import shutil

    return shutil.which("python") or "python"


def _ingest_commands(opts: PipelineOptions | None = None) -> list[tuple[str, str]]:
    """Full re-ingestion of all seasons (2000–current year)."""
    from datetime import datetime as dt

    opts = opts or PipelineOptions()
    python = _python_bin()
    year = dt.now(timezone.utc).year

    if opts.seasons:
        seasons = " ".join(str(s) for s in opts.seasons)
        season_label = ", ".join(str(s) for s in opts.seasons)
    else:
        seasons = " ".join(str(s) for s in range(2000, year + 1))
        season_label = f"2000–{year}"

    refresh_flags = ""
    if opts.refresh_mlbapi:
        refresh_flags += " --refresh-mlbapi"
    if opts.refresh_retro:
        refresh_flags += " --refresh-retro"

    preseason_flag = "" if opts.include_preseason else " --no-preseason"

    return [
        (
            f"Ingest schedules & gamelogs ({season_label})",
            f"{python} scripts/ingest_all.py --seasons {seasons}{refresh_flags}{preseason_flag}",
        ),
        (
            f"Ingest pitcher stats ({season_label})",
            f"{python} scripts/ingest_pitcher_stats.py --seasons {seasons}"
            + (" --refresh" if opts.refresh_mlbapi else ""),
        ),
        (
            f"Ingest FanGraphs metrics ({season_label})",
            f"{python} scripts/ingest_fangraphs.py --seasons {seasons}",
        ),
        (
            f"Ingest player data ({season_label})",
            f"{python} scripts/ingest_player_data.py --seasons {seasons}",
        ),
        (
            f"Ingest weather data ({season_label})",
            f"{python} scripts/ingest_weather.py --seasons {seasons}",
        ),
        (
            f"Build feature matrices ({season_label})",
            f"{python} scripts/build_features.py --seasons {seasons}",
        ),
        (
            f"Build spring training features ({season_label})",
            f"{python} scripts/build_spring_features.py --seasons {seasons}",
        ),
        ("Build 2026 pre-season features (if needed)", f"{python} scripts/build_features_2026.py"),
    ]


def _update_commands(opts: PipelineOptions | None = None) -> list[tuple[str, str]]:
    """Update current season only (non-destructive)."""
    from datetime import datetime as dt

    opts = opts or PipelineOptions()
    python = _python_bin()

    if opts.seasons:
        year = " ".join(str(s) for s in opts.seasons)
        season_label = ", ".join(str(s) for s in opts.seasons)
    else:
        yr = str(dt.now(timezone.utc).year)
        year = yr
        season_label = yr

    preseason_flag = "" if opts.include_preseason else " --no-preseason"
    refresh_schedule = " --refresh-mlbapi" if opts.refresh_mlbapi else ""
    refresh_retro = " --refresh" if opts.refresh_retro else ""

    return [
        (
            f"Refresh schedule ({season_label})",
            f"{python} scripts/ingest_schedule.py --seasons {year}{refresh_schedule}{preseason_flag}",
        ),
        (
            f"Refresh Retrosheet gamelogs ({season_label})",
            f"{python} scripts/ingest_retrosheet_gamelogs.py --seasons {year}{refresh_retro}",
        ),
        (
            f"Rebuild crosswalk ({season_label})",
            f"{python} scripts/build_crosswalk.py --seasons {year}",
        ),
        (
            f"Refresh pitcher stats ({season_label})",
            f"{python} scripts/ingest_pitcher_stats.py --seasons {year}"
            + (" --refresh" if opts.refresh_mlbapi else ""),
        ),
        (
            f"Refresh FanGraphs metrics ({season_label})",
            f"{python} scripts/ingest_fangraphs.py --seasons {year}",
        ),
        (
            f"Refresh player data ({season_label})",
            f"{python} scripts/ingest_player_data.py --seasons {year}",
        ),
        (
            f"Refresh weather data ({season_label})",
            f"{python} scripts/ingest_weather.py --seasons {year}",
        ),
        (
            f"Rebuild feature matrix ({season_label})",
            f"{python} scripts/build_features.py --seasons {year}",
        ),
        (
            f"Rebuild spring training features ({season_label})",
            f"{python} scripts/build_spring_features.py --seasons {year}",
        ),
        ("Build 2026 pre-season features (if needed)", f"{python} scripts/build_features_2026.py"),
    ]


def _retrain_commands(
    opts: PipelineOptions | None = None,
    *,
    bootstrap: bool = False,
    training_tier: str = "full",
) -> list[tuple[str, str]]:
    """Retrain all production models.

    CV is always skipped for dashboard-triggered retrains because it is
    evaluation-only (trains 5 models x 24 seasons) and exceeds the
    container memory budget.

    Stage 1 (player embeddings) is included for **full**-tier retrains
    so that the neural-network features improve the stacked ensemble.
    The Docker image must be built with ``TORCH_SOURCE=1`` to compile
    PyTorch for SSE4.2 CPUs; the CI workflow does this automatically for
    non-PR builds.

    Quick-tier and bootstrap retrains skip Stage 1 entirely.
    """
    python = _python_bin()
    models = "logistic lightgbm xgboost catboost mlp stacked"
    effective_tier = "quick" if bootstrap else training_tier
    flags = f" --tier {effective_tier} --skip-cv"
    if effective_tier == "quick":
        flags += " --no-stage1"
    tier_label = "quick" if effective_tier == "quick" else "full"
    return [
        (
            f"Train all production models ({tier_label})",
            f"{python} scripts/train_model.py --models {models}{flags}",
        ),
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
    opts: PipelineOptions | None = None,
    *,
    bootstrap: bool = False,
    training_tier: str = "full",
) -> None:
    """Execute a pipeline (ingest or retrain) as a background task.

    After successful completion, calls ``on_success`` (typically a model reload)
    and writes a timestamp marker.  When *bootstrap* is True, retrain uses
    quick tier.  *training_tier* selects ``"quick"`` or ``"full"`` storage.
    """
    state = get_state(kind)

    blocker = conflicting_pipeline()
    if blocker is not None:
        logger.warning("Pipeline %s blocked — %s is already running", kind.value, blocker.value)
        return

    state.reset()

    effective_tier = "quick" if bootstrap else training_tier

    try:
        if kind == PipelineKind.INGEST:
            state.append_log(">>> Clearing all processed data…")
            _clean_processed_data(state)
        elif kind == PipelineKind.RETRAIN:
            state.append_log(f">>> Archiving {effective_tier} model artifacts…")
            _archive_models(state, tier=effective_tier)

        if kind == PipelineKind.INGEST:
            commands = _ingest_commands(opts)
        elif kind == PipelineKind.UPDATE:
            commands = _update_commands(opts)
        else:
            commands = _retrain_commands(
                opts,
                bootstrap=bootstrap,
                training_tier=training_tier,
            )

        state.init_steps([desc for desc, _ in commands])

        for step_idx, (desc, cmd) in enumerate(commands):
            state.begin_step(step_idx)
            state.append_log(f">>> {desc}")
            logger.info("[%s] %s", kind.value, desc)
            step_t0 = time.monotonic()
            rc = await _stream_process(cmd, state)
            step_elapsed = time.monotonic() - step_t0
            state.append_log(f"    [{step_elapsed:.1f}s]")
            logger.info("[%s] %s completed in %.1fs", kind.value, desc, step_elapsed)
            if rc != 0:
                state.fail_step(step_idx)
                state.finish(ok=False, error=f"Step '{desc}' exited with code {rc}")
                return
            state.complete_step(step_idx, step_elapsed)

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


def _scan_model_dir(d: Path, tier: str = "legacy") -> list[dict[str, Any]]:
    """Scan a single directory for model artifacts, tagging each with its tier."""
    found: list[dict[str, Any]] = []
    if not d.exists():
        return found
    for item in sorted(d.iterdir()):
        if not item.is_dir() or not (item / "model.joblib").exists():
            continue
        meta_path = item / "metadata.json"
        meta: dict[str, Any] = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                pass
        found.append(
            {
                "name": item.name,
                "model_type": meta.get("model_type", item.name.split("_")[0]),
                "version": meta.get("model_version", meta.get("version", "unknown")),
                "training_tier": meta.get("training_tier", tier),
                "training_seasons": meta.get("training_seasons"),
                "trained_at": meta.get("trained_at"),
            }
        )
    return found


def gather_model_status() -> dict[str, Any]:
    """Collect information about trained models for the dashboard.

    Scans full/, quick/, and legacy top-level directories.
    Reports archived model count for drift analysis.
    """
    last_retrain: str | None = None
    marker = _MODEL_DIR / ".last_retrain"
    if marker.exists():
        last_retrain = marker.read_text().strip()

    models_found: list[dict[str, Any]] = []
    models_found.extend(_scan_model_dir(_MODEL_DIR / "full", tier="full"))
    models_found.extend(_scan_model_dir(_MODEL_DIR / "quick", tier="quick"))
    models_found.extend(_scan_model_dir(_MODEL_DIR, tier="legacy"))

    archive_count = 0
    archive_dir = _MODEL_DIR / "archive"
    if archive_dir.exists():
        archive_count = sum(
            1 for d in archive_dir.iterdir() if d.is_dir() and (d / "model.joblib").exists()
        )

    cv_summary: list[dict[str, Any]] = []
    for name in (
        "cv_summary_v4.json",
        "cv_summary_v3.json",
        "cv_summary_v2.json",
        "cv_summary.json",
    ):
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
        tier = m.get("training_tier", "legacy")
        key = f"{mt}_{tier}"
        if key not in latest_per_type or m["name"] > latest_per_type[key]["name"]:
            latest_per_type[key] = m

    return {
        "last_retrain": last_retrain,
        "models_found": models_found,
        "production_models": list(latest_per_type.values()),
        "total_artifacts": len(models_found),
        "archived_artifacts": archive_count,
        "cv_summary": cv_summary,
    }


# ---------------------------------------------------------------------------
# WebSocket shell runner
# ---------------------------------------------------------------------------


async def ws_shell_run(websocket: Any) -> None:
    """Run shell commands received via WebSocket, streaming output back line-by-line.

    Protocol (JSON messages):
      Client -> Server: {"cmd": "python scripts/ingest_schedule.py --seasons 2026"}
      Server -> Client: {"type": "stdout", "data": "...line..."}
      Server -> Client: {"type": "exit", "code": 0}
      Client -> Server: {"type": "kill"}   (sends SIGTERM to running process)
    """
    await websocket.accept()
    proc: asyncio.subprocess.Process | None = None

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            if msg.get("type") == "kill" and proc is not None:
                proc.terminate()
                await websocket.send_text(json.dumps({"type": "stdout", "data": "[killed]\n"}))
                continue

            cmd = msg.get("cmd", "").strip()
            if not cmd:
                await websocket.send_text(
                    json.dumps({"type": "exit", "code": -1, "error": "Empty command"})
                )
                continue

            await websocket.send_text(json.dumps({"type": "stdout", "data": f"$ {cmd}\n"}))

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
                await websocket.send_text(
                    json.dumps({"type": "stdout", "data": line.decode("utf-8", errors="replace")})
                )
            await proc.wait()
            await websocket.send_text(json.dumps({"type": "exit", "code": proc.returncode or 0}))
            proc = None
    except Exception:
        pass
    finally:
        if proc is not None:
            proc.terminate()


# ---------------------------------------------------------------------------
# WebSocket Python REPL
# ---------------------------------------------------------------------------


class _ReplConsole(code.InteractiveConsole):
    """InteractiveConsole subclass that captures output to a buffer."""

    def __init__(self) -> None:
        super().__init__(locals={"__name__": "__repl__", "__builtins__": __builtins__})
        self._buf = io.StringIO()

    def execute(self, source: str) -> tuple[str, bool]:
        """Execute source code; return (output_text, more_input_needed)."""
        self._buf = io.StringIO()
        more = False
        with redirect_stdout(self._buf), redirect_stderr(self._buf):
            try:
                more = self.push(source)
            except SystemExit:
                self._buf.write("[SystemExit caught — REPL remains alive]\n")
        return self._buf.getvalue(), more


_repl_sessions: dict[str, _ReplConsole] = {}


def _get_repl(session_id: str = "default") -> _ReplConsole:
    """Return (or create) a REPL console for the given session ID."""
    if session_id not in _repl_sessions:
        _repl_sessions[session_id] = _ReplConsole()
    return _repl_sessions[session_id]


async def ws_repl_run(websocket: Any) -> None:
    """Interactive Python REPL over WebSocket.

    Protocol (JSON messages):
      Client -> Server: {"code": "import pandas as pd", "session": "default"}
      Server -> Client: {"type": "output", "data": "...", "more": false, "session": "default"}
      Client -> Server: {"type": "reset", "session": "default"}
      Server -> Client: {"type": "output", "data": "[session reset]\\n", "more": false, ...}
    """
    await websocket.accept()

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            session_id = msg.get("session", "default")

            if msg.get("type") == "reset":
                _repl_sessions.pop(session_id, None)
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "output",
                            "data": "[session reset]\n",
                            "more": False,
                            "session": session_id,
                        }
                    )
                )
                continue

            source = msg.get("code", "")
            console = _get_repl(session_id)

            loop = asyncio.get_event_loop()
            output, more = await loop.run_in_executor(None, console.execute, source)

            await websocket.send_text(
                json.dumps({"type": "output", "data": output, "more": more, "session": session_id})
            )
    except Exception:
        pass
