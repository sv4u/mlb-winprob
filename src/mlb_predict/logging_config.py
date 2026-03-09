"""Centralized logging configuration for the MLB Prediction System.

Provides a single ``setup_logging()`` entry-point that every script and the
web server should call once at startup.  Supports two output formats
(human-readable and structured JSON) and dual destinations (stdout + rotating
log file).
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_LOG_DIR = _REPO_ROOT / "logs"


class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[1] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


_HUMAN_FMT = "%(asctime)s  %(levelname)-8s  %(name)-32s  %(message)s"
_HUMAN_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    *,
    level: int | str = logging.INFO,
    log_format: Literal["human", "json", "auto"] = "auto",
    log_dir: Path | str | None = None,
    log_filename: str = "mlb_predict.log",
    max_bytes: int = 20 * 1024 * 1024,
    backup_count: int = 5,
    verbose: bool | None = None,
) -> None:
    """Configure the root logger for the entire application.

    Parameters
    ----------
    level:
        Base logging level.  Overridden by the ``MLB_PREDICT_LOG_LEVEL``
        environment variable if set.
    log_format:
        ``"human"`` for development-friendly output, ``"json"`` for
        structured production logs, ``"auto"`` to choose based on the
        ``MLB_PREDICT_LOG_FORMAT`` env var (defaulting to ``"human"``).
    log_dir:
        Directory for the rotating log file.  Defaults to ``<repo>/logs/``.
        Overridden by ``MLB_PREDICT_LOG_DIR``.
    log_filename:
        Name of the log file inside *log_dir*.
    max_bytes:
        Max size per log file before rotation.
    backup_count:
        Number of rotated backups to keep.
    verbose:
        When *True*, force ``DEBUG`` level regardless of *level*.
        Overridden by ``MLB_PREDICT_VERBOSE=1``.
    """
    env_level = os.environ.get("MLB_PREDICT_LOG_LEVEL", "").upper()
    if env_level:
        level = getattr(logging, env_level, logging.INFO)

    env_verbose = os.environ.get("MLB_PREDICT_VERBOSE", "")
    if verbose is None:
        verbose = env_verbose in ("1", "true", "yes")
    if verbose:
        level = logging.DEBUG

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    env_format = os.environ.get("MLB_PREDICT_LOG_FORMAT", "").lower()
    if log_format == "auto":
        log_format = "json" if env_format == "json" else "human"

    log_dir_path = Path(os.environ.get("MLB_PREDICT_LOG_DIR", str(log_dir or _DEFAULT_LOG_DIR)))
    log_dir_path.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    if log_format == "json":
        formatter: logging.Formatter = _JsonFormatter()
    else:
        formatter = logging.Formatter(_HUMAN_FMT, datefmt=_HUMAN_DATE_FMT)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    root.addHandler(stdout_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        str(log_dir_path / log_filename),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    root.debug(
        "Logging initialised: level=%s format=%s file=%s verbose=%s",
        logging.getLevelName(level),
        log_format,
        log_dir_path / log_filename,
        verbose,
    )
