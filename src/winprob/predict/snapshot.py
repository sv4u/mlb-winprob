"""Immutable prediction snapshot writer.

Snapshot schema (per AGENTS.md §5):
  game_pk, home_team, away_team, predicted_home_win_prob,
  run_ts_utc, model_version, schedule_hash, feature_hash,
  lineup_param_hash, starter_param_hash, git_commit, tag
"""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _git_commit() -> str:
    """Return the current HEAD commit hash, or 'unknown' if not a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _file_hash(path: Path) -> str:
    """SHA-256 of a single file."""
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_snapshot(
    predictions: pd.DataFrame,
    *,
    season: int,
    model_version: str,
    model_type: str,
    feature_file: Path,
    schedule_file: Path,
    tag: str | None = None,
    snapshot_dir: Path = Path("data/processed/predictions"),
) -> Path:
    """Write an immutable prediction snapshot parquet.

    Parameters
    ----------
    predictions:
        DataFrame with columns ``game_pk``, ``home_team``, ``away_team``,
        ``predicted_home_win_prob``, ``feature_hash``.
    season:
        The season these predictions cover.
    model_version:
        Model version string (e.g. ``"v1"``).
    model_type:
        ``"logistic"`` or ``"lightgbm"``.
    feature_file:
        Path to the feature parquet used for this prediction.
    schedule_file:
        Path to the schedule parquet for provenance hashing.
    tag:
        Optional human-readable label (e.g. ``"opening-day"``).
    snapshot_dir:
        Root of the predictions directory.

    Returns
    -------
    Path
        The written parquet file path.
    """
    run_ts = datetime.now(timezone.utc).isoformat()
    out_dir = snapshot_dir / f"season={season}" / "snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)

    snap = predictions.copy()
    snap["run_ts_utc"] = run_ts
    snap["model_version"] = f"{model_type}_{model_version}"
    snap["schedule_hash"] = _file_hash(schedule_file)
    snap["feature_hash"] = snap.get("feature_hash", "")
    snap["lineup_param_hash"] = None   # placeholder for future lineup module
    snap["starter_param_hash"] = None  # placeholder for future pitcher module
    snap["git_commit"] = _git_commit()
    snap["tag"] = tag

    # Enforce schema order
    cols = [
        "game_pk", "home_team", "away_team", "predicted_home_win_prob",
        "run_ts_utc", "model_version", "schedule_hash", "feature_hash",
        "lineup_param_hash", "starter_param_hash", "git_commit", "tag",
    ]
    for c in cols:
        if c not in snap.columns:
            snap[c] = None

    ts_safe = run_ts.replace(":", "-").replace("+", "p")
    out_path = out_dir / f"run_ts={ts_safe}_{model_type}.parquet"
    snap[cols].to_parquet(out_path, index=False)
    return out_path
