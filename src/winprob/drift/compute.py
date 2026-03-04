"""Prediction drift computation (AGENTS.md §6).

For each model run, computes:
  1. Incremental diff  — current snapshot vs. the previous snapshot.
  2. Baseline diff     — current snapshot vs. the first snapshot of the season.

Diff schema per game:
  game_pk, p_old, p_new, delta, abs_delta, direction

Run metrics:
  mean_abs_delta, p95_abs_delta, max_abs_delta,
  pct_gt_0p01, pct_gt_0p02, pct_gt_0p05
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from winprob.errors import DriftComputationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DriftMetrics:
    """Summary drift statistics for one (old, new) snapshot pair."""

    run_ts_utc: str
    model_version: str
    season: int
    n_games: int
    mean_abs_delta: float
    p95_abs_delta: float
    max_abs_delta: float
    pct_gt_0p01: float
    pct_gt_0p02: float
    pct_gt_0p05: float


def _diff_snapshots(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Compute per-game prediction deltas between two snapshots."""
    merged = new[["game_pk", "predicted_home_win_prob"]].merge(
        old[["game_pk", "predicted_home_win_prob"]],
        on="game_pk",
        suffixes=("_new", "_old"),
    )
    merged["delta"] = merged["predicted_home_win_prob_new"] - merged["predicted_home_win_prob_old"]
    merged["abs_delta"] = merged["delta"].abs()
    merged["direction"] = np.where(
        merged["delta"] > 0, "up", np.where(merged["delta"] < 0, "down", "unchanged")
    )
    return merged.rename(
        columns={
            "predicted_home_win_prob_old": "p_old",
            "predicted_home_win_prob_new": "p_new",
        }
    )[["game_pk", "p_old", "p_new", "delta", "abs_delta", "direction"]]


def _metrics_from_diff(
    diff: pd.DataFrame, *, run_ts: str, model_version: str, season: int
) -> DriftMetrics:
    ad = diff["abs_delta"]
    if ad.empty:
        return DriftMetrics(
            run_ts_utc=run_ts,
            model_version=model_version,
            season=season,
            n_games=0,
            mean_abs_delta=0.0,
            p95_abs_delta=0.0,
            max_abs_delta=0.0,
            pct_gt_0p01=0.0,
            pct_gt_0p02=0.0,
            pct_gt_0p05=0.0,
        )
    return DriftMetrics(
        run_ts_utc=run_ts,
        model_version=model_version,
        season=season,
        n_games=int(len(diff)),
        mean_abs_delta=float(ad.mean()),
        p95_abs_delta=float(ad.quantile(0.95)),
        max_abs_delta=float(ad.max()),
        pct_gt_0p01=float((ad > 0.01).mean()),
        pct_gt_0p02=float((ad > 0.02).mean()),
        pct_gt_0p05=float((ad > 0.05).mean()),
    )


def compute_drift(
    *,
    season: int,
    model_type: str = "xgboost",
    snapshot_dir: Path = Path("data/processed/predictions"),
    drift_dir: Path = Path("data/processed/drift"),
) -> dict[str, DriftMetrics]:
    """Compute incremental and baseline drift for the latest snapshot.

    Snapshots are expected in ``snapshot_dir/season=YYYY/snapshots/``.
    Only snapshots whose filename ends with ``_{model_type}.parquet`` are
    considered, preventing cross-model comparisons.

    Returns
    -------
    dict
        Keys ``"incremental"`` and ``"baseline"``, each mapping to
        ``DriftMetrics``.  Returns an empty dict if fewer than 2 snapshots exist.
    """
    snap_path = snapshot_dir / f"season={season}" / "snapshots"
    snaps = sorted(snap_path.glob(f"*_{model_type}.parquet"))
    if len(snaps) < 2:
        return {}

    try:
        current = pd.read_parquet(snaps[-1])
        previous = pd.read_parquet(snaps[-2])
        baseline = pd.read_parquet(snaps[0])
    except Exception as exc:
        raise DriftComputationError(
            f"Failed to read snapshot parquets for season={season} model={model_type}: {exc}"
        ) from exc

    required_col = "predicted_home_win_prob"
    for label, df in [("current", current), ("previous", previous), ("baseline", baseline)]:
        if required_col not in df.columns:
            raise DriftComputationError(
                f"Snapshot '{label}' missing required column '{required_col}'"
            )

    run_ts = str(current["run_ts_utc"].iloc[0])
    model_version = str(current["model_version"].iloc[0])

    inc_diff = _diff_snapshots(previous, current)
    base_diff = _diff_snapshots(baseline, current)

    drift_dir.mkdir(parents=True, exist_ok=True)

    inc_metrics = _metrics_from_diff(
        inc_diff, run_ts=run_ts, model_version=model_version, season=season
    )
    base_metrics = _metrics_from_diff(
        base_diff, run_ts=run_ts, model_version=model_version, season=season
    )

    # Append to season-level run_metrics
    _append_run_metrics(inc_metrics, drift_dir / f"run_metrics_{season}.parquet")
    _append_run_metrics(base_metrics, drift_dir / f"baseline_metrics_{season}.parquet")

    # Append to global run_metrics (deduplicated by season + run_ts_utc + model_version)
    _append_global_metrics(inc_metrics, drift_dir / "global_run_metrics.parquet")

    return {"incremental": inc_metrics, "baseline": base_metrics}


def _append_run_metrics(metrics: DriftMetrics, path: Path) -> None:
    new_row = pd.DataFrame([metrics.__dict__])
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row
    combined.to_parquet(path, index=False)


def _append_global_metrics(metrics: DriftMetrics, path: Path) -> None:
    new_row = pd.DataFrame([metrics.__dict__])
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new_row], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["season", "run_ts_utc", "model_version"], keep="last"
        )
    else:
        combined = new_row
    combined.to_parquet(path, index=False)
