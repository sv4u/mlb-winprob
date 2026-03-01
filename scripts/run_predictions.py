"""Generate and snapshot win probability predictions for a target season.

Uses the production model trained on all seasons prior to the target season.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from winprob.model.artifacts import latest_artifact, load_model
from winprob.model.train import FEATURE_COLS
from winprob.predict.snapshot import write_snapshot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--season",
        type=int,
        required=True,
        help="Season to generate predictions for",
    )
    ap.add_argument(
        "--model-type",
        choices=["logistic", "lightgbm"],
        default="lightgbm",
        help="Which model type to use for predictions",
    )
    ap.add_argument(
        "--features-dir",
        type=Path,
        default=Path("data/processed/features"),
    )
    ap.add_argument(
        "--model-dir",
        type=Path,
        default=Path("data/models"),
    )
    ap.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path("data/processed/predictions"),
    )
    ap.add_argument("--tag", type=str, default=None)
    args = ap.parse_args()

    features_path = args.features_dir / f"features_{args.season}.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Feature file not found: {features_path}")

    artifact_dir = latest_artifact(args.model_type, model_dir=args.model_dir)
    if artifact_dir is None:
        raise RuntimeError(f"No {args.model_type} model artifact found in {args.model_dir}")

    print(f"Loading model from {artifact_dir}")
    model, meta = load_model(artifact_dir)

    feat_df = pd.read_parquet(features_path)
    clean = feat_df.dropna(subset=FEATURE_COLS)
    X_df = clean[FEATURE_COLS].astype(float)
    # LightGBM expects a named DataFrame; sklearn Pipelines expect numpy arrays.
    is_lgbm = hasattr(model, "booster_")
    y_prob = model.predict_proba(X_df if is_lgbm else X_df.values)[:, 1]

    predictions = clean[["game_pk", "home_retro", "away_retro", "feature_hash"]].copy()
    predictions = predictions.rename(columns={"home_retro": "home_team", "away_retro": "away_team"})
    predictions["predicted_home_win_prob"] = y_prob

    schedule_path = Path("data/processed/schedule") / f"games_{args.season}.parquet"
    snap_path = write_snapshot(
        predictions,
        season=args.season,
        model_version=meta.model_version,
        model_type=args.model_type,
        feature_file=features_path,
        schedule_file=schedule_path,
        tag=args.tag,
        snapshot_dir=args.snapshot_dir,
    )

    print(f"Snapshot written → {snap_path}")
    print(f"  n_games = {len(predictions)}")
    print(f"  mean predicted home win prob = {y_prob.mean():.4f}")


if __name__ == "__main__":
    main()
