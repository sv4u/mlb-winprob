"""Generate and snapshot win probability predictions for a target season.

Uses the production model trained on all seasons prior to the target season.
For v4 models, runs Stage 1 inference to populate player features before
running Stage 2 predictions.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from mlb_predict.model.artifacts import latest_artifact, load_model
from mlb_predict.model.train import _predict_proba
from mlb_predict.predict.snapshot import write_snapshot

logger = logging.getLogger(__name__)

_FEATURE_VERSION = "v4"


def _inject_stage1_at_inference(
    feat_df: pd.DataFrame,
    model_dir: Path,
    gamelogs_dir: Path,
    player_data_dir: Path,
    season: int,
) -> pd.DataFrame:
    """Run Stage 1 inference and inject features into the prediction DataFrame.

    Loads the saved production Stage 1 model, prepares tensors from gamelogs
    and player rolling data, and replaces the zero-filled Stage 1 columns.
    """
    from mlb_predict.player.embeddings import STAGE1_FEATURE_NAMES, load_stage1_model
    from mlb_predict.player.biographical import build_biographical_df, build_bio_lookup
    from mlb_predict.player.rolling import build_batter_rolling, build_pitcher_rolling
    from mlb_predict.player.pitcher_gamelogs import load_pitcher_gamelogs
    from mlb_predict.player.lineup_model import (
        prepare_game_tensors,
        generate_stage1_features,
        stage1_features_to_df,
    )

    s1_dirs = sorted(model_dir.glob(f"player_embedding_{_FEATURE_VERSION}_*"))
    if not s1_dirs:
        logger.warning("No Stage 1 model found in %s; using zero features", model_dir)
        return feat_df
    s1_dir = s1_dirs[-1]

    try:
        model, vocab = load_stage1_model(s1_dir)
    except Exception as exc:
        logger.warning("Failed to load Stage 1 model from %s: %s", s1_dir, exc)
        return feat_df

    bio_df = build_biographical_df(cache_dir=player_data_dir)
    if bio_df.empty:
        logger.warning("No biographical data; using zero Stage 1 features")
        return feat_df

    bio_lookup = build_bio_lookup(bio_df)
    retro_to_mlbam: dict[str, int] = {}
    for _, row in bio_df.iterrows():
        retro_id = row.get("retro_id")
        mlbam = row.get("mlbam_id")
        if pd.notna(retro_id) and pd.notna(mlbam):
            retro_to_mlbam[str(retro_id).strip().lower()] = int(mlbam)

    gl_path = gamelogs_dir / f"gamelogs_{season}.parquet"
    if not gl_path.exists():
        logger.warning("No gamelogs for %d; using zero Stage 1 features", season)
        return feat_df

    gamelogs = pd.read_parquet(gl_path)

    all_seasons = sorted(int(f.stem.split("_")[1]) for f in gamelogs_dir.glob("gamelogs_*.parquet"))
    gl_frames = [pd.read_parquet(gamelogs_dir / f"gamelogs_{s}.parquet") for s in all_seasons]
    all_gl = pd.concat(gl_frames, ignore_index=True)

    pitcher_game_logs = load_pitcher_gamelogs(player_data_dir, all_seasons)

    batter_rolling = build_batter_rolling(all_gl, retro_to_mlbam=retro_to_mlbam)
    pitcher_rolling = build_pitcher_rolling(
        all_gl,
        retro_to_mlbam=retro_to_mlbam,
        pitcher_game_logs=pitcher_game_logs if not pitcher_game_logs.empty else None,
    )

    tensors = prepare_game_tensors(
        gamelogs,
        batter_rolling,
        pitcher_rolling,
        bio_lookup,
        retro_to_mlbam,
        vocab,
        train_mode=False,
    )

    if tensors is None:
        logger.warning("Stage 1 tensor prep failed; using zero features")
        return feat_df

    features = generate_stage1_features(model, tensors)
    features_df = stage1_features_to_df(features)

    result = feat_df.copy()
    if len(features_df) == len(gamelogs) and len(gamelogs) == len(result):
        for col in STAGE1_FEATURE_NAMES:
            if col in features_df.columns:
                result[col] = features_df[col].values
        logger.info("Injected Stage 1 features for %d games", len(features_df))
    else:
        logger.warning(
            "Stage 1 count mismatch (features=%d, gamelogs=%d, feat_df=%d); using zeros",
            len(features_df),
            len(gamelogs),
            len(result),
        )

    return result


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
        choices=["logistic", "lightgbm", "xgboost", "catboost", "mlp", "stacked"],
        default="stacked",
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
    ap.add_argument(
        "--gamelogs-dir",
        type=Path,
        default=Path("data/processed/retrosheet"),
    )
    ap.add_argument(
        "--player-data-dir",
        type=Path,
        default=Path("data/processed/player"),
    )
    ap.add_argument(
        "--no-stage1",
        action="store_true",
        help="Skip Stage 1 inference (use zero features)",
    )
    ap.add_argument("--tag", type=str, default=None)
    args = ap.parse_args()

    features_path = args.features_dir / f"features_{args.season}.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Feature file not found: {features_path}")

    artifact_dir = latest_artifact(
        args.model_type, model_dir=args.model_dir, version=_FEATURE_VERSION
    )
    if artifact_dir is None:
        artifact_dir = latest_artifact(args.model_type, model_dir=args.model_dir, version="v3")
        if artifact_dir is None:
            raise RuntimeError(f"No {args.model_type} model artifact found in {args.model_dir}")
        logger.warning("Using v3 model (no v4 found); Stage 1 features will be zeros")

    print(f"Loading model from {artifact_dir}")
    model, meta = load_model(artifact_dir)
    feat_cols = meta.feature_cols

    feat_df = pd.read_parquet(features_path)
    if "is_spring" not in feat_df.columns:
        feat_df["is_spring"] = 0.0
    else:
        feat_df["is_spring"] = feat_df["is_spring"].fillna(0.0)

    # Stage 1 inference: populate player features before Stage 2 prediction
    if not args.no_stage1 and meta.feature_set_version == _FEATURE_VERSION:
        feat_df = _inject_stage1_at_inference(
            feat_df,
            model_dir=args.model_dir,
            gamelogs_dir=args.gamelogs_dir,
            player_data_dir=args.player_data_dir,
            season=args.season,
        )

    clean = feat_df.dropna(subset=feat_cols)
    X_df = clean[feat_cols].astype(float)
    y_prob = _predict_proba(model, X_df)

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
