"""Compute feature importance (SHAP or permutation) for trained tree models.

Writes a report of mean absolute SHAP per feature and flags low-importance
features for potential pruning. Use after training to guide feature selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlb_predict.features.builder import FEATURE_COLS
from mlb_predict.model.artifacts import load_model, latest_artifact
from mlb_predict.model.train import PlattCalibrator


def _get_base_model(model: object) -> object:
    """Return the underlying tree model for SHAP (unwrap PlattCalibrator)."""
    if isinstance(model, PlattCalibrator):
        return model.base
    return model


def run_shap_importance(
    model: object,
    X: np.ndarray | pd.DataFrame,
    feature_cols: list[str],
    *,
    max_samples: int = 5000,
) -> dict[str, float]:
    """Compute mean absolute SHAP value per feature.

    Returns
    -------
    dict
        Feature name -> mean absolute SHAP value (higher = more important).
    """
    import shap

    base = _get_base_model(model)
    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X, columns=feature_cols)
    else:
        X_df = X[feature_cols] if feature_cols else X

    if len(X_df) > max_samples:
        X_df = X_df.sample(n=max_samples, random_state=42)

    explainer = shap.TreeExplainer(base)
    shap_vals = explainer.shap_values(X_df)
    if isinstance(shap_vals, list):
        # Binary: index 1 = positive class
        arr = np.asarray(shap_vals[1])
    else:
        arr = np.asarray(shap_vals)

    mean_abs = np.abs(arr).mean(axis=0)
    return dict(zip(feature_cols, mean_abs.tolist()))


def run_permutation_importance(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: list[str],
    *,
    n_repeats: int = 5,
    max_samples: int = 3000,
    random_state: int = 42,
) -> dict[str, float]:
    """Compute permutation importance (sklearn) for any model.

    Returns
    -------
    dict
        Feature name -> importance (higher = more important).
    """
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import brier_score_loss

    if len(X) > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X, y = X[idx], y[idx]

    def _scorer(est: object, X_val: np.ndarray, y_val: np.ndarray) -> float:
        base = _get_base_model(est)
        if hasattr(base, "booster_"):
            X_val = pd.DataFrame(X_val, columns=feature_cols)
        proba = est.predict_proba(X_val)
        p = proba[:, 1] if proba.ndim == 2 else proba
        return -float(brier_score_loss(y_val, p))

    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=_scorer,
    )
    return dict(zip(feature_cols, result.importances_mean.tolist()))


def main() -> None:
    ap = argparse.ArgumentParser(description="Feature importance for win-prob models")
    ap.add_argument("--model-dir", type=Path, default=Path("data/models"))
    ap.add_argument("--features-dir", type=Path, default=Path("data/processed/features"))
    ap.add_argument("--model-type", choices=["lightgbm", "xgboost"], default="lightgbm")
    ap.add_argument("--method", choices=["shap", "permutation"], default="shap")
    ap.add_argument("--max-samples", type=int, default=5000)
    ap.add_argument("--out", type=Path, default=None, help="Write report JSON here")
    ap.add_argument(
        "--threshold", type=float, default=0.0, help="Flag features below this importance"
    )
    args = ap.parse_args()

    artifact_dir = latest_artifact(args.model_type, model_dir=args.model_dir, version="v3")
    if not artifact_dir or not artifact_dir.exists():
        raise SystemExit(f"No artifact found for {args.model_type} in {args.model_dir}")

    model, meta = load_model(artifact_dir)
    feature_cols = meta.feature_cols or FEATURE_COLS

    # Load recent feature data
    parquets = sorted(args.features_dir.glob("features_*.parquet"))
    if not parquets:
        raise SystemExit(f"No feature files in {args.features_dir}")
    df = pd.read_parquet(parquets[-1])
    df = df.dropna(subset=feature_cols + ["home_win"])
    X = df[feature_cols].values.astype(float)
    y = df["home_win"].values.astype(float)

    if args.method == "shap":
        importance = run_shap_importance(model, X, feature_cols, max_samples=args.max_samples)
    else:
        importance = run_permutation_importance(
            model, X, y, feature_cols, max_samples=args.max_samples
        )

    # Rank and flag low-importance
    ranked = sorted(importance.items(), key=lambda x: -x[1])
    report = {
        "model_type": args.model_type,
        "method": args.method,
        "n_samples": len(X),
        "importance": importance,
        "ranked": [{"feature": f, "importance": round(v, 6)} for f, v in ranked],
        "low_importance": [f for f, v in importance.items() if v <= args.threshold and v >= 0],
    }

    out_path = (
        args.out or args.model_dir / f"feature_importance_{args.model_type}_{args.method}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report → {out_path}")
    print("\nTop 15 features:")
    for r in report["ranked"][:15]:
        print(f"  {r['feature']}: {r['importance']:.5f}")
    if report["low_importance"]:
        print(f"\nFeatures at or below threshold {args.threshold}: {report['low_importance']}")


if __name__ == "__main__":
    main()
