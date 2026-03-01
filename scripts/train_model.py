"""Train win probability models with Optuna HPO, then expanding-window CV.

Pipeline
--------
1. (Optional) Run Optuna HPO on LightGBM and XGBoost — 60 trials each,
   evaluated on the last 3 seasons.  Results saved to data/models/hpo_*.json.
2. Run expanding-window CV for all seasons using the best hyperparameters.
3. Train production models on all available data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from winprob.model.train import run_expanding_cv, run_optuna_hpo, train_production_model


def _load_hpo(model_dir: Path, model_type: str) -> dict | None:
    p = model_dir / f"hpo_{model_type}.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", type=Path, default=Path("data/processed/features"))
    ap.add_argument("--model-dir", type=Path, default=Path("data/models"))
    ap.add_argument("--min-train-seasons", type=int, default=2)
    ap.add_argument(
        "--models",
        nargs="*",
        choices=["logistic", "lightgbm", "xgboost", "stacked"],
        default=["logistic", "lightgbm", "xgboost", "stacked"],
    )
    ap.add_argument("--hpo", action="store_true", help="Run Optuna HPO before CV")
    ap.add_argument("--hpo-trials", type=int, default=60)
    ap.add_argument("--skip-cv", action="store_true", help="Skip CV, only run HPO + production")
    args = ap.parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)

    # Load season data
    season_dfs = {
        int(f.stem.split("_")[1]): pd.read_parquet(f)
        for f in sorted(args.features_dir.glob("features_*.parquet"))
    }

    lgb_params = _load_hpo(args.model_dir, "lightgbm")
    xgb_params = _load_hpo(args.model_dir, "xgboost")

    # -------------------------------------------------------------------------
    # Optional Optuna HPO
    # -------------------------------------------------------------------------
    if args.hpo:
        tree_models = [m for m in args.models if m in ("lightgbm", "xgboost")]
        for mt in tree_models:
            print(f"\nRunning Optuna HPO for {mt} ({args.hpo_trials} trials)…")
            best = run_optuna_hpo(
                season_dfs,
                model_type=mt,
                n_trials=args.hpo_trials,
                model_dir=args.model_dir,
            )
            print(f"  Best {mt} params: {best}")
            if mt == "lightgbm":
                lgb_params = best
            else:
                xgb_params = best

    # -------------------------------------------------------------------------
    # Expanding-window CV
    # -------------------------------------------------------------------------
    if not args.skip_cv:
        print("\nRunning expanding-window cross-validation…")
        cv_results = run_expanding_cv(
            args.features_dir,
            args.model_dir,
            min_train_seasons=args.min_train_seasons,
            model_types=args.models,
            lgb_params=lgb_params,
            xgb_params=xgb_params,
        )

        all_rows = [row for rows in cv_results.values() for row in rows]
        summary_df = pd.DataFrame(all_rows).sort_values(["season", "model"])

        print("\n=== Mean CV metrics by model ===")
        print(summary_df.groupby("model")[["brier", "accuracy", "cal_err"]].mean().round(4).to_string())

        print("\n=== Best / worst season by Brier ===")
        for mt, grp in summary_df.groupby("model"):
            best = grp.loc[grp["brier"].idxmin()]
            worst = grp.loc[grp["brier"].idxmax()]
            print(
                f"  {mt}: best {int(best.season)} ({best.brier:.4f}) "
                f"| worst {int(worst.season)} ({worst.brier:.4f})"
            )

        cv_path = args.model_dir / "cv_summary_v3.json"
        cv_path.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
        print(f"\nCV summary → {cv_path}")

    # -------------------------------------------------------------------------
    # Production models
    # -------------------------------------------------------------------------
    print("\nTraining production models on all available seasons…")
    train_production_model(
        args.features_dir,
        args.model_dir,
        model_types=args.models,
        lgb_params=lgb_params,
        xgb_params=xgb_params,
    )
    print("Done.")


if __name__ == "__main__":
    main()
