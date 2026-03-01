"""Model training — v4.

Models
------
- Logistic regression (sklearn Pipeline with StandardScaler)
- LightGBM (gradient-boosted trees, Optuna-tuned)
- XGBoost  (gradient-boosted trees, Optuna-tuned)
- Stacked ensemble (meta-learner trained on calibrated base-model probabilities)

Training improvements
---------------------
- **Time-weighted training**: exponential decay puts more weight on recent seasons
  so the model adapts to changes in the game (shift to analytics, rule changes, etc.).
- **Platt calibration**: a sigmoid meta-layer fitted on a held-out calibration
  set corrects systematic over/under-confidence.
- **Optuna HPO**: 60-trial study for LightGBM and XGBoost using a fast 3-season
  expanding-window evaluation to find the best hyperparameters before the
  full expanding-CV run.

Expanding-window evaluation
----------------------------
For each season N in [first_eval_season … last_season]:
  Train on seasons < N (time-weighted) → calibrate on last 20% → evaluate on N.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb

from winprob.features.builder import FEATURE_COLS
from winprob.model.artifacts import ModelMetadata, save_model
from winprob.model.evaluate import evaluate, EvalResult

logger = logging.getLogger(__name__)

_FEATURE_VERSION = "v3"

_LR_PARAMS: dict[str, Any] = {
    "C": 1.0,
    "max_iter": 1000,
    "solver": "lbfgs",
    "random_state": 42,
}

# Default params (overridden by Optuna if HPO has been run)
_LGB_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "n_estimators": 500,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "verbose": -1,
}

_XGB_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "learning_rate": 0.05,
    "max_depth": 6,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbosity": 0,
    "use_label_encoder": False,
}

_META_PARAMS: dict[str, Any] = {
    "C": 0.5,
    "max_iter": 500,
    "solver": "lbfgs",
    "random_state": 42,
}

_TIME_DECAY: float = 0.12  # exponential decay rate per season


# ---------------------------------------------------------------------------
# Platt calibrator
# ---------------------------------------------------------------------------

class PlattCalibrator:
    """Platt scaling: fits a sigmoid on the base model's log-odds output."""

    def __init__(self, base: Any) -> None:
        self.base = base
        self._sigmoid = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        raw = _raw_proba(self.base, X)
        log_odds = logit(np.clip(raw, 1e-7, 1 - 1e-7)).reshape(-1, 1)
        self._sigmoid.fit(log_odds, y)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        raw = _raw_proba(self.base, X)
        log_odds = logit(np.clip(raw, 1e-7, 1 - 1e-7)).reshape(-1, 1)
        return self._sigmoid.predict_proba(log_odds)

    @property
    def booster_(self) -> Any:
        return getattr(self.base, "booster_", None)


# ---------------------------------------------------------------------------
# Time weighting
# ---------------------------------------------------------------------------

def _season_weights(seasons: pd.Series, decay: float = _TIME_DECAY) -> np.ndarray:
    """Exponential weights: most-recent season = 1.0, older → exp(-decay*gap)."""
    max_s = seasons.max()
    return np.exp(-decay * (max_s - seasons).clip(lower=0))


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _build_lr() -> Pipeline:
    return Pipeline(
        [("scaler", StandardScaler()), ("lr", LogisticRegression(**_LR_PARAMS))]
    )


def _build_lgbm(params: dict[str, Any] | None = None) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(**(params or _LGB_PARAMS))


def _build_xgb(params: dict[str, Any] | None = None) -> xgb.XGBClassifier:
    p = dict(params or _XGB_PARAMS)
    p.pop("use_label_encoder", None)
    return xgb.XGBClassifier(**p)


# ---------------------------------------------------------------------------
# Predict helpers
# ---------------------------------------------------------------------------

def _raw_proba(model: Any, X: np.ndarray | pd.DataFrame) -> np.ndarray:
    is_lgbm = hasattr(model, "booster_")
    is_xgb = isinstance(model, xgb.XGBClassifier)
    if (is_lgbm or is_xgb) and not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=FEATURE_COLS)
    return model.predict_proba(X)[:, 1]


def _predict_proba(model: Any, X: np.ndarray | pd.DataFrame) -> np.ndarray:
    if isinstance(model, PlattCalibrator):
        return model.predict_proba(X)[:, 1]
    return _raw_proba(model, X)


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def _prep(
    df: pd.DataFrame,
    *,
    time_weighted: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X, y, sample_weights from a feature DataFrame."""
    clean = df.dropna(subset=FEATURE_COLS + ["home_win"])
    X = clean[FEATURE_COLS].values.astype(float)
    y = clean["home_win"].values.astype(float)
    if time_weighted and "season" in clean.columns:
        w = _season_weights(clean["season"])
    else:
        w = np.ones(len(y))
    return X, y, w


def _calibrate(base: Any, X_cal: np.ndarray, y_cal: np.ndarray) -> PlattCalibrator:
    cal = PlattCalibrator(base)
    cal.fit(X_cal, y_cal)
    return cal


def _fit_model(
    mt: str,
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    params: dict[str, Any] | None = None,
) -> Any:
    if mt == "logistic":
        model = _build_lr()
        model.fit(X, y, lr__sample_weight=w)
    elif mt == "lightgbm":
        model = _build_lgbm(params)
        X_df = pd.DataFrame(X, columns=FEATURE_COLS)
        model.fit(X_df, y, sample_weight=w)
    elif mt == "xgboost":
        model = _build_xgb(params)
        X_df = pd.DataFrame(X, columns=FEATURE_COLS)
        model.fit(X_df, y, sample_weight=w)
    else:
        raise ValueError(f"Unknown model type: {mt}")
    return model


# ---------------------------------------------------------------------------
# Optuna HPO
# ---------------------------------------------------------------------------

def run_optuna_hpo(
    season_dfs: dict[int, pd.DataFrame],
    *,
    model_type: str = "lightgbm",
    n_trials: int = 60,
    eval_seasons: list[int] | None = None,
    model_dir: Path = Path("data/models"),
) -> dict[str, Any]:
    """Find optimal hyperparameters via Optuna on a subset of expanding seasons.

    The objective is mean Brier score on the last 3 seasons of the eval window.
    Returns the best params dict and saves it to ``model_dir/hpo_{model_type}.json``.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    seasons = sorted(season_dfs)
    if eval_seasons is None:
        eval_seasons = seasons[-3:]  # evaluate on last 3 seasons for speed

    def _objective(trial: "optuna.Trial") -> float:  # type: ignore[name-defined]
        if model_type == "lightgbm":
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "verbose": -1,
                "random_state": 42,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 127),
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            }
        else:  # xgboost
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "verbosity": 0,
                "random_state": 42,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 9),
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            }

        briers = []
        for eval_s in eval_seasons:
            train_seasons = [s for s in seasons if s < eval_s]
            if len(train_seasons) < 2:
                continue
            train_df = pd.concat([season_dfs[s] for s in train_seasons], ignore_index=True)
            X_train, y_train, w_train = _prep(train_df)
            eval_df = season_dfs[eval_s]
            eval_clean = eval_df.dropna(subset=FEATURE_COLS + ["home_win"])
            X_eval = eval_clean[FEATURE_COLS].values.astype(float)
            y_eval = eval_clean["home_win"].values.astype(float)

            cal_size = max(int(0.2 * len(X_train)), 300)
            X_fit, y_fit, w_fit = X_train[:-cal_size], y_train[:-cal_size], w_train[:-cal_size]
            X_cal, y_cal = X_train[-cal_size:], y_train[-cal_size:]

            try:
                model = _fit_model(model_type, X_fit, y_fit, w_fit, params)
                cal_model = _calibrate(model, X_cal, y_cal)
                y_prob = _predict_proba(cal_model, X_eval)
                er = evaluate(y_eval, y_prob)
                briers.append(er.brier_score)
            except Exception:
                return 1.0  # penalize failures

        return float(np.mean(briers)) if briers else 1.0

    study = optuna.create_study(direction="minimize")
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    logger.info("Optuna HPO (%s): best Brier=%.4f params=%s", model_type, study.best_value, best)

    # Merge with defaults to ensure all keys present
    if model_type == "lightgbm":
        final = {**_LGB_PARAMS, **best}
    else:
        final = {**_XGB_PARAMS, **best}

    model_dir.mkdir(parents=True, exist_ok=True)
    out = model_dir / f"hpo_{model_type}.json"
    out.write_text(json.dumps(final, indent=2))
    return final


# ---------------------------------------------------------------------------
# Expanding-window CV
# ---------------------------------------------------------------------------

def run_expanding_cv(
    features_dir: Path = Path("data/processed/features"),
    model_dir: Path = Path("data/models"),
    *,
    min_train_seasons: int = 2,
    model_types: list[str] | None = None,
    lgb_params: dict[str, Any] | None = None,
    xgb_params: dict[str, Any] | None = None,
) -> dict[str, list[dict]]:
    """Expanding-window CV with time-weighted training and Platt calibration."""
    model_types = model_types or ["logistic", "lightgbm", "xgboost", "stacked"]

    season_dfs: dict[int, pd.DataFrame] = {
        int(f.stem.split("_")[1]): pd.read_parquet(f)
        for f in sorted(features_dir.glob("features_*.parquet"))
    }
    if not season_dfs:
        raise RuntimeError(f"No feature files found in {features_dir}")

    seasons = sorted(season_dfs)
    results: dict[str, list[dict]] = {mt: [] for mt in model_types}

    for i, eval_season in enumerate(seasons):
        if i < min_train_seasons:
            continue

        train_seasons = seasons[:i]
        train_df = pd.concat([season_dfs[s] for s in train_seasons], ignore_index=True)
        X_train, y_train, w_train = _prep(train_df)

        eval_clean = season_dfs[eval_season].dropna(subset=FEATURE_COLS + ["home_win"])
        X_eval = eval_clean[FEATURE_COLS].values.astype(float)
        y_eval = eval_clean["home_win"].values.astype(float)

        if len(X_train) < 100 or len(X_eval) == 0:
            continue

        cal_size = max(int(0.2 * len(X_train)), 500)
        X_fit, y_fit, w_fit = X_train[:-cal_size], y_train[:-cal_size], w_train[:-cal_size]
        X_cal, y_cal = X_train[-cal_size:], y_train[-cal_size:]

        fitted_models: dict[str, Any] = {}

        for mt in model_types:
            if mt == "stacked":
                continue
            params = lgb_params if mt == "lightgbm" else (xgb_params if mt == "xgboost" else None)
            model = _fit_model(mt, X_fit, y_fit, w_fit, params)
            cal = _calibrate(model, X_cal, y_cal)
            fitted_models[mt] = cal

            y_prob = _predict_proba(cal, X_eval)
            er = evaluate(y_eval, y_prob)

            meta = ModelMetadata(
                model_version=_FEATURE_VERSION,
                model_type=mt,
                training_seasons=train_seasons,
                hyperparameters=params or (_LR_PARAMS if mt == "logistic" else _LGB_PARAMS),
                feature_set_version=_FEATURE_VERSION,
                feature_cols=FEATURE_COLS,
                train_brier=float(er.brier_score),
                train_n_games=len(X_fit),
            )
            save_model(cal, meta, model_dir=model_dir)
            results[mt].append(_result_row(mt, eval_season, len(X_train), er))

        # Stacked ensemble
        base_keys = [k for k in ["logistic", "lightgbm", "xgboost"] if k in fitted_models]
        if "stacked" in model_types and len(base_keys) >= 2:
            cal_preds = np.column_stack([_predict_proba(fitted_models[k], X_cal) for k in base_keys])
            meta_lr = LogisticRegression(**_META_PARAMS)
            meta_lr.fit(cal_preds, y_cal)
            eval_preds = np.column_stack([_predict_proba(fitted_models[k], X_eval) for k in base_keys])
            y_prob_stack = meta_lr.predict_proba(eval_preds)[:, 1]
            er = evaluate(y_eval, y_prob_stack)
            results["stacked"].append(_result_row("stacked", eval_season, len(X_train), er))

        status = " | ".join(
            f"{mt}: brier={results[mt][-1]['brier']:.4f} acc={results[mt][-1]['accuracy']:.4f}"
            for mt in model_types
            if results[mt]
        )
        print(f"  {eval_season}: n_train={len(X_train):,} | {status}")

    return results


def _result_row(model_type: str, season: int, n_train: int, er: EvalResult) -> dict:
    return {
        "model": model_type,
        "season": season,
        "n_train": n_train,
        "n_eval": er.n_games,
        "brier": round(er.brier_score, 4),
        "log_loss": round(er.log_loss, 4),
        "accuracy": round(er.accuracy, 4),
        "cal_err": round(er.calibration_mean_err, 4),
    }


# ---------------------------------------------------------------------------
# Production model training
# ---------------------------------------------------------------------------

def train_production_model(
    features_dir: Path = Path("data/processed/features"),
    model_dir: Path = Path("data/models"),
    *,
    model_types: list[str] | None = None,
    lgb_params: dict[str, Any] | None = None,
    xgb_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train calibrated production models on all available seasons."""
    model_types = model_types or ["logistic", "lightgbm", "xgboost", "stacked"]

    frames = [pd.read_parquet(f) for f in sorted(features_dir.glob("features_*.parquet"))]
    all_data = pd.concat(frames, ignore_index=True)
    X_all, y_all, w_all = _prep(all_data)
    seasons = sorted({int(f.stem.split("_")[1]) for f in features_dir.glob("features_*.parquet")})

    cal_size = max(int(0.15 * len(X_all)), 500)
    X_fit, y_fit, w_fit = X_all[:-cal_size], y_all[:-cal_size], w_all[:-cal_size]
    X_cal, y_cal = X_all[-cal_size:], y_all[-cal_size:]

    trained: dict[str, Any] = {}
    base_models: dict[str, Any] = {}

    for mt in model_types:
        if mt == "stacked":
            continue
        params = lgb_params if mt == "lightgbm" else (xgb_params if mt == "xgboost" else None)
        model = _fit_model(mt, X_fit, y_fit, w_fit, params)
        cal = _calibrate(model, X_cal, y_cal)
        base_models[mt] = cal

        meta = ModelMetadata(
            model_version=_FEATURE_VERSION,
            model_type=mt,
            training_seasons=seasons,
            hyperparameters=params or (_LR_PARAMS if mt == "logistic" else _LGB_PARAMS),
            feature_set_version=_FEATURE_VERSION,
            feature_cols=FEATURE_COLS,
            train_brier=0.0,
            train_n_games=len(X_fit),
        )
        save_model(cal, meta, model_dir=model_dir)
        trained[mt] = cal
        print(f"  {mt} production model saved")

    base_keys = [k for k in ["logistic", "lightgbm", "xgboost"] if k in base_models]
    if "stacked" in model_types and len(base_keys) >= 2:
        cal_preds = np.column_stack([_predict_proba(base_models[k], X_cal) for k in base_keys])
        meta_lr = LogisticRegression(**_META_PARAMS)
        meta_lr.fit(cal_preds, y_cal)
        trained["stacked"] = (base_models, meta_lr, base_keys)
        print("  stacked ensemble production model saved (in-memory)")

    return trained
