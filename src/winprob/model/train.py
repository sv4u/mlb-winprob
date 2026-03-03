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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from winprob.features.builder import FEATURE_COLS
from winprob.model.artifacts import ModelMetadata, load_model, save_model
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

_CATB_PARAMS: dict[str, Any] = {
    "loss_function": "Logloss",
    "learning_rate": 0.05,
    "depth": 6,
    "iterations": 500,
    "l2_leaf_reg": 3.0,
    "random_seed": 42,
    "verbose": False,
}

_MLP_PARAMS: dict[str, Any] = {
    "hidden_layer_sizes": (128, 64, 32),
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.001,
    "batch_size": 256,
    "max_iter": 300,
    "random_state": 42,
    "early_stopping": True,
    "validation_fraction": 0.1,
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

    def __init__(self, base: Any, C: float = 1.0) -> None:
        self.base = base
        self._sigmoid = LogisticRegression(C=C, solver="lbfgs", max_iter=500)

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


class StackedEnsemble:
    """Meta-learner over calibrated base-model probabilities.

    Wraps ``base_models`` (dict of PlattCalibrator-wrapped models) and a
    logistic-regression meta-learner so the whole ensemble can be persisted
    as a single joblib artifact and loaded by ``data_cache.startup()``.
    """

    def __init__(
        self,
        base_models: dict[str, Any],
        meta_lr: LogisticRegression,
        base_keys: list[str],
    ) -> None:
        self.base_models = base_models
        self.meta_lr = meta_lr
        self.base_keys = base_keys

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return shape (n, 2) probability array (mirrors sklearn API)."""
        preds = np.column_stack([_predict_proba(self.base_models[k], X) for k in self.base_keys])
        return self.meta_lr.predict_proba(preds)


# ---------------------------------------------------------------------------
# Time weighting
# ---------------------------------------------------------------------------


def _season_weights(
    seasons: pd.Series,
    decay: float = _TIME_DECAY,
) -> np.ndarray:
    """Exponential weights: most-recent season = 1.0, older → exp(-decay*gap)."""
    max_s = seasons.max()
    return np.exp(-decay * (max_s - seasons).clip(lower=0))


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def _build_lr() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(**_LR_PARAMS))])


def _build_lgbm(params: dict[str, Any] | None = None) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(**(params or _LGB_PARAMS))


def _build_xgb(params: dict[str, Any] | None = None) -> xgb.XGBClassifier:
    p = dict(params or _XGB_PARAMS)
    p.pop("use_label_encoder", None)
    return xgb.XGBClassifier(**p)


def _build_catboost(params: dict[str, Any] | None = None) -> CatBoostClassifier:
    return CatBoostClassifier(**(params or _CATB_PARAMS))


def _build_mlp(params: dict[str, Any] | None = None) -> Pipeline:
    """Feed-forward MLP with StandardScaler (2–3 hidden layers, ReLU, sigmoid output)."""
    p = dict(params or _MLP_PARAMS)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(**p)),
    ])


# ---------------------------------------------------------------------------
# Predict helpers
# ---------------------------------------------------------------------------


def _raw_proba(model: Any, X: np.ndarray | pd.DataFrame) -> np.ndarray:
    is_lgbm = hasattr(model, "booster_")
    is_xgb = isinstance(model, xgb.XGBClassifier)
    is_catb = isinstance(model, CatBoostClassifier)
    is_mlp = hasattr(model, "named_steps") and "mlp" in getattr(model, "named_steps", {})
    if (is_lgbm or is_xgb or is_catb or is_mlp) and not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=FEATURE_COLS)
    return model.predict_proba(X)[:, 1]


def _predict_proba(model: Any, X: np.ndarray | pd.DataFrame) -> np.ndarray:
    if isinstance(model, (PlattCalibrator, StackedEnsemble)):
        return model.predict_proba(X)[:, 1]
    return _raw_proba(model, X)


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------


def _prep(
    df: pd.DataFrame,
    *,
    time_weighted: bool = True,
    time_decay: float = _TIME_DECAY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X, y, sample_weights from a feature DataFrame."""
    clean = df.dropna(subset=FEATURE_COLS + ["home_win"])
    X = clean[FEATURE_COLS].values.astype(float)
    y = clean["home_win"].values.astype(float)
    if time_weighted and "season" in clean.columns:
        w = _season_weights(clean["season"], decay=time_decay)
    else:
        w = np.ones(len(y))
    return X, y, w


def _calibrate(
    base: Any,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    platt_C: float = 1.0,
) -> PlattCalibrator:
    cal = PlattCalibrator(base, C=platt_C)
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
    elif mt == "catboost":
        model = _build_catboost(params)
        X_df = pd.DataFrame(X, columns=FEATURE_COLS)
        model.fit(X_df, y, sample_weight=w)
    elif mt == "mlp":
        model = _build_mlp(params)
        model.fit(X, y, mlp__sample_weight=w)
    else:
        raise ValueError(f"Unknown model type: {mt}")
    return model


# ---------------------------------------------------------------------------
# Optuna HPO
# ---------------------------------------------------------------------------


# Keys in HPO result that are training globals, not model constructor params.
_HPO_TRAINING_KEYS = frozenset({"time_decay", "platt_C"})


def _model_params_only(params: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a copy of params with training keys removed for _fit_model."""
    if params is None:
        return None
    return {k: v for k, v in params.items() if k not in _HPO_TRAINING_KEYS}


def run_optuna_hpo(
    season_dfs: dict[int, pd.DataFrame],
    *,
    model_type: str = "lightgbm",
    n_trials: int = 200,
    eval_seasons: list[int] | None = None,
    model_dir: Path = Path("data/models"),
) -> dict[str, Any]:
    """Find optimal hyperparameters via Optuna on a subset of expanding seasons.

    The objective is mean Brier score on the last 5 seasons of the eval window.
    Returns the best params dict (including time_decay, platt_C) and saves to
    ``model_dir/hpo_{model_type}.json``. Callers should pass only model params
    to _fit_model (e.g. pop _HPO_TRAINING_KEYS before passing to LGBM/XGB).
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    seasons = sorted(season_dfs)
    if eval_seasons is None:
        eval_seasons = seasons[-5:] if len(seasons) >= 5 else seasons[-3:]

    def _objective(trial: "optuna.Trial") -> float:
        time_decay = trial.suggest_float("time_decay", 0.05, 0.25)
        platt_C = trial.suggest_float("platt_C", 0.1, 10.0, log=True)

        if model_type == "lightgbm":
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "verbose": -1,
                "random_state": 42,
                "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
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
            X_train, y_train, w_train = _prep(train_df, time_decay=time_decay)
            eval_df = season_dfs[eval_s]
            eval_clean = eval_df.dropna(subset=FEATURE_COLS + ["home_win"])
            X_eval = eval_clean[FEATURE_COLS].values.astype(float)
            y_eval = eval_clean["home_win"].values.astype(float)

            cal_size = max(int(0.2 * len(X_train)), 300)
            X_fit, y_fit, w_fit = X_train[:-cal_size], y_train[:-cal_size], w_train[:-cal_size]
            X_cal, y_cal = X_train[-cal_size:], y_train[-cal_size:]

            try:
                model = _fit_model(model_type, X_fit, y_fit, w_fit, params)
                cal_model = _calibrate(model, X_cal, y_cal, platt_C=platt_C)
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

    # Merge with defaults; keep training keys (time_decay, platt_C) in saved JSON
    if model_type == "lightgbm":
        model_defaults = _LGB_PARAMS
    else:
        model_defaults = _XGB_PARAMS
    final = {**model_defaults, **best}

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
    catboost_params: dict[str, Any] | None = None,
    mlp_params: dict[str, Any] | None = None,
    time_decay: float | None = None,
    platt_C: float | None = None,
) -> dict[str, list[dict]]:
    """Expanding-window CV with time-weighted training and Platt calibration."""
    model_types = model_types or ["logistic", "lightgbm", "xgboost", "catboost", "mlp", "stacked"]
    decay = time_decay if time_decay is not None else _TIME_DECAY
    cal_C = platt_C if platt_C is not None else 1.0

    raw_season_dfs: dict[int, pd.DataFrame] = {
        int(f.stem.split("_")[1]): pd.read_parquet(f)
        for f in sorted(features_dir.glob("features_*.parquet"))
    }
    if not raw_season_dfs:
        raise RuntimeError(f"No feature files found in {features_dir}")

    # Only keep seasons whose feature matrix has all FEATURE_COLS (e.g. drop stale 2026)
    required = set(FEATURE_COLS) | {"home_win"}
    season_dfs = {
        s: df for s, df in raw_season_dfs.items()
        if required.issubset(df.columns)
    }
    if not season_dfs:
        raise RuntimeError(
            f"No feature files with all required columns in {features_dir}. "
            f"Required: {len(required)} columns."
        )

    seasons = sorted(season_dfs)
    results: dict[str, list[dict]] = {mt: [] for mt in model_types}

    for i, eval_season in enumerate(seasons):
        if i < min_train_seasons:
            continue

        train_seasons = seasons[:i]
        train_df = pd.concat([season_dfs[s] for s in train_seasons], ignore_index=True)
        X_train, y_train, w_train = _prep(train_df, time_decay=decay)

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
            params = (
                lgb_params if mt == "lightgbm"
                else (xgb_params if mt == "xgboost" else (catboost_params if mt == "catboost" else (mlp_params if mt == "mlp" else None)))
            )
            model = _fit_model(mt, X_fit, y_fit, w_fit, _model_params_only(params))
            cal = _calibrate(model, X_cal, y_cal, platt_C=cal_C)
            fitted_models[mt] = cal

            y_prob = _predict_proba(cal, X_eval)
            er = evaluate(y_eval, y_prob)

            meta = ModelMetadata(
                model_version=_FEATURE_VERSION,
                model_type=mt,
                training_seasons=train_seasons,
                hyperparameters=_model_params_only(params)
                or (
                    _LR_PARAMS if mt == "logistic"
                    else (_LGB_PARAMS if mt == "lightgbm" else (_XGB_PARAMS if mt == "xgboost" else (_CATB_PARAMS if mt == "catboost" else _MLP_PARAMS))),
                ),
                feature_set_version=_FEATURE_VERSION,
                feature_cols=FEATURE_COLS,
                train_brier=float(er.brier_score),
                train_n_games=len(X_fit),
            )
            save_model(cal, meta, model_dir=model_dir)
            results[mt].append(_result_row(mt, eval_season, len(X_train), er))

        # Stacked ensemble
        base_keys = [k for k in ["logistic", "lightgbm", "xgboost", "catboost", "mlp"] if k in fitted_models]
        if "stacked" in model_types and len(base_keys) >= 2:
            cal_preds = np.column_stack(
                [_predict_proba(fitted_models[k], X_cal) for k in base_keys]
            )
            meta_lr = LogisticRegression(**_META_PARAMS)
            meta_lr.fit(cal_preds, y_cal)
            eval_preds = np.column_stack(
                [_predict_proba(fitted_models[k], X_eval) for k in base_keys]
            )
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
    catboost_params: dict[str, Any] | None = None,
    mlp_params: dict[str, Any] | None = None,
    time_decay: float | None = None,
    platt_C: float | None = None,
) -> dict[str, Any]:
    """Train calibrated production models on all available seasons."""
    model_types = model_types or ["logistic", "lightgbm", "xgboost", "catboost", "mlp", "stacked"]
    decay = time_decay if time_decay is not None else _TIME_DECAY
    cal_C = platt_C if platt_C is not None else 1.0

    frames = [pd.read_parquet(f) for f in sorted(features_dir.glob("features_*.parquet"))]
    all_data = pd.concat(frames, ignore_index=True)
    X_all, y_all, w_all = _prep(all_data, time_decay=decay)
    seasons = sorted({int(f.stem.split("_")[1]) for f in features_dir.glob("features_*.parquet")})

    cal_size = max(int(0.15 * len(X_all)), 500)
    X_fit, y_fit, w_fit = X_all[:-cal_size], y_all[:-cal_size], w_all[:-cal_size]
    X_cal, y_cal = X_all[-cal_size:], y_all[-cal_size:]

    trained: dict[str, Any] = {}
    base_models: dict[str, Any] = {}

    for mt in model_types:
        if mt == "stacked":
            continue
        params = (
            lgb_params if mt == "lightgbm"
            else (xgb_params if mt == "xgboost" else (catboost_params if mt == "catboost" else (mlp_params if mt == "mlp" else None)))
        )
        model = _fit_model(mt, X_fit, y_fit, w_fit, _model_params_only(params))
        cal = _calibrate(model, X_cal, y_cal, platt_C=cal_C)
        base_models[mt] = cal

        meta = ModelMetadata(
            model_version=_FEATURE_VERSION,
            model_type=mt,
            training_seasons=seasons,
            hyperparameters=_model_params_only(params)
            or (
                _LR_PARAMS if mt == "logistic"
                else (_LGB_PARAMS if mt == "lightgbm" else (_XGB_PARAMS if mt == "xgboost" else (_CATB_PARAMS if mt == "catboost" else _MLP_PARAMS))),
            ),
            feature_set_version=_FEATURE_VERSION,
            feature_cols=FEATURE_COLS,
            train_brier=0.0,
            train_n_games=len(X_fit),
        )
        save_model(cal, meta, model_dir=model_dir)
        trained[mt] = cal
        print(f"  {mt} production model saved")

    # For stacked: supplement base_models with any already-saved artifacts so
    # that `--models stacked --skip-cv` works without retraining base models.
    if "stacked" in model_types:
        from winprob.model.artifacts import latest_artifact as _latest

        for _bk in ["logistic", "lightgbm", "xgboost", "catboost", "mlp"]:
            if _bk not in base_models:
                _art = _latest(_bk, model_dir=model_dir, version=_FEATURE_VERSION)
                if _art is not None:
                    base_models[_bk], _ = load_model(_art)
                    print(f"  loaded existing {_bk} artifact for stacking")

    base_keys = [k for k in ["logistic", "lightgbm", "xgboost", "catboost"] if k in base_models]
    if "stacked" in model_types and len(base_keys) >= 2:
        cal_preds = np.column_stack([_predict_proba(base_models[k], X_cal) for k in base_keys])
        meta_lr = LogisticRegression(**_META_PARAMS)
        meta_lr.fit(cal_preds, y_cal)
        ensemble = StackedEnsemble(
            base_models={k: base_models[k] for k in base_keys},
            meta_lr=meta_lr,
            base_keys=base_keys,
        )
        meta = ModelMetadata(
            model_version=_FEATURE_VERSION,
            model_type="stacked",
            training_seasons=seasons,
            hyperparameters={"meta": _META_PARAMS},
            feature_set_version=_FEATURE_VERSION,
            feature_cols=FEATURE_COLS,
            train_brier=0.0,
            train_n_games=len(X_cal),
        )
        save_model(ensemble, meta, model_dir=model_dir)
        trained["stacked"] = ensemble
        print("  stacked production model saved")

    return trained
