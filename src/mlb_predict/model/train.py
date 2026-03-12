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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from mlb_predict.features.builder import FEATURE_COLS
from mlb_predict.model.artifacts import ModelMetadata, load_model, save_model
from mlb_predict.model.evaluate import evaluate, EvalResult
from mlb_predict.player.embeddings import STAGE1_FEATURE_NAMES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1 integration
# ---------------------------------------------------------------------------


def _load_stage1_inputs(
    gamelogs_dir: Path,
    player_dir: Path,
    seasons: list[int],
) -> dict[str, Any] | None:
    """Load gamelogs, rolling stats, and bio data needed for Stage 1.

    Returns None if player data is unavailable (graceful v3 fallback).
    """

    from mlb_predict.player.biographical import build_bio_lookup, build_biographical_df
    from mlb_predict.player.rolling import build_batter_rolling, build_pitcher_rolling

    bio_path = player_dir / "biographical.parquet"
    if not bio_path.exists():
        bio_df = build_biographical_df(cache_dir=player_dir)
    else:
        bio_df = pd.read_parquet(bio_path)

    if bio_df.empty:
        logger.info("No biographical data available; skipping Stage 1")
        return None

    bio_lookup = build_bio_lookup(bio_df)

    retro_to_mlbam: dict[str, int] = {}
    for _, row in bio_df.iterrows():
        retro_id = row.get("retro_id")
        mlbam = row.get("mlbam_id")
        if pd.notna(retro_id) and pd.notna(mlbam):
            retro_to_mlbam[str(retro_id).strip().lower()] = int(mlbam)

    gamelogs_frames = []
    for s in seasons:
        path = gamelogs_dir / f"gamelogs_{s}.parquet"
        if path.exists():
            gamelogs_frames.append(pd.read_parquet(path))

    if not gamelogs_frames:
        logger.info("No gamelogs found for Stage 1; skipping")
        return None

    all_gamelogs = pd.concat(gamelogs_frames, ignore_index=True)

    # Load per-pitcher game logs (high-fidelity MLB API data) if available
    from mlb_predict.player.pitcher_gamelogs import load_pitcher_gamelogs

    pitcher_game_logs = load_pitcher_gamelogs(player_dir, seasons)
    if not pitcher_game_logs.empty:
        logger.info("Loaded %d per-pitcher game log entries", len(pitcher_game_logs))

    batter_rolling = build_batter_rolling(all_gamelogs, retro_to_mlbam=retro_to_mlbam)
    pitcher_rolling = build_pitcher_rolling(
        all_gamelogs,
        retro_to_mlbam=retro_to_mlbam,
        pitcher_game_logs=pitcher_game_logs if not pitcher_game_logs.empty else None,
    )

    return {
        "gamelogs": all_gamelogs,
        "batter_rolling": batter_rolling,
        "pitcher_rolling": pitcher_rolling,
        "bio_lookup": bio_lookup,
        "retro_to_mlbam": retro_to_mlbam,
    }


def _train_stage1_and_generate(
    stage1_data: dict[str, Any],
    train_gamelogs: pd.DataFrame,
    target_gamelogs: pd.DataFrame,
    target_season_df: pd.DataFrame,
) -> pd.DataFrame:
    """Train Stage 1 on train_gamelogs, generate features for target_season_df.

    Returns a copy of target_season_df with Stage 1 features populated.
    Aggressively frees intermediate PyTorch tensors to limit peak memory.
    """
    import gc

    from mlb_predict.player.embeddings import PlayerVocab
    from mlb_predict.player.lineup_model import (
        generate_stage1_features,
        prepare_game_tensors,
        stage1_features_to_df,
        train_stage1,
    )

    vocab = PlayerVocab()
    bio_lookup = stage1_data["bio_lookup"]
    retro_to_mlbam = stage1_data["retro_to_mlbam"]
    batter_rolling = stage1_data["batter_rolling"]
    pitcher_rolling = stage1_data["pitcher_rolling"]

    train_tensors = prepare_game_tensors(
        train_gamelogs,
        batter_rolling,
        pitcher_rolling,
        bio_lookup,
        retro_to_mlbam,
        vocab,
        train_mode=True,
    )

    if train_tensors is None or train_tensors["valid_mask"].sum() < 100:
        logger.warning("Insufficient Stage 1 training data; using zero features")
        return target_season_df

    val_size = max(int(0.15 * train_tensors["valid_mask"].sum()), 200)
    n_valid = int(train_tensors["valid_mask"].sum())
    if n_valid > val_size + 200:
        valid_indices = train_tensors["valid_mask"].nonzero(as_tuple=True)[0]
        val_indices = valid_indices[-val_size:]
        val_mask = train_tensors["valid_mask"].clone()
        val_mask[:] = False
        val_mask[val_indices] = True
        train_mask = train_tensors["valid_mask"].clone()
        train_mask[val_indices] = False

        val_tensors = {k: v.clone() for k, v in train_tensors.items()}
        val_tensors["valid_mask"] = val_mask
        train_tensors["valid_mask"] = train_mask
    else:
        val_tensors = None

    model = train_stage1(train_tensors, vocab, val_tensors=val_tensors)
    del train_tensors, val_tensors

    target_tensors = prepare_game_tensors(
        target_gamelogs,
        batter_rolling,
        pitcher_rolling,
        bio_lookup,
        retro_to_mlbam,
        vocab,
        train_mode=False,
    )

    if target_tensors is None:
        del model
        gc.collect()
        return target_season_df

    features = generate_stage1_features(model, target_tensors)
    del model, target_tensors
    features_df = stage1_features_to_df(features)
    del features
    gc.collect()

    result = target_season_df.copy()
    if len(features_df) == len(result):
        for col in STAGE1_FEATURE_NAMES:
            if col in features_df.columns:
                result[col] = features_df[col].values
    else:
        logger.warning(
            "Stage 1 feature count mismatch: %d features vs %d games; using zeros",
            len(features_df),
            len(result),
        )

    return result


_FEATURE_VERSION = "v4"

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
    "train_dir": "/tmp/catboost_info",
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

# Features that are highly correlated with others (r > 0.95) and can be dropped
# to reduce noise for tree models without losing predictive information.
# Identified from correlation analysis of the 96-feature set.
_REDUNDANT_FEATURES: frozenset[str] = frozenset(
    {
        "home_win_pct_14",
        "away_win_pct_14",
        "home_run_diff_14",
        "away_run_diff_14",
        "home_pythag_14",
        "away_pythag_14",
        "home_win_pct_7",
        "away_win_pct_7",
        "home_run_diff_7",
        "away_run_diff_7",
        "home_pythag_7",
        "away_pythag_7",
    }
)


def select_features(
    feature_cols: list[str],
    *,
    drop_redundant: bool = True,
) -> list[str]:
    """Return a pruned feature list with correlated/redundant features removed."""
    if not drop_redundant:
        return feature_cols
    return [c for c in feature_cols if c not in _REDUNDANT_FEATURES]


# ---------------------------------------------------------------------------
# Platt calibrator
# ---------------------------------------------------------------------------


class PlattCalibrator:
    """Platt scaling: fits a sigmoid on the base model's log-odds output."""

    def __init__(self, base: Any, C: float = 1.0) -> None:
        self.base = base
        self._sigmoid = LogisticRegression(C=C, solver="lbfgs", max_iter=500)

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        raw = _raw_proba(self.base, X)
        log_odds = logit(np.clip(raw, 1e-7, 1 - 1e-7)).reshape(-1, 1)
        self._sigmoid.fit(log_odds, y)
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        raw = _raw_proba(self.base, X)
        log_odds = logit(np.clip(raw, 1e-7, 1 - 1e-7)).reshape(-1, 1)
        return self._sigmoid.predict_proba(log_odds)

    @property
    def booster_(self) -> Any:
        return getattr(self.base, "booster_", None)


class IsotonicCalibrator:
    """Isotonic calibration: non-parametric monotonic mapping from raw probs to calibrated probs.

    Works better than Platt scaling when the relationship between raw scores
    and true probabilities is not perfectly sigmoid (common with tree models).
    """

    def __init__(self, base: Any) -> None:
        self.base = base
        self._iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        raw = _raw_proba(self.base, X)
        self._iso.fit(raw, y)
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        raw = _raw_proba(self.base, X)
        cal = self._iso.predict(raw)
        return np.column_stack([1.0 - cal, cal])

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
    p = dict(params or _CATB_PARAMS)
    p.setdefault("train_dir", "/tmp/catboost_info")
    return CatBoostClassifier(**p)


def _build_mlp(params: dict[str, Any] | None = None) -> Pipeline:
    """Feed-forward MLP with StandardScaler (2–3 hidden layers, ReLU, sigmoid output)."""
    p = dict(params or _MLP_PARAMS)
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(**p)),
        ]
    )


# ---------------------------------------------------------------------------
# Predict helpers
# ---------------------------------------------------------------------------


def _raw_proba(
    model: Any,
    X: np.ndarray | pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> np.ndarray:
    if not isinstance(X, pd.DataFrame):
        cols = feature_cols or FEATURE_COLS
        X = pd.DataFrame(X, columns=cols)
    return model.predict_proba(X)[:, 1]


def _predict_proba(model: Any, X: np.ndarray | pd.DataFrame) -> np.ndarray:
    if isinstance(model, (PlattCalibrator, IsotonicCalibrator, StackedEnsemble)):
        return model.predict_proba(X)[:, 1]
    return _raw_proba(model, X)


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------


def _available_features(
    dfs: dict[int, pd.DataFrame],
) -> list[str]:
    """Return FEATURE_COLS that exist in *every* season DataFrame."""
    if not dfs:
        return list(FEATURE_COLS)
    common = set.intersection(*(set(df.columns) for df in dfs.values()))
    avail = [c for c in FEATURE_COLS if c in common]
    dropped = set(FEATURE_COLS) - set(avail)
    if dropped:
        logger.info(
            "Dropped %d features missing from some seasons: %s",
            len(dropped),
            sorted(dropped),
        )
    return avail


_DEFAULT_SPRING_WEIGHT: float = 0.25


def _prep(
    df: pd.DataFrame,
    *,
    feature_cols: list[str] | None = None,
    time_weighted: bool = True,
    time_decay: float = _TIME_DECAY,
    spring_weight: float = _DEFAULT_SPRING_WEIGHT,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return X (DataFrame), y, sample_weights from a feature DataFrame.

    Spring training games (``is_spring == 1.0``) receive a reduced sample
    weight of ``spring_weight`` (default 0.25) so they contribute signal
    without dominating regular-season patterns.
    """
    cols = feature_cols or FEATURE_COLS
    clean = df.dropna(subset=cols + ["home_win"])
    X = clean[cols].astype(float)
    y = clean["home_win"].values.astype(float)
    if time_weighted and "season" in clean.columns:
        w = _season_weights(clean["season"], decay=time_decay)
    else:
        w = np.ones(len(y))
    if "is_spring" in clean.columns:
        is_spring = clean["is_spring"].values.astype(float)
        w = w * np.where(is_spring == 1.0, spring_weight, 1.0)
    return X, y, w


def _calibrate(
    base: Any,
    X_cal: pd.DataFrame | np.ndarray,
    y_cal: np.ndarray,
    platt_C: float = 1.0,
    calibration: str = "platt",
) -> PlattCalibrator | IsotonicCalibrator:
    """Calibrate a fitted model. Use 'isotonic' for tree models with enough data."""
    if calibration == "isotonic" and len(y_cal) >= 500:
        cal: PlattCalibrator | IsotonicCalibrator = IsotonicCalibrator(base)
    else:
        cal = PlattCalibrator(base, C=platt_C)
    cal.fit(X_cal, y_cal)
    return cal


def _fit_model(
    mt: str,
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    params: dict[str, Any] | None = None,
) -> Any:
    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    if mt == "logistic":
        model = _build_lr()
        model.fit(X_df, y, lr__sample_weight=w)
    elif mt == "lightgbm":
        model = _build_lgbm(params)
        model.fit(X_df, y, sample_weight=w)
    elif mt == "xgboost":
        model = _build_xgb(params)
        model.fit(X_df, y, sample_weight=w)
    elif mt == "catboost":
        model = _build_catboost(params)
        model.fit(X_df, y, sample_weight=w)
    elif mt == "mlp":
        model = _build_mlp(params)
        model.fit(X_df, y, mlp__sample_weight=w)
    else:
        raise ValueError(f"Unknown model type: {mt}")
    return model


# ---------------------------------------------------------------------------
# Optuna HPO
# ---------------------------------------------------------------------------


# Keys in HPO result that are training globals, not model constructor params.
_HPO_TRAINING_KEYS = frozenset({"time_decay", "platt_C", "calibration"})


def _model_params_only(params: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a copy of params with training keys removed for _fit_model."""
    if params is None:
        return None
    return {k: v for k, v in params.items() if k not in _HPO_TRAINING_KEYS}


_DEFAULT_CAL: dict[str, str] = {
    "logistic": "platt",
    "lightgbm": "isotonic",
    "xgboost": "isotonic",
    "catboost": "isotonic",
    "mlp": "platt",
}


def _cal_method_for(mt: str, params: dict[str, Any] | None) -> str:
    """Return the calibration method for *mt*, honouring HPO-tuned value if present."""
    if params and "calibration" in params:
        return str(params["calibration"])
    return _DEFAULT_CAL.get(mt, "platt")


def run_optuna_hpo(
    season_dfs: dict[int, pd.DataFrame],
    *,
    model_type: str = "lightgbm",
    n_trials: int = 200,
    eval_seasons: list[int] | None = None,
    model_dir: Path = Path("data/models"),
    spring_weight: float = _DEFAULT_SPRING_WEIGHT,
) -> dict[str, Any]:
    """Find optimal hyperparameters via Optuna on a subset of expanding seasons.

    The objective is mean Brier score on the last 5 seasons of the eval window.
    Returns the best params dict (including time_decay, platt_C, calibration)
    and saves to ``model_dir/hpo_{model_type}.json``.  Callers must strip
    _HPO_TRAINING_KEYS before passing params to model constructors; use
    _cal_method_for() to read the tuned calibration preference.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    feat_cols = _available_features(season_dfs)
    seasons = sorted(season_dfs)
    if eval_seasons is None:
        eval_seasons = seasons[-5:] if len(seasons) >= 5 else seasons[-3:]

    def _objective(trial: "optuna.Trial") -> float:
        time_decay = trial.suggest_float("time_decay", 0.05, 0.25)
        platt_C = trial.suggest_float("platt_C", 0.1, 10.0, log=True)
        cal_method = trial.suggest_categorical("calibration", ["platt", "isotonic"])

        if model_type == "lightgbm":
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "verbose": -1,
                "random_state": 42,
                "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 127),
                "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 2.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            }
        elif model_type == "xgboost":
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "verbosity": 0,
                "random_state": 42,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 9),
                "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 2.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            }
        elif model_type == "catboost":
            params = {
                "loss_function": "Logloss",
                "random_seed": 42,
                "verbose": False,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "depth": trial.suggest_int("depth", 4, 10),
                "iterations": trial.suggest_int("iterations", 200, 1000),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
                "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            }
        else:
            return 1.0

        briers = []
        for eval_s in eval_seasons:
            train_seasons = [s for s in seasons if s < eval_s]
            if len(train_seasons) < 2:
                continue
            train_df = pd.concat([season_dfs[s] for s in train_seasons], ignore_index=True)
            X_train, y_train, w_train = _prep(
                train_df,
                feature_cols=feat_cols,
                time_decay=time_decay,
                spring_weight=spring_weight,
            )
            eval_df = season_dfs[eval_s]
            eval_clean = eval_df.dropna(subset=feat_cols + ["home_win"])
            X_eval = eval_clean[feat_cols].astype(float)
            y_eval = eval_clean["home_win"].values.astype(float)

            cal_size = max(int(0.2 * len(X_train)), 300)
            X_fit, y_fit, w_fit = X_train[:-cal_size], y_train[:-cal_size], w_train[:-cal_size]
            X_cal, y_cal = X_train[-cal_size:], y_train[-cal_size:]

            try:
                model = _fit_model(model_type, X_fit, y_fit, w_fit, params)
                cal_model = _calibrate(
                    model,
                    X_cal,
                    y_cal,
                    platt_C=platt_C,
                    calibration=cal_method,
                )
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

    _default_map = {
        "lightgbm": _LGB_PARAMS,
        "xgboost": _XGB_PARAMS,
        "catboost": _CATB_PARAMS,
    }
    model_defaults = _default_map.get(model_type, {})
    final = {**model_defaults, **best}

    model_dir.mkdir(parents=True, exist_ok=True)
    out = model_dir / f"hpo_{model_type}.json"
    out.write_text(json.dumps(final, indent=2))
    return final


# ---------------------------------------------------------------------------
# Expanding-window CV
# ---------------------------------------------------------------------------


def _load_all_feature_files(features_dir: Path) -> dict[int, pd.DataFrame]:
    """Load regular-season and spring training feature files, merging by season.

    Adds ``is_spring = 0.0`` to DataFrames that lack the column for backward
    compatibility with feature files built before this column was introduced.
    """
    season_dfs: dict[int, list[pd.DataFrame]] = {}

    for f in sorted(features_dir.glob("features_*.parquet")):
        stem = f.stem
        if stem.startswith("features_spring_"):
            season = int(stem.split("_")[2])
        else:
            season = int(stem.split("_")[1])
        df = pd.read_parquet(f)
        if "is_spring" not in df.columns:
            df["is_spring"] = 0.0
        season_dfs.setdefault(season, []).append(df)

    return {s: pd.concat(dfs, ignore_index=True) for s, dfs in season_dfs.items()}


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
    spring_weight: float = _DEFAULT_SPRING_WEIGHT,
    gamelogs_dir: Path = Path("data/processed/retrosheet"),
    player_data_dir: Path = Path("data/processed/player"),
    enable_stage1: bool = True,
) -> dict[str, list[dict]]:
    """Expanding-window CV with time-weighted training and Platt calibration.

    When *enable_stage1* is True and player data files exist, the Stage 1
    player embedding model is trained at each fold (on train seasons) and its
    17 features are injected before fitting Stage 2 models.
    """
    model_types = model_types or ["logistic", "lightgbm", "xgboost", "catboost", "mlp", "stacked"]
    decay = time_decay if time_decay is not None else _TIME_DECAY
    cal_C = platt_C if platt_C is not None else 1.0

    raw_season_dfs = _load_all_feature_files(features_dir)
    if not raw_season_dfs:
        raise RuntimeError(f"No feature files found in {features_dir}")

    # Keep seasons that have home_win and at least some features
    season_dfs = {s: df for s, df in raw_season_dfs.items() if "home_win" in df.columns}
    if not season_dfs:
        raise RuntimeError(f"No feature files with 'home_win' column in {features_dir}")

    feat_cols = _available_features(season_dfs)
    logger.info("CV using %d features (of %d FEATURE_COLS)", len(feat_cols), len(FEATURE_COLS))

    seasons = sorted(season_dfs)
    results: dict[str, list[dict]] = {mt: [] for mt in model_types}

    # Stage 1: load player data and pre-compute features for ALL seasons in
    # one O(n) expanding-window pass.  Each season's features are generated by
    # a model trained only on prior seasons, so there is no data leakage.
    # Features are injected directly into season_dfs to avoid doubling memory.
    import gc

    if enable_stage1:
        stage1_data: dict[str, Any] | None = None
        try:
            stage1_data = _load_stage1_inputs(gamelogs_dir, player_data_dir, seasons)
        except Exception as exc:
            logger.warning("Stage 1 data loading failed: %s; proceeding without", exc)

        if stage1_data is not None:
            train_gl = stage1_data["gamelogs"]
            train_gl["date"] = pd.to_datetime(train_gl["date"])
            n_pre = 0
            for i, season in enumerate(seasons):
                if i < min_train_seasons:
                    continue
                earlier = seasons[:i]
                earlier_mask = train_gl["date"].dt.year.isin(earlier)
                season_mask = train_gl["date"].dt.year == season
                if earlier_mask.any() and season_mask.any():
                    season_dfs[season] = _train_stage1_and_generate(
                        stage1_data,
                        train_gamelogs=train_gl[earlier_mask],
                        target_gamelogs=train_gl[season_mask],
                        target_season_df=season_dfs[season],
                    )
                    n_pre += 1
            logger.info("Pre-computed Stage 1 features for %d seasons", n_pre)
            del stage1_data, train_gl
            gc.collect()

    for i, eval_season in enumerate(seasons):
        if i < min_train_seasons:
            continue

        train_seasons = seasons[:i]

        train_df = pd.concat([season_dfs[s] for s in train_seasons], ignore_index=True)
        X_train, y_train, w_train = _prep(
            train_df,
            feature_cols=feat_cols,
            time_decay=decay,
            spring_weight=spring_weight,
        )

        eval_clean = season_dfs[eval_season].dropna(subset=feat_cols + ["home_win"])
        X_eval = eval_clean[feat_cols].astype(float)
        y_eval = eval_clean["home_win"].values.astype(float)

        if len(X_train) < 100 or len(X_eval) == 0:
            continue

        cal_size = max(int(0.2 * len(X_train)), 500)
        X_fit, y_fit, w_fit = X_train[:-cal_size], y_train[:-cal_size], w_train[:-cal_size]
        X_cal_full, y_cal_full = X_train[-cal_size:], y_train[-cal_size:]

        # Split calibration data: first half calibrates base models, second
        # half trains the meta-learner to prevent data leakage.
        cal_mid = len(X_cal_full) // 2
        X_cal, y_cal = X_cal_full.iloc[:cal_mid], y_cal_full[:cal_mid]
        X_meta, y_meta = X_cal_full.iloc[cal_mid:], y_cal_full[cal_mid:]

        fitted_models: dict[str, Any] = {}
        _params_map = {
            "lightgbm": lgb_params,
            "xgboost": xgb_params,
            "catboost": catboost_params,
            "mlp": mlp_params,
        }

        for mt in model_types:
            if mt == "stacked":
                continue
            params = _params_map.get(mt)
            model = _fit_model(mt, X_fit, y_fit, w_fit, _model_params_only(params))
            cal_method = _cal_method_for(mt, params)
            cal = _calibrate(model, X_cal, y_cal, platt_C=cal_C, calibration=cal_method)
            fitted_models[mt] = cal

            y_prob = _predict_proba(cal, X_eval)
            er = evaluate(y_eval, y_prob)

            _hp_defaults = {
                "logistic": _LR_PARAMS,
                "lightgbm": _LGB_PARAMS,
                "xgboost": _XGB_PARAMS,
                "catboost": _CATB_PARAMS,
                "mlp": _MLP_PARAMS,
            }
            hp: dict[str, Any] = _model_params_only(params) or _hp_defaults.get(mt, {})
            meta = ModelMetadata(
                model_version=_FEATURE_VERSION,
                model_type=mt,
                training_seasons=train_seasons,
                hyperparameters=hp,
                feature_set_version=_FEATURE_VERSION,
                feature_cols=feat_cols,
                eval_brier=float(er.brier_score),
                train_n_games=len(X_fit),
            )
            save_model(cal, meta, model_dir=model_dir)
            results[mt].append(_result_row(mt, eval_season, len(X_train), er))

        # Stacked ensemble: train meta-learner on X_meta (unseen by calibrators)
        base_keys = [
            k for k in ["logistic", "lightgbm", "xgboost", "catboost", "mlp"] if k in fitted_models
        ]
        if "stacked" in model_types and len(base_keys) >= 2:
            meta_preds = np.column_stack(
                [_predict_proba(fitted_models[k], X_meta) for k in base_keys]
            )
            meta_lr = LogisticRegression(**_META_PARAMS)
            meta_lr.fit(meta_preds, y_meta)
            eval_preds = np.column_stack(
                [_predict_proba(fitted_models[k], X_eval) for k in base_keys]
            )
            y_prob_stack = meta_lr.predict_proba(eval_preds)[:, 1]
            er = evaluate(y_eval, y_prob_stack)
            results["stacked"].append(_result_row("stacked", eval_season, len(X_train), er))

            # Also evaluate a simple weighted average for comparison
            avg_prob = np.mean(
                [_predict_proba(fitted_models[k], X_eval) for k in base_keys], axis=0
            )
            er_avg = evaluate(y_eval, avg_prob)
            if "avg_ensemble" not in results:
                results["avg_ensemble"] = []
            results["avg_ensemble"].append(
                _result_row("avg_ensemble", eval_season, len(X_train), er_avg)
            )

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
    spring_weight: float = _DEFAULT_SPRING_WEIGHT,
    gamelogs_dir: Path = Path("data/processed/retrosheet"),
    player_data_dir: Path = Path("data/processed/player"),
    enable_stage1: bool = True,
) -> dict[str, Any]:
    """Train calibrated production models on all available seasons.

    When *enable_stage1* is True, trains the Stage 1 player embedding model
    on all seasons and saves the model alongside Stage 2 artifacts.
    """
    model_types = model_types or ["logistic", "lightgbm", "xgboost", "catboost", "mlp", "stacked"]
    decay = time_decay if time_decay is not None else _TIME_DECAY
    cal_C = platt_C if platt_C is not None else 1.0

    season_frames = _load_all_feature_files(features_dir)

    # --- Stage 1 for production: train on all data, inject features ----------
    import gc

    if enable_stage1:
        try:
            seasons_list = sorted(season_frames.keys())
            stage1_data = _load_stage1_inputs(gamelogs_dir, player_data_dir, seasons_list)
            if stage1_data is not None:
                train_gl = stage1_data["gamelogs"]
                train_gl["date"] = pd.to_datetime(train_gl["date"])
                for s in seasons_list:
                    s_gl_mask = train_gl["date"].dt.year == s
                    earlier = [ss for ss in seasons_list if ss < s]
                    if len(earlier) >= 2 and s_gl_mask.any():
                        earlier_gl_mask = train_gl["date"].dt.year.isin(earlier)
                        season_frames[s] = _train_stage1_and_generate(
                            stage1_data,
                            train_gamelogs=train_gl[earlier_gl_mask],
                            target_gamelogs=train_gl[s_gl_mask],
                            target_season_df=season_frames[s],
                        )
                # Save Stage 1 production model
                from mlb_predict.player.embeddings import (
                    PlayerVocab,
                    save_stage1_model,
                )
                from mlb_predict.player.lineup_model import (
                    prepare_game_tensors,
                    train_stage1,
                )

                vocab = PlayerVocab()
                all_tensors = prepare_game_tensors(
                    train_gl,
                    stage1_data["batter_rolling"],
                    stage1_data["pitcher_rolling"],
                    stage1_data["bio_lookup"],
                    stage1_data["retro_to_mlbam"],
                    vocab,
                    train_mode=True,
                )
                del stage1_data, train_gl
                if all_tensors is not None:
                    prod_model = train_stage1(all_tensors, vocab)
                    del all_tensors
                    s1_dir = (
                        model_dir / f"player_embedding_{_FEATURE_VERSION}_train{max(seasons_list)}"
                    )
                    save_stage1_model(
                        prod_model,
                        vocab,
                        s1_dir,
                        metadata={"training_seasons": seasons_list},
                    )
                    del prod_model
                    logger.info("Stage 1 production model saved to %s", s1_dir)
                else:
                    del all_tensors
                gc.collect()
        except Exception as exc:
            logger.warning("Stage 1 production training failed: %s; proceeding without", exc)
            gc.collect()
    feat_cols = _available_features(season_frames)
    logger.info(
        "Production training using %d features (of %d FEATURE_COLS)",
        len(feat_cols),
        len(FEATURE_COLS),
    )

    all_data = pd.concat(season_frames.values(), ignore_index=True)
    X_all, y_all, w_all = _prep(
        all_data,
        feature_cols=feat_cols,
        time_decay=decay,
        spring_weight=spring_weight,
    )
    seasons = sorted(season_frames)

    cal_size = max(int(0.15 * len(X_all)), 500)
    X_fit, y_fit, w_fit = X_all[:-cal_size], y_all[:-cal_size], w_all[:-cal_size]
    X_cal_full, y_cal_full = X_all[-cal_size:], y_all[-cal_size:]

    # Split calibration data: first half calibrates base models, second
    # half trains the meta-learner to prevent data leakage.
    cal_mid = len(X_cal_full) // 2
    X_cal, y_cal = X_cal_full.iloc[:cal_mid], y_cal_full[:cal_mid]
    X_meta, y_meta = X_cal_full.iloc[cal_mid:], y_cal_full[cal_mid:]

    trained: dict[str, Any] = {}
    base_models: dict[str, Any] = {}
    _params_map = {
        "lightgbm": lgb_params,
        "xgboost": xgb_params,
        "catboost": catboost_params,
        "mlp": mlp_params,
    }

    for mt in model_types:
        if mt == "stacked":
            continue
        params = _params_map.get(mt)
        model = _fit_model(mt, X_fit, y_fit, w_fit, _model_params_only(params))
        cal_method = _cal_method_for(mt, params)
        cal = _calibrate(model, X_cal, y_cal, platt_C=cal_C, calibration=cal_method)
        base_models[mt] = cal

        y_prob_fit = _predict_proba(cal, X_fit)
        er_fit = evaluate(y_fit, y_prob_fit)

        _hp_defaults = {
            "logistic": _LR_PARAMS,
            "lightgbm": _LGB_PARAMS,
            "xgboost": _XGB_PARAMS,
            "catboost": _CATB_PARAMS,
            "mlp": _MLP_PARAMS,
        }
        hp: dict[str, Any] = _model_params_only(params) or _hp_defaults.get(mt, {})
        meta = ModelMetadata(
            model_version=_FEATURE_VERSION,
            model_type=mt,
            training_seasons=seasons,
            hyperparameters=hp,
            feature_set_version=_FEATURE_VERSION,
            feature_cols=feat_cols,
            eval_brier=float(er_fit.brier_score),
            train_n_games=len(X_fit),
        )
        save_model(cal, meta, model_dir=model_dir)
        trained[mt] = cal
        print(f"  {mt} production model saved")

    if "stacked" in model_types:
        from mlb_predict.model.artifacts import latest_artifact as _latest

        for _bk in ["logistic", "lightgbm", "xgboost", "catboost", "mlp"]:
            if _bk not in base_models:
                _art = _latest(_bk, model_dir=model_dir, version=_FEATURE_VERSION)
                if _art is not None:
                    base_models[_bk], _ = load_model(_art)
                    print(f"  loaded existing {_bk} artifact for stacking")

    base_keys = [
        k for k in ["logistic", "lightgbm", "xgboost", "catboost", "mlp"] if k in base_models
    ]
    if "stacked" in model_types and len(base_keys) >= 2:
        # Train meta-learner on X_meta (unseen by calibrators)
        meta_preds = np.column_stack([_predict_proba(base_models[k], X_meta) for k in base_keys])
        meta_lr = LogisticRegression(**_META_PARAMS)
        meta_lr.fit(meta_preds, y_meta)
        ensemble = StackedEnsemble(
            base_models={k: base_models[k] for k in base_keys},
            meta_lr=meta_lr,
            base_keys=base_keys,
        )
        y_prob_stack = ensemble.predict_proba(X_fit)[:, 1]
        er_stack = evaluate(y_fit, y_prob_stack)
        meta = ModelMetadata(
            model_version=_FEATURE_VERSION,
            model_type="stacked",
            training_seasons=seasons,
            hyperparameters={"meta": _META_PARAMS},
            feature_set_version=_FEATURE_VERSION,
            feature_cols=feat_cols,
            eval_brier=float(er_stack.brier_score),
            train_n_games=len(X_fit),
        )
        save_model(ensemble, meta, model_dir=model_dir)
        trained["stacked"] = ensemble
        print("  stacked production model saved")

    return trained
