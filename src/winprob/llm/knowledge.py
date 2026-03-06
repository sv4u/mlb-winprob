"""Knowledge base for the LLM: feature descriptions, model docs, sabermetric glossary.

Sourced from scripts/query_game.py _FEATURE_LABELS and the technical wiki.
Queried on demand by the describe_feature tool and used in the system prompt.
"""

from __future__ import annotations

# Feature name → short description (keep in sync with scripts/query_game.py _FEATURE_LABELS)
FEATURE_LABELS: dict[str, str] = {
    "home_elo": "Home team Elo rating",
    "away_elo": "Away team Elo rating",
    "elo_diff": "Elo advantage (home − away)",
    "home_win_pct_30": "Home team 30-game win%",
    "away_win_pct_30": "Away team 30-game win%",
    "home_pythag_30": "Home team 30-game Pythagorean",
    "away_pythag_30": "Away team 30-game Pythagorean",
    "home_win_pct_ewm": "Home team recent form (EWMA)",
    "away_win_pct_ewm": "Away team recent form (EWMA)",
    "home_pythag_ewm": "Home team recent Pythagorean (EWMA)",
    "away_pythag_ewm": "Away team recent Pythagorean (EWMA)",
    "home_win_pct_home_only": "Home team at-home win%",
    "away_win_pct_away_only": "Away team on-road win%",
    "home_sp_era": "Home starter ERA (prior season)",
    "away_sp_era": "Away starter ERA (prior season)",
    "home_sp_k9": "Home starter K/9 (prior season)",
    "away_sp_k9": "Away starter K/9 (prior season)",
    "home_bat_woba": "Home team wOBA (prior season)",
    "away_bat_woba": "Away team wOBA (prior season)",
    "home_pit_fip": "Home team FIP (prior season)",
    "away_pit_fip": "Away team FIP (prior season)",
    "pythag_diff_30": "Pythagorean edge (home − away, 30g)",
    "pythag_diff_ewm": "Recent Pythagorean edge (EWMA)",
    "home_away_split_diff": "Home/road performance edge",
    "sp_era_diff": "Pitcher ERA edge (lower = home advantage)",
    "woba_diff": "Batting quality edge (wOBA)",
    "fip_diff": "Pitching quality edge (FIP)",
    "park_run_factor": "Park run factor (1.0 = neutral)",
    "season_progress": "Point in season (0=opening, 1=end)",
    "home_streak": "Home team win streak (+) / loss streak (−)",
    "away_streak": "Away team win streak (+) / loss streak (−)",
    "home_rest_days": "Home team rest days",
    "away_rest_days": "Away team rest days",
}

# Sabermetric / stat terms (from wiki)
GLOSSARY: dict[str, str] = {
    "elo": "Elo rating: chess-style team strength. Starts at 1500; winner gains points, loser drops. Home gets +24. Positive elo_diff favors home.",
    "pythagorean": "Pythagorean expectation: expected win% from runs scored and allowed, RS²/(RS²+RA²). Teams above it are often 'lucky'.",
    "ewma": "Exponentially weighted moving average: recent games weighted more than older ones. Captures hot/cold streaks.",
    "woba": "Weighted On-Base Average: overall batting value per PA. Better than AVG because it weights hits by run value.",
    "fip": "Field Independent Pitching: ERA-like measure using only K, BB, HR. Strips out defense and batted-ball luck.",
    "era": "Earned Run Average: earned runs per 9 innings. Lower is better for pitchers.",
    "k9": "Strikeouts per 9 innings. Higher usually means dominant stuff.",
    "park factor": "Runs at this ballpark vs league average. 1.0 = neutral; Coors ~1.3, pitcher parks ~0.85.",
    "shap": "SHAP values: each feature's contribution to the prediction. Positive = pushes toward home win, negative toward away.",
    "brier": "Brier score: mean squared error of probabilities. Lower is better. 0.25 = random.",
}

# One-paragraph model descriptions (for system prompt / get_model_info)
MODEL_DOCS: dict[str, str] = {
    "logistic": "Logistic regression baseline. Linear combination of features passed through sigmoid. Z-score standardised, L2 regularisation. Interpretable coefficients.",
    "lightgbm": "LightGBM gradient boosted trees. Leaf-wise growth, histogram binning. Captures non-linear interactions. Optuna-tuned (num_leaves, learning_rate, n_estimators).",
    "xgboost": "XGBoost gradient boosted trees. Different regularisation from LightGBM. Often best single-model Brier. Optuna-tuned (max_depth, learning_rate, n_estimators).",
    "catboost": "CatBoost gradient boosted trees. Ordered boosting and symmetric trees. Third tree model for ensemble diversity. Optuna-tuned.",
    "mlp": "Multi-layer perceptron: 3 hidden layers (128→64→32), ReLU, Adam. Z-score normalised inputs. L2 weight decay. Captures different non-linear patterns than trees.",
    "stacked": "Production default. Meta-learner (logistic) on top of five calibrated base model probabilities. No raw features. Disjoint calibration/meta split to prevent leakage. C=0.5.",
}


def get_feature_description(feature_name: str) -> str:
    """Return a short description for a feature, or the name if unknown."""
    key = (feature_name or "").strip()
    return FEATURE_LABELS.get(key, key or "unknown feature")


def get_glossary_term(term: str) -> str | None:
    """Return the glossary definition for a term (lowercased), or None if not found."""
    key = (term or "").strip().lower()
    return GLOSSARY.get(key)


def get_model_docs(model_type: str) -> str:
    """Return a one-paragraph description for a model type, or a generic message."""
    key = (model_type or "").strip().lower()
    return MODEL_DOCS.get(key, f"No specific documentation for model '{model_type}'.")
