# AGENTS.md

## MLB Prediction System

### Full Engineering Specification (v3)

------------------------------------------------------------------------

# 1. System Overview

This repository implements a research-grade, reproducible MLB win
probability modeling platform covering seasons 2000–2026 (regular season
and spring training).

The system is designed to:

-   Ingest authoritative historical data
-   Maintain deterministic provenance
-   Support lineup-aware and pitcher-aware modeling
-   Persist prediction history immutably
-   Track model drift across time
-   Enable long-term calibration and research analysis

This document defines mandatory architectural, operational, and
governance rules for all agents contributing to this repository.

------------------------------------------------------------------------

# 2. Architectural Principles

All agents must preserve the following invariants:

1.  Determinism — identical inputs produce identical outputs.
2.  Provenance — every derived artifact must be traceable.
3.  Immutability — historical prediction snapshots are never mutated.
4.  Observability — all stages emit structured artifacts.
5.  Rate Safety — external APIs must be called through bounded,
    throttled clients.
6.  Multi-season support — scripts must accept multiple seasons per run.
7.  Fail-fast correctness — ambiguous states must raise errors.
8.  Coverage enforcement — minimum 99.0% crosswalk coverage.
9.  Storage redundancy — Parquet + CSV where appropriate.
10. Forward extensibility — no architectural dead-ends.

------------------------------------------------------------------------

# 3. Data Layer Specification

## 3.1 MLB Stats API Wrapper

Location: src/mlb_predict/mlbapi/

Requirements:

-   Async aiohttp client
-   TokenBucket(rate=5.0, burst=10.0)
-   Bounded concurrency
-   Retry with exponential backoff
-   429 handling
-   Raw JSON cache
-   Metadata JSONL logging

Direct API calls outside wrapper are forbidden.

### Cache Metadata Schema

Each request must log:

-   ts_unix
-   url
-   params
-   cache_key
-   endpoint
-   status

------------------------------------------------------------------------

## 3.2 Retrosheet Game Logs

Sources:

-   Chadwick GitHub mirror (primary)
-   Retrosheet.org ZIP (fallback)

On fallback:

-   source_used
-   url_used
-   fallback_reason
-   raw_sha256

Raw TXT must always be preserved.

Attribution requirements per Retrosheet notice must be respected.

------------------------------------------------------------------------

# 4. Data Schemas

## 4.1 Schedule Schema

games\_<season>.parquet

Columns:

-   game_pk (int)
-   season (int)
-   game_date_utc (ISO string)
-   game_date_local (ISO string)
-   home_mlb_id (int)
-   away_mlb_id (int)
-   home_abbrev (str)
-   away_abbrev (str)
-   venue_id (int)
-   local_timezone (str)
-   double_header (str)
-   game_number (int)
-   status (str)
- game_type (str: R=regular, S=spring training)
- home_score (int, nullable: final home score for completed games)
- away_score (int, nullable: final away score for completed games)

Checksum file:

games\_<season>.checksum.json

Includes:

- row_count
- game_types
-   parquet_sha256
-   csv_sha256
-   raw_payloads_sha256
-   mlbapi_config

------------------------------------------------------------------------

## 4.2 Retrosheet Schema

gamelogs\_<season>.parquet

Columns derived from official GL format.

Mandatory normalized fields:

-   date (date)
-   game_num (int)
-   visiting_team (str)
-   home_team (str)
-   visiting_score (int)
-   home_score (int)
-   visiting_starting_pitcher_id
-   home_starting_pitcher_id

Checksum file required.

------------------------------------------------------------------------

## 4.3 Crosswalk Schema

game_id_map\_<season>.parquet

Columns:

-   date
-   home_mlb_id
-   away_mlb_id
-   home_retro
-   away_retro
-   dh_game_num
-   status (matched\|missing\|ambiguous)
-   mlb_game_pk
-   match_confidence
-   notes

Coverage report:

crosswalk_coverage_report.parquet

Coverage threshold: 99.0% minimum.

------------------------------------------------------------------------

# 5. Prediction Snapshot Specification

Location:

data/processed/predictions/season=YYYY/snapshots/

Filename:

run_ts=<ISO>_<model_type>.parquet

Mandatory columns:

-   game_pk
-   home_team
-   away_team
-   predicted_home_win_prob
-   run_ts_utc
-   model_version
-   schedule_hash
-   feature_hash
-   lineup_param_hash
-   starter_param_hash
-   git_commit
-   tag (nullable)

Snapshots are immutable.

------------------------------------------------------------------------

# 6. Drift Specification

Each run must compute:

1.  Incremental diff (vs previous snapshot)
2.  Baseline diff (vs first snapshot of season)

Diff schema:

-   game_pk
-   p_old
-   p_new
-   delta
-   abs_delta
-   direction

Run metrics schema:

-   mean_abs_delta
-   p95_abs_delta
-   max_abs_delta
-   pct_gt_0p01
-   pct_gt_0p02
-   pct_gt_0p05

Logs:

- run_metrics_<season>.parquet (per-season, keyed by model_type + run_ts_utc)
- global_run_metrics.parquet (deduplicated by season + run_ts_utc + model_type)

run_ts_utc is auto-generated and immutable.

------------------------------------------------------------------------

# 7. Feature Engineering Contract

Future feature modules must:

- Accept season-scoped data
- Produce deterministic feature matrices
- Persist feature_hash
- Be reproducible from raw inputs

No feature randomness allowed without seeded RNG recorded in metadata.

------------------------------------------------------------------------

# 8. Modeling Contract

Implemented models (all available via `--model` / `--model-type` flags):

- `logistic` — L2 logistic regression baseline
- `lightgbm` — gradient boosted trees, Optuna-tuned
- `xgboost` — gradient boosted trees, Optuna-tuned
- `catboost` — gradient boosted trees, Optuna-tuned
- `mlp` — multi-layer perceptron neural network
- `stacked` — meta-learner (logistic) over calibrated base-model outputs (default production model)

All models apply:

- Probability calibration after training: isotonic calibration (non-parametric monotonic mapping) for tree models (LightGBM, XGBoost, CatBoost); Platt calibration (sigmoid meta-layer) for linear and neural models (logistic, MLP). Optuna HPO may override the default per model.
- Expanding-window cross-validation (train on seasons ≤ N-1, evaluate on N)
- Time-weighted training (exponential decay sample weights for recency)
- Stacked ensemble uses a disjoint calibration/meta-learner split: first half of held-out data calibrates base models, second half trains the meta-learner to prevent data leakage.

Model artifacts must include:

- model_version
- training_seasons
- hyperparameters
- feature_set_version
- eval_brier (formerly train_brier; legacy artifacts are auto-migrated on load)

------------------------------------------------------------------------

# 9. Error Taxonomy

All failures must classify into:

- IngestionError — raw data download or parsing failures
- APIError — MLB Stats API communication failures (canonical base class in errors.py; MLBAPIError in mlbapi/client.py)
- CoverageError — crosswalk coverage below required threshold
- SchemaError — unexpected column sets, types, or missing mandatory fields
- DriftComputationError — snapshot diff or metrics computation failures
- SnapshotIntegrityError — immutable snapshot corruption or schema violations

Silent failure is forbidden.

------------------------------------------------------------------------

# 10. Governance Rules

Agents must NOT:

- Modify historical snapshots
- Delete drift logs
- Change coverage threshold
- Introduce uncontrolled API concurrency
- Remove provenance metadata
- Introduce nondeterministic randomness

------------------------------------------------------------------------

# 11. Roadmap

Implemented modules:

1. Feature engineering pipeline (119 features, multi-window rolling, EWMA, Elo, home/away splits, FanGraphs, Statcast, bullpen, weather, Vegas, is_spring)
2. Pitcher modeling (season-level ERA, K/9, BB/9, WHIP via MLB Stats API)
3. Calibration engine (isotonic calibration for tree models, Platt calibration for linear/neural models)
4. Explanation interface (SHAP for tree models; coefficient ranking for logistic)
5. Web dashboard (FastAPI / Jinja2) with game browser, SHAP charts, upsets, 2026 season page, technical wiki, and admin dashboard (update season, full reingest, retrain with cleanup, fine-grained pipeline options, shell runner, Python REPL)
6. Live standings comparison (predicted vs actual divisional standings, league leaders, team batting/pitching stats via MLB Stats API; `src/mlb_predict/standings.py` + `src/mlb_predict/mlbapi/standings.py`)
7. Sitemap (HTML visual sitemap at `/sitemap` + XML sitemap at `/sitemap.xml`; linked from all page navigation)
8. CLI query tool (`scripts/query_game.py`)
9. Daily automation (`scripts/update_daily.sh` + cron)
10. EV Calculator (expected value, implied probability, edge, ROI, break-even probability, Kelly criterion; standalone page at `/tools/ev-calculator` + embedded widget on game detail pages with auto-populated model probabilities)
11. Pre-season / spring training support (`game_type` column: `R`=regular, `S`=spring; schedule ingestion with `--include-preseason` as default; `--no-preseason` to opt out; odds from both `baseball_mlb` and `baseball_mlb_preseason` sport keys; UI badges and caveat for pre-season predictions)
12. Admin console enhancements: fine-grained pipeline controls (season picker, `--include-preseason` toggle, refresh flags per pipeline); WebSocket shell runner (`WS /ws/admin/shell`) for streaming command execution; WebSocket Python REPL (`WS /ws/admin/repl`) with persistent session state
13. Spring training feature builder (`scripts/build_spring_features.py`): builds features for spring training games from MLB Stats API schedule scores + prior-season team state; `is_spring` binary feature; configurable `--spring-weight` (default 0.25) for sample weighting during training
14. Pre-training data validation: validates all required data files (schedule, regular-season features) exist for seasons 2000–current before training; lenient on spring training gaps

Planned modules:

1. Lineup expectation engine (batter quality from play-by-play / Statcast)
2. Monte Carlo simulation (full lineup-aware game simulation)
3. Market comparison module (line movement vs. predicted probability)
4. Hierarchical team priors

------------------------------------------------------------------------

# 12. Long-Term Intent

This repository is designed as a durable sabermetric research system.

Primary goals:

-   Auditability
-   Longitudinal drift study
-   Model explainability
-   Stable evolution over many seasons

Agents must preserve system integrity across time.

------------------------------------------------------------------------

END OF AGENTS.md (v3)
