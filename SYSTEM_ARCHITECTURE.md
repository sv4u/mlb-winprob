# SYSTEM_ARCHITECTURE.md
## MLB Win Probability Modeling System — Architecture and Data Flow

---

# 1. High-Level Components

1. **Ingestion Layer** — fetch and cache external data
   - MLB Stats API client (async, rate-limited, cached)
   - Retrosheet downloader (Chadwick primary / retrosheet.org ZIP fallback)
   - FanGraphs team metrics (via `pybaseball`)
   - Individual pitcher stats (MLB Stats API)
   - Spring training schedule scores (MLB Stats API)

2. **Processing Layer** — normalize and link raw data
   - Schedule normalization
   - Retrosheet GL parsing
   - Crosswalk builder (Retrosheet → MLB `game_pk`)

3. **Feature Layer** — deterministic 119-feature matrix per game
   - Elo ratings (sequential, cross-season)
   - Multi-window rolling stats (15 / 30 / 60 games)
   - EWMA rolling stats (span=20)
   - Home/away performance splits
   - Pitcher stats (prior-season ERA, K/9, BB/9)
   - FanGraphs team advanced metrics (prior-season wOBA, FIP, xFIP, …)
   - Park run factors
   - Differential features
   - Spring training features (from schedule scores + prior-season state)
   - Game type indicator (is_spring)
   - Spring training feature builder (schedule scores + prior-season team state)

4. **Model Layer** — six trained classifiers with probability calibration (isotonic for tree models, Platt for linear/neural)
   - Logistic regression (interpretable baseline)
   - LightGBM (Optuna-tuned gradient boosting)
   - XGBoost (Optuna-tuned gradient boosting)
   - CatBoost (Optuna-tuned gradient boosting)
   - MLP (multi-layer perceptron neural network)
   - Stacked ensemble (meta-logistic on base-model probabilities)

5. **Scoring Layer** — immutable prediction snapshots
   - Daily snapshot Parquet files (by season + run timestamp)
   - Reproducible from raw inputs given the same model artifact

6. **Monitoring Layer** — drift detection across runs
   - Incremental diff (vs previous snapshot)
   - Baseline diff (vs first snapshot of season)
   - Per-season `run_metrics.parquet` and global metrics log

7. **Serving Layer** — web dashboard + CLI
   - FastAPI / Jinja2 web dashboard (port 30087)
   - Human-centric CLI query tool (`scripts/query_game.py`)

---

# 2. Data Flow

## 2.1 Full pipeline

```text
MLB Stats API ──► data/raw/mlb_api/{schedule,stats,teams}/
             │
             ├─► data/processed/schedule/games_YYYY.parquet
             └─► data/processed/teams/teams_YYYY.parquet

Retrosheet (Chadwick primary / retrosheet.org fallback):
             ──► data/raw/retrosheet/gamelogs/GL<YYYY>.TXT
             └─► data/processed/retrosheet/gamelogs_YYYY.parquet

FanGraphs (via pybaseball):
             └─► data/processed/fangraphs/fangraphs_YYYY.parquet

MLB Stats API (pitcher stats):
             └─► data/processed/pitcher_stats/pitchers_YYYY.parquet

schedule + gamelogs
   └─► data/processed/crosswalk/game_id_map_YYYY.parquet

crosswalk + gamelogs + pitcher_stats + fangraphs
   └─► data/processed/features/features_YYYY.parquet   (119 features, historical)
   └─► data/processed/features/features_2026.parquet   (pre-season, from team state)

schedule (spring training scores) + prior-season features
   └─► data/processed/features/features_spring_YYYY.parquet  (spring training)

features ──► model training ──► data/models/{type}_v3_train{season}/

features + model
   └─► data/processed/predictions/season=YYYY/snapshots/run_ts=<iso>.parquet
   └─► data/processed/drift/{run_metrics_YYYY,global_run_metrics}.parquet
```

## 2.2 Daily automated update (`scripts/update_daily.sh`)

```text
1. ingest_schedule.py  (regular + spring training)
2. ingest_retrosheet_gamelogs.py  (current season)
3. build_crosswalk.py  (current season)
4. build_features.py  (current season)
5. build_spring_features.py  (current season)
6. build_features_2026.py  (update 2026 pre-season state)
7. kill server → start fresh uvicorn instance
```

---

# 3. Execution Model

## 3.1 Async + rate limiting

All external HTTP calls must:

- go through async clients (`src/winprob/mlbapi/client.py`)
- be token-bucket throttled (rate=5.0 req/s, burst=10)
- use bounded concurrency
- implement retries with exponential backoff

No direct synchronous HTTP calls for external sources.

## 3.2 Multi-season runs

All ingestion and feature scripts accept multiple seasons per invocation
via `--seasons` (space-separated list or shell expansion).

Orchestration script (`scripts/ingest_all.py`) runs all ingestion steps
in sequence and exits nonzero on any failure.

---

# 4. Module Responsibilities

## 4.1 `src/winprob/mlbapi`

- Async Stats API wrapper
- Cache responses keyed by endpoint + params (SHA256 filename)
- Append-only `metadata.jsonl` for audit trail
- Clients: `schedule.py`, `pitcher_stats.py`, `teams.py`

## 4.2 `src/winprob/retrosheet`

- Download + parse Retrosheet GL logs
- Multiple source support with automatic fallback
- Persist provenance metadata (source URL, raw SHA256)

## 4.3 `src/winprob/crosswalk`

- Deterministically map Retrosheet game rows to MLB `game_pk`
- Emit unresolved lists per season
- Produce coverage report; enforce ≥ 99.0% match threshold

## 4.4 `src/winprob/statcast`

- FanGraphs team advanced metrics via `pybaseball`
- Persists `fangraphs_YYYY.parquet` (wOBA, FIP, xFIP, K%, BB%, …)

## 4.5 `src/winprob/features`

- `elo.py` — sequential cross-season Elo with home-field HFA offset
- `team_stats.py` — rolling windows (15/30/60 games), EWMA, home/away splits, streaks, rest
- `pitcher_stats.py` — gamelog-based pitcher ERA assembly
- `park_factors.py` — median runs-per-game park factor from historical gamelogs
- `builder.py` — assembles the 119-feature matrix with is_spring indicator and saves per-season Parquet

## 4.6 `src/winprob/model`

- `train.py` — logistic, LightGBM, XGBoost, CatBoost, MLP, stacked ensemble; calibration (isotonic for tree models, Platt for linear/neural);
  time-weighted sample weights; Optuna HPO; expanding-window cross-validation;
  spring training weighting; pre-training data validation; combined regular + spring feature loading
- `evaluate.py` — Brier score, accuracy, calibration error
- `artifacts.py` — save / load model artifacts (joblib + JSON metadata)

## 4.7 `src/winprob/predict`

- `snapshot.py` — produces immutable prediction Parquet files with
  provenance hashes (`model_version`, `feature_hash`, `schedule_hash`, `git_commit`)

## 4.8 `src/winprob/drift`

- `compute.py` — incremental and baseline diffs; per-season and global run metrics

## 4.9 `src/winprob/app`

- `main.py` — FastAPI application: game browser, 2026 season page, game detail,
  upsets, CV summary; SHAP attribution on game detail
- `data_cache.py` — loads all feature Parquet files and the production model
  once at startup; normalizes date types; pre-computes probabilities for all games
- `templates/` — Jinja2 HTML templates (`index.html`, `game.html`, `season_2026.html`, `chat.html`, `odds_hub.html`, `ev_calculator.html`)

---

# 5. Failure Modes and Handling

| Failure | Classification | Handling |
| --- | --- | --- |
| API 429 | `APIError` | Respect `Retry-After`; exponential backoff |
| API 5xx | `APIError` | Exponential retry; eventual `APIError` with diagnostics |
| Retrosheet download failure | `IngestionError` | Fallback source; log `fallback_reason` |
| Crosswalk coverage < 99% | `CoverageError` | Emit report; flag in `crosswalk_seasons_below_threshold.csv` |
| Schema mismatch | `SchemaError` | Raise with column-level diagnostics |
| Drift computation failure | `DriftComputationError` | Raise; do not silence |
| Snapshot integrity failure | `SnapshotIntegrityError` | Raise; never overwrite existing snapshot |

Silent failure is forbidden in all modules.

---

# 6. Security / Compliance

- Preserve Retrosheet attribution requirements (`docs/RETROSHEET_ATTRIBUTION.md`).
- Do not store secrets; MLB Stats API access is anonymous.
- FanGraphs data is fetched via `pybaseball` for personal research use.

---

# 7. Extensibility Guidelines

Any new module must:

- Update `DATA_SCHEMA.md` when new datasets are introduced
- Define deterministic hashes for all derived artifacts
- Provide tests for schema stability (`tests/unit/`)
- Document provenance and versioning behavior
- Route all external HTTP through `src/winprob/mlbapi/client.py`
