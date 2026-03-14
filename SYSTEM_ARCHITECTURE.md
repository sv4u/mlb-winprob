# SYSTEM_ARCHITECTURE.md

## MLB Prediction System — Architecture and Data Flow

---

# 1. High-Level Components

1. **Ingestion Layer** — fetch and cache external data
   - MLB Stats API client (async, rate-limited, cached)
   - Retrosheet downloader (Chadwick primary / retrosheet.org ZIP fallback)
   - FanGraphs team metrics (via `pybaseball`)
   - Individual pitcher stats (MLB Stats API)
   - Player data (biographical, gamelogs, Statcast batter/pitcher stats)
   - Spring training schedule scores (MLB Stats API)

2. **Processing Layer** — normalize and link raw data
   - Schedule normalization
   - Retrosheet GL parsing
   - Crosswalk builder (Retrosheet → MLB `game_pk`)

3. **Storage Layer** — hybrid DuckDB + Parquet
   - **DuckDB** (`data/processed/mlb_predict.duckdb`) — primary analytical store for fast reads
     (10-50x faster than scanning individual Parquet files for multi-season queries)
   - **Parquet files** — canonical export format for interoperability, snapshots, and checksums
   - Feature data is ingested into DuckDB after build; training and serving read from DuckDB first
   - Automatic fallback to direct Parquet reads when DuckDB store is unavailable

4. **Feature Layer** — deterministic 136-feature matrix per game (119 team + 17 Stage 1 player)
   - Elo ratings (sequential, cross-season)
   - Multi-window rolling stats (15 / 30 / 60 games)
   - EWMA rolling stats (span=20)
   - Home/away performance splits
   - Pitcher stats (prior-season ERA, K/9, BB/9)
   - FanGraphs team advanced metrics (prior-season wOBA, FIP, xFIP, …)
   - Statcast lineup-weighted and pitcher stats (prior-season player-level)
   - Park run factors
   - Differential features
   - Spring training features (from schedule scores + prior-season state)
   - Game type indicator (is_spring)
   - Stage 1 player embedding features (17 game-level features from PyTorch model)
   - Feature assembly uses `pd.concat(axis=1)` to avoid DataFrame fragmentation

5. **Model Layer** — six trained classifiers with probability calibration (isotonic for tree models, Platt for linear/neural)
   - Logistic regression (interpretable baseline)
   - LightGBM (Optuna-tuned gradient boosting, 60 trials)
   - XGBoost (Optuna-tuned gradient boosting, 60 trials)
   - CatBoost (Optuna-tuned gradient boosting, 60 trials)
   - MLP (multi-layer perceptron neural network)
   - Stacked ensemble (meta-logistic on base-model probabilities)

6. **Scoring Layer** — immutable prediction snapshots
   - Daily snapshot Parquet files (by season + run timestamp)
   - Reproducible from raw inputs given the same model artifact

7. **Monitoring Layer** — drift detection across runs
   - Incremental diff (vs previous snapshot)
   - Baseline diff (vs first snapshot of season)
   - Per-season `run_metrics.parquet` and global metrics log

8. **Serving Layer** — web dashboard + CLI
   - FastAPI / Jinja2 web dashboard (port 30087)
   - gRPC server (port 50051) with gateway
   - MCP server (Streamable HTTP at /mcp)
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

Player data (biographical, gamelogs, Statcast):
             └─► data/processed/player/biographical.parquet
             └─► data/processed/player/{batter,pitcher}_stats_YYYY.parquet
             └─► data/processed/player/pitcher_gamelogs_YYYY.parquet

schedule + gamelogs
   └─► data/processed/crosswalk/game_id_map_YYYY.parquet

crosswalk + gamelogs + pitcher_stats + fangraphs + statcast + vegas + weather
   └─► data/processed/features/features_YYYY.parquet   (136 features, historical)
   └─► data/processed/features/features_2026.parquet   (pre-season, from team state)

schedule (spring training scores) + prior-season features
   └─► data/processed/features/features_spring_YYYY.parquet  (spring training)

features ──► DuckDB store (data/processed/mlb_predict.duckdb)
         └─► model training ──► data/models/{type}_v4_train{season}/

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

- go through async clients (`src/mlb_predict/mlbapi/client.py`)
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

## 4.1 `src/mlb_predict/mlbapi`

- Async Stats API wrapper
- Cache responses keyed by endpoint + params (SHA256 filename)
- Append-only `metadata.jsonl` for audit trail
- Clients: `schedule.py`, `pitcher_stats.py`, `teams.py`

## 4.2 `src/mlb_predict/retrosheet`

- Download + parse Retrosheet GL logs
- Multiple source support with automatic fallback
- Persist provenance metadata (source URL, raw SHA256)

## 4.3 `src/mlb_predict/crosswalk`

- Deterministically map Retrosheet game rows to MLB `game_pk`
- Emit unresolved lists per season
- Produce coverage report; enforce ≥ 99.0% match threshold

## 4.4 `src/mlb_predict/statcast`

- FanGraphs team advanced metrics via `pybaseball`
- Persists `fangraphs_YYYY.parquet` (wOBA, FIP, xFIP, K%, BB%, …)

## 4.5 `src/mlb_predict/storage`

- `duckdb_store.py` — hybrid DuckDB + Parquet storage layer
- Singleton `get_store()` provides thread-safe access to the DuckDB database
- `ingest_parquet()` / `ingest_all_features()` for Parquet → DuckDB ingestion
- `query_features()` / `query_training_data()` for fast analytical reads
- `export_parquet()` for DuckDB → Parquet export (snapshots, interoperability)
- Automatic fallback: if DuckDB is unavailable, callers fall back to direct Parquet reads

## 4.6 `src/mlb_predict/features`

- `elo.py` — sequential cross-season Elo with home-field HFA offset
- `team_stats.py` — rolling windows (15/30/60 games), EWMA, home/away splits, streaks, rest
- `pitcher_stats.py` — gamelog-based pitcher ERA assembly
- `park_factors.py` — median runs-per-game park factor from historical gamelogs
- `builder.py` — assembles the 136-feature matrix (119 team + 17 Stage 1 player)
  using `pd.concat(axis=1)` to avoid DataFrame fragmentation; saves per-season Parquet

## 4.7 `src/mlb_predict/model`

- `train.py` — logistic, LightGBM, XGBoost, CatBoost, MLP, stacked ensemble; calibration (isotonic for tree models, Platt for linear/neural);
  time-weighted sample weights; Optuna HPO (60 trials); expanding-window cross-validation;
  spring training weighting; DuckDB-accelerated feature loading with Parquet fallback;
  Stage 1 player embedding integration; pre-training data validation
- `evaluate.py` — Brier score, accuracy, calibration error
- `artifacts.py` — save / load model artifacts (joblib + JSON metadata)

## 4.8 `src/mlb_predict/predict`

- `snapshot.py` — produces immutable prediction Parquet files with
  provenance hashes (`model_version`, `feature_hash`, `schedule_hash`, `git_commit`)

## 4.9 `src/mlb_predict/drift`

- `compute.py` — incremental and baseline diffs; per-season and global run metrics

## 4.10 `src/mlb_predict/app`

- `main.py` — FastAPI application: game browser, 2026 season page, game detail,
  upsets, CV summary; SHAP attribution on game detail; auto-bootstrap with
  DuckDB population after ingest
- `data_cache.py` — loads features via DuckDB store (fast path) or Parquet files
  (fallback); normalizes date types; pre-computes probabilities for all games
- `admin.py` — pipeline runner with player data ingest step
- `templates/` — Jinja2 HTML templates (`index.html`, `game.html`, `season_2026.html`, `odds_hub.html`)

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
- Route all external HTTP through `src/mlb_predict/mlbapi/client.py`
