# MLB Prediction System

Research-grade MLB prediction system for pre-game win probability, standings, and related analysis — regular season and spring training, 2000–2026.

## Model performance (v3 — out-of-sample, expanding-window CV)

| Model               | Mean Brier | Mean Accuracy | Cal. Error | Best season |
| ------------------- | ---------- | ------------- | ---------- | ----------- |
| Logistic regression  | 0.2444     | 56.1%         | 0.029      | 2019: 58.4% |
| LightGBM (Optuna)   | 0.2455     | 55.8%         | 0.034      | 2019: 57.9% |
| XGBoost (Optuna)     | 0.2449     | 56.0%         | 0.032      | 2019: 58.5% |
| CatBoost (Optuna)    | 0.2470     | 54.9%         | 0.031      | 2010: 57.1% |
| MLP (Neural Network) | 0.2464     | 55.1%         | 0.031      | 2019: 59.2% |
| Avg. ensemble        | 0.2445     | 56.1%         | 0.029      | 2019: 58.5% |
| Stacked ensemble     | **0.2446** | **56.1%**     | **0.029**  | 2019: 58.5% |

All metrics are fully out-of-sample: train on seasons < N, evaluate on season N.
The stacked ensemble (blending all five base models) is the default production model.
Tree models use **isotonic calibration**; logistic and MLP use **Platt calibration**.
The training pipeline dynamically selects features available across all seasons.

---

## Models

The system trains six models on 119 pre-game features using an **expanding-window protocol** — each season N is evaluated using a model trained exclusively on seasons before N, so all reported metrics are fully out-of-sample. Every model goes through **probability calibration** (isotonic for tree models, Platt sigmoid for linear/neural) and **time-weighted training** (exponential decay rate 0.12 per season, so 2024 weight = 1.0, 2020 weight ≈ 0.61, 2015 weight ≈ 0.30). Features are dynamically selected based on availability across seasons.

### Logistic Regression

A regularised linear model that serves as the interpretable baseline. All 119 features are z-score standardised before fitting. Because the decision boundary is a hyperplane, the model captures additive effects — for example, a larger Elo differential increases home-win probability by a fixed amount regardless of the other features. Its simplicity makes it fast, stable, and easy to audit.

- **Regularisation**: L2 (ridge), `C=1.0`
- **Solver**: L-BFGS with up to 1 000 iterations
- **Interpretability**: SHAP attributions are computed directly from `coef × z-score` — no approximate explainer needed
- **When to use**: Speed-critical inference, auditing individual predictions, baseline comparison

### LightGBM

Microsoft's LightGBM grows an ensemble of shallow decision trees in sequence, where each tree corrects the residual errors of the ones before it. Unlike logistic regression, it captures **non-linear interactions** — for example, a high Elo differential combined with a good home/away split may carry a larger joint effect than either feature alone.

- **Hyperparameters**: 60-trial Optuna Bayesian search minimising out-of-sample Brier score (typical result: `num_leaves≈63`, `learning_rate≈0.05`, `n_estimators≈500`)
- **Interpretability**: Tree-based SHAP values via `shap.TreeExplainer`
- **When to use**: Fast batch inference at scale; often competitive with XGBoost

### XGBoost

DMLC's XGBoost is the other dominant gradient-boosted tree library. Its regularisation scheme (`min_child_weight`, separate L1/L2 penalties on leaf weights) and histogram-based split finding produce probability estimates that are **complementary** to LightGBM — they tend to disagree most on uncertain games near 50%, which makes them useful ensemble partners. XGBoost typically achieves the best **single-model** Brier score in this system.

- **Hyperparameters**: 60-trial Optuna Bayesian search (typical result: `max_depth≈6`, `learning_rate≈0.05`, `n_estimators≈500`)
- **Interpretability**: Tree-based SHAP values via `shap.TreeExplainer`
- **When to use**: Highest standalone accuracy; default choice when not ensembling

### CatBoost

Yandex's CatBoost uses ordered boosting and symmetric (oblivious) decision trees. Its unique training procedure reduces prediction shift, and symmetric trees tend to generalise well on tabular data. Acts as a third complementary tree model in the stacked ensemble, providing low-variance predictions that differ structurally from LightGBM and XGBoost.

- **Regularisation**: L2 leaf regularisation, learning rate decay
- **Architecture**: Symmetric (oblivious) trees with ordered boosting
- **When to use**: Robustness-focused inference; low-variance ensemble partner

### Neural Network (MLP)

A multi-layer perceptron classifier with three hidden layers (128, 64, 32 units) and ReLU activations. Captures nonlinear feature interactions that tree models may miss. Features are z-score normalised before training. Provides model diversity for stacking since its error surface is fundamentally different from tree-based learners.

- **Architecture**: 128 → 64 → 32 → 1, Adam optimiser
- **Regularisation**: L2 weight decay (alpha)
- **When to use**: Ensemble diversity; capturing non-tree-like nonlinearities

### Stacked Ensemble (default production model)

The stacked ensemble never sees raw features. Instead, it takes the **calibrated probability outputs** of all five base models as its five inputs and trains a logistic-regression **meta-learner** to find the optimal blend. Because each base model makes different errors, the meta-learner learns to up-weight whichever model is most confident in each probability range.

```
 Logistic prob  ─┐
 LightGBM prob  ─┤
 XGBoost prob   ─┼──▶  Logistic meta-learner  ──▶  P(home win)
 CatBoost prob  ─┤
 MLP prob       ─┘
```

- **Meta-learner**: `LogisticRegression(C=0.5)` — slight regularisation prevents over-fitting to the calibration set
- **Training**: The meta-learner is fit on the same held-out calibration split used for Platt scaling, so base-model probabilities are out-of-sample relative to the meta-learner
- **When to use**: Always — this is the default and achieves the best Brier score and calibration

### Training techniques

| Technique | What it does |
| --- | --- |
| **Probability calibration** | **Isotonic calibration** (non-parametric monotonic mapping) for tree models (LightGBM, XGBoost, CatBoost); **Platt calibration** (sigmoid) for logistic and MLP. Both use a held-out calibration set so predicted 65% games actually win ~65% of the time. |
| **Time-weighted training** | Exponential decay (`rate=0.12` per season) gives recent seasons more influence. This adapts the model to baseball rule changes — the 2023 shift ban, pitch clock, and larger bases shift team-level stats in ways that older seasons do not reflect. |
| **Optuna HPO** | Bayesian hyperparameter search (200 trials per model type) over a 5-season expanding-window objective. Searches `learning_rate`, tree depth, `n_estimators`, `subsample`, `colsample_bytree`, L1/L2 regularisation, and calibration method. Supports LightGBM, XGBoost, and CatBoost. |
| **Expanding-window CV** | For evaluation season N, the model is trained on all seasons before N. No future data ever leaks into training or calibration. |
| **Dynamic feature selection** | The pipeline automatically detects the intersection of available features across all season DataFrames and trains using only those features, ensuring robustness to missing columns in older seasons. |
| **Spring training weighting** | Spring training games are down-weighted via `--spring-weight` (default 0.25) so regular-season performance drives the model more strongly. |

---

## Features (119 total)

### Team performance (27 features)

- **Elo rating** (home, away, diff) — sequential cross-season rating with regression-to-mean at each season start; accounts for opponent quality
- **Multi-window rolling** (7 / 14 / 15 / 30 / 60 games, cross-season warm-start): win%, run differential, Pythagorean expectation
- **EWMA rolling** (span=20): exponentially-weighted recent-form metrics
- **Home/away performance splits**: team win% and Pythagorean computed separately in home games vs. road games

### Run distribution (4 features)

- **Scoring variance** (30-game window): run standard deviation for each team
- **One-run game win%** (30-game window): close-game resilience metric

### Context & fatigue (7 features)

- **Streak**: current win (+) or loss (−) streak for each team
- **Rest days**: calendar days since last game (capped at 10)
- **Season progress**: 0 = opener, 1 = final day
- **Day/night**: 1 = day game, 0 = night game
- **Interleague**: 1 = interleague matchup
- **Day of week**: 0 (Monday) – 6 (Sunday)

### Pitcher quality (8 features)

- **Prior-season SP ERA, K/9, BB/9, WHIP** from the MLB Stats API — one row per pitcher per season, joined by name

### Statcast individual player features (6 features)

- **Lineup-weighted batter xwOBA** (home, away) — prior-season Statcast expected wOBA averaged across the 9-man lineup; uses Chadwick Register for Retrosheet → MLBAM ID mapping
- **Lineup-weighted barrel%** (home, away) — prior-season barrel rate averaged across the lineup
- **Starting pitcher expected wOBA allowed** (home, away) — prior-season Statcast xwOBA for the opposing starter

### Advanced team metrics (FanGraphs, prior season, 20 features)

- **Batting**: wOBA, Barrel%, Hard Hit%, ISO, BABIP, xwOBA
- **Pitching**: FIP, xFIP, K%, BB%, HR/FB, WHIP

### Bullpen (8 features)

- **Bullpen usage** (15 / 30 game window): rolling average of relief innings pitched
- **Bullpen ERA proxy** (15 / 30 game window): rolling average of earned runs allowed by the bullpen

### Lineup (2 features)

- **Lineup continuity** (home, away) — fraction of the prior game's lineup retained

### Park & venue (1 feature)

- **Park run factor** — historical runs per game at the venue vs. league average

### Game type (1 feature)

- **is_spring** — binary: 1.0 for spring training, 0.0 for regular season

### Vegas odds (2 features)

- **Implied home win probability** — converted from money-line odds (defaults to 0.5 when unavailable)
- **Line movement** — change from opening to closing implied probability

### Weather (3 features)

- **Game temperature** (°F), **wind speed** (mph), **humidity** (%) — fetched from Open-Meteo historical API using park geo-coordinates

### Differential features (9 features)

- Pythagorean diff, EWMA Pythagorean diff, home/road split diff, SP ERA diff, wOBA diff, FIP diff, xwOBA diff, WHIP diff, ISO diff

---

## Quick start

### Install

```bash
git clone <repo>
cd mlb-predict
pip install -e .
```

### Full data ingestion (first run)

```bash
# 1. Fetch MLB schedules (2000–2026) — includes preseason by default; use --no-preseason to opt out
python scripts/ingest_schedule.py --seasons $(seq 2000 2026)

# 2. Fetch Retrosheet gamelogs (historical + current season)
python scripts/ingest_retrosheet_gamelogs.py --seasons $(seq 2000 2025)

# 3. Build Retrosheet ↔ MLB crosswalk
python scripts/build_crosswalk.py --seasons $(seq 2000 2025)

# 4. Fetch individual pitcher season stats
python scripts/ingest_pitcher_stats.py --seasons $(seq 2000 2025)

# 5. Fetch FanGraphs team advanced metrics
python scripts/ingest_fangraphs.py --seasons $(seq 2002 2025)
```

### Build features

```bash
# Historical seasons (2000–2025)
python scripts/build_features.py --seasons $(seq 2000 2025)

# Spring training features (schedule scores + prior-season team state)
python scripts/build_spring_features.py --seasons $(seq 2000 2026)

# 2026 pre-season predictions (uses 2025 end-of-season team strength)
python scripts/build_features_2026.py
```

### Ingest external data (optional — Vegas odds and weather)

```bash
# Vegas odds (requires a CSV of historical money lines)
python scripts/ingest_vegas.py --input odds.csv

# Weather data (backfills from Open-Meteo API based on gamelogs)
python scripts/ingest_weather.py
```

### Train models (with Optuna HPO)

```bash
python scripts/train_model.py --hpo --hpo-trials 60
```

Skip HPO if you just want to re-train with existing hyperparameters:

```bash
# Train all 6 models: logistic, lightgbm, xgboost, catboost, mlp, stacked
# --spring-weight 0.25 (default) down-weights spring training games
python scripts/train_model.py

# Train a subset
python scripts/train_model.py --models logistic xgboost stacked
```

### Launch the web dashboard

```bash
python scripts/serve.py                   # default: stacked ensemble, http://localhost:30087
python scripts/serve.py --model xgboost   # use XGBoost model
python scripts/serve.py --model catboost  # use CatBoost model
python scripts/serve.py --model mlp       # use MLP (neural network) model
```

Open:

- `http://localhost:30087` — all-seasons games browser
- `http://localhost:30087/season/2026` — 2026 schedule and predictions
- `http://localhost:30087/standings` — predicted vs actual standings with team stats
- `http://localhost:30087/dashboard` — admin dashboard (update season, full reingest, retrain, system status)
- `http://localhost:30087/sitemap` — complete page and API index

### CLI query tool

```bash
# Game detail with SHAP attribution
python scripts/query_game.py --game-pk 745444

# Dodgers vs. Padres on opening day 2024
python scripts/query_game.py --home SDP --away LAD --season 2024 --date 2024-03-20

# All 2024 Dodgers home games (compact)
python scripts/query_game.py --home LAD --season 2024 --show-schedule

# Biggest upsets of 2024
python scripts/query_game.py --season 2024 --show-upsets --top-n 10

# Brief one-line output
python scripts/query_game.py --home NYY --season 2025 --brief
```

---

## Server management

### Start in foreground (development)

```bash
python scripts/serve.py                   # stacked ensemble (default)
python scripts/serve.py --model xgboost   # explicit model selection
```

### Start in background (production)

```bash
mkdir -p logs
nohup python scripts/serve.py >> logs/server.log 2>&1 &
echo $! > server.pid
```

The server PID is saved to `server.pid` so it can be stopped cleanly later.

### Stop the server

```bash
# Graceful stop using saved PID
kill $(cat server.pid)

# Force stop using saved PID (if graceful stop hangs)
kill -9 $(cat server.pid)

# Stop by port number (no PID file needed)
kill $(lsof -ti:30087)

# Force stop by port number
kill -9 $(lsof -ti:30087)

# Stop all uvicorn/serve.py processes
pkill -f "serve.py"
```

### Restart the server

```bash
kill $(lsof -ti:30087) 2>/dev/null; sleep 2
nohup python scripts/serve.py >> logs/server.log 2>&1 &
echo $! > server.pid
```

### Check server status

```bash
# Is the server running?
lsof -i:30087

# Tail the server log
tail -f logs/server.log

# Check the PID file
cat server.pid && kill -0 $(cat server.pid) && echo "running" || echo "not running"
```

---

## Daily update (cron job)

The `scripts/update_daily.sh` script refreshes game results, rebuilds features, and restarts the server. It is designed to run at 01:00 each night after Retrosheet publishes the previous day's results.

### What the script does

| Step | Action |
| --- | --- |
| 1 | Refresh the current-season MLB schedule (picks up postponements and rescheduled games) |
| 2 | Refresh the current-season Retrosheet gamelogs (yesterday's results) |
| 3 | Rebuild the Retrosheet ↔ MLB crosswalk for the current season |
| 4 | Rebuild the 119-feature matrix for the current season (incl. Statcast, Vegas, weather) |
| 5 | Build spring training features for the current season |
| 6 | Rebuild 2026 pre-season predictions from the updated team state |
| 7 | Kill the running server and start a fresh instance to load the new data |

All output is appended to `logs/cron.log`; the server log goes to `logs/server.log`.

### One-time setup

```bash
# 1. Make the script executable
chmod +x scripts/update_daily.sh

# 2. Create the logs directory
mkdir -p logs

# 3. Test it manually first
scripts/update_daily.sh
```

### Install the cron job

```bash
crontab -e
```

Add this line (replace the path with your actual project root):

```cron
0 1 * * * /path/to/mlb-predict/scripts/update_daily.sh >> /path/to/mlb-predict/logs/cron.log 2>&1
```

The format is `minute hour day month weekday command`:

| Field | Value | Meaning |
| --- | --- | --- |
| `0` | minute | at the top of the hour |
| `1` | hour | 1 AM local time |
| `*` | day | every day |
| `*` | month | every month |
| `*` | weekday | every day of the week |

### Verify the cron job is registered

```bash
crontab -l
```

### Remove the cron job

```bash
crontab -e
# Delete the update_daily.sh line, save and exit
```

### Override environment variables

The script respects the following environment variables, which can be set inline:

```bash
# Use a different Python or model
PYTHON=/usr/local/bin/python3 MODEL=stacked scripts/update_daily.sh

# Run for a specific season only (useful for backfilling)
# Edit update_daily.sh YEAR variable or export:
YEAR=2025 scripts/update_daily.sh
```

---

## Docker

The entire workflow — data ingestion, model training, web server, and scheduled re-runs — can be run as a single Docker container with the data volume mounted on the host.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) ≥ 24
- [Docker Compose](https://docs.docker.com/compose/install/) v2 (bundled with Docker Desktop)
- **Minimum 4 GB RAM allocated to the container** (8 GB recommended for training). The ML stack — pandas, LightGBM, XGBoost, CatBoost, scikit-learn MLP, SHAP — requires ~2 GB at startup; training all 6 models concurrently can peak higher.
- Supported platforms: `linux/amd64` and `linux/arm64` (Synology/QNAP NAS, AWS Graviton, Oracle Ampere, Apple Silicon via Rosetta)

### Quick start

```bash
# 1. Build the image and start the container
docker compose up --build

# 2. Open the dashboard (once the bootstrap is complete)
open http://localhost:30087
```

> **First-run notice**: On a cold start (no `data/` directory on the host) the container runs the full bootstrap pipeline — ingesting 25+ years of historical data and training all 6 models. **This can take several hours.** Subsequent starts are fast because the data volume persists on the host.

### Detached / daemon mode

```bash
docker compose up --build -d        # start in background
docker compose logs -f mlb-predict      # follow all logs
docker compose logs -f mlb-predict | grep '\[server\]'   # server logs only
```

### Stop / restart

```bash
docker compose down                  # stop and remove container (data is preserved)
docker compose restart mlb-predict       # restart without rebuilding
docker compose up -d                 # start again
```

### Environment overrides

| Variable | Default   | Description |
| -------- | --------- | ----------- |
| `MODEL`  | `stacked` | Model served: `logistic \| lightgbm \| xgboost \| catboost \| mlp \| stacked` |
| `PORT`   | `30087`   | Host port the dashboard is exposed on |
| `MEM_LIMIT` | `1536m` | Container memory limit (use `2g` for training/bootstrap) |
| `CPUS`   | `2`      | Container CPU limit |
| `MLB_PREDICT_LIVE_API` | `1` | Set to `0` to disable live MLB Stats API and Odds API calls at runtime (minimize network) |

```bash
# Serve on port 9000 with the XGBoost model
PORT=9000 MODEL=xgboost docker compose up -d
```

### Minimize network footprint

To reduce outbound traffic when the app is running:

1. **Disable live API calls at runtime** — set `MLB_PREDICT_LIVE_API=0`. The dashboard will serve predicted standings and game lists from precomputed data only; play-by-play, live standings, league leaders, team stats, and odds refresh will not call external APIs. Standings pages show predicted standings without live actuals; other live endpoints return a “live data disabled” message.
2. **Reduce scheduled network use** — the container runs cron at 01:00 UTC (ingest) and 20:00 UTC (retrain). To avoid that traffic, you can comment out the two entries in `docker/crontab` and rebuild the image, or run the container without supercronic (custom entrypoint). Data will then only change when you run ingest/train manually or via the admin pipeline.

### Live Odds (optional)

To enable live game and futures odds from [The Odds API](https://the-odds-api.com), set an API key in one of these ways:

- **Environment:** `ODDS_API_KEY=your-key`
- **Admin dashboard:** Dashboard → “Live Odds API Key” → enter key → Save (writes `data/processed/odds/config.json`)
- **Config file:** Create `data/processed/odds/config.json` with `{"api_key": "your-key"}` (the `data/` directory is git-ignored)

Without a key, the app runs normally; odds features are simply unavailable. See the [Wiki → Data Sources → Live Odds](http://localhost:30087/wiki#ds-live-odds) for full instructions.

### Force a full re-bootstrap

Deleting the model artifacts directory causes the entrypoint to re-run the complete pipeline on the next start:

```bash
docker compose down
rm -rf data/models/          # remove all trained model artifacts
docker compose up -d         # triggers full re-ingest + re-train
```

### Scheduled jobs (inside the container)

The container runs two cron jobs via `supercronic`:

| Schedule   | Script                      | What it does |
| ---------- | --------------------------- | ------------ |
| 01:00 UTC  | `docker/ingest_daily.sh`    | Refresh current-season schedule and gamelogs (incl. spring training), rebuild 119-feature matrix and spring features, restart server |
| 20:00 UTC  | `docker/retrain_daily.sh`   | Retrain all 6 models on fresh data, restart server |

Logs are written to `./logs/ingest_daily.log` and `./logs/retrain_daily.log` on the host.

### Inspect or restart processes inside the container

```bash
docker exec -it mlb-predict supervisorctl status
docker exec -it mlb-predict supervisorctl restart mlb-predict-server
docker exec -it mlb-predict supervisorctl tail -f mlb-predict-server
```

### Data volume layout

Both `./data` and `./logs` on the host are bind-mounted into `/app/data` and `/app/logs` inside the container.

```
./data/          ←→  /app/data     (raw + processed data, trained models)
./logs/          ←→  /app/logs     (server, cron, bootstrap, supervisord logs)
```

All data is accessible on the host machine at all times. The container itself is stateless — removing and recreating it leaves all data intact.

### Build the image without Compose

```bash
docker build -t mlb-predict .
docker run -p 30087:30087 \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/logs:/app/logs" \
    -e MODEL=stacked \
    mlb-predict
```

### GitHub Container Registry (GHCR)

The CI pipeline automatically builds and publishes the production image to GHCR on every push to `main` and on version tags:

```bash
# Pull the latest image from GHCR
docker pull ghcr.io/sv4u/mlb-predict:main

# Run directly from GHCR (no local build needed)
docker run -p 30087:30087 \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/logs:/app/logs" \
    ghcr.io/sv4u/mlb-predict:main

# Or use the image-only Compose file (same env/volumes as docker-compose.yml, no build)
docker compose -f docker-compose.image.yml pull
docker compose -f docker-compose.image.yml up -d

| Git event | Image tag(s) published |
|---|---|
| Push to `main` | `:main`, `:sha-<short>` |
| Tag `v1.2.3` | `:1.2.3`, `:1.2`, `:sha-<short>` |
| Pull request | Image is built but **not** pushed |

### Docker image stages

The `Dockerfile` uses a multi-stage build:

| Stage | Built by | Platforms | Contents |
|---|---|---|---|
| `base` | both | amd64 + arm64 | System deps, supercronic (arch-aware), editable Python package |
| `test` | CI only | amd64 only | `base` + dev deps (`ruff`, `mypy`, `pytest`) + `tests/` |
| `production` | CI + local | amd64 + arm64 | `base` + `scripts/`, `docker/` helpers, entrypoint |

`supercronic` is downloaded for the correct architecture at build time using Docker BuildKit's `TARGETARCH` built-in — no manual configuration needed.

To build only the test stage locally:

```bash
docker build --target test -t mlb-predict:test .
docker run --rm --entrypoint python mlb-predict:test -m pytest tests/ -v
```

---

## Data pipeline

```
MLB Stats API     Retrosheet gamelogs   FanGraphs      Statcast (pybaseball)
      │                  │                  │                  │
      ▼                  ▼                  ▼                  ▼
 schedule/          retrosheet/        fangraphs/        statcast_player/
 games_YYYY         gamelogs_YYYY      fangraphs_YYYY    batter/pitcher stats
      │                  │
      └──── crosswalk ───┘
             game_id_map_YYYY

 Open-Meteo API      Vegas odds CSV
       │                   │
       ▼                   ▼
   weather/             vegas/
 by_park_date         vegas_YYYY

                    │
                    ▼
              features/
     features_YYYY.parquet      ←── 119 features per game (build_features.py)
     features_spring_YYYY.parquet ←── spring training (build_spring_features.py)
     features_2026.parquet      ←── pre-season 2026 (from build_features_2026.py)
                    │
                    ▼
               models/
  logistic_v3_train2026/    lightgbm_v3_train2026/
  xgboost_v3_train2026/     catboost_v3_train2026/
  stacked_v3_train2026/     cv_summary_v3.json
```

---

## Data locations

| Path                               | Contents                                                   |
| ---------------------------------- | ---------------------------------------------------------- |
| `data/raw/mlb_api/schedule/`       | Raw MLB Stats API JSON responses (schedule endpoint)       |
| `data/raw/mlb_api/stats/`          | Raw MLB Stats API JSON responses (pitcher stats endpoint)  |
| `data/raw/mlb_api/teams/`          | Raw MLB Stats API JSON responses (teams endpoint)          |
| `data/raw/retrosheet/gamelogs/`    | Raw Retrosheet GL text files (`GL<YYYY>.TXT`)              |
| `data/processed/schedule/`         | `games_YYYY.parquet` + CSV + checksums                     |
| `data/processed/retrosheet/`       | `gamelogs_YYYY.parquet` + CSV + checksums                  |
| `data/processed/crosswalk/`        | `game_id_map_YYYY.parquet`, coverage report, failed lists  |
| `data/processed/teams/`            | `teams_YYYY.parquet` (MLB team roster metadata)            |
| `data/processed/pitcher_stats/`    | `pitchers_YYYY.parquet` (MLB API individual pitcher stats) |
| `data/processed/fangraphs/`        | `fangraphs_YYYY.parquet` (FanGraphs team advanced metrics) |
| `data/processed/statcast_player/`  | Statcast individual batter and pitcher stats (via pybaseball) |
| `data/processed/vegas/`            | `vegas_YYYY.parquet` (implied probabilities from money lines) |
| `data/processed/weather/`          | `by_park_date.parquet` (historical temp, wind, humidity per game) |
| `data/processed/features/`         | `features_YYYY.parquet` (119-feature matrix per season), `features_spring_YYYY.parquet` (spring training) |
| `data/models/`                     | Trained model artifacts + HPO results + CV summaries       |
| `data/processed/predictions/`      | Immutable prediction snapshots (Parquet, by season)        |
| `data/processed/drift/`            | Drift monitoring logs (`run_metrics_YYYY.parquet`, global) |
| `logs/server.log`                  | Web server stdout/stderr                                   |
| `logs/cron.log`                    | Daily cron job output                                      |
| `server.pid`                       | PID of the running server process                          |

---

## Querying predictions (Python)

```python
import pandas as pd
from pathlib import Path

# Load all predictions (features_*.parquet + features_spring_*.parquet)
frames = [pd.read_parquet(f) for f in sorted(Path("data/processed/features").glob("features_*.parquet"))]
df = pd.concat(frames, ignore_index=True)

# Backward compat: ensure is_spring exists (0.0 for older feature files)
if "is_spring" not in df.columns:
    df["is_spring"] = 0.0

from mlb_predict.model.artifacts import latest_artifact, load_model
from mlb_predict.model.train import _predict_proba

model, meta = load_model(latest_artifact("logistic", version="v3"))
df["prob"] = _predict_proba(model, df[meta.feature_cols].fillna(0.5))

# 2024 games with high home-team probability
df24 = df[df["season"] == 2024].sort_values("prob", ascending=False)
print(df24[["date","home_retro","away_retro","prob","home_win"]].head(10))

# 2026 pre-season predictions
df26 = df[df["season"] == 2026].sort_values("date")
print(df26[["date","home_retro","away_retro","prob"]].head(10))

# Accuracy by favourite probability bucket (historical seasons only)
dfh = df[df["home_win"].notna()]
dfh["fav_won"] = ((dfh["prob"] >= 0.5) == (dfh["home_win"] == 1)).astype(float)
print(dfh.groupby(pd.cut(dfh["prob"].clip(0.5, 0.99), 5))["fav_won"].mean())
```

---

## Web dashboard

Start the dashboard with `python scripts/serve.py`, then open `http://localhost:30087`.

### Pages

| URL | Description |
| --- | --- |
| `http://localhost:30087/` | All-seasons games browser (2000–2026) |
| `http://localhost:30087/season/2026` | 2026 schedule, pre-season predictions, standings summary, and Elo power rankings |
| `http://localhost:30087/standings` | Predicted vs actual divisional standings, league leaders, team batting and pitching stats |
| `http://localhost:30087/game/{game_pk}` | Individual game detail with SHAP feature attribution and embedded EV calculator |
| `http://localhost:30087/tools/ev-calculator` | Expected value calculator for sports bets (American, decimal, fractional odds; edge, ROI, Kelly criterion) |
| `http://localhost:30087/wiki` | Technical wiki: models, data sources, features, training pipeline |
| `http://localhost:30087/dashboard` | Admin dashboard: update season, full reingest, retrain models, system status |
| `http://localhost:30087/sitemap` | Complete index of all pages and API endpoints |
| `http://localhost:30087/sitemap.xml` | XML sitemap for search engine crawlers |

### Features

- **Games browser** — filter by season, home team, away team, or date; paginated; links to game detail
- **2026 season page** — full 2,430-game schedule with pre-season win probabilities, countdown, favourite/toss-up badges, a sticky Elo power rankings sidebar, and a standings summary with predicted league leaders and divisional standings
- **Standings page** — predicted vs actual divisional standings for all 6 divisions, predicted and actual league leaders (AL/NL), and tabbed team batting and pitching statistics fetched live from the MLB Stats API
- **Game detail** — probability bars, SHAP factor attribution chart, key stats comparison
- **Biggest upsets** — all-time or by season, filterable by home/away team and minimum favourite probability
- **CV accuracy chart** — out-of-sample accuracy trend across all 6 model types
- **Models explained** — collapsible cards describing each model with live Brier/Accuracy from CV data
- **Technical wiki** — comprehensive documentation of all models, baseball statistics, data sources, feature engineering, training pipeline, calibration, evaluation metrics, prediction snapshots, drift monitoring, error handling, and system architecture
- **Admin dashboard** — three pipeline controls: "Update Season" (non-destructive current-year refresh), "Full Reingest" (clears all processed data and re-ingests every season from scratch), and "Retrain Models" (clears all model artifacts and retrains from scratch). Destructive actions require confirmation. All pipelines run async with real-time log streaming, status badges, trained model inventory, CV performance table, and data coverage stats. Pipelines auto-reload the server on completion.
- **EV Calculator** — expected value calculator for sports bets. Supports American, decimal, and fractional odds with real-time computation of EV, implied probability, edge, ROI, break-even probability, and Kelly criterion with adjustable fraction slider. Available as a standalone page (`/tools/ev-calculator`) and as an embedded widget on each game detail page with auto-populated model probabilities and home/away team toggle
- **Sitemap** — complete index of all pages and API endpoints with descriptions; also available as XML (`/sitemap.xml`) for search engine crawlers

### API endpoints

| Endpoint                                          | Description                                      |
| ------------------------------------------------- | ------------------------------------------------ |
| `GET /api/version`                                | Application version and git commit hash          |
| `GET /api/seasons`                                | List available seasons                           |
| `GET /api/teams`                                  | List all teams (Retrosheet codes + names)        |
| `GET /api/games?season=&home=&away=&date=`        | Paginated game list with predictions             |
| `GET /api/games/{game_pk}`                        | Full detail + SHAP attribution for one game      |
| `GET /api/upsets?season=&home=&away=&min_prob=`   | Biggest upsets, filterable by team               |
| `GET /api/cv-summary`                             | Model CV results by season                       |
| `GET /api/standings?season=`                      | Predicted vs actual standings by division with league leaders |
| `GET /api/team-stats?season=`                     | Team batting and pitching statistics from MLB Stats API |
| `GET /api/admin/status`                           | Full system status (data, models, pipelines)     |
| `POST /api/admin/ingest`                          | Full re-ingestion: clear all data + re-ingest all seasons (async) |
| `POST /api/admin/update`                          | Update current season only — non-destructive (async) |
| `POST /api/admin/retrain`                         | Clear all models + retrain from scratch (async)  |

---

## Project structure

```
mlb-predict/
├── src/mlb_predict/
│   ├── mlbapi/          # MLB Stats API client (async, rate-limited)
│   │   ├── client.py
│   │   ├── schedule.py
│   │   ├── teams.py
│   │   ├── pitcher_stats.py
│   │   └── standings.py     # Live standings, team batting/pitching stats
│   ├── statcast/        # Statcast / FanGraphs advanced metrics
│   │   ├── fangraphs.py     # FanGraphs team-level stats (via pybaseball)
│   │   └── player_stats.py  # Statcast individual batter/pitcher stats + ID mapping
│   ├── external/        # External data sources
│   │   ├── vegas.py         # Money-line → implied probability conversion
│   │   └── weather.py       # Open-Meteo historical weather API client + cache
│   ├── features/        # Feature engineering pipeline
│   │   ├── elo.py           # Sequential Elo rating
│   │   ├── team_stats.py    # Multi-window (7/14/15/30/60), EWMA, home/away splits
│   │   ├── pitcher_stats.py # Gamelog-based pitcher ERA
│   │   ├── park_factors.py
│   │   ├── bullpen.py       # Bullpen usage and ERA proxy features
│   │   ├── lineup.py        # Lineup continuity features
│   │   └── builder.py       # Assembles 119-feature matrix
│   ├── model/           # Model training and evaluation
│   │   ├── train.py     # LR + LightGBM + XGBoost + CatBoost + MLP + stacked
│   │   ├── evaluate.py
│   │   └── artifacts.py # Save / load model artifacts
│   ├── predict/         # Prediction snapshots
│   │   └── snapshot.py
│   ├── drift/           # Drift monitoring
│   │   └── compute.py
│   ├── standings.py     # Division mappings, predicted standings, merge logic
│   ├── errors.py        # Structured error taxonomy (WinProbError hierarchy)
│   └── app/             # FastAPI web dashboard
│       ├── main.py          # Routes and API endpoints
│       ├── data_cache.py    # In-memory feature and model cache
│       ├── admin.py         # Background pipeline runner + system status
│       └── templates/
│           ├── index.html        # All-seasons games browser
│           ├── game.html         # Individual game detail + SHAP
│           ├── season_2026.html  # 2026 season schedule + predictions + standings summary
│           ├── standings.html    # Predicted vs actual standings + team stats
│           ├── wiki.html         # Technical wiki (models, data, training)
│           ├── dashboard.html    # Admin dashboard (update, reingest, retrain, status)
│           └── sitemap.html      # Complete page and API endpoint index
├── scripts/
│   ├── ingest_schedule.py              # MLB Stats API schedule ingestion
│   ├── ingest_retrosheet_gamelogs.py   # Retrosheet game log ingestion
│   ├── build_crosswalk.py              # Retrosheet ↔ MLB ID crosswalk
│   ├── ingest_pitcher_stats.py         # MLB Stats API individual pitcher stats
│   ├── ingest_fangraphs.py             # FanGraphs team advanced metrics
│   ├── ingest_vegas.py                 # Vegas money-line odds → implied probabilities
│   ├── ingest_weather.py               # Open-Meteo historical weather backfill
│   ├── ingest_all.py                   # Orchestrate all ingestion steps
│   ├── build_features.py               # Build 119-feature matrices (historical)
│   ├── build_spring_features.py        # Build spring training feature matrices
│   ├── build_features_2026.py          # Build 2026 pre-season feature matrix
│   ├── train_model.py                  # Optuna HPO + expanding-window CV + 6 production models
│   ├── feature_importance.py           # SHAP-based feature importance analysis
│   ├── run_predictions.py              # Snapshot predictions
│   ├── compute_drift.py                # Drift monitoring
│   ├── query_game.py                   # Human-centric CLI query tool
│   ├── serve.py                        # Launch FastAPI dashboard
│   └── update_daily.sh                 # Daily cron: refresh data + restart server (host)
├── docker/
│   ├── entrypoint.sh                   # Container startup: bootstrap check + supervisord
│   ├── supervisord.conf                # Process manager config (server + cron)
│   ├── crontab                         # supercronic schedule (1am ingest, 11pm retrain)
│   ├── ingest_daily.sh                 # Daily 1am data refresh
│   └── retrain_daily.sh                # Daily 11pm model retrain (all 6 models)
├── Dockerfile                          # Multi-stage image (base → test → production)
├── docker-compose.yml                  # Compose config (volumes, ports, env vars)
├── .dockerignore                       # Excludes data/, .git/, .venv/, caches from build context
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── schedule/
│   │   ├── retrosheet/
│   │   ├── crosswalk/
│   │   ├── pitcher_stats/
│   │   ├── fangraphs/
│   │   ├── statcast_player/    # Statcast batter/pitcher individual stats
│   │   ├── vegas/              # Implied probabilities from money lines
│   │   ├── weather/            # Open-Meteo historical weather cache
│   │   ├── features/        # features_YYYY.parquet, features_spring_YYYY.parquet
│   │   ├── predictions/
│   │   └── drift/
│   └── models/
├── logs/
│   ├── server.log          # Web server output
│   ├── cron.log            # Daily cron output (host-based cron)
│   ├── ingest_daily.log    # Docker daily ingest output
│   ├── retrain_daily.log   # Docker daily retrain output
│   ├── bootstrap.log       # Docker first-run bootstrap output
│   └── supervisord.log     # Docker process manager output
├── server.pid       # PID of the running server (host-based only)
├── pyproject.toml
└── README.md
```

---

## Changelog (v3.1 — Code Review Fixes)

Fixes from the comprehensive code review (22 new tests added, 228 total passing):

### Critical

- **team_stats.py**: `away_win_pct_away_only` now correctly computes away-only rolling stats (was using home-only stats for the away side)
- **lineup.py**: Lineup continuity now tracks across home/away venue transitions (was fragmenting tracking by venue)
- **train.py**: Stacked meta-learner uses disjoint calibration split — first half calibrates base models, second half trains the meta-learner (fixes data leakage)

### High

- **train.py**: Production model now computes real `eval_brier` instead of hardcoded `0.0`; metadata field renamed from `train_brier` to `eval_brier` (legacy auto-migration on load)
- **compute.py**: Global drift dedup now includes `model_version`; baseline metrics are now persisted; empty diffs produce zero metrics instead of NaN
- **client.py**: Non-retryable 4xx errors (400, 401, 403) raise immediately instead of retrying; TokenBucket rejects requests exceeding capacity
- **bullpen.py**: Bullpen fatigue now tracks across home and away games (was fragmenting by venue); ER fill value corrected to game-level scale (2.5, not ERA-scale 4.5)
- **weather.py**: Game-hour estimation now uses longitude-based timezone offset instead of hardcoded UTC hour 19
- **data_cache.py**: Added threading lock for thread-safe cache reads/writes during hot reloads

### Medium

- **standings.py**: Null rank values no longer crash `int()` conversion
- **teams.py**: NaN abbreviations are filtered out; empty DataFrames handled gracefully
- **builder.py**: Feature hash handles `inf`/`-inf` values; crosswalk join drop count is logged
- **snapshot.py**: Collision detection prevents overwriting immutable snapshots; deduplicates hashing with `util/hashing.py`
- **errors.py**: `APIError` now defined and exported; `MLBAPIError` inherits from it
- **artifacts.py**: Legacy `train_brier` metadata migrated to `eval_brier` on load
- **run_predictions.py**: Uses model's trained `feature_cols` from metadata (not the global `FEATURE_COLS`); adds `catboost` and `mlp` to `--model-type` choices

### Low

- **hashing.py**: Uses POSIX paths for cross-platform determinism; added docstrings
- **ingest_all.py**: Replaced deprecated `datetime.utcnow()` and `get_event_loop()`; uses `sys.executable` for subprocess calls

---

## Attribution

Game log data from **Retrosheet** (retrosheet.org).

> The information used here was obtained free of charge from and is copyrighted by Retrosheet.
> Interested parties may contact Retrosheet at 20 Sunset Rd., Newark, DE 19711.

Advanced metrics from **FanGraphs** (fangraphs.com) via the `pybaseball` library.
Statcast individual player data from **Baseball Savant** (baseballsavant.mlb.com) via `pybaseball`.
Schedule and player data from the **MLB Stats API** (statsapi.mlb.com).
Historical weather data from the **Open-Meteo API** (open-meteo.com).
Player ID mapping via the **Chadwick Baseball Bureau** register.
