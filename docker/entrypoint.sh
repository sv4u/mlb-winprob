#!/usr/bin/env bash
# =============================================================================
# entrypoint.sh — container startup script
#
# On every container start:
#   1. If no trained model artifacts exist in /app/data/models, run the full
#      bootstrap pipeline (ingest all historical data + train every model).
#      This can take several hours on a cold first run.
#   2. Start supervisord, which launches:
#        - mlb-predict-server  (uvicorn FastAPI dashboard, port 30087)
#        - cron            (supercronic executing docker/crontab)
#
# Skip the bootstrap by pre-populating ./data on the host before the first
# `docker compose up`.  Delete ./data/models/ to force a full re-bootstrap.
# =============================================================================

set -euo pipefail

cd /app

export MODEL="${MODEL:-stacked}"
export PORT="${PORT:-30087}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [entrypoint] $*"; }
die() { log "ERROR: $*"; exit 1; }

run_step() {
    local desc="$1"; shift
    log "→ $desc"
    "$@" || die "'$desc' failed"
    log "  ✓ $desc"
}

# ---------------------------------------------------------------------------
# Ensure runtime directories exist on the mounted volume
# ---------------------------------------------------------------------------
mkdir -p \
    data/raw/mlb_api/schedule \
    data/raw/mlb_api/stats \
    data/raw/retrosheet/gamelogs \
    data/processed/schedule \
    data/processed/gamelogs \
    data/processed/crosswalk \
    data/processed/features \
    data/processed/predictions \
    data/processed/fangraphs \
    data/processed/pitcher_stats \
    data/processed/statcast_player \
    data/processed/vegas \
    data/processed/weather \
    data/models \
    logs

# ---------------------------------------------------------------------------
# Bootstrap check
#
# We look for any stacked production artifact.  If one exists the volume was
# either pre-populated or a previous run succeeded, so we skip the expensive
# initial pipeline.
# ---------------------------------------------------------------------------
if ls data/models/stacked_v3_train*/model.joblib 2>/dev/null | grep -q .; then
    log "Existing model artifacts found — skipping bootstrap."
    log "  To force a full re-bootstrap: delete ./data/models/ and restart."
else
    log "========================================================"
    log "  FIRST-RUN BOOTSTRAP"
    log "  No trained model found in data/models/."
    log "  Running full ingestion and training pipeline."
    log "  This can take SEVERAL HOURS on a cold start."
    log "  Progress is logged here and to logs/bootstrap.log"
    log "========================================================"

    {
        run_step "Ingest all historical data (2000–$(date +%Y))" \
            python scripts/ingest_all.py

        YEAR=$(date +%Y)

        run_step "Build pitcher stats" \
            python scripts/ingest_pitcher_stats.py --seasons $(seq -s ' ' 2000 "$YEAR")

        run_step "Build FanGraphs team metrics" \
            python scripts/ingest_fangraphs.py

        run_step "Build historical feature matrices (incl. Statcast, Vegas, weather)" \
            python scripts/build_features.py

        run_step "Build spring training features (all seasons)" \
            python scripts/build_spring_features.py

        if [ "$YEAR" = "2026" ] && [ -f scripts/build_features_2026.py ]; then
            run_step "Build 2026 pre-season features" \
                python scripts/build_features_2026.py
        fi

        run_step "Train all models (logistic, lightgbm, xgboost, catboost, mlp, stacked)" \
            python scripts/train_model.py --models logistic lightgbm xgboost catboost mlp stacked

        log "Bootstrap complete."

    } 2>&1 | tee logs/bootstrap.log
fi

# ---------------------------------------------------------------------------
# Hand off to supervisord (server + cron)
# ---------------------------------------------------------------------------
log "Starting supervisord (server on port ${PORT}, model=${MODEL})..."
exec supervisord -c /app/docker/supervisord.conf
