#!/usr/bin/env bash
# =============================================================================
# entrypoint.sh — container startup script
#
# On every container start:
#   1. Create required runtime directories on the mounted volume.
#   2. Start supervisord immediately, which launches:
#        - mlb-predict-server  (uvicorn FastAPI dashboard, port 30087)
#        - cron            (supercronic executing docker/crontab)
#
# If no trained model artifacts exist, the FastAPI app auto-bootstraps in the
# background (ingest + train) while serving a real-time progress UI at "/".
# Visit http://localhost:PORT/ to monitor bootstrap progress.
#
# Skip the auto-bootstrap by pre-populating ./data on the host before the
# first `docker compose up`.  Delete ./data/models/ to force a re-bootstrap.
# =============================================================================

set -euo pipefail

cd /app

export MODEL="${MODEL:-stacked}"
export PORT="${PORT:-30087}"
export HOME="${HOME:-/root}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
mkdir -p "$HOME" "$MPLCONFIGDIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [entrypoint] $*"; }

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
    data/processed/player \
    data/processed/vegas \
    data/processed/weather \
    data/models \
    logs

# ---------------------------------------------------------------------------
# Seed the Retrosheet↔MLB team ID crosswalk if missing (first run on a fresh
# volume).  Every ingest run depends on this file (scripts/build_crosswalk.py)
# and it is deliberately preserved across destructive re-ingests, so it must
# exist before the first ingest ever runs.  Never overwrites an existing file.
# ---------------------------------------------------------------------------
if [ ! -f data/processed/team_id_map_retro_to_mlb.csv ] && [ -f docs/team_id_map_template.csv ]; then
    mkdir -p data/processed
    cp docs/team_id_map_template.csv data/processed/team_id_map_retro_to_mlb.csv
    log "Seeded data/processed/team_id_map_retro_to_mlb.csv from docs/team_id_map_template.csv"
fi

# ---------------------------------------------------------------------------
# Status check (informational only — no blocking bootstrap)
# ---------------------------------------------------------------------------
if ls data/models/stacked_v*_train*/model.joblib 2>/dev/null | grep -q .; then
    log "Existing model artifacts found — app will be ready immediately."
else
    log "No trained models found in data/models/."
    log "The web server will start now and auto-bootstrap in the background."
    log "Visit http://localhost:${PORT}/ for real-time progress."
fi

# ---------------------------------------------------------------------------
# Hand off to supervisord (server + cron)
# ---------------------------------------------------------------------------
log "Starting supervisord (server on port ${PORT}, model=${MODEL})..."
exec supervisord -c /app/docker/supervisord.conf
