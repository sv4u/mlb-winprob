#!/usr/bin/env bash
# =============================================================================
# retrain_daily.sh — daily model retrain (runs at 23:00 UTC via supercronic)
#
# Retrains the full production model stack on the freshest available data,
# then restarts the web server to load the new artifacts.
#
# The server continues serving the previous model during training; it is only
# restarted (and therefore briefly unavailable) after training completes.
# =============================================================================

set -euo pipefail

cd /app

MODEL="${MODEL:-stacked}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [retrain-daily] $*"; }
die() { log "ERROR: $*"; exit 1; }

log "========================================================"
log "  Daily retrain — $(date -u '+%Y-%m-%d %H:%M UTC')"
log "  Model: $MODEL"
log "========================================================"

log "→ Training all production models..."
python scripts/train_model.py \
    || die "Model training failed — server will continue with previous artifacts"
log "  ✓ Training complete"

log "→ Restarting web server to load new model artifacts..."
supervisorctl -c /app/docker/supervisord.conf restart winprob-server \
    || die "supervisorctl restart failed"
log "  ✓ Web server restarted with new model"

log "========================================================"
log "  Daily retrain complete"
log "========================================================"
