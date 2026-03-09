#!/usr/bin/env bash
# =============================================================================
# ingest_daily.sh — daily data refresh (runs at 01:00 UTC via supercronic)
#
# Refreshes the current season's schedule, Retrosheet gamelogs, crosswalk,
# and feature matrices, then restarts the web server to serve fresh data.
# =============================================================================

set -euo pipefail

cd /app

YEAR=$(date +%Y)
MODEL="${MODEL:-stacked}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ingest-daily] $*"; }
die() { log "ERROR: $*"; exit 1; }

run_step() {
    local desc="$1"; shift
    log "→ $desc"
    "$@" || die "'$desc' failed"
    log "  ✓ $desc"
}

log "========================================================"
log "  Daily ingest — $YEAR — $(date -u '+%Y-%m-%d %H:%M UTC')"
log "========================================================"

run_step "Refresh $YEAR schedule — regular + spring training (MLB Stats API)" \
    python scripts/ingest_schedule.py --seasons "$YEAR" --refresh-mlbapi

run_step "Refresh $YEAR Retrosheet gamelogs" \
    python scripts/ingest_retrosheet_gamelogs.py --seasons "$YEAR" --refresh

run_step "Rebuild $YEAR crosswalk" \
    python scripts/build_crosswalk.py --seasons "$YEAR"

run_step "Rebuild $YEAR feature matrix (incl. Statcast, Vegas, weather)" \
    python scripts/build_features.py --seasons "$YEAR"

run_step "Build $YEAR spring training features" \
    python scripts/build_spring_features.py --seasons "$YEAR"

if [ "$YEAR" = "2026" ] && [ -f scripts/build_features_2026.py ]; then
    run_step "Rebuild 2026 pre-season features" \
        python scripts/build_features_2026.py
fi

# Write a timestamp marker so the dashboard can show last-ingest time
date -u '+%Y-%m-%dT%H:%M:%SZ' > /app/data/processed/.last_ingest

log "→ Restarting web server to reload updated features..."
supervisorctl -c /app/docker/supervisord.conf restart mlb-predict-server \
    || die "supervisorctl restart failed"
log "  ✓ Web server restarted"

log "========================================================"
log "  Daily ingest complete"
log "========================================================"
