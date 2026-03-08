#!/usr/bin/env bash
# =============================================================================
# update_daily.sh — MLB Win Probability daily data refresh
#
# Refreshes yesterday's game results, rebuilds features, and restarts the
# web server.  Designed to be run from cron at 01:00 each day during the
# MLB regular season (roughly March–September).
#
# Setup
# -----
# 1. Make executable:
#       chmod +x scripts/update_daily.sh
#
# 2. Install the cron job (runs at 01:00 daily):
#       crontab -e
#    Add the following line (adjust REPO path):
#       0 1 * * * /Users/sasank.vishnubhatla/Documents/personal-dev/mlb-winprob/scripts/update_daily.sh >> /Users/sasank.vishnubhatla/Documents/personal-dev/mlb-winprob/logs/cron.log 2>&1
#
# 3. Verify cron is registered:
#       crontab -l
#
# Environment
# -----------
# REPO   — absolute path to the project root (default: directory of this script)
# PYTHON — python3 binary to use (default: pyenv version in .python-version)
# PORT   — dashboard port (default: 30087)
# MODEL  — model type: logistic | lightgbm | xgboost (default: xgboost)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override with env vars)
# ---------------------------------------------------------------------------
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-/Users/sasank.vishnubhatla/.pyenv/versions/3.11.14/bin/python3}"
PORT="${PORT:-30087}"
MODEL="${MODEL:-stacked}"
LOG_DIR="$REPO/logs"
PID_FILE="$REPO/server.pid"
YEAR=$(date +%Y)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

die() { log "ERROR: $*"; exit 1; }

run() {
    local desc="$1"; shift
    log "→ $desc"
    "$@" || die "'$desc' failed (exit $?)"
    log "  ✓ $desc"
}

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
cd "$REPO" || die "Cannot cd to $REPO"
mkdir -p "$LOG_DIR"

[[ -x "$PYTHON" ]] || die "Python not found at $PYTHON. Set PYTHON env var."

log "============================================================"
log "Daily update — season $YEAR"
log "  REPO   = $REPO"
log "  PYTHON = $PYTHON"
log "  MODEL  = $MODEL"
log "  PORT   = $PORT"
log "============================================================"

# ---------------------------------------------------------------------------
# 1. Refresh current-season schedule
#    (picks up any rescheduled or added games)
# ---------------------------------------------------------------------------
run "Ingest $YEAR schedule (regular + spring training)" \
    "$PYTHON" scripts/ingest_schedule.py --seasons "$YEAR" --refresh-mlbapi

# ---------------------------------------------------------------------------
# 2. Refresh current-season Retrosheet gamelogs
#    (Retrosheet typically publishes the previous day's results by ~midnight)
# ---------------------------------------------------------------------------
run "Ingest $YEAR Retrosheet gamelogs" \
    "$PYTHON" scripts/ingest_retrosheet_gamelogs.py --seasons "$YEAR" --refresh

# ---------------------------------------------------------------------------
# 3. Rebuild crosswalk for current season
# ---------------------------------------------------------------------------
run "Build $YEAR crosswalk" \
    "$PYTHON" scripts/build_crosswalk.py --seasons "$YEAR"

# ---------------------------------------------------------------------------
# 4. Rebuild feature matrix for current season
# ---------------------------------------------------------------------------
run "Build $YEAR features" \
    "$PYTHON" scripts/build_features.py --seasons "$YEAR"

# ---------------------------------------------------------------------------
# 5. Rebuild spring training features for current season
# ---------------------------------------------------------------------------
run "Build $YEAR spring training features" \
    "$PYTHON" scripts/build_spring_features.py --seasons "$YEAR"

# ---------------------------------------------------------------------------
# 6. Rebuild 2026 pre-season features
#    During the 2026 season this updates predictions game-by-game using the
#    end-of-last-game team state; before the season it uses 2025 end-of-year.
# ---------------------------------------------------------------------------
if [ "$YEAR" = "2026" ] && [ -f scripts/build_features_2026.py ]; then
    run "Build 2026 pre-season features" \
        "$PYTHON" scripts/build_features_2026.py
fi

# ---------------------------------------------------------------------------
# 7. Restart the web server
#    Kills the existing process (if any) then starts a new one in the
#    background, redirecting output to logs/server.log.
# ---------------------------------------------------------------------------
log "→ Restarting web server on port $PORT"

# Kill existing server
EXISTING=$(lsof -ti:"$PORT" 2>/dev/null || true)
if [[ -n "$EXISTING" ]]; then
    log "  Stopping PID(s): $EXISTING"
    kill "$EXISTING" 2>/dev/null || true
    sleep 3
fi

# Also check the PID file
if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE")
    kill "$OLD_PID" 2>/dev/null || true
    rm -f "$PID_FILE"
fi

# Start fresh
nohup "$PYTHON" scripts/serve.py \
    --model "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    >> "$LOG_DIR/server.log" 2>&1 &

SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

# Wait for the server to bind to the port before declaring success
sleep 5
if ! lsof -ti:"$PORT" > /dev/null 2>&1; then
    die "Server failed to start on port $PORT — check $LOG_DIR/server.log"
fi
log "  ✓ Server started (PID $SERVER_PID) — http://127.0.0.1:$PORT"

log "============================================================"
log "Daily update complete"
log "============================================================"
