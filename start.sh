#!/usr/bin/env bash
# =============================================================================
# MLB Win Probability — Local Setup & Start Script (macOS / Linux)
#
# Usage:
#   ./start.sh              # Full setup + start
#   ./start.sh --stop       # Stop running containers
#   ./start.sh --status     # Check container status
#   ./start.sh --rebuild    # Force rebuild the Docker image
#   ./start.sh --model xgboost  # Start with a specific model type
#
# Prerequisites installed by this script:
#   - Docker Desktop (checks installation)
#   - Docker Compose (bundled with Docker Desktop)
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="${MODEL:-stacked}"
PORT="${PORT:-30087}"
REBUILD=false
ACTION="start"

print_banner() {
  echo ""
  echo -e "${BLUE}${BOLD}  ⚾  MLB Win Probability${NC}"
  echo -e "${BLUE}  ─────────────────────────────────────${NC}"
  echo ""
}

log_info()  { echo -e "  ${GREEN}✓${NC} $1"; }
log_warn()  { echo -e "  ${YELLOW}!${NC} $1"; }
log_error() { echo -e "  ${RED}✗${NC} $1"; }
log_step()  { echo -e "\n  ${BOLD}$1${NC}"; }

# ── Parse arguments ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --stop)     ACTION="stop";    shift ;;
    --status)   ACTION="status";  shift ;;
    --rebuild)  REBUILD=true;     shift ;;
    --model)    MODEL="$2";       shift 2 ;;
    --port)     PORT="$2";        shift 2 ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --stop          Stop running containers"
      echo "  --status        Show container status"
      echo "  --rebuild       Force rebuild the Docker image"
      echo "  --model TYPE    Model type: logistic|lightgbm|xgboost|catboost|mlp|stacked (default: stacked)"
      echo "  --port PORT     Host port (default: 30087)"
      echo "  -h, --help      Show this help"
      exit 0
      ;;
    *) log_error "Unknown option: $1"; exit 1 ;;
  esac
done

# ── Stop action ──────────────────────────────────────────────────────────
if [[ "$ACTION" == "stop" ]]; then
  print_banner
  log_step "Stopping containers…"
  docker compose down 2>/dev/null || docker-compose down 2>/dev/null
  log_info "Containers stopped."
  exit 0
fi

# ── Status action ────────────────────────────────────────────────────────
if [[ "$ACTION" == "status" ]]; then
  print_banner
  log_step "Container status"
  docker compose ps 2>/dev/null || docker-compose ps 2>/dev/null
  echo ""
  if curl -sf "http://localhost:${PORT}/api/version" > /dev/null 2>&1; then
    VERSION=$(curl -sf "http://localhost:${PORT}/api/version")
    log_info "Server is running at http://localhost:${PORT}"
    echo "       $VERSION"
  else
    log_warn "Server is not responding on port ${PORT}."
  fi
  exit 0
fi

# ── Start action ─────────────────────────────────────────────────────────
print_banner

# Step 1: Check Docker is installed
log_step "Step 1/5 — Checking prerequisites"

if ! command -v docker &> /dev/null; then
  log_error "Docker is not installed."
  echo ""
  echo "  Please install Docker Desktop:"
  if [[ "$(uname)" == "Darwin" ]]; then
    echo "    https://docs.docker.com/desktop/install/mac-install/"
    echo ""
    echo "  Or install via Homebrew:"
    echo "    brew install --cask docker"
  else
    echo "    https://docs.docker.com/desktop/install/linux-install/"
  fi
  echo ""
  exit 1
fi
log_info "Docker is installed: $(docker --version)"

# Check Docker daemon is running
if ! docker info &> /dev/null; then
  log_error "Docker daemon is not running."
  echo ""
  if [[ "$(uname)" == "Darwin" ]]; then
    echo "  Please start Docker Desktop from your Applications folder."
  else
    echo "  Please start the Docker service:"
    echo "    sudo systemctl start docker"
  fi
  echo ""
  exit 1
fi
log_info "Docker daemon is running."

# Check docker compose
if docker compose version &> /dev/null; then
  COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
  COMPOSE_CMD="docker-compose"
else
  log_error "Docker Compose is not available."
  echo "  Docker Compose is included with Docker Desktop."
  echo "  Please update Docker Desktop to the latest version."
  exit 1
fi
log_info "Docker Compose is available: $($COMPOSE_CMD version --short 2>/dev/null || echo 'OK')"

# Step 2: Create required directories
log_step "Step 2/5 — Creating directories"

mkdir -p data/raw data/processed data/models logs
log_info "data/ and logs/ directories ready."

# Step 3: Set environment
log_step "Step 3/5 — Configuring environment"

export MODEL="$MODEL"
export PORT="$PORT"

GIT_COMMIT="unknown"
if command -v git &> /dev/null && git rev-parse --git-dir &> /dev/null 2>&1; then
  GIT_COMMIT=$(git rev-parse --short=8 HEAD 2>/dev/null || echo "unknown")
fi
export GIT_COMMIT

log_info "Model type: ${BOLD}${MODEL}${NC}"
log_info "Host port:  ${BOLD}${PORT}${NC}"
log_info "Git commit: ${GIT_COMMIT}"

# Step 4: Build / pull image
log_step "Step 4/5 — Building Docker image"

BUILD_ARGS="--build-arg GIT_COMMIT=${GIT_COMMIT}"
if [[ "$REBUILD" == true ]]; then
  log_warn "Forcing rebuild (--rebuild flag)"
  $COMPOSE_CMD build --no-cache $BUILD_ARGS
else
  $COMPOSE_CMD build $BUILD_ARGS
fi
log_info "Docker image built successfully."

# Step 5: Start containers
log_step "Step 5/5 — Starting server"

$COMPOSE_CMD up -d
echo ""
log_info "Server starting at ${BOLD}http://localhost:${PORT}${NC}"
echo ""
echo -e "  ${BLUE}Useful commands:${NC}"
echo "    $COMPOSE_CMD logs -f         # Follow server logs"
echo "    $0 --status               # Check server status"
echo "    $0 --stop                 # Stop the server"
echo "    $0 --model xgboost        # Restart with a different model"
echo ""

# Wait for health check
log_step "Waiting for server to become healthy…"
echo -e "  (This may take a few minutes on first run while data is ingested.)"
echo ""

MAX_WAIT=120
ELAPSED=0
while [[ $ELAPSED -lt $MAX_WAIT ]]; do
  if curl -sf "http://localhost:${PORT}/api/version" > /dev/null 2>&1; then
    log_info "Server is healthy and ready!"
    echo ""
    echo -e "  ${GREEN}${BOLD}Open your browser: http://localhost:${PORT}${NC}"
    echo ""
    exit 0
  fi
  sleep 5
  ELAPSED=$((ELAPSED + 5))
  printf "  Waiting… (%ds / %ds)\r" "$ELAPSED" "$MAX_WAIT"
done

echo ""
log_warn "Server hasn't responded within ${MAX_WAIT}s."
echo "  On first run, the initial data ingestion can take much longer."
echo "  Check progress with: $COMPOSE_CMD logs -f"
echo ""
echo -e "  The server will be available at ${BOLD}http://localhost:${PORT}${NC}"
echo "  once the initial setup completes."
echo ""
