# =============================================================================
# MLB Win Probability — multi-stage Dockerfile
#
# Stages
# ------
#   base        System deps + supercronic + uv + Python package install
#   test        base + dev dependencies + tests/ — used by CI to run pytest
#   production  base + scripts + docker helpers — what runs in production
#               (default build target; pushed to GHCR)
#
# Build examples
# --------------
#   docker build .                               # production image (default)
#   docker build --target test .                 # test image for CI
#   docker compose up --build                    # production via Compose
#
# Multi-platform
# --------------
# The image is published for linux/amd64 and linux/arm64.  supercronic is
# downloaded for the correct architecture automatically via TARGETARCH (a
# BuildKit built-in ARG that matches the target platform's architecture).
# =============================================================================


# =============================================================================
# Stage 1: base
# =============================================================================
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ---------------------------------------------------------------------------
# System dependencies
#   supervisor  — process manager (web server + cron side-by-side)
#   curl        — used to download supercronic and by health checks
#   libgomp1    — GCC OpenMP runtime required by LightGBM (and XGBoost on
#                 some platforms).  Not present in python:3.11-slim by default;
#                 omitting it causes "libgomp.so.1: cannot open shared object
#                 file" when train.py imports lightgbm.
# ---------------------------------------------------------------------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# supercronic — Docker-friendly cron daemon that logs to stdout/stderr
# ---------------------------------------------------------------------------
ARG TARGETARCH=amd64
ARG SUPERCRONIC_VERSION=0.2.33

RUN curl -fsSL \
    "https://github.com/aptible/supercronic/releases/download/v${SUPERCRONIC_VERSION}/supercronic-linux-${TARGETARCH}" \
    -o /usr/local/bin/supercronic \
    && chmod +x /usr/local/bin/supercronic

# ---------------------------------------------------------------------------
# uv — fast Python package installer (10-100x faster than pip)
# ---------------------------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# ---------------------------------------------------------------------------
# Python dependencies — cached layer
#
# Strategy: copy pyproject.toml + create a minimal package stub so that
# `uv pip install -e .` succeeds and caches the heavy dependency layer.
# The real source code is copied afterwards; only that COPY rebuilds when
# code changes (deps stay cached).
# ---------------------------------------------------------------------------
COPY pyproject.toml .
RUN mkdir -p src/winprob && touch src/winprob/__init__.py
RUN uv pip install --system --no-cache --compile-bytecode -e .

# ---------------------------------------------------------------------------
# Source code — this layer rebuilds on every code change, but all deps
# above are already cached so it's nearly instant.
# ---------------------------------------------------------------------------
COPY src/ src/


# =============================================================================
# Stage 2: test
# Used by GitHub Actions to run pytest inside the same environment that the
# production image is built from.  Never pushed to GHCR.
# =============================================================================
FROM base AS test

RUN uv pip install --system --no-cache --compile-bytecode -e ".[dev]"

COPY tests/ tests/


# =============================================================================
# Stage 3: production  (default target)
# =============================================================================
FROM base AS production

# ---------------------------------------------------------------------------
# Application scripts and Docker helpers
# ---------------------------------------------------------------------------
COPY scripts/ scripts/
COPY docker/  docker/
COPY proto/   proto/

# Proto codegen — generate gRPC stubs (grpcio-tools required at build time).
# grpcio-tools left installed; uv pip uninstall has no non-interactive flag in this image.
RUN uv pip install --system --no-cache grpcio-tools \
    && PYTHON=python ./scripts/gen_proto.sh

# Bake the git commit hash into the image.
# Priority: explicit --build-arg > loose ref > packed-refs > detached HEAD.
# Modern Git stores branch tips as loose files under refs/heads/ and only
# populates packed-refs on `git gc`.  We copy both so resolution works
# regardless of the repo's pack state.
ARG GIT_COMMIT=unknown
COPY .git/HEA[D] .git/packed-ref[s] /tmp/gitinfo/
COPY .git/refs/heads/ /tmp/gitinfo/refs/heads/
RUN set -e; commit="$GIT_COMMIT"; \
    if [ "$commit" = "unknown" ] && [ -f /tmp/gitinfo/HEAD ]; then \
        ref=$(cat /tmp/gitinfo/HEAD); \
        if printf '%s' "$ref" | grep -q '^ref:'; then \
            rp=$(printf '%s' "$ref" | sed 's/^ref: //'); \
    loose="/tmp/gitinfo/$rp"; \
    if [ -f "$loose" ]; then \
    commit=$(head -c 8 "$loose"); \
    elif [ -f /tmp/gitinfo/packed-refs ]; then \
                commit=$(grep "$rp" /tmp/gitinfo/packed-refs | head -1 | cut -c1-8); \
            fi; \
        else commit=$(printf '%s' "$ref" | head -c 8); fi; \
    fi; \
    echo "$commit" > /app/GIT_COMMIT; \
    rm -rf /tmp/gitinfo

RUN chmod +x docker/entrypoint.sh \
    docker/ingest_daily.sh \
    docker/retrain_daily.sh

# ---------------------------------------------------------------------------
# Runtime directories
# ---------------------------------------------------------------------------
RUN mkdir -p data/raw data/processed data/processed/statcast_player \
    data/processed/vegas data/processed/weather data/models logs

VOLUME ["/app/data", "/app/logs"]

EXPOSE 8087 50051

ENV MODEL=stacked \
    PORT=8087 \
    GRPC_PORT=50051 \
    WINPROB_GRPC_ENABLED=1 \
    OLLAMA_HOST=http://localhost:11434 \
    OLLAMA_CHAT_MODEL=llama3.1:8b

ENTRYPOINT ["/app/docker/entrypoint.sh"]
