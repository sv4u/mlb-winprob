# =============================================================================
# MLB Win Probability — multi-stage Dockerfile
#
# Stages
# ------
#   base        System deps + supercronic + Python package (editable install)
#   test        base + dev dependencies + tests/ — used by CI to run pytest
#   production  base + scripts + docker helpers — what runs in production
#               (default build target; pushed to GHCR)
#
# Build examples
# --------------
#   docker build .                               # production image (default)
#   docker build --target test .                 # test image for CI
#   docker compose up --build                    # production via Compose
# =============================================================================


# =============================================================================
# Stage 1: base
# =============================================================================
FROM python:3.11-slim AS base

# ---------------------------------------------------------------------------
# System dependencies
#   supervisor  — process manager (web server + cron side-by-side)
#   curl        — used to download supercronic
# ---------------------------------------------------------------------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    curl \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# supercronic — Docker-friendly cron daemon that logs to stdout/stderr
# ---------------------------------------------------------------------------
ARG SUPERCRONIC_VERSION=0.2.33
RUN curl -fsSL \
    "https://github.com/aptible/supercronic/releases/download/v${SUPERCRONIC_VERSION}/supercronic-linux-amd64" \
    -o /usr/local/bin/supercronic \
    && chmod +x /usr/local/bin/supercronic

WORKDIR /app

# ---------------------------------------------------------------------------
# Python package
#
# Editable install is intentional: data_cache.py resolves data/ and models/
# relative to __file__, which only works when source lives at /app/src/.
# Layer order: pyproject.toml + src/ first so script-only rebuilds reuse cache.
# ---------------------------------------------------------------------------
COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir -e .


# =============================================================================
# Stage 2: test
# Used by GitHub Actions to run lint and pytest inside the same environment
# that the production image is built from.
# =============================================================================
FROM base AS test

RUN pip install --no-cache-dir -e ".[dev]"

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

RUN chmod +x docker/entrypoint.sh \
    docker/ingest_daily.sh \
    docker/retrain_daily.sh

# ---------------------------------------------------------------------------
# Runtime directories
# /app/data and /app/logs are declared VOLUME mount points so docker-compose
# (or `docker run -v`) can bind them to host paths.
# ---------------------------------------------------------------------------
RUN mkdir -p data/raw data/processed data/models logs

VOLUME ["/app/data", "/app/logs"]

EXPOSE 8087

ENV MODEL=stacked \
    PORT=8087

ENTRYPOINT ["/app/docker/entrypoint.sh"]
