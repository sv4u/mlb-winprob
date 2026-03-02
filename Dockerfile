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

# ---------------------------------------------------------------------------
# Python environment flags
#   PYTHONUNBUFFERED=1           — flush stdout/stderr immediately so
#                                  `docker logs -f` shows output in real time
#   PYTHONDONTWRITEBYTECODE=1    — skip .pyc file creation inside the image
# ---------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ---------------------------------------------------------------------------
# System dependencies
#   supervisor  — process manager (web server + cron side-by-side)
#   curl        — used to download supercronic and by health checks
# ---------------------------------------------------------------------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    curl \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# supercronic — Docker-friendly cron daemon that logs to stdout/stderr
#
# TARGETARCH is a BuildKit built-in ARG (set automatically by --platform).
# It maps platform slugs to supercronic binary names:
#   linux/amd64  → TARGETARCH=amd64  → supercronic-linux-amd64
#   linux/arm64  → TARGETARCH=arm64  → supercronic-linux-arm64
# ---------------------------------------------------------------------------
ARG TARGETARCH=amd64
ARG SUPERCRONIC_VERSION=0.2.33

RUN curl -fsSL \
    "https://github.com/aptible/supercronic/releases/download/v${SUPERCRONIC_VERSION}/supercronic-linux-${TARGETARCH}" \
    -o /usr/local/bin/supercronic \
    && chmod +x /usr/local/bin/supercronic

WORKDIR /app

# ---------------------------------------------------------------------------
# Python package
#
# Editable install is intentional: data_cache.py resolves data/ and models/
# relative to __file__, which only works when source lives at /app/src/.
# Layout: /app/src/winprob/app/data_cache.py → .parent×4 = /app  ✓
#
# Layer order: pyproject.toml + src/ first so script-only rebuilds reuse cache.
# ---------------------------------------------------------------------------
COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir -e .


# =============================================================================
# Stage 2: test
# Used by GitHub Actions to run pytest inside the same environment that the
# production image is built from.  Never pushed to GHCR.
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
#
# Created before VOLUME so they exist in the image layer.  When docker-compose
# provides bind mounts for /app/data and /app/logs the bind mounts take
# precedence and the entrypoint creates any missing subdirectories.
# ---------------------------------------------------------------------------
RUN mkdir -p data/raw data/processed data/models logs

VOLUME ["/app/data", "/app/logs"]

EXPOSE 8087

ENV MODEL=stacked \
    PORT=8087

ENTRYPOINT ["/app/docker/entrypoint.sh"]
