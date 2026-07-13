# =============================================================================
# MLB Win Probability — multi-stage Dockerfile
#
# Stages
# ------
#   pytorch-builder  (optional) Build PyTorch from source for SSE4.2 CPUs
#   gitlog           Extract commit history into a plain-text file
#   base             System deps + supercronic + uv + Python package install
#   test             base + dev dependencies + tests/ — used by CI to run pytest
#   production       base + scripts + docker helpers — what runs in production
#                    (default build target; pushed to GHCR)
#
# Build examples
# --------------
#   docker build .                               # production image (default, CPU-only torch)
#   docker build --build-arg TORCH_CPU=0 .       # with CUDA-enabled torch (adds ~1.5 GB)
#   docker build --build-arg TORCH_SOURCE=1 .    # build PyTorch from source for SSE4.2 CPUs
#   docker build --target test .                 # test image for CI
#   docker compose up --build                    # production via Compose
#
# PyTorch source build (TORCH_SOURCE=1)
# --------------------------------------
# Pre-built PyTorch wheels require AVX2 instructions.  CPUs that lack AVX2
# (e.g. Intel Celeron J4125) crash with SIGILL when running Stage 1 player
# embeddings.  Setting TORCH_SOURCE=1 compiles PyTorch from source targeting
# only SSE4.2, which works on all x86-64 CPUs.  The build is slow (~30–90 min)
# but is cached by Docker BuildKit / GHA cache after the first run.
#
# Multi-platform
# --------------
# The image is published for linux/amd64 and linux/arm64.  supercronic is
# downloaded for the correct architecture automatically via TARGETARCH (a
# BuildKit built-in ARG that matches the target platform's architecture).
# =============================================================================


# =============================================================================
# Stage 0a: pytorch-builder — compile PyTorch from source for SSE4.2 CPUs
#
# Only performs real work when TORCH_SOURCE=1.  When TORCH_SOURCE=0 (default)
# the stage simply creates an empty /wheels/ directory and exits instantly.
#
# Key build flags:
#   USE_MKLDNN=0    Disables MKL-DNN (primary source of SIGILL on non-AVX CPUs)
#   USE_FBGEMM=0    Disables FBGEMM (uses AVX2 matrix kernels)
#   CFLAGS/CXXFLAGS -march=nehalem  Targets SSE4.2 (no AVX/AVX2/AVX-512)
#   BLAS=Eigen      Portable BLAS backend (no MKL dependency)
# =============================================================================
FROM python:3.11-slim AS pytorch-builder

ARG TORCH_SOURCE=0
ARG PYTORCH_VERSION=2.6.0
ARG PYTORCH_BUILD_JOBS=2

WORKDIR /build
RUN mkdir -p /wheels

RUN set -e; \
    if [ "$TORCH_SOURCE" = "1" ]; then \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git ca-certificates \
    libblas-dev liblapack-dev && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir \
    numpy pyyaml setuptools cffi typing_extensions wheel astunparse && \
    git clone --branch "v${PYTORCH_VERSION}" --depth 1 \
    --recurse-submodules --shallow-submodules \
    https://github.com/pytorch/pytorch.git /pytorch && \
    cd /pytorch && \
    USE_CUDA=0 \
    USE_CUDNN=0 \
    USE_MKLDNN=0 \
    USE_FBGEMM=0 \
    USE_NNPACK=0 \
    USE_QNNPACK=0 \
    USE_PYTORCH_QNNPACK=0 \
    USE_XNNPACK=0 \
    USE_DISTRIBUTED=0 \
    USE_TENSORPIPE=0 \
    USE_GLOO=0 \
    USE_MPI=0 \
    USE_NCCL=0 \
    USE_KINETO=0 \
    USE_ITT=0 \
    BUILD_TEST=0 \
    BUILD_CAFFE2=0 \
    BUILD_CAFFE2_OPS=0 \
    BLAS=Eigen \
    MAX_JOBS="${PYTORCH_BUILD_JOBS}" \
    CFLAGS="-march=nehalem" \
    CXXFLAGS="-march=nehalem" \
    PYTORCH_BUILD_VERSION="${PYTORCH_VERSION}" \
    PYTORCH_BUILD_NUMBER=0 \
    python setup.py bdist_wheel && \
    cp dist/*.whl /wheels/ && \
    rm -rf /pytorch; \
    fi


# =============================================================================
# Stage 0b: gitlog — extract commit history into a plain-text file
# Uses alpine/git (~8 MB) so the production image needs neither git nor .git/.
# The output format matches what get_changelog() already parses.
# =============================================================================
FROM alpine/git:latest AS gitlog
WORKDIR /repo
COPY .git/ .git/
RUN git log --format="%H|%ad|%s" --date=short --reverse > /changelog.txt 2>/dev/null || true


# =============================================================================
# Stage 1: base
# =============================================================================
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/root \
    MPLCONFIGDIR=/tmp/matplotlib

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
# Pinned to a specific version for reproducible builds (mlb-predict-pipeline.Rmd §2 rule 1).
# ---------------------------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:0.10.8 /uv /usr/local/bin/uv

WORKDIR /app

# ---------------------------------------------------------------------------
# Python dependencies — cached layer
#
# Strategy: copy pyproject.toml + create a minimal package stub so that
# `uv pip install -e .` succeeds and caches the heavy dependency layer.
# The real source code is copied afterwards; only that COPY rebuilds when
# code changes (deps stay cached).
#
# PyTorch installation (three modes, selected by build args):
#
#   TORCH_SOURCE=1              Custom wheel from pytorch-builder stage,
#                               compiled for SSE4.2 (works on all x86-64).
#   TORCH_SOURCE=0, TORCH_CPU=1 Pre-built CPU-only wheel from PyTorch index
#                               (default; requires AVX2).
#   TORCH_SOURCE=0, TORCH_CPU=0 CUDA-enabled wheel from PyPI (adds ~1.5 GB).
# ---------------------------------------------------------------------------
ARG TORCH_CPU=1
COPY pyproject.toml .
RUN mkdir -p src/mlb_predict && touch src/mlb_predict/__init__.py
COPY --from=pytorch-builder /wheels/ /tmp/pytorch-wheels/
RUN set -e; \
    CUSTOM_TORCH=0; \
    if ls /tmp/pytorch-wheels/*.whl 1>/dev/null 2>&1; then \
    uv pip install --system --no-cache /tmp/pytorch-wheels/*.whl; \
    CUSTOM_TORCH=1; \
    fi; \
    rm -rf /tmp/pytorch-wheels; \
    if [ "$CUSTOM_TORCH" = "0" ] && [ "$TORCH_CPU" = "1" ]; then \
    UV_INDEX_ARGS="--extra-index-url https://download.pytorch.org/whl/cpu"; \
    else \
    UV_INDEX_ARGS=""; \
    fi; \
    uv pip install --system --no-cache --compile-bytecode $UV_INDEX_ARGS -e .; \
    if [ "$CUSTOM_TORCH" = "0" ] && [ "$TORCH_CPU" = "1" ]; then \
    uv pip uninstall --system nvidia-nccl-cu12 nvidia-cuda-runtime-cu12 \
    nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 \
    nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cudnn-cu12 \
    nvidia-nvtx-cu12 nvidia-nvjitlink-cu12 2>/dev/null || true; \
    fi

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
COPY docs/team_id_map_template.csv docs/team_id_map_template.csv

# Proto codegen — generate gRPC stubs (grpcio-tools required at build time).
# After codegen, remove build-only tools (uv, pip, setuptools) that are not
# needed at runtime.  Saves ~75 MB.
RUN uv pip install --system --no-cache grpcio-tools \
    && PYTHON=python ./scripts/gen_proto.sh \
    && uv pip uninstall --system grpcio-tools \
    && rm -f /usr/local/bin/uv \
    && rm -rf /usr/local/lib/python3.11/site-packages/pip \
    /usr/local/lib/python3.11/site-packages/pip-*.dist-info \
    /usr/local/lib/python3.11/site-packages/setuptools \
    /usr/local/lib/python3.11/site-packages/setuptools-*.dist-info \
    /usr/local/bin/pip*

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

COPY --from=gitlog /changelog.txt /app/CHANGELOG.txt

RUN chmod +x docker/entrypoint.sh \
    docker/ingest_daily.sh \
    docker/retrain_daily.sh

# ---------------------------------------------------------------------------
# Runtime directories
# ---------------------------------------------------------------------------
RUN mkdir -p data/raw data/processed data/processed/statcast_player \
    data/processed/vegas data/processed/weather data/models logs

VOLUME ["/app/data", "/app/logs"]

EXPOSE 30087 50051

ENV MODEL=stacked \
    PORT=30087 \
    GRPC_PORT=50051 \
    MLB_PREDICT_GRPC_ENABLED=1 \
    MLB_PREDICT_LIVE_API=1

ENTRYPOINT ["/app/docker/entrypoint.sh"]
