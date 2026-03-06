#!/usr/bin/env bash
# Generate Python gRPC and protobuf code from proto/winprob/v1/*.proto.
# Run from repository root. Requires grpcio-tools (pip install -e ".[dev]").
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PROTO_DIR="$REPO_ROOT/proto"
OUT_DIR="$REPO_ROOT/src/winprob/grpc/generated"

# Prefer uv run python so project dev deps (grpcio-tools) are used
if command -v uv >/dev/null 2>&1 && uv run python -c "import grpc_tools.protoc" 2>/dev/null; then
  PYTHON="uv run python"
elif ! python -c "import grpc_tools.protoc" 2>/dev/null; then
  echo "grpcio-tools not installed. Run: uv pip install -e '.[dev]' or pip install -e '.[dev]'" >&2
  exit 1
else
  PYTHON="python"
fi

if [[ ! -d "$PROTO_DIR" ]]; then
  echo "Proto directory not found: $PROTO_DIR" >&2
  exit 1
fi

# Generate *_pb2.py and *_pb2_grpc.py into OUT_DIR.
# -I proto makes proto/ the root for imports; package winprob.v1 -> winprob/v1/ under OUT_DIR.
"$PYTHON" -m grpc_tools.protoc \
  -I "$PROTO_DIR" \
  --python_out="$OUT_DIR" \
  --grpc_python_out="$OUT_DIR" \
  "$PROTO_DIR"/winprob/v1/*.proto

echo "Generated code in $OUT_DIR"
