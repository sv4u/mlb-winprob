"""Start the MLB Win Probability web dashboard.

Runs HTTP (FastAPI) and gRPC in a single process: the app lifespan starts the
gRPC server on GRPC_PORT (default 50051) when MLB_PREDICT_GRPC_ENABLED=1. Set
MLB_PREDICT_GRPC_ENABLED=0 or use --no-grpc to run pure FastAPI (fallback mode).

Usage
-----
    python scripts/serve.py                   # default: localhost:30087, gRPC :50051
    python scripts/serve.py --port 9000       # custom HTTP port
    python scripts/serve.py --no-grpc         # disable gRPC (pure FastAPI)
    python scripts/serve.py --model xgboost   # use a different model type
    python scripts/serve.py --model lightgbm  # LightGBM model
    python scripts/serve.py --reload          # auto-reload on code changes (dev)
    python scripts/serve.py --verbose         # DEBUG-level logging

Background / daemon
-------------------
    # Start in background and save PID
    nohup python scripts/serve.py --model xgboost >> logs/server.log 2>&1 &
    echo $! > server.pid

    # Kill the server
    kill $(cat server.pid)              # graceful
    kill -9 $(cat server.pid)           # force
    kill $(lsof -ti:30087)              # by port
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "src"))


def main() -> None:
    """Parse arguments, configure logging, and start the Uvicorn server."""
    ap = argparse.ArgumentParser(description="Start the MLB Prediction System dashboard.")
    default_host = (
        "0.0.0.0" if os.environ.get("MLB_PREDICT_LISTEN_ALL", "").strip() == "1" else "127.0.0.1"
    )
    ap.add_argument(
        "--host",
        default=default_host,
        help="Bind address (default: 127.0.0.1; set MLB_PREDICT_LISTEN_ALL=1 for 0.0.0.0 to allow MCP/dashboard from home network).",
    )
    ap.add_argument("--port", type=int, default=30087)
    ap.add_argument(
        "--no-grpc",
        action="store_true",
        help="Disable gRPC server (MLB_PREDICT_GRPC_ENABLED=0); run pure FastAPI.",
    )
    ap.add_argument(
        "--grpc-port",
        type=int,
        default=50051,
        help="gRPC server port (default 50051). Set GRPC_PORT env to override.",
    )
    ap.add_argument(
        "--model",
        default="stacked",
        choices=["logistic", "lightgbm", "xgboost", "catboost", "mlp", "stacked"],
    )
    ap.add_argument("--reload", action="store_true")
    ap.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level logging for all modules.",
    )
    ap.add_argument(
        "--log-format",
        default="auto",
        choices=["human", "json", "auto"],
        help="Log output format (default: auto — human locally, json in production).",
    )
    args = ap.parse_args()

    os.environ.setdefault("MLB_PREDICT_MODEL_TYPE", args.model)
    if args.no_grpc:
        os.environ["MLB_PREDICT_GRPC_ENABLED"] = "0"
    else:
        os.environ.setdefault("MLB_PREDICT_GRPC_ENABLED", "1")
    os.environ.setdefault("GRPC_PORT", str(args.grpc_port))

    from mlb_predict.logging_config import setup_logging

    setup_logging(verbose=args.verbose or None, log_format=args.log_format)

    import uvicorn  # type: ignore

    log_level = "debug" if args.verbose else "info"

    uvicorn.run(
        "mlb_predict.app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()
