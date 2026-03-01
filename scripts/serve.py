"""Start the MLB Win Probability web dashboard.

Usage
-----
    python scripts/serve.py                   # default: localhost:8000
    python scripts/serve.py --port 8080       # custom port
    python scripts/serve.py --model lightgbm  # use a different model type
    python scripts/serve.py --reload          # auto-reload on code changes
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure src/ is in the path when running as a script
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Start the MLB Win Probability dashboard.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="logistic", choices=["logistic", "lightgbm", "xgboost"])
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("WINPROB_MODEL_TYPE", args.model)

    import uvicorn  # type: ignore

    uvicorn.run(
        "winprob.app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
