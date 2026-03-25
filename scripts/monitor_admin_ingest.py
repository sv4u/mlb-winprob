#!/usr/bin/env python3
"""Poll ``/api/admin/status`` until ingest finishes successfully or fails.

Writes human-readable lines to stdout and optionally to a log file. Intended
for long full ingests (10-minute cadence by default).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any


def _utc_now() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_status(url: str, timeout_sec: float) -> dict[str, Any]:
    """GET admin status JSON from the running app."""
    req = urllib.request.Request(
        url.rstrip("/") + "/api/admin/status",
        headers={"Accept": "application/json"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def ingest_snapshot(data: dict[str, Any]) -> dict[str, Any]:
    """Extract ingest pipeline fields for logging."""
    ing = data.get("pipelines", {}).get("ingest", {})
    steps = ing.get("steps") or []
    running = [s for s in steps if s.get("status") == "running"]
    failed = [s for s in steps if s.get("status") == "failed"]
    return {
        "status": ing.get("status"),
        "error": ing.get("error"),
        "current_step_index": ing.get("current_step_index"),
        "total_steps": ing.get("total_steps"),
        "running_descriptions": [s.get("description", "") for s in running],
        "failed_descriptions": [s.get("description", "") for s in failed],
        "log_tail": ing.get("log_tail") or [],
    }


def format_line(snap: dict[str, Any]) -> str:
    """One-line summary for a poll."""
    st = snap["status"]
    idx = snap["current_step_index"]
    total = snap["total_steps"]
    err = snap["error"]
    running = snap["running_descriptions"]
    run_hint = running[0][:60] if running else "—"
    parts = [
        f"ingest={st}",
        f"step={idx + 1 if isinstance(idx, int) and idx >= 0 else '?'}/{total}",
        f"active={run_hint!r}",
    ]
    if err:
        parts.append(f"error={err!r}")
    return " | ".join(parts)


def main() -> int:
    """Parse CLI args and poll until ingest success, failure, or max iterations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:30087",
        help="App base URL (no trailing path)",
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=600.0,
        help="Seconds between polls (default: 600 = 10 minutes)",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=30.0,
        help="HTTP timeout per request",
    )
    parser.add_argument(
        "--log-file",
        default="",
        help="Append poll lines to this file (e.g. logs/ingest_monitor.log)",
    )
    parser.add_argument(
        "--max-polls",
        type=int,
        default=0,
        help="Stop after N polls (0 = unlimited)",
    )
    args = parser.parse_args()

    def log(msg: str) -> None:
        print(msg, flush=True)
        if args.log_file:
            with open(args.log_file, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    poll_n = 0
    seen_running = False

    while True:
        poll_n += 1
        prefix = f"[{_utc_now()}] poll={poll_n}"
        try:
            data = fetch_status(args.base_url, args.timeout_sec)
            snap = ingest_snapshot(data)
            if snap["status"] == "running":
                seen_running = True
            line = f"{prefix} {format_line(snap)}"
            log(line)
            if snap["status"] == "failed" or snap["error"]:
                log(f"{prefix} EXIT: ingest failed")
                tail = snap["log_tail"][-40:]
                for tline in tail:
                    log(f"  log: {tline}")
                return 2
            if snap["status"] == "success":
                log(f"{prefix} EXIT: ingest success")
                return 0
            if snap["status"] == "idle" and seen_running:
                # Server restart or state reset mid-run
                log(f"{prefix} WARNING: ingest idle after run was seen — stopping monitor")
                return 3
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
            log(f"{prefix} HTTP/parse error: {e!r}")

        if args.max_polls and poll_n >= args.max_polls:
            log(f"{prefix} EXIT: max polls reached")
            return 4

        time.sleep(max(1.0, float(args.interval_sec)))

    return 0


if __name__ == "__main__":
    sys.exit(main())
