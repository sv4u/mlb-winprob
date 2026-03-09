"""SHA-256 hashing utilities for provenance tracking (AGENTS.md §2)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence


def sha256_file(path: Path) -> str:
    """Return the hex-encoded SHA-256 digest of *path*'s contents."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_aggregate_of_files(paths: Sequence[Path]) -> str:
    """Return a single SHA-256 digest representing all files in *paths*.

    Files are sorted by their POSIX path representation (forward slashes)
    so the result is platform-independent.
    """
    hashes: list[bytes] = []
    for p in sorted(paths, key=lambda x: x.as_posix()):
        hashes.append(sha256_file(p).encode("ascii"))
    return hashlib.sha256(b"".join(hashes)).hexdigest()
