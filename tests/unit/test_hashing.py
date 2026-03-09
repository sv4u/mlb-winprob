"""Tests for mlb_predict.util.hashing — sha256 utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    pass

from mlb_predict.util.hashing import sha256_aggregate_of_files, sha256_file


def _write(path: Path, content: bytes) -> Path:
    """Helper: write bytes to path and return it."""
    path.write_bytes(content)
    return path


# ---------------------------------------------------------------------------
# sha256_file
# ---------------------------------------------------------------------------


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    """sha256_file must return the same digest as a direct hashlib call."""
    data = b"hello, MLB!"
    p = _write(tmp_path / "a.bin", data)
    expected = hashlib.sha256(data).hexdigest()
    assert sha256_file(p) == expected


def test_sha256_file_empty(tmp_path: Path) -> None:
    """sha256_file on an empty file must match the empty-string sha256."""
    p = _write(tmp_path / "empty.bin", b"")
    assert sha256_file(p) == hashlib.sha256(b"").hexdigest()


def test_sha256_file_large_multi_chunk(tmp_path: Path) -> None:
    """sha256_file must correctly hash data that spans multiple 1 MB chunks."""
    data = b"x" * (3 * 1024 * 1024 + 7)  # ~3 MB + a few bytes
    p = _write(tmp_path / "large.bin", data)
    expected = hashlib.sha256(data).hexdigest()
    assert sha256_file(p) == expected


def test_sha256_file_different_content_differs(tmp_path: Path) -> None:
    """Two files with different contents must produce different digests."""
    p1 = _write(tmp_path / "a.bin", b"content-a")
    p2 = _write(tmp_path / "b.bin", b"content-b")
    assert sha256_file(p1) != sha256_file(p2)


def test_sha256_file_returns_hex_string(tmp_path: Path) -> None:
    """sha256_file must return a 64-character lowercase hex string."""
    p = _write(tmp_path / "c.bin", b"test")
    digest = sha256_file(p)
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


# ---------------------------------------------------------------------------
# sha256_aggregate_of_files
# ---------------------------------------------------------------------------


def test_sha256_aggregate_single_file(tmp_path: Path) -> None:
    """Aggregate of one file equals the sha256 of its individual digest."""
    data = b"single-file"
    p = _write(tmp_path / "single.bin", data)
    individual_hex = hashlib.sha256(data).hexdigest()
    expected = hashlib.sha256(individual_hex.encode("ascii")).hexdigest()
    assert sha256_aggregate_of_files([p]) == expected


def test_sha256_aggregate_order_independent(tmp_path: Path) -> None:
    """File order does not affect the aggregate digest (paths are sorted)."""
    p_a = _write(tmp_path / "a.bin", b"aaa")
    p_b = _write(tmp_path / "b.bin", b"bbb")
    assert sha256_aggregate_of_files([p_a, p_b]) == sha256_aggregate_of_files([p_b, p_a])


def test_sha256_aggregate_changes_when_content_changes(tmp_path: Path) -> None:
    """Changing a file's content must change the aggregate digest."""
    p_a = _write(tmp_path / "a.bin", b"original-a")
    p_b = _write(tmp_path / "b.bin", b"original-b")
    digest_before = sha256_aggregate_of_files([p_a, p_b])

    p_a.write_bytes(b"modified-a")
    assert sha256_aggregate_of_files([p_a, p_b]) != digest_before


def test_sha256_aggregate_two_files_differ_from_one(tmp_path: Path) -> None:
    """Aggregate of two distinct files differs from aggregate of either alone."""
    p_a = _write(tmp_path / "a.bin", b"alpha")
    p_b = _write(tmp_path / "b.bin", b"beta")
    agg_both = sha256_aggregate_of_files([p_a, p_b])
    assert agg_both != sha256_aggregate_of_files([p_a])
    assert agg_both != sha256_aggregate_of_files([p_b])


def test_sha256_aggregate_deterministic(tmp_path: Path) -> None:
    """Calling the function twice with the same inputs yields the same result."""
    p_a = _write(tmp_path / "a.bin", b"determinism")
    p_b = _write(tmp_path / "b.bin", b"matters")
    first = sha256_aggregate_of_files([p_a, p_b])
    second = sha256_aggregate_of_files([p_a, p_b])
    assert first == second
