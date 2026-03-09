"""Tests for mlb_predict.retrosheet.gamelogs."""

from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from aioresponses import aioresponses as aioresponses_ctx

if TYPE_CHECKING:
    pass

from mlb_predict.mlbapi.client import TokenBucket
from mlb_predict.retrosheet.gamelogs import (
    GAMELOG_COLUMNS,
    RetrosheetGLSource,
    _extract_gl_txt_from_zip,
    download_gamelog_txt,
    parse_gamelog_txt,
    sha256_bytes,
)


# ---------------------------------------------------------------------------
# sha256_bytes
# ---------------------------------------------------------------------------


def test_sha256_bytes_matches_hashlib() -> None:
    """sha256_bytes must match the hashlib reference implementation."""
    data = b"test data"
    assert sha256_bytes(data) == hashlib.sha256(data).hexdigest()


def test_sha256_bytes_empty() -> None:
    """sha256_bytes on empty bytes must match the hashlib empty-bytes digest."""
    assert sha256_bytes(b"") == hashlib.sha256(b"").hexdigest()


def test_sha256_bytes_returns_hex_string() -> None:
    """sha256_bytes must return a 64-character lowercase hex string."""
    digest = sha256_bytes(b"MLB")
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


# ---------------------------------------------------------------------------
# _extract_gl_txt_from_zip
# ---------------------------------------------------------------------------


def _make_zip_with(filename: str, content: bytes) -> bytes:
    """Build an in-memory ZIP containing one file."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(filename, content)
    return buf.getvalue()


def test_extract_gl_txt_exact_name() -> None:
    """Must extract the file when the ZIP entry name matches exactly."""
    content = b"data,data"
    zip_bytes = _make_zip_with("GL2024.TXT", content)
    result = _extract_gl_txt_from_zip(zip_bytes, 2024)
    assert result == content


def test_extract_gl_txt_case_insensitive() -> None:
    """Must extract the file even when the ZIP entry uses a different case."""
    content = b"lower,case"
    zip_bytes = _make_zip_with("gl2024.txt", content)
    result = _extract_gl_txt_from_zip(zip_bytes, 2024)
    assert result == content


def test_extract_gl_txt_not_found_raises() -> None:
    """Must raise FileNotFoundError when the expected file is absent from the ZIP."""
    zip_bytes = _make_zip_with("GL2023.TXT", b"wrong year")
    with pytest.raises(FileNotFoundError, match="GL2024.TXT"):
        _extract_gl_txt_from_zip(zip_bytes, 2024)


def test_extract_gl_txt_multiple_files() -> None:
    """Must return the correct file when the ZIP contains multiple entries."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("README.txt", b"readme")
        zf.writestr("GL2024.TXT", b"correct")
    zip_bytes = buf.getvalue()
    result = _extract_gl_txt_from_zip(zip_bytes, 2024)
    assert result == b"correct"


# ---------------------------------------------------------------------------
# RetrosheetGLSource — URL generation
# ---------------------------------------------------------------------------


def test_retrosheetsource_chadwick_url() -> None:
    """url_for('chadwick') must return the correct GitHub raw URL."""
    src = RetrosheetGLSource(primary="chadwick")
    url = src.url_for(2024, "chadwick")
    assert "chadwickbureau" in url
    assert "GL2024.TXT" in url
    assert "2024" in url


def test_retrosheetsource_retrosheet_url() -> None:
    """url_for('retrosheet') must return the correct Retrosheet ZIP URL."""
    src = RetrosheetGLSource(primary="retrosheet")
    url = src.url_for(2024, "retrosheet")
    assert "retrosheet.org" in url
    assert "gl2024.zip" in url


def test_retrosheetsource_unknown_kind_raises() -> None:
    """url_for must raise ValueError for an unknown kind."""
    src = RetrosheetGLSource()
    with pytest.raises(ValueError, match="Unknown kind"):
        src.url_for(2024, "unknown")


# ---------------------------------------------------------------------------
# parse_gamelog_txt
# ---------------------------------------------------------------------------


def test_parse_gamelog_txt_basic(gamelog_txt_path: Path) -> None:
    """parse_gamelog_txt must return a DataFrame with 161 columns."""
    df = parse_gamelog_txt(gamelog_txt_path)
    assert len(df) == 2
    assert list(df.columns) == GAMELOG_COLUMNS


def test_parse_gamelog_txt_date_parsing(gamelog_txt_path: Path) -> None:
    """The date column must be converted from YYYYMMDD integer string to date objects."""
    df = parse_gamelog_txt(gamelog_txt_path)
    assert df["date"].iloc[0] == date(2024, 4, 1)


def test_parse_gamelog_txt_numeric_columns(gamelog_txt_path: Path) -> None:
    """visiting_score, home_score, and game_num must be numeric (not string)."""
    df = parse_gamelog_txt(gamelog_txt_path)
    assert df["visiting_score"].iloc[0] == 3
    assert df["home_score"].iloc[0] == 5
    import pandas as pd

    assert pd.api.types.is_numeric_dtype(df["game_num"])


def test_parse_gamelog_txt_team_codes_preserved(gamelog_txt_path: Path) -> None:
    """Team codes must be read as strings and match the written values."""
    df = parse_gamelog_txt(gamelog_txt_path)
    assert df["visiting_team"].iloc[0] == "OAK"
    assert df["home_team"].iloc[0] == "BOS"


def test_parse_gamelog_txt_wrong_column_count_raises(tmp_path: Path) -> None:
    """parse_gamelog_txt must raise ValueError when column count doesn't match."""
    bad = tmp_path / "bad.TXT"
    bad.write_text('"col1","col2","col3"\n')
    with pytest.raises(ValueError, match="Unexpected column count"):
        parse_gamelog_txt(bad)


# ---------------------------------------------------------------------------
# download_gamelog_txt
# ---------------------------------------------------------------------------


async def test_download_uses_disk_cache(gamelog_txt_path: Path, tmp_path: Path) -> None:
    """download_gamelog_txt must return cached=True when the file already exists."""
    out = tmp_path / "GL2024.TXT"
    out.write_bytes(gamelog_txt_path.read_bytes())

    result = await download_gamelog_txt(season=2024, out_path=out)

    assert result["cached"] is True
    assert result["source_used"] == "cache"
    assert result["season"] == 2024
    assert result["sha256"] == sha256_bytes(out.read_bytes())


async def test_download_chadwick_primary_success(tmp_path: Path, gamelog_txt_path: Path) -> None:
    """download_gamelog_txt must fetch from chadwick and write the file."""
    out = tmp_path / "download" / "GL2024.TXT"
    raw_content = gamelog_txt_path.read_bytes()
    src = RetrosheetGLSource(primary="chadwick", enable_fallback=False)
    chadwick_url = src.url_for(2024, "chadwick")

    with aioresponses_ctx() as m:
        m.get(chadwick_url, body=raw_content)
        bucket = TokenBucket(rate=10_000.0, capacity=10_000.0)
        result = await download_gamelog_txt(
            season=2024,
            out_path=out,
            source=src,
            bucket_chadwick=bucket,
        )

    assert out.exists()
    assert result["cached"] is False
    assert result["source_used"] == "chadwick"
    assert result["sha256"] == sha256_bytes(raw_content)


async def test_download_fallback_on_primary_failure(tmp_path: Path, gamelog_txt_path: Path) -> None:
    """When the primary source fails, the fallback source must be tried."""
    out = tmp_path / "fallback" / "GL2024.TXT"
    raw_content = gamelog_txt_path.read_bytes()

    src = RetrosheetGLSource(primary="chadwick", enable_fallback=True)
    chadwick_url = src.url_for(2024, "chadwick")

    # Build a ZIP for the Retrosheet fallback
    zip_content = io.BytesIO()
    with zipfile.ZipFile(zip_content, "w") as zf:
        zf.writestr("GL2024.TXT", raw_content)
    retrosheet_url = src.url_for(2024, "retrosheet")

    bucket = TokenBucket(rate=10_000.0, capacity=10_000.0)
    with aioresponses_ctx() as m:
        m.get(chadwick_url, status=404)
        m.get(retrosheet_url, body=zip_content.getvalue())
        result = await download_gamelog_txt(
            season=2024,
            out_path=out,
            source=src,
            bucket_chadwick=bucket,
            bucket_retrosheet=bucket,
        )

    assert result["source_used"] == "retrosheet"
    assert result["fallback_reason"] is not None
    assert out.exists()


async def test_download_no_fallback_raises_on_failure(tmp_path: Path) -> None:
    """When fallback is disabled and primary fails, the error must propagate."""
    out = tmp_path / "nofallback" / "GL2024.TXT"
    src = RetrosheetGLSource(primary="chadwick", enable_fallback=False)
    chadwick_url = src.url_for(2024, "chadwick")

    bucket = TokenBucket(rate=10_000.0, capacity=10_000.0)
    with aioresponses_ctx() as m:
        m.get(chadwick_url, status=500)
        with pytest.raises(Exception):
            await download_gamelog_txt(
                season=2024,
                out_path=out,
                source=src,
                bucket_chadwick=bucket,
            )


async def test_download_refresh_refetches(tmp_path: Path, gamelog_txt_path: Path) -> None:
    """With refresh=True, download must re-fetch even when the file exists."""
    out = tmp_path / "refresh" / "GL2024.TXT"
    out.parent.mkdir(parents=True, exist_ok=True)
    old_content = b"old data" + b"," * 160  # placeholder
    out.write_bytes(old_content)

    new_content = gamelog_txt_path.read_bytes()
    src = RetrosheetGLSource(primary="chadwick", enable_fallback=False)
    chadwick_url = src.url_for(2024, "chadwick")

    bucket = TokenBucket(rate=10_000.0, capacity=10_000.0)
    with aioresponses_ctx() as m:
        m.get(chadwick_url, body=new_content)
        result = await download_gamelog_txt(
            season=2024,
            out_path=out,
            source=src,
            refresh=True,
            bucket_chadwick=bucket,
        )

    assert result["cached"] is False
    assert out.read_bytes() == new_content


async def test_download_creates_output_directory(tmp_path: Path, gamelog_txt_path: Path) -> None:
    """download_gamelog_txt must create any missing parent directories."""
    out = tmp_path / "nested" / "dir" / "GL2024.TXT"
    raw_content = gamelog_txt_path.read_bytes()

    src = RetrosheetGLSource(primary="chadwick", enable_fallback=False)
    chadwick_url = src.url_for(2024, "chadwick")
    bucket = TokenBucket(rate=10_000.0, capacity=10_000.0)

    with aioresponses_ctx() as m:
        m.get(chadwick_url, body=raw_content)
        await download_gamelog_txt(season=2024, out_path=out, source=src, bucket_chadwick=bucket)

    assert out.exists()
