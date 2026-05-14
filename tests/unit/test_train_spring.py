"""Tests for spring training integration in mlb_predict.model.train.

Covers ``_load_all_feature_files``, ``_prep`` with ``spring_weight``,
backward compatibility for the ``is_spring`` column, and
``_validate_data_completeness`` from ``scripts/train_model.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    pass

from mlb_predict.features.builder import FEATURE_COLS
from mlb_predict.model.train import (
    _DEFAULT_SPRING_WEIGHT,
    _load_all_feature_files,
    _prep,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_feature_df(
    season: int,
    n_rows: int = 10,
    *,
    include_is_spring: bool = True,
    is_spring_val: float = 0.0,
) -> pd.DataFrame:
    """Build a minimal feature DataFrame for testing."""
    data: dict = {"season": [season] * n_rows, "home_win": [1.0, 0.0] * (n_rows // 2)}
    for col in FEATURE_COLS:
        if col == "is_spring" and include_is_spring:
            data[col] = [is_spring_val] * n_rows
        elif col == "is_spring":
            continue
        else:
            data[col] = [0.5] * n_rows
    return pd.DataFrame(data)


@pytest.fixture
def features_dir(tmp_path: Path) -> Path:
    """Temp directory with regular and spring feature files."""
    d = tmp_path / "features"
    d.mkdir()

    _make_feature_df(2023, 20).to_parquet(d / "features_2023.parquet", index=False)
    _make_feature_df(2024, 20).to_parquet(d / "features_2024.parquet", index=False)

    spring_df = _make_feature_df(2024, 6, is_spring_val=1.0)
    spring_df.to_parquet(d / "features_spring_2024.parquet", index=False)

    return d


@pytest.fixture
def features_dir_no_is_spring(tmp_path: Path) -> Path:
    """Temp directory with legacy feature files (no is_spring column)."""
    d = tmp_path / "features_legacy"
    d.mkdir()

    _make_feature_df(2023, 20, include_is_spring=False).to_parquet(
        d / "features_2023.parquet", index=False
    )
    _make_feature_df(2024, 20, include_is_spring=False).to_parquet(
        d / "features_2024.parquet", index=False
    )

    return d


# ---------------------------------------------------------------------------
# _load_all_feature_files
# ---------------------------------------------------------------------------


def test_load_all_feature_files_merges_by_season(features_dir: Path) -> None:
    """_load_all_feature_files must merge regular and spring files by season."""
    result = _load_all_feature_files(features_dir)
    assert 2023 in result
    assert 2024 in result
    assert len(result[2023]) == 20
    assert len(result[2024]) == 26


def test_load_all_feature_files_season_parsing(features_dir: Path) -> None:
    """_load_all_feature_files must correctly parse season from both filename patterns."""
    result = _load_all_feature_files(features_dir)
    for season, df in result.items():
        assert (df["season"] == season).all()


def test_load_all_feature_files_adds_is_spring_backcompat(
    features_dir_no_is_spring: Path,
) -> None:
    """_load_all_feature_files must add is_spring=0.0 for legacy files."""
    result = _load_all_feature_files(features_dir_no_is_spring)
    for df in result.values():
        assert "is_spring" in df.columns
        assert (df["is_spring"] == 0.0).all()


def test_load_all_feature_files_empty_dir(tmp_path: Path) -> None:
    """_load_all_feature_files must return empty dict for empty directory."""
    d = tmp_path / "empty"
    d.mkdir()
    result = _load_all_feature_files(d)
    assert result == {}


def test_load_all_feature_files_preserves_spring_flag(features_dir: Path) -> None:
    """Spring feature files must have is_spring=1.0 after loading."""
    result = _load_all_feature_files(features_dir)
    df_2024 = result[2024]
    spring_rows = df_2024[df_2024["is_spring"] == 1.0]
    regular_rows = df_2024[df_2024["is_spring"] == 0.0]
    assert len(spring_rows) == 6
    assert len(regular_rows) == 20


# ---------------------------------------------------------------------------
# _prep with spring_weight
# ---------------------------------------------------------------------------


def test_prep_spring_weight_reduces_spring_weights() -> None:
    """_prep must apply spring_weight multiplier to spring training rows."""
    df = pd.concat(
        [
            _make_feature_df(2024, 10, is_spring_val=0.0),
            _make_feature_df(2024, 10, is_spring_val=1.0),
        ],
        ignore_index=True,
    )
    _, _, w = _prep(df, time_weighted=False, spring_weight=0.25)
    assert len(w) == 20
    regular_weights = w[:10]
    spring_weights = w[10:]
    assert np.allclose(regular_weights, 1.0)
    assert np.allclose(spring_weights, 0.25)


def test_prep_spring_weight_default() -> None:
    """_prep default spring_weight must match _DEFAULT_SPRING_WEIGHT."""
    df = _make_feature_df(2024, 4, is_spring_val=1.0)
    _, _, w = _prep(df, time_weighted=False)
    assert np.allclose(w, _DEFAULT_SPRING_WEIGHT)


def test_prep_spring_weight_one_means_equal() -> None:
    """spring_weight=1.0 should make spring and regular rows equally weighted."""
    df = pd.concat(
        [
            _make_feature_df(2024, 4, is_spring_val=0.0),
            _make_feature_df(2024, 4, is_spring_val=1.0),
        ],
        ignore_index=True,
    )
    _, _, w = _prep(df, time_weighted=False, spring_weight=1.0)
    assert np.allclose(w, 1.0)


def test_prep_spring_weight_combined_with_time_decay() -> None:
    """spring_weight must be multiplied with time-decay weights, not replace them."""
    df = pd.concat(
        [
            _make_feature_df(2023, 4, is_spring_val=0.0),
            _make_feature_df(2024, 4, is_spring_val=1.0),
        ],
        ignore_index=True,
    )
    _, _, w_arr = _prep(df, time_weighted=True, time_decay=0.12, spring_weight=0.5)
    w = np.asarray(w_arr, dtype=float)
    regular_2023 = w[:4]
    spring_2024 = w[4:]
    assert np.all(regular_2023 < 1.0)
    assert np.all(spring_2024 < regular_2023)


def test_prep_no_is_spring_column_does_not_crash() -> None:
    """_prep must handle DataFrames without is_spring column gracefully.

    When is_spring is missing, _prep should still work if explicit
    feature_cols are provided that exclude is_spring.
    """
    df = _make_feature_df(2024, 10, include_is_spring=False)
    cols_without_spring = [c for c in FEATURE_COLS if c != "is_spring"]
    X, y, w = _prep(df, feature_cols=cols_without_spring, time_weighted=False)
    assert len(w) == 10
    assert np.allclose(w, 1.0)


# ---------------------------------------------------------------------------
# is_spring in FEATURE_COLS
# ---------------------------------------------------------------------------


def test_is_spring_in_feature_cols() -> None:
    """is_spring must be present in FEATURE_COLS."""
    assert "is_spring" in FEATURE_COLS


def test_feature_cols_no_duplicates() -> None:
    """FEATURE_COLS must not contain duplicates."""
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS))


# ---------------------------------------------------------------------------
# _validate_data_completeness (from scripts/train_model.py)
# ---------------------------------------------------------------------------


def test_validate_data_completeness_passes(tmp_path: Path) -> None:
    """Validation passes when all required files exist."""
    from scripts.train_model import _validate_data_completeness

    sched_dir = tmp_path / "schedule"
    feat_dir = tmp_path / "features"
    sched_dir.mkdir(parents=True)
    feat_dir.mkdir(parents=True)

    seasons = [2023, 2024]
    for s in seasons:
        (sched_dir / f"games_{s}.parquet").write_bytes(b"")
        (feat_dir / f"features_{s}.parquet").write_bytes(b"")

    assert _validate_data_completeness(tmp_path, feat_dir, seasons) is True


def test_validate_data_completeness_missing_schedule(tmp_path: Path) -> None:
    """Validation fails when schedule files are missing."""
    from scripts.train_model import _validate_data_completeness

    sched_dir = tmp_path / "schedule"
    feat_dir = tmp_path / "features"
    sched_dir.mkdir(parents=True)
    feat_dir.mkdir(parents=True)

    seasons = [2023, 2024]
    (feat_dir / "features_2023.parquet").write_bytes(b"")
    (feat_dir / "features_2024.parquet").write_bytes(b"")

    assert _validate_data_completeness(tmp_path, feat_dir, seasons) is False


def test_validate_data_completeness_missing_features(tmp_path: Path) -> None:
    """Validation fails when regular-season feature files are missing."""
    from scripts.train_model import _validate_data_completeness

    sched_dir = tmp_path / "schedule"
    feat_dir = tmp_path / "features"
    sched_dir.mkdir(parents=True)
    feat_dir.mkdir(parents=True)

    seasons = [2023, 2024]
    (sched_dir / "games_2023.parquet").write_bytes(b"")
    (sched_dir / "games_2024.parquet").write_bytes(b"")
    (feat_dir / "features_2023.parquet").write_bytes(b"")

    assert _validate_data_completeness(tmp_path, feat_dir, seasons) is False


def test_validate_data_completeness_counts_spring_files(tmp_path: Path) -> None:
    """Validation reports spring training file count without requiring them."""
    from scripts.train_model import _validate_data_completeness

    sched_dir = tmp_path / "schedule"
    feat_dir = tmp_path / "features"
    sched_dir.mkdir(parents=True)
    feat_dir.mkdir(parents=True)

    seasons = [2023]
    (sched_dir / "games_2023.parquet").write_bytes(b"")
    (feat_dir / "features_2023.parquet").write_bytes(b"")
    (feat_dir / "features_spring_2023.parquet").write_bytes(b"")

    assert _validate_data_completeness(tmp_path, feat_dir, seasons) is True
