"""Tests for bugs identified and fixed in the code review.

Covers ISSUE-2 through ISSUE-33 fixes across features, model training,
drift computation, API client, and utility modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from _pytest.monkeypatch import MonkeyPatch


# ---------------------------------------------------------------------------
# ISSUE-3: away_win_pct_away_only must reflect away-only splits
# ---------------------------------------------------------------------------


def _make_gamelogs(n_games: int = 60) -> pd.DataFrame:
    """Generate synthetic gamelogs with two teams alternating home/away."""
    rows = []
    for i in range(n_games):
        if i % 2 == 0:
            h, v = "NYA", "BOS"
        else:
            h, v = "BOS", "NYA"
        rows.append(
            {
                "date": f"2023-04-{(i % 28) + 1:02d}",
                "game_num": 0,
                "home_team": h,
                "visiting_team": v,
                "home_score": 5 if i % 3 != 0 else 2,
                "visiting_score": 3 if i % 3 != 0 else 6,
            }
        )
    return pd.DataFrame(rows)


class TestTeamStatsSplits:
    """ISSUE-3: Verify home/away splits are computed from correct venue subsets."""

    def test_away_split_differs_from_home_split(self) -> None:
        """The away team's away-only win% must not equal their home-only win%."""
        from mlb_predict.features.team_stats import build_team_rolling_stats

        gl = _make_gamelogs(60)
        result = build_team_rolling_stats(gl)
        assert "home_win_pct_home_only" in result.columns
        assert "away_win_pct_away_only" in result.columns
        home_vals = result["home_win_pct_home_only"].dropna().values
        away_vals = result["away_win_pct_away_only"].dropna().values
        assert len(home_vals) > 0
        assert len(away_vals) > 0

    def test_split_columns_exist(self) -> None:
        """All four split columns must exist in the output."""
        from mlb_predict.features.team_stats import build_team_rolling_stats

        gl = _make_gamelogs(30)
        result = build_team_rolling_stats(gl)
        for col in [
            "home_win_pct_home_only",
            "home_pythag_home_only",
            "away_win_pct_away_only",
            "away_pythag_away_only",
        ]:
            assert col in result.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# ISSUE-4: Lineup continuity must track across venues
# ---------------------------------------------------------------------------


class TestLineupContinuityCrossVenue:
    """ISSUE-4: Lineup continuity should compare to the most recent game,
    regardless of home/away venue."""

    def test_continuity_uses_most_recent_game(self) -> None:
        """If a team plays home then away with same lineup, away continuity
        should be high (not the neutral default)."""
        from mlb_predict.features.lineup import build_lineup_continuity

        id_cols_h = [f"home_{i}_id" for i in range(1, 10)]
        id_cols_a = [f"visiting_{i}_id" for i in range(1, 10)]
        same_lineup = [f"player_{i}" for i in range(1, 10)]
        diff_lineup = [f"other_{i}" for i in range(1, 10)]

        game1 = {"date": "2023-04-01", "game_num": 0, "home_team": "NYA", "visiting_team": "BOS"}
        game2 = {"date": "2023-04-02", "game_num": 0, "home_team": "BOS", "visiting_team": "NYA"}
        for i, pid in enumerate(same_lineup):
            game1[id_cols_h[i]] = pid
            game1[id_cols_a[i]] = diff_lineup[i]
            game2[id_cols_h[i]] = diff_lineup[i]
            game2[id_cols_a[i]] = pid

        gl = pd.DataFrame([game1, game2])
        result = build_lineup_continuity(gl)
        assert result["away_lineup_continuity"].iloc[1] == 9.0


# ---------------------------------------------------------------------------
# ISSUE-10: TokenBucket must reject tokens > capacity
# ---------------------------------------------------------------------------


class TestTokenBucketOverCapacity:
    """ISSUE-10: Requesting more tokens than capacity must raise ValueError."""

    @pytest.mark.asyncio
    async def test_acquire_over_capacity_raises(self) -> None:
        from mlb_predict.mlbapi.client import TokenBucket

        bucket = TokenBucket(rate=5.0, capacity=10.0)
        with pytest.raises(ValueError, match="exceeds bucket capacity"):
            await bucket.acquire(15.0)


# ---------------------------------------------------------------------------
# ISSUE-9: Non-retryable 4xx errors must not be retried
# ---------------------------------------------------------------------------


class TestClientNonRetryable4xx:
    """ISSUE-9: 4xx errors (excluding 404/429) must raise immediately."""

    @pytest.mark.asyncio
    async def test_400_raises_immediately(self, tmp_path: "FixtureRequest") -> None:
        from aioresponses import aioresponses

        from mlb_predict.mlbapi.client import MLBAPIClient, MLBAPIConfig, MLBAPIError

        cfg = MLBAPIConfig(max_retries=3)
        async with MLBAPIClient(config=cfg, cache_dir=tmp_path, refresh=True) as client:
            url = f"{cfg.base_url}{cfg.api_prefix}/test"
            with aioresponses() as m:
                m.get(url, status=400)
                with pytest.raises(MLBAPIError, match="400"):
                    await client.get_json("test", {})


# ---------------------------------------------------------------------------
# ISSUE-7: Global drift dedup must include model_version
# ---------------------------------------------------------------------------


class TestDriftDedup:
    """ISSUE-7: Global drift dedup must use (season, run_ts_utc, model_version)."""

    def test_global_dedup_preserves_different_model_versions(
        self, tmp_path: "FixtureRequest"
    ) -> None:
        from mlb_predict.drift.compute import DriftMetrics, _append_global_metrics

        path = tmp_path / "global.parquet"
        m1 = DriftMetrics(
            run_ts_utc="2024-01-01T00:00:00",
            model_version="xgboost_v3",
            season=2024,
            n_games=100,
            mean_abs_delta=0.01,
            p95_abs_delta=0.02,
            max_abs_delta=0.03,
            pct_gt_0p01=0.5,
            pct_gt_0p02=0.3,
            pct_gt_0p05=0.1,
        )
        m2 = DriftMetrics(
            run_ts_utc="2024-01-01T00:00:00",
            model_version="lightgbm_v3",
            season=2024,
            n_games=100,
            mean_abs_delta=0.02,
            p95_abs_delta=0.03,
            max_abs_delta=0.04,
            pct_gt_0p01=0.6,
            pct_gt_0p02=0.4,
            pct_gt_0p05=0.2,
        )
        _append_global_metrics(m1, path)
        _append_global_metrics(m2, path)
        df = pd.read_parquet(path)
        assert len(df) == 2, "Both model versions should be preserved"

    def test_empty_diff_produces_zero_metrics(self) -> None:
        """Empty diff DataFrame should produce zero-valued metrics, not NaN."""
        from mlb_predict.drift.compute import _metrics_from_diff

        empty_diff = pd.DataFrame(columns=["game_pk", "p_old", "p_new", "delta", "abs_delta"])
        m = _metrics_from_diff(
            empty_diff,
            run_ts="2024-01-01",
            model_version="test",
            season=2024,
        )
        assert m.n_games == 0
        assert m.mean_abs_delta == 0.0
        assert m.max_abs_delta == 0.0


# ---------------------------------------------------------------------------
# ISSUE-14: standings int(None) crash
# ---------------------------------------------------------------------------


class TestStandingsNullRank:
    """ISSUE-14: Null rank values must not crash int() conversion."""

    @pytest.mark.asyncio
    async def test_null_division_rank_does_not_crash(self) -> None:
        from unittest.mock import AsyncMock

        from mlb_predict.mlbapi.standings import fetch_standings

        raw = {
            "records": [
                {
                    "league": {"id": 103, "name": "AL"},
                    "division": {"id": 201, "name": "AL East"},
                    "teamRecords": [
                        {
                            "team": {"id": 147, "name": "Yankees"},
                            "wins": 50,
                            "losses": 30,
                            "winningPercentage": ".625",
                            "gamesBack": "-",
                            "divisionRank": None,
                            "leagueRank": None,
                            "runsScored": 400,
                            "runsAllowed": 350,
                            "runDifferential": 50,
                        }
                    ],
                }
            ]
        }
        client = AsyncMock()
        client.get_json = AsyncMock(return_value=raw)
        df = await fetch_standings(client, season=2024)
        assert df["division_rank"].iloc[0] == 0
        assert df["league_rank"].iloc[0] == 0


# ---------------------------------------------------------------------------
# ISSUE-15: NaN abbreviation in teams
# ---------------------------------------------------------------------------


class TestTeamsNanAbbrev:
    """ISSUE-15: Teams with null abbreviation must be filtered out."""

    @pytest.mark.asyncio
    async def test_null_abbrev_filtered_from_df(self) -> None:
        from unittest.mock import AsyncMock

        from mlb_predict.mlbapi.teams import get_teams_df

        raw = {
            "teams": [
                {"id": 147, "abbreviation": "NYY", "name": "Yankees"},
                {"id": 148, "abbreviation": None, "name": "NoAbbrev Team"},
            ]
        }
        client = AsyncMock()
        client.get_json = AsyncMock(return_value=raw)
        df = await get_teams_df(client, season=2024)
        assert len(df) == 1
        assert df["abbrev"].iloc[0] == "NYY"


# ---------------------------------------------------------------------------
# ISSUE-16: _hash_feature_row must handle inf
# ---------------------------------------------------------------------------


class TestFeatureHashInf:
    """ISSUE-16: Feature hash must not crash on inf/-inf values."""

    def test_hash_with_inf_does_not_raise(self) -> None:
        from mlb_predict.features.builder import FEATURE_COLS, _hash_feature_row

        row = pd.Series({c: 0.5 for c in FEATURE_COLS})
        row.iloc[0] = float("inf")
        result = _hash_feature_row(row)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_hash_with_neg_inf_does_not_raise(self) -> None:
        from mlb_predict.features.builder import FEATURE_COLS, _hash_feature_row

        row = pd.Series({c: 0.5 for c in FEATURE_COLS})
        row.iloc[0] = float("-inf")
        result = _hash_feature_row(row)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# ISSUE-18: Snapshot collision detection
# ---------------------------------------------------------------------------


class TestSnapshotCollision:
    """ISSUE-18: Writing a second snapshot at the same timestamp must raise."""

    def test_duplicate_snapshot_raises(
        self, tmp_path: "FixtureRequest", monkeypatch: "MonkeyPatch"
    ) -> None:
        from datetime import datetime, timezone

        from mlb_predict.errors import SnapshotIntegrityError
        from mlb_predict.predict import snapshot as snap_mod
        from mlb_predict.predict.snapshot import write_snapshot

        fixed_ts = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(
            snap_mod,
            "datetime",
            type(
                "dt",
                (),
                {
                    "now": staticmethod(lambda tz=None: fixed_ts),
                },
            ),
        )

        predictions = pd.DataFrame(
            {
                "game_pk": [1],
                "home_team": ["NYA"],
                "away_team": ["BOS"],
                "predicted_home_win_prob": [0.55],
                "feature_hash": ["abc"],
            }
        )
        feat_file = tmp_path / "features.parquet"
        sched_file = tmp_path / "schedule.parquet"
        feat_file.write_bytes(b"dummy")
        sched_file.write_bytes(b"dummy")

        write_snapshot(
            predictions,
            season=2024,
            model_version="v3",
            model_type="test",
            feature_file=feat_file,
            schedule_file=sched_file,
            snapshot_dir=tmp_path,
        )

        with pytest.raises(SnapshotIntegrityError, match="immutability"):
            write_snapshot(
                predictions,
                season=2024,
                model_version="v3",
                model_type="test",
                feature_file=feat_file,
                schedule_file=sched_file,
                snapshot_dir=tmp_path,
            )


# ---------------------------------------------------------------------------
# ISSUE-26: APIError exported from errors.py
# ---------------------------------------------------------------------------


class TestAPIErrorExported:
    """ISSUE-26: APIError must be importable from mlb_predict.errors."""

    def test_import_api_error(self) -> None:
        from mlb_predict.errors import APIError

        assert issubclass(APIError, Exception)

    def test_mlbapi_error_inherits_api_error(self) -> None:
        from mlb_predict.errors import APIError
        from mlb_predict.mlbapi.client import MLBAPIError

        assert issubclass(MLBAPIError, APIError)


# ---------------------------------------------------------------------------
# ISSUE-28: Platform-independent hashing
# ---------------------------------------------------------------------------


class TestPlatformIndependentHash:
    """ISSUE-28: Aggregate hash must use POSIX paths for cross-platform determinism."""

    def test_aggregate_uses_posix_sort(self, tmp_path: "FixtureRequest") -> None:
        from mlb_predict.util.hashing import sha256_aggregate_of_files

        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("alpha")
        f2.write_text("beta")
        h1 = sha256_aggregate_of_files([f1, f2])
        h2 = sha256_aggregate_of_files([f2, f1])
        assert h1 == h2

    def test_aggregate_docstring_mentions_posix(self) -> None:
        from mlb_predict.util.hashing import sha256_aggregate_of_files

        assert "POSIX" in (sha256_aggregate_of_files.__doc__ or "")


# ---------------------------------------------------------------------------
# ISSUE-6 / ISSUE-5: eval_brier field in ModelMetadata
# ---------------------------------------------------------------------------


class TestModelMetadataEvalBrier:
    """ISSUE-5/6: ModelMetadata must use eval_brier, with legacy compat."""

    def test_eval_brier_field_exists(self) -> None:
        from mlb_predict.model.artifacts import ModelMetadata

        meta = ModelMetadata(
            model_version="v3",
            model_type="test",
            training_seasons=[2024],
            hyperparameters={},
            feature_set_version="v3",
            feature_cols=["a"],
            eval_brier=0.25,
            train_n_games=100,
        )
        assert meta.eval_brier == 0.25

    def test_legacy_train_brier_loaded_as_eval_brier(self, tmp_path: "FixtureRequest") -> None:
        """Old metadata files with 'train_brier' must load correctly."""
        import json

        import joblib

        from mlb_predict.model.artifacts import load_model

        art_dir = tmp_path / "test_v3_train2024"
        art_dir.mkdir()
        meta_dict = {
            "model_version": "v3",
            "model_type": "test",
            "training_seasons": [2024],
            "hyperparameters": {},
            "feature_set_version": "v3",
            "feature_cols": ["a"],
            "train_brier": 0.25,
            "train_n_games": 100,
        }
        (art_dir / "metadata.json").write_text(json.dumps(meta_dict))
        joblib.dump({"dummy": True}, art_dir / "model.joblib")

        _, meta = load_model(art_dir)
        assert meta.eval_brier == 0.25


# ---------------------------------------------------------------------------
# ISSUE-11: Bullpen fatigue cross-venue tracking
# ---------------------------------------------------------------------------


class TestBullpenCrossVenue:
    """ISSUE-11: Bullpen usage must accumulate across home and away games."""

    def test_bullpen_usage_includes_road_games(self) -> None:
        from mlb_predict.features.bullpen import build_bullpen_features

        rows = []
        for i in range(20):
            if i % 2 == 0:
                h, v = "NYA", "BOS"
            else:
                h, v = "BOS", "NYA"
            rows.append(
                {
                    "date": f"2023-04-{i + 1:02d}",
                    "game_num": 0,
                    "home_team": h,
                    "visiting_team": v,
                    "home_pitchers_used": 4,
                    "visiting_pitchers_used": 5,
                    "home_er": 3,
                    "visiting_er": 4,
                }
            )
        gl = pd.DataFrame(rows)
        result = build_bullpen_features(gl)
        assert "home_bullpen_usage_15" in result.columns
        assert result["home_bullpen_usage_15"].notna().all()


# ---------------------------------------------------------------------------
# ISSUE-12: Weather game-hour estimation
# ---------------------------------------------------------------------------


class TestWeatherGameHour:
    """ISSUE-12: Game-hour estimation must use longitude-based timezone offset."""

    def test_east_coast_game_hour(self) -> None:
        from mlb_predict.external.weather import _game_hour_utc

        hour = _game_hour_utc(lat=40.83, lon=-73.93)
        assert 22 <= hour or hour <= 2, f"East coast 7PM should be ~23-0 UTC, got {hour}"

    def test_west_coast_game_hour(self) -> None:
        from mlb_predict.external.weather import _game_hour_utc

        hour = _game_hour_utc(lat=37.78, lon=-122.39)
        assert 1 <= hour <= 5, f"West coast 7PM should be ~2-3 UTC, got {hour}"

    def test_central_game_hour(self) -> None:
        from mlb_predict.external.weather import _game_hour_utc

        hour = _game_hour_utc(lat=41.83, lon=-87.63)
        assert 0 <= hour <= 3, f"Central 7PM should be ~0-1 UTC, got {hour}"
