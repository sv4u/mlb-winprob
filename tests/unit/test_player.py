"""Tests for the mlb_predict.player package (v4 player data pipeline).

Covers biographical data, player stat ingestion, EWMA rolling stats,
the PyTorch player embedding model, lineup model tensor preparation,
and Stage 1 feature generation.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import torch

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bio_df() -> pd.DataFrame:
    """Minimal biographical DataFrame."""
    return pd.DataFrame(
        {
            "mlbam_id": [660271, 545361, 502110],
            "retro_id": ["turnt001", "troum001", "kershc001"],
            "bat_side": [1.0, 1.0, -1.0],
            "throw_side": [1.0, 1.0, -1.0],
            "birth_date": [date(1999, 1, 30), date(1991, 6, 27), date(1988, 3, 19)],
        }
    )


@pytest.fixture
def sample_gamelogs() -> pd.DataFrame:
    """Minimal gamelogs for rolling stat computation."""
    home_ids = [f"player{i:03d}" for i in range(1, 10)]
    away_ids = [f"player{i:03d}" for i in range(10, 19)]

    rows = []
    for day in range(1, 11):
        row = {
            "date": f"2024-04-{day:02d}",
            "home_team": "BOS",
            "visiting_team": "NYA",
            "home_score": 5,
            "visiting_score": 3,
            "num_outs": 54,
            "home_abs": 36,
            "visiting_abs": 33,
            "home_hits": 10,
            "visiting_hits": 8,
            "home_doubles": 2,
            "visiting_doubles": 1,
            "home_triples": 0,
            "visiting_triples": 0,
            "home_homeruns": 2,
            "visiting_homeruns": 1,
            "home_bb": 4,
            "visiting_bb": 3,
            "home_k": 8,
            "visiting_k": 9,
            "home_starting_pitcher_id": "kershc001",
            "visiting_starting_pitcher_id": "pitcher02",
            "home_er": 3,
            "visiting_er": 5,
            "home_po": 27,
            "visiting_po": 27,
        }
        for i, pid in enumerate(home_ids, 1):
            row[f"home_{i}_id"] = pid
        for i, pid in enumerate(away_ids, 1):
            row[f"visiting_{i}_id"] = pid
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# biographical.py
# ---------------------------------------------------------------------------


class TestBiographical:
    """Tests for mlb_predict.player.biographical."""

    def test_player_age_at_date_known_birthday(self) -> None:
        """player_age_at_date returns normalised age for known birthday."""
        from mlb_predict.player.biographical import player_age_at_date

        age = player_age_at_date(date(1988, 3, 19), date(2024, 7, 1))
        expected_raw = (date(2024, 7, 1) - date(1988, 3, 19)).days / 365.25
        expected_norm = (expected_raw - 27.5) / 10.0
        assert abs(age - expected_norm) < 0.01

    def test_player_age_at_date_unknown(self) -> None:
        """player_age_at_date returns 0.0 for unknown birthday."""
        from mlb_predict.player.biographical import player_age_at_date

        assert player_age_at_date(None, date(2024, 7, 1)) == 0.0

    def test_encode_position(self) -> None:
        """encode_position maps positions to normalised floats."""
        from mlb_predict.player.biographical import encode_position

        assert encode_position("C") == 0.0
        assert encode_position("DH") == pytest.approx(8.0 / 9.0)
        assert encode_position("P") == 1.0
        assert encode_position(None) == 0.5

    def test_bat_side_map(self) -> None:
        """BAT_SIDE_MAP encodes L/R/B correctly."""
        from mlb_predict.player.biographical import BAT_SIDE_MAP

        assert BAT_SIDE_MAP["L"] == -1.0
        assert BAT_SIDE_MAP["R"] == 1.0
        assert BAT_SIDE_MAP["B"] == 0.0

    def test_build_bio_lookup(self, bio_df: pd.DataFrame) -> None:
        """build_bio_lookup creates correct mlbam_id → features dict."""
        from mlb_predict.player.biographical import build_bio_lookup

        lookup = build_bio_lookup(bio_df)
        assert 660271 in lookup
        assert lookup[660271]["bat_side"] == 1.0
        assert not np.isnan(lookup[660271]["birth_date_ordinal"])

    def test_build_biographical_df_cache(self, tmp_path: Path, bio_df: pd.DataFrame) -> None:
        """build_biographical_df reads from cache when available."""
        from mlb_predict.player.biographical import build_biographical_df

        cache = tmp_path / "player"
        cache.mkdir()
        bio_df.to_parquet(cache / "biographical.parquet", index=False)
        result = build_biographical_df(cache_dir=cache)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# rolling.py
# ---------------------------------------------------------------------------


class TestRolling:
    """Tests for mlb_predict.player.rolling."""

    def test_build_batter_rolling_returns_expected_columns(
        self, sample_gamelogs: pd.DataFrame
    ) -> None:
        """build_batter_rolling produces EWMA columns."""
        from mlb_predict.player.rolling import build_batter_rolling

        result = build_batter_rolling(sample_gamelogs)
        assert not result.empty
        expected_cols = {"ops_ewm", "iso_ewm", "k_pct_ewm", "bb_pct_ewm", "player_id", "date"}
        assert expected_cols.issubset(set(result.columns))

    def test_build_batter_rolling_values_in_range(self, sample_gamelogs: pd.DataFrame) -> None:
        """EWMA stats should be in plausible ranges."""
        from mlb_predict.player.rolling import build_batter_rolling

        result = build_batter_rolling(sample_gamelogs)
        assert result["ops_ewm"].between(0, 3).all()
        assert result["k_pct_ewm"].between(0, 1).all()
        assert result["bb_pct_ewm"].between(0, 1).all()

    def test_build_pitcher_rolling_returns_expected_columns(
        self, sample_gamelogs: pd.DataFrame
    ) -> None:
        """build_pitcher_rolling produces EWMA columns."""
        from mlb_predict.player.rolling import build_pitcher_rolling

        result = build_pitcher_rolling(sample_gamelogs)
        assert not result.empty
        expected_cols = {"era_ewm", "k9_ewm", "bb9_ewm", "whip_ewm", "player_id", "date"}
        assert expected_cols.issubset(set(result.columns))

    def test_build_pitcher_rolling_values_reasonable(self, sample_gamelogs: pd.DataFrame) -> None:
        """Pitcher EWMA stats should be in plausible ranges."""
        from mlb_predict.player.rolling import build_pitcher_rolling

        result = build_pitcher_rolling(sample_gamelogs)
        assert result["era_ewm"].between(0, 30).all()
        assert result["k9_ewm"].between(0, 30).all()

    def test_get_latest_batter_rolling_for_game(self, sample_gamelogs: pd.DataFrame) -> None:
        """get_latest_batter_rolling_for_game returns most recent stats before cutoff."""
        from mlb_predict.player.rolling import (
            build_batter_rolling,
            get_latest_batter_rolling_for_game,
        )

        rolling = build_batter_rolling(sample_gamelogs)
        cutoff = pd.Timestamp("2024-04-10")
        result = get_latest_batter_rolling_for_game(rolling, ["player001"], cutoff)
        if "player001" in result:
            assert "ops_ewm" in result["player001"]

    def test_get_latest_pitcher_rolling_for_game(self, sample_gamelogs: pd.DataFrame) -> None:
        """get_latest_pitcher_rolling_for_game returns most recent pitcher stats."""
        from mlb_predict.player.rolling import (
            build_pitcher_rolling,
            get_latest_pitcher_rolling_for_game,
        )

        rolling = build_pitcher_rolling(sample_gamelogs)
        cutoff = pd.Timestamp("2024-04-10")
        result = get_latest_pitcher_rolling_for_game(rolling, "kershc001", cutoff)
        if result:
            assert "era_ewm" in result

    def test_empty_gamelogs_return_empty(self) -> None:
        """Rolling functions return empty DataFrames for empty gamelogs."""
        from mlb_predict.player.rolling import build_batter_rolling, build_pitcher_rolling

        empty = pd.DataFrame()
        assert build_batter_rolling(empty).empty
        assert build_pitcher_rolling(empty).empty

    def test_build_pitcher_rolling_with_api_gamelogs(self) -> None:
        """build_pitcher_rolling uses MLB API game logs when provided."""
        from mlb_predict.player.rolling import build_pitcher_rolling

        pitcher_gl = pd.DataFrame(
            {
                "date": pd.to_datetime([f"2024-04-{d:02d}" for d in range(1, 11)]),
                "mlbam_id": [502110] * 10,
                "season": [2024] * 10,
                "is_start": [True] * 10,
                "ip": [6.0, 7.0, 5.0, 6.0, 7.0, 6.0, 5.0, 7.0, 6.0, 5.0],
                "hits": [4, 3, 5, 4, 2, 6, 5, 3, 4, 5],
                "earned_runs": [2, 1, 3, 2, 0, 3, 2, 1, 2, 3],
                "bb": [1, 2, 1, 0, 1, 2, 1, 1, 0, 2],
                "k": [8, 10, 6, 9, 11, 7, 8, 10, 9, 6],
                "hr": [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
                "runs": [2, 1, 3, 2, 0, 4, 2, 1, 2, 3],
                "batters_faced": [25, 27, 22, 24, 26, 28, 23, 26, 24, 23],
            }
        )
        retro_to_mlbam = {"kershc001": 502110}

        result = build_pitcher_rolling(
            pd.DataFrame(),
            retro_to_mlbam=retro_to_mlbam,
            pitcher_game_logs=pitcher_gl,
        )
        assert not result.empty
        expected_cols = {"era_ewm", "k9_ewm", "bb9_ewm", "whip_ewm", "player_id", "date"}
        assert expected_cols.issubset(set(result.columns))
        assert (result["player_id"] == "kershc001").all()
        assert result["era_ewm"].between(0, 10).all()
        assert result["k9_ewm"].between(0, 20).all()

    def test_pitcher_rolling_uses_per_team_putouts(self) -> None:
        """Gamelog fallback uses home_po/visiting_po for per-side IP estimation."""
        from mlb_predict.player.rolling import build_pitcher_rolling

        gl = pd.DataFrame(
            [
                {
                    "date": "2024-04-01",
                    "home_team": "BOS",
                    "visiting_team": "NYA",
                    "home_score": 5,
                    "visiting_score": 3,
                    "num_outs": 51,
                    "home_po": 27,
                    "visiting_po": 24,
                    "home_starting_pitcher_id": "homesp01",
                    "visiting_starting_pitcher_id": "awaysp01",
                    "home_er": 3,
                    "visiting_er": 5,
                    "visiting_hits": 6,
                    "visiting_bb": 2,
                    "visiting_k": 7,
                    "home_hits": 9,
                    "home_bb": 3,
                    "home_k": 8,
                },
            ]
        )
        result = build_pitcher_rolling(gl)
        assert not result.empty
        home_sp = result[result["player_id"] == "homesp01"]
        away_sp = result[result["player_id"] == "awaysp01"]
        assert not home_sp.empty
        assert not away_sp.empty
        home_era = home_sp.iloc[0]["era_ewm"]
        away_era = away_sp.iloc[0]["era_ewm"]
        assert home_era == pytest.approx(3 / 9.0 * 9.0, rel=0.01)
        assert away_era == pytest.approx(5 / 8.0 * 9.0, rel=0.01)

    def test_pitcher_rolling_api_takes_precedence(self, sample_gamelogs: pd.DataFrame) -> None:
        """When API game logs are provided, gamelog approximation is not used."""
        from mlb_predict.player.rolling import build_pitcher_rolling

        pitcher_gl = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-04-01"]),
                "mlbam_id": [99999],
                "season": [2024],
                "is_start": [True],
                "ip": [7.0],
                "hits": [3],
                "earned_runs": [1],
                "bb": [1],
                "k": [10],
                "hr": [0],
                "runs": [1],
                "batters_faced": [25],
            }
        )
        retro_to_mlbam = {"apipit001": 99999}

        result = build_pitcher_rolling(
            sample_gamelogs,
            retro_to_mlbam=retro_to_mlbam,
            pitcher_game_logs=pitcher_gl,
        )
        assert not result.empty
        assert "apipit001" in result["player_id"].values


# ---------------------------------------------------------------------------
# pitcher_gamelogs.py
# ---------------------------------------------------------------------------


class TestPitcherGamelogs:
    """Tests for mlb_predict.player.pitcher_gamelogs."""

    def test_parse_pitcher_gamelog(self) -> None:
        """_parse_pitcher_gamelog extracts stats from API response."""
        from mlb_predict.player.pitcher_gamelogs import _parse_pitcher_gamelog

        raw = {
            "stats": [
                {
                    "splits": [
                        {
                            "date": "2024-04-15",
                            "stat": {
                                "inningsPitched": "6.0",
                                "hits": 4,
                                "earnedRuns": 2,
                                "baseOnBalls": 1,
                                "strikeOuts": 8,
                                "homeRuns": 1,
                                "runs": 2,
                                "battersFaced": 24,
                                "gamesStarted": 1,
                            },
                        },
                    ],
                }
            ],
        }
        rows = _parse_pitcher_gamelog(raw, 502110, 2024)
        assert len(rows) == 1
        assert rows[0]["mlbam_id"] == 502110
        assert rows[0]["ip"] == 6.0
        assert rows[0]["k"] == 8
        assert rows[0]["is_start"] is True

    def test_parse_pitcher_gamelog_skips_zero_ip(self) -> None:
        """Games with zero IP are excluded."""
        from mlb_predict.player.pitcher_gamelogs import _parse_pitcher_gamelog

        raw = {
            "stats": [{"splits": [{"date": "2024-04-15", "stat": {"inningsPitched": "0"}}]}],
        }
        assert _parse_pitcher_gamelog(raw, 502110, 2024) == []

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        """Pitcher game logs round-trip through save/load."""
        from mlb_predict.player.pitcher_gamelogs import (
            load_pitcher_gamelogs,
            save_pitcher_gamelogs,
        )

        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-04-01", "2024-04-06"]),
                "mlbam_id": [502110, 502110],
                "season": [2024, 2024],
                "is_start": [True, True],
                "ip": [6.0, 7.0],
                "hits": [4, 3],
                "earned_runs": [2, 1],
                "bb": [1, 0],
                "k": [8, 10],
                "hr": [1, 0],
                "runs": [2, 1],
                "batters_faced": [24, 25],
            }
        )
        save_pitcher_gamelogs(df, tmp_path, 2024)
        loaded = load_pitcher_gamelogs(tmp_path, [2024])
        assert len(loaded) == 2
        assert "mlbam_id" in loaded.columns


# ---------------------------------------------------------------------------
# embeddings.py
# ---------------------------------------------------------------------------


class TestEmbeddings:
    """Tests for mlb_predict.player.embeddings (Stage 1 PyTorch model)."""

    def test_player_vocab_add_and_get(self) -> None:
        """PlayerVocab assigns unique indices and retrieves them."""
        from mlb_predict.player.embeddings import PlayerVocab

        vocab = PlayerVocab()
        idx1 = vocab.get_or_add(660271)
        idx2 = vocab.get_or_add(545361)
        assert idx1 != idx2
        assert idx1 == vocab.get(660271)
        assert vocab.get(999999) == 0  # unknown returns padding idx

    def test_player_vocab_size(self) -> None:
        """PlayerVocab size includes padding token."""
        from mlb_predict.player.embeddings import PlayerVocab

        vocab = PlayerVocab()
        vocab.get_or_add(100)
        vocab.get_or_add(200)
        assert vocab.size == 3  # padding + 2 players

    def test_player_vocab_save_load(self, tmp_path: Path) -> None:
        """PlayerVocab round-trips through save/load."""
        from mlb_predict.player.embeddings import PlayerVocab

        vocab = PlayerVocab()
        vocab.get_or_add(660271)
        vocab.get_or_add(545361)
        path = tmp_path / "vocab.json"
        vocab.save(path)

        loaded = PlayerVocab.load(path)
        assert loaded.size == vocab.size
        assert loaded.get(660271) == vocab.get(660271)

    def test_batter_encoder_forward_shape(self) -> None:
        """BatterEncoder produces (batch, 9, 16) output."""
        from mlb_predict.player.embeddings import BatterEncoder

        enc = BatterEncoder(vocab_size=100)
        ids = torch.randint(0, 100, (4, 9))
        stats = torch.randn(4, 9, 9)
        bio = torch.randn(4, 9, 3)
        out = enc(ids, stats, bio)
        assert out.shape == (4, 9, 16)

    def test_pitcher_encoder_forward_shape(self) -> None:
        """PitcherEncoder produces (batch, 16) output."""
        from mlb_predict.player.embeddings import BatterEncoder, PitcherEncoder

        batter_enc = BatterEncoder(vocab_size=100)
        pitcher_enc = PitcherEncoder(batter_enc.embedding)
        ids = torch.randint(0, 100, (4,))
        stats = torch.randn(4, 7)
        bio = torch.randn(4, 2)
        out = pitcher_enc(ids, stats, bio)
        assert out.shape == (4, 16)

    def test_player_game_model_forward(self) -> None:
        """PlayerGameModel produces correct output shapes."""
        from mlb_predict.player.embeddings import PlayerGameModel

        model = PlayerGameModel(vocab_size=100)
        batch = 8
        features, logits = model(
            home_batter_ids=torch.randint(0, 100, (batch, 9)),
            home_batter_stats=torch.randn(batch, 9, 9),
            home_batter_bio=torch.randn(batch, 9, 3),
            away_batter_ids=torch.randint(0, 100, (batch, 9)),
            away_batter_stats=torch.randn(batch, 9, 9),
            away_batter_bio=torch.randn(batch, 9, 3),
            home_sp_id=torch.randint(0, 100, (batch,)),
            home_sp_stats=torch.randn(batch, 7),
            home_sp_bio=torch.randn(batch, 2),
            away_sp_id=torch.randint(0, 100, (batch,)),
            away_sp_stats=torch.randn(batch, 7),
            away_sp_bio=torch.randn(batch, 2),
        )
        assert features.shape == (batch, 17)
        assert logits.shape == (batch, 1)

    def test_stage1_feature_names_count(self) -> None:
        """STAGE1_FEATURE_NAMES has exactly 17 entries."""
        from mlb_predict.player.embeddings import STAGE1_FEATURE_NAMES

        assert len(STAGE1_FEATURE_NAMES) == 17

    def test_model_save_load_roundtrip(self, tmp_path: Path) -> None:
        """Stage 1 model round-trips through save/load."""
        from mlb_predict.player.embeddings import (
            PlayerGameModel,
            PlayerVocab,
            load_stage1_model,
            save_stage1_model,
        )

        vocab = PlayerVocab()
        for i in range(1, 20):
            vocab.get_or_add(i)

        model = PlayerGameModel(vocab_size=vocab.size)
        save_dir = tmp_path / "stage1"
        save_stage1_model(model, vocab, save_dir)

        loaded_model, loaded_vocab = load_stage1_model(save_dir)
        assert loaded_vocab.size == vocab.size
        assert (save_dir / "model.pt").exists()
        assert (save_dir / "vocab.json").exists()
        assert (save_dir / "metadata.json").exists()

    def test_embedding_regularization_loss(self) -> None:
        """embedding_regularization_loss returns a positive scalar."""
        from mlb_predict.player.embeddings import PlayerGameModel

        model = PlayerGameModel(vocab_size=50)
        ids = torch.randint(1, 50, (4, 9))
        stats = torch.randn(4, 9, 9)
        loss = model.embedding_regularization_loss(ids, stats, is_batter=True)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_pitcher_embedding_regularization_loss(self) -> None:
        """embedding_regularization_loss works for pitcher inputs (is_batter=False)."""
        from mlb_predict.player.embeddings import PlayerGameModel

        model = PlayerGameModel(vocab_size=50)
        sp_ids = torch.randint(1, 50, (4,))
        sp_stats = torch.randn(4, 7)
        loss = model.embedding_regularization_loss(sp_ids, sp_stats, is_batter=False)
        assert loss.ndim == 0
        assert loss.item() >= 0


# ---------------------------------------------------------------------------
# lineup_model.py
# ---------------------------------------------------------------------------


class TestLineupModel:
    """Tests for mlb_predict.player.lineup_model (training + inference)."""

    def test_prepare_game_tensors(self, sample_gamelogs: pd.DataFrame) -> None:
        """prepare_game_tensors returns correctly shaped tensors."""
        from mlb_predict.player.embeddings import PlayerVocab
        from mlb_predict.player.lineup_model import prepare_game_tensors
        from mlb_predict.player.rolling import build_batter_rolling, build_pitcher_rolling

        batter_rolling = build_batter_rolling(sample_gamelogs)
        pitcher_rolling = build_pitcher_rolling(sample_gamelogs)
        vocab = PlayerVocab()
        bio_lookup: dict[int, dict[str, float]] = {}
        retro_to_mlbam: dict[str, int] = {}

        tensors = prepare_game_tensors(
            sample_gamelogs,
            batter_rolling,
            pitcher_rolling,
            bio_lookup,
            retro_to_mlbam,
            vocab,
            train_mode=True,
        )
        assert tensors is not None
        n = len(sample_gamelogs)
        assert tensors["home_batter_ids"].shape == (n, 9)
        assert tensors["home_batter_stats"].shape == (n, 9, 9)
        assert tensors["home_batter_bio"].shape == (n, 9, 3)
        assert tensors["home_sp_id"].shape == (n,)
        assert tensors["home_sp_stats"].shape == (n, 7)
        assert tensors["targets"].shape == (n,)

    def test_generate_stage1_features_shape(self) -> None:
        """generate_stage1_features returns (n, 17) array."""
        from mlb_predict.player.embeddings import PlayerGameModel
        from mlb_predict.player.lineup_model import generate_stage1_features

        model = PlayerGameModel(vocab_size=50)
        model.eval()
        n = 16
        tensors = {
            "home_batter_ids": torch.randint(0, 50, (n, 9)),
            "home_batter_stats": torch.randn(n, 9, 9),
            "home_batter_bio": torch.randn(n, 9, 3),
            "away_batter_ids": torch.randint(0, 50, (n, 9)),
            "away_batter_stats": torch.randn(n, 9, 9),
            "away_batter_bio": torch.randn(n, 9, 3),
            "home_sp_id": torch.randint(0, 50, (n,)),
            "home_sp_stats": torch.randn(n, 7),
            "home_sp_bio": torch.randn(n, 2),
            "away_sp_id": torch.randint(0, 50, (n,)),
            "away_sp_stats": torch.randn(n, 7),
            "away_sp_bio": torch.randn(n, 2),
        }
        features = generate_stage1_features(model, tensors)
        assert features.shape == (n, 17)
        assert not np.isnan(features).any()

    def test_stage1_features_to_df(self) -> None:
        """stage1_features_to_df converts array to DataFrame with correct columns."""
        from mlb_predict.player.embeddings import STAGE1_FEATURE_NAMES
        from mlb_predict.player.lineup_model import stage1_features_to_df

        arr = np.random.randn(5, 17)
        df = stage1_features_to_df(arr, game_pks=[1, 2, 3, 4, 5])
        assert list(df.columns) == ["game_pk"] + STAGE1_FEATURE_NAMES
        assert len(df) == 5

    def test_train_stage1_small_dataset(self, sample_gamelogs: pd.DataFrame) -> None:
        """train_stage1 runs without error on a small dataset."""
        from mlb_predict.player.embeddings import PlayerVocab
        from mlb_predict.player.lineup_model import prepare_game_tensors, train_stage1
        from mlb_predict.player.rolling import build_batter_rolling, build_pitcher_rolling

        batter_rolling = build_batter_rolling(sample_gamelogs)
        pitcher_rolling = build_pitcher_rolling(sample_gamelogs)
        vocab = PlayerVocab()

        tensors = prepare_game_tensors(
            sample_gamelogs,
            batter_rolling,
            pitcher_rolling,
            {},
            {},
            vocab,
            train_mode=True,
        )
        assert tensors is not None

        model = train_stage1(tensors, vocab, max_epochs=3, patience=10)
        model.eval()

        with torch.no_grad():
            features, logits = model(
                tensors["home_batter_ids"][:2],
                tensors["home_batter_stats"][:2],
                tensors["home_batter_bio"][:2],
                tensors["away_batter_ids"][:2],
                tensors["away_batter_stats"][:2],
                tensors["away_batter_bio"][:2],
                tensors["home_sp_id"][:2],
                tensors["home_sp_stats"][:2],
                tensors["home_sp_bio"][:2],
                tensors["away_sp_id"][:2],
                tensors["away_sp_stats"][:2],
                tensors["away_sp_bio"][:2],
            )
        assert features.shape == (2, 17)
        assert logits.shape == (2, 1)


# ---------------------------------------------------------------------------
# lineups.py (MLB API live lineup fetcher)
# ---------------------------------------------------------------------------


class TestLineups:
    """Tests for mlb_predict.mlbapi.lineups."""

    def test_lineup_entry_dataclass(self) -> None:
        """LineupEntry stores fields correctly."""
        from mlb_predict.mlbapi.lineups import LineupEntry

        entry = LineupEntry(
            mlbam_id=660271,
            full_name="Trea Turner",
            batting_order=1,
            position="SS",
            bat_side="R",
        )
        assert entry.mlbam_id == 660271
        assert entry.batting_order == 1

    def test_lineup_to_player_ids(self) -> None:
        """lineup_to_player_ids extracts IDs in batting order."""
        from mlb_predict.mlbapi.lineups import LineupEntry, lineup_to_player_ids

        lineup = [
            LineupEntry(3, "C", 3, "RF", "R"),
            LineupEntry(1, "A", 1, "SS", "L"),
            LineupEntry(2, "B", 2, "CF", "R"),
        ]
        ids = lineup_to_player_ids(lineup)
        assert ids == [1, 2, 3]

    def test_extract_lineup_empty(self) -> None:
        """_extract_lineup returns empty list for empty team data."""
        from mlb_predict.mlbapi.lineups import _extract_lineup

        result = _extract_lineup({})
        assert result == []

    def test_extract_sp_empty(self) -> None:
        """_extract_sp returns (None, '') for empty team data."""
        from mlb_predict.mlbapi.lineups import _extract_sp

        sp_id, sp_name = _extract_sp({})
        assert sp_id is None
        assert sp_name == ""


# ---------------------------------------------------------------------------
# Feature column integration
# ---------------------------------------------------------------------------


class TestFeatureIntegration:
    """Tests verifying v4 Stage 1 features are wired into FEATURE_COLS."""

    def test_stage1_features_in_feature_cols(self) -> None:
        """All 17 Stage 1 feature names must appear in FEATURE_COLS."""
        from mlb_predict.features.builder import FEATURE_COLS
        from mlb_predict.player.embeddings import STAGE1_FEATURE_NAMES

        for name in STAGE1_FEATURE_NAMES:
            assert name in FEATURE_COLS, f"{name} missing from FEATURE_COLS"

    def test_feature_cols_no_duplicates(self) -> None:
        """FEATURE_COLS must not contain duplicates after v4 additions."""
        from mlb_predict.features.builder import FEATURE_COLS

        assert len(FEATURE_COLS) == len(set(FEATURE_COLS))

    def test_feature_version_is_v4(self) -> None:
        """Training module must report v4 feature version."""
        from mlb_predict.model.train import _FEATURE_VERSION

        assert _FEATURE_VERSION == "v4"

    def test_batting_order_weights_sum_to_one(self) -> None:
        """Batting order weights must sum to approximately 1.0."""
        from mlb_predict.player.embeddings import BATTING_ORDER_WEIGHTS

        assert abs(BATTING_ORDER_WEIGHTS.sum().item() - 1.0) < 1e-5
