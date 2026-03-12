"""Stage 1: PyTorch player embedding model for lineup/pitcher quality estimation.

Architecture
------------
Per-Batter Encoder:
    player_id → Embedding(vocab_size, 32)
    + rolling stats (9 features: OPS, ISO, K%, BB%, xwOBA, barrel%, hard_hit%, wRC+, sprint_speed)
    + bio features (3: bat_side, age, position)
    → FC(64, ReLU, Dropout=0.2) → FC(32, ReLU) → player_vector (16-dim)

Per-Pitcher Encoder:
    player_id → Embedding(shared vocab, 32)
    + rolling stats (7: ERA, FIP, K/9, BB/9, xwOBA allowed, WHIP, swinging strike%)
    + bio features (2: throw_side, age)
    → FC(32, ReLU, Dropout=0.2) → FC(16, ReLU) → sp_vector (16-dim)

Lineup Aggregation:
    9 player_vectors × batting_order_weights → weighted sum → lineup_vector (16-dim)

Game-Level Head:
    [home_lineup_vec, away_sp_vec, away_lineup_vec, home_sp_vec] (64-dim)
    → FC(64, ReLU) → FC(32, ReLU) → game_features (17) + win_prob (1)

Cold-Start:
    A stat-to-embedding projection network maps observable stats → embedding space.
    Regularized during training so unseen players get reasonable embeddings.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_BATTER_STAT_DIM: int = 9
_BATTER_BIO_DIM: int = 3
_PITCHER_STAT_DIM: int = 7
_PITCHER_BIO_DIM: int = 2
_EMBEDDING_DIM: int = 32
_PLAYER_VEC_DIM: int = 16
_LINEUP_SIZE: int = 9
_GAME_HEAD_DIM: int = 64
_GAME_FEATURE_DIM: int = 17

BATTING_ORDER_WEIGHTS = torch.tensor(
    [1.20, 1.15, 1.10, 1.05, 1.00, 0.95, 0.90, 0.85, 0.80],
    dtype=torch.float32,
)
BATTING_ORDER_WEIGHTS = BATTING_ORDER_WEIGHTS / BATTING_ORDER_WEIGHTS.sum()

STAGE1_FEATURE_NAMES: list[str] = [
    "home_lineup_strength",
    "away_lineup_strength",
    "home_top3_quality",
    "away_top3_quality",
    "home_bottom3_quality",
    "away_bottom3_quality",
    "home_lineup_variance",
    "away_lineup_variance",
    "home_platoon_advantage",
    "away_platoon_advantage",
    "home_sp_quality",
    "away_sp_quality",
    "home_lineup_vs_sp",
    "away_lineup_vs_sp",
    "lineup_strength_diff",
    "sp_quality_diff",
    "matchup_advantage_diff",
]


class BatterEncoder(nn.Module):
    """Encode a single batter's embedding + stats + bio into a player vector."""

    def __init__(self, vocab_size: int, embedding_dim: int = _EMBEDDING_DIM) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        input_dim = embedding_dim + _BATTER_STAT_DIM + _BATTER_BIO_DIM
        self.fc1 = nn.Linear(input_dim, 64)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, _PLAYER_VEC_DIM)

    def forward(
        self,
        player_ids: torch.Tensor,
        stats: torch.Tensor,
        bio: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        player_ids : (batch, 9) int tensor of MLBAM IDs (mapped to vocab indices)
        stats : (batch, 9, 9) batter rolling stats
        bio : (batch, 9, 3) batter biographical features

        Returns
        -------
        (batch, 9, 16) player vectors
        """
        emb = self.embedding(player_ids)
        x = torch.cat([emb, stats, bio], dim=-1)
        x = self.dropout(F.relu(self.fc1(x)))
        return F.relu(self.fc2(x))


class PitcherEncoder(nn.Module):
    """Encode a starting pitcher's embedding + stats + bio into a pitcher vector."""

    def __init__(self, embedding: nn.Embedding) -> None:
        super().__init__()
        self.embedding = embedding
        input_dim = _EMBEDDING_DIM + _PITCHER_STAT_DIM + _PITCHER_BIO_DIM
        self.fc1 = nn.Linear(input_dim, 32)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, _PLAYER_VEC_DIM)

    def forward(
        self,
        player_id: torch.Tensor,
        stats: torch.Tensor,
        bio: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        player_id : (batch,) int tensor
        stats : (batch, 7) pitcher rolling stats
        bio : (batch, 2) pitcher bio features

        Returns
        -------
        (batch, 16) pitcher vector
        """
        emb = self.embedding(player_id)
        x = torch.cat([emb, stats, bio], dim=-1)
        x = self.dropout(F.relu(self.fc1(x)))
        return F.relu(self.fc2(x))


class StatToEmbeddingProjection(nn.Module):
    """Fallback: project observable stats into the embedding space for unseen players."""

    def __init__(self, stat_dim: int, embedding_dim: int = _EMBEDDING_DIM) -> None:
        super().__init__()
        self.fc = nn.Linear(stat_dim, embedding_dim)

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        """Map stat features to embedding space."""
        return self.fc(stats)


class PlayerGameModel(nn.Module):
    """Full Stage 1 model: batters + pitchers → game-level player features.

    Takes both teams' lineups (9 batters each) and starting pitchers,
    produces 17 game-level features + an auxiliary win probability.
    """

    order_weights: torch.Tensor

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

        self.batter_encoder = BatterEncoder(vocab_size)
        self.pitcher_encoder = PitcherEncoder(self.batter_encoder.embedding)

        self.batter_stat_proj = StatToEmbeddingProjection(_BATTER_STAT_DIM)
        self.pitcher_stat_proj = StatToEmbeddingProjection(_PITCHER_STAT_DIM)

        self.game_fc1 = nn.Linear(_GAME_HEAD_DIM, 64)
        self.game_fc2 = nn.Linear(64, 32)
        self.feature_head = nn.Linear(32, _GAME_FEATURE_DIM)
        self.win_head = nn.Linear(32, 1)

        self.register_buffer("order_weights", BATTING_ORDER_WEIGHTS.clone())

    def forward(
        self,
        home_batter_ids: torch.Tensor,
        home_batter_stats: torch.Tensor,
        home_batter_bio: torch.Tensor,
        away_batter_ids: torch.Tensor,
        away_batter_stats: torch.Tensor,
        away_batter_bio: torch.Tensor,
        home_sp_id: torch.Tensor,
        home_sp_stats: torch.Tensor,
        home_sp_bio: torch.Tensor,
        away_sp_id: torch.Tensor,
        away_sp_stats: torch.Tensor,
        away_sp_bio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass producing game features and win probability.

        Returns
        -------
        features : (batch, 17) game-level player features
        win_logit : (batch, 1) raw logit for home win probability
        """
        home_pv = self.batter_encoder(home_batter_ids, home_batter_stats, home_batter_bio)
        away_pv = self.batter_encoder(away_batter_ids, away_batter_stats, away_batter_bio)

        home_sp_vec = self.pitcher_encoder(home_sp_id, home_sp_stats, home_sp_bio)
        away_sp_vec = self.pitcher_encoder(away_sp_id, away_sp_stats, away_sp_bio)

        weights = self.order_weights.unsqueeze(0).unsqueeze(-1)
        home_lineup_vec = (home_pv * weights).sum(dim=1)
        away_lineup_vec = (away_pv * weights).sum(dim=1)

        game_input = torch.cat([home_lineup_vec, away_sp_vec, away_lineup_vec, home_sp_vec], dim=-1)

        h = F.relu(self.game_fc1(game_input))
        h = F.relu(self.game_fc2(h))

        raw_features = self.feature_head(h)
        win_logit = self.win_head(h)

        home_pv_norms = home_pv.norm(dim=-1)
        away_pv_norms = away_pv.norm(dim=-1)

        features = self._assemble_features(
            raw_features,
            home_lineup_vec,
            away_lineup_vec,
            home_sp_vec,
            away_sp_vec,
            home_pv_norms,
            away_pv_norms,
            home_batter_bio,
            away_batter_bio,
            home_sp_bio,
            away_sp_bio,
        )

        return features, win_logit

    def _assemble_features(
        self,
        raw: torch.Tensor,
        home_lv: torch.Tensor,
        away_lv: torch.Tensor,
        home_sp: torch.Tensor,
        away_sp: torch.Tensor,
        home_norms: torch.Tensor,
        away_norms: torch.Tensor,
        home_bat_bio: torch.Tensor,
        away_bat_bio: torch.Tensor,
        home_sp_bio: torch.Tensor,
        away_sp_bio: torch.Tensor,
    ) -> torch.Tensor:
        """Build the 17 interpretable features from model internals."""
        home_strength = torch.sigmoid(home_lv.norm(dim=-1, keepdim=True))
        away_strength = torch.sigmoid(away_lv.norm(dim=-1, keepdim=True))

        home_top3 = home_norms[:, :3].mean(dim=-1, keepdim=True)
        away_top3 = away_norms[:, :3].mean(dim=-1, keepdim=True)
        home_bot3 = home_norms[:, 6:].mean(dim=-1, keepdim=True)
        away_bot3 = away_norms[:, 6:].mean(dim=-1, keepdim=True)

        home_var = home_norms.std(dim=-1, keepdim=True)
        away_var = away_norms.std(dim=-1, keepdim=True)

        # Platoon advantage: interaction of batter handedness with opposing SP handedness
        home_bat_side = home_bat_bio[:, :, 0].mean(dim=-1, keepdim=True)
        away_bat_side = away_bat_bio[:, :, 0].mean(dim=-1, keepdim=True)
        away_sp_throw = away_sp_bio[:, 0:1]
        home_sp_throw = home_sp_bio[:, 0:1]
        home_platoon = home_bat_side * away_sp_throw * -1.0
        away_platoon = away_bat_side * home_sp_throw * -1.0

        home_sp_q = torch.sigmoid(home_sp.norm(dim=-1, keepdim=True))
        away_sp_q = torch.sigmoid(away_sp.norm(dim=-1, keepdim=True))

        home_matchup = raw[:, 0:1]
        away_matchup = raw[:, 1:2]

        strength_diff = home_strength - away_strength
        sp_diff = home_sp_q - away_sp_q
        matchup_diff = home_matchup - away_matchup

        features = torch.cat(
            [
                home_strength,
                away_strength,
                home_top3,
                away_top3,
                home_bot3,
                away_bot3,
                home_var,
                away_var,
                home_platoon,
                away_platoon,
                home_sp_q,
                away_sp_q,
                home_matchup,
                away_matchup,
                strength_diff,
                sp_diff,
                matchup_diff,
            ],
            dim=-1,
        )

        return features

    def embedding_regularization_loss(
        self,
        player_ids: torch.Tensor,
        stats: torch.Tensor,
        is_batter: bool = True,
    ) -> torch.Tensor:
        """Compute ||e_learned - e_predicted||² regularization.

        Ensures the learned embeddings are coherent with observable stats,
        so unseen players get reasonable fallback embeddings.
        """
        emb = self.batter_encoder.embedding(player_ids)
        proj = self.batter_stat_proj if is_batter else self.pitcher_stat_proj
        predicted = proj(stats)

        if emb.dim() == 3:
            predicted = predicted.unsqueeze(1).expand_as(emb) if predicted.dim() == 2 else predicted

        return F.mse_loss(emb.detach(), predicted)


# ---------------------------------------------------------------------------
# Vocabulary management
# ---------------------------------------------------------------------------


class PlayerVocab:
    """Maps MLBAM player IDs to embedding indices.

    Index 0 is reserved for unknown/padding.
    """

    def __init__(self, id_to_idx: dict[int, int] | None = None) -> None:
        self._id_to_idx: dict[int, int] = id_to_idx or {}
        self._next_idx: int = max(self._id_to_idx.values(), default=0) + 1

    @property
    def size(self) -> int:
        """Total vocabulary size including the padding token."""
        return self._next_idx

    def get_or_add(self, mlbam_id: int) -> int:
        """Get existing index or assign new one."""
        if mlbam_id in self._id_to_idx:
            return self._id_to_idx[mlbam_id]
        idx = self._next_idx
        self._id_to_idx[mlbam_id] = idx
        self._next_idx += 1
        return idx

    def get(self, mlbam_id: int) -> int:
        """Look up index; returns 0 (unknown) if not found."""
        return self._id_to_idx.get(mlbam_id, 0)

    def save(self, path: Path) -> None:
        """Save vocabulary to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"id_to_idx": {str(k): v for k, v in self._id_to_idx.items()}}
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "PlayerVocab":
        """Load vocabulary from JSON."""
        data = json.loads(path.read_text())
        id_to_idx = {int(k): v for k, v in data["id_to_idx"].items()}
        return cls(id_to_idx)


# ---------------------------------------------------------------------------
# Save / load model
# ---------------------------------------------------------------------------


def save_stage1_model(
    model: PlayerGameModel,
    vocab: PlayerVocab,
    model_dir: Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save Stage 1 model checkpoint, vocabulary, and metadata."""
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "model.pt")
    vocab.save(model_dir / "vocab.json")
    meta = {
        "vocab_size": vocab.size,
        "embedding_dim": _EMBEDDING_DIM,
        "player_vec_dim": _PLAYER_VEC_DIM,
        "feature_names": STAGE1_FEATURE_NAMES,
        **(metadata or {}),
    }
    (model_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    logger.info("Saved Stage 1 model to %s (vocab_size=%d)", model_dir, vocab.size)
    return model_dir


def load_stage1_model(model_dir: Path) -> tuple[PlayerGameModel, PlayerVocab]:
    """Load a saved Stage 1 model and vocabulary."""
    vocab = PlayerVocab.load(model_dir / "vocab.json")
    model = PlayerGameModel(vocab_size=vocab.size)
    state = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, vocab
