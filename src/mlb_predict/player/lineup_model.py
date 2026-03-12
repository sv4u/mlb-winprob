"""Stage 1 lineup model: training, inference, and feature generation.

Orchestrates the PlayerGameModel (embeddings.py) with data from rolling stats,
biographical data, and gamelogs to produce the 17 game-level player features
that feed into the Stage 2 game ensemble.

Training uses the same expanding-window CV protocol as Stage 2:
  - For season N, train Stage 1 on seasons < N.
  - Generate Stage 1 features for season N using the model trained on < N.

Inference: given a game's lineup (9 batter IDs + 1 SP per team), look up
each player's rolling EWMA stats and bio data, run them through the model,
and return 17 features.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mlb_predict.player.embeddings import (
    STAGE1_FEATURE_NAMES,
    PlayerGameModel,
    PlayerVocab,
    _BATTER_BIO_DIM,
    _BATTER_STAT_DIM,
    _PITCHER_BIO_DIM,
    _PITCHER_STAT_DIM,
)
from mlb_predict.player.rolling import (
    _AVG_BB9,
    _AVG_BB_PCT,
    _AVG_BARREL_PCT,
    _AVG_ERA,
    _AVG_FIP,
    _AVG_HARD_HIT_PCT,
    _AVG_ISO,
    _AVG_K9,
    _AVG_K_PCT,
    _AVG_OPS,
    _AVG_PIT_XWOBA,
    _AVG_SPRINT_SPEED,
    _AVG_SWSTR_PCT,
    _AVG_WHIP,
    _AVG_WRC_PLUS,
    _AVG_XWOBA,
)

logger = logging.getLogger(__name__)

_SEED: int = 42
_DEFAULT_LR: float = 1e-3
_DEFAULT_WEIGHT_DECAY: float = 1e-4
_DEFAULT_BATCH_SIZE: int = 512
_DEFAULT_MAX_EPOCHS: int = 50
_DEFAULT_PATIENCE: int = 5
_DEFAULT_EMB_REG_LAMBDA: float = 0.01

_HOME_LINEUP_COLS = [f"home_{i}_id" for i in range(1, 10)]
_AWAY_LINEUP_COLS = [f"visiting_{i}_id" for i in range(1, 10)]

# Batter stat columns in the rolling DataFrame (order matters)
_BATTER_STAT_KEYS = [
    "ops_ewm",
    "iso_ewm",
    "k_pct_ewm",
    "bb_pct_ewm",
    "xwoba_ewm",
    "barrel_pct_ewm",
    "hard_hit_pct_ewm",
    "wrc_plus_ewm",
    "sprint_speed",
]
_BATTER_DEFAULTS = [
    _AVG_OPS,
    _AVG_ISO,
    _AVG_K_PCT,
    _AVG_BB_PCT,
    _AVG_XWOBA,
    _AVG_BARREL_PCT,
    _AVG_HARD_HIT_PCT,
    _AVG_WRC_PLUS,
    _AVG_SPRINT_SPEED,
]

# Pitcher stat columns in the rolling DataFrame (order matters)
_PITCHER_STAT_KEYS = [
    "era_ewm",
    "fip_ewm",
    "k9_ewm",
    "bb9_ewm",
    "xwoba_allowed_ewm",
    "whip_ewm",
    "swstr_pct_ewm",
]
_PITCHER_DEFAULTS = [
    _AVG_ERA,
    _AVG_FIP,
    _AVG_K9,
    _AVG_BB9,
    _AVG_PIT_XWOBA,
    _AVG_WHIP,
    _AVG_SWSTR_PCT,
]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def prepare_game_tensors(
    gamelogs: pd.DataFrame,
    batter_rolling: pd.DataFrame,
    pitcher_rolling: pd.DataFrame,
    bio_lookup: dict[int, dict[str, float]],
    retro_to_mlbam: dict[str, int],
    vocab: PlayerVocab,
    *,
    train_mode: bool = True,
) -> dict[str, torch.Tensor] | None:
    """Convert gamelogs + rolling stats into model input tensors.

    Returns dict of tensors keyed by model input names, or None if no valid games.
    """
    gl = gamelogs.copy()
    gl["date"] = pd.to_datetime(gl["date"])

    n_games = len(gl)
    home_bat_ids = torch.zeros(n_games, 9, dtype=torch.long)
    home_bat_stats = torch.zeros(n_games, 9, _BATTER_STAT_DIM)
    home_bat_bio = torch.zeros(n_games, 9, _BATTER_BIO_DIM)
    away_bat_ids = torch.zeros(n_games, 9, dtype=torch.long)
    away_bat_stats = torch.zeros(n_games, 9, _BATTER_STAT_DIM)
    away_bat_bio = torch.zeros(n_games, 9, _BATTER_BIO_DIM)
    home_sp_id = torch.zeros(n_games, dtype=torch.long)
    home_sp_stats = torch.zeros(n_games, _PITCHER_STAT_DIM)
    home_sp_bio = torch.zeros(n_games, _PITCHER_BIO_DIM)
    away_sp_id = torch.zeros(n_games, dtype=torch.long)
    away_sp_stats = torch.zeros(n_games, _PITCHER_STAT_DIM)
    away_sp_bio = torch.zeros(n_games, _PITCHER_BIO_DIM)
    targets = torch.full((n_games,), float("nan"))

    valid_mask = torch.ones(n_games, dtype=torch.bool)

    for i, (_, row) in enumerate(gl.iterrows()):
        game_date = row["date"]

        _fill_lineup(
            row,
            _HOME_LINEUP_COLS,
            i,
            home_bat_ids,
            home_bat_stats,
            home_bat_bio,
            batter_rolling,
            bio_lookup,
            retro_to_mlbam,
            vocab,
            game_date,
            train_mode,
        )
        _fill_lineup(
            row,
            _AWAY_LINEUP_COLS,
            i,
            away_bat_ids,
            away_bat_stats,
            away_bat_bio,
            batter_rolling,
            bio_lookup,
            retro_to_mlbam,
            vocab,
            game_date,
            train_mode,
        )

        _fill_pitcher(
            row,
            "home_starting_pitcher_id",
            i,
            home_sp_id,
            home_sp_stats,
            home_sp_bio,
            pitcher_rolling,
            bio_lookup,
            retro_to_mlbam,
            vocab,
            game_date,
            train_mode,
        )
        _fill_pitcher(
            row,
            "visiting_starting_pitcher_id",
            i,
            away_sp_id,
            away_sp_stats,
            away_sp_bio,
            pitcher_rolling,
            bio_lookup,
            retro_to_mlbam,
            vocab,
            game_date,
            train_mode,
        )

        home_score = _safe_num(row.get("home_score"))
        away_score = _safe_num(row.get("visiting_score"))
        if not np.isnan(home_score) and not np.isnan(away_score):
            targets[i] = 1.0 if home_score > away_score else 0.0
        else:
            valid_mask[i] = False

    if train_mode:
        valid_mask = valid_mask & ~targets.isnan()

    return {
        "home_batter_ids": home_bat_ids,
        "home_batter_stats": home_bat_stats,
        "home_batter_bio": home_bat_bio,
        "away_batter_ids": away_bat_ids,
        "away_batter_stats": away_bat_stats,
        "away_batter_bio": away_bat_bio,
        "home_sp_id": home_sp_id,
        "home_sp_stats": home_sp_stats,
        "home_sp_bio": home_sp_bio,
        "away_sp_id": away_sp_id,
        "away_sp_stats": away_sp_stats,
        "away_sp_bio": away_sp_bio,
        "targets": targets,
        "valid_mask": valid_mask,
    }


def _fill_lineup(
    row: pd.Series,
    id_cols: list[str],
    game_idx: int,
    ids_tensor: torch.Tensor,
    stats_tensor: torch.Tensor,
    bio_tensor: torch.Tensor,
    batter_rolling: pd.DataFrame,
    bio_lookup: dict[int, dict[str, float]],
    retro_to_mlbam: dict[str, int],
    vocab: PlayerVocab,
    game_date: pd.Timestamp,
    train_mode: bool,
) -> None:
    """Fill one team's lineup tensors for a single game."""
    for slot, col in enumerate(id_cols):
        retro_id = row.get(col)
        if pd.isna(retro_id) or str(retro_id).strip() == "":
            continue

        retro_key = str(retro_id).strip().lower()
        mlbam = retro_to_mlbam.get(retro_key, 0)
        vocab_idx = vocab.get_or_add(mlbam) if train_mode else vocab.get(mlbam)
        ids_tensor[game_idx, slot] = vocab_idx

        stats = _lookup_batter_stats(batter_rolling, retro_key, game_date)
        stats_tensor[game_idx, slot] = torch.tensor(stats, dtype=torch.float32)

        bio = _lookup_bio(bio_lookup, mlbam, game_date)
        bio_tensor[game_idx, slot] = torch.tensor(bio, dtype=torch.float32)


def _fill_pitcher(
    row: pd.Series,
    pid_col: str,
    game_idx: int,
    id_tensor: torch.Tensor,
    stats_tensor: torch.Tensor,
    bio_tensor: torch.Tensor,
    pitcher_rolling: pd.DataFrame,
    bio_lookup: dict[int, dict[str, float]],
    retro_to_mlbam: dict[str, int],
    vocab: PlayerVocab,
    game_date: pd.Timestamp,
    train_mode: bool,
) -> None:
    """Fill one team's starting pitcher tensors for a single game."""
    retro_id = row.get(pid_col)
    if pd.isna(retro_id) or str(retro_id).strip() == "":
        return

    retro_key = str(retro_id).strip().lower()
    mlbam = retro_to_mlbam.get(retro_key, 0)
    vocab_idx = vocab.get_or_add(mlbam) if train_mode else vocab.get(mlbam)
    id_tensor[game_idx] = vocab_idx

    stats = _lookup_pitcher_stats(pitcher_rolling, retro_key, game_date)
    stats_tensor[game_idx] = torch.tensor(stats, dtype=torch.float32)

    bio = _lookup_pitcher_bio(bio_lookup, mlbam, game_date)
    bio_tensor[game_idx] = torch.tensor(bio, dtype=torch.float32)


def _lookup_batter_stats(
    rolling: pd.DataFrame,
    player_id: str,
    game_date: pd.Timestamp,
) -> list[float]:
    """Get the most recent EWMA batter stats before game_date."""
    if rolling.empty:
        return list(_BATTER_DEFAULTS)

    mask = (rolling["player_id"] == player_id) & (rolling["date"] < game_date)
    rows = rolling.loc[mask]
    if rows.empty:
        return list(_BATTER_DEFAULTS)

    latest = rows.iloc[-1]
    return [float(latest.get(k, d)) for k, d in zip(_BATTER_STAT_KEYS, _BATTER_DEFAULTS)]


def _lookup_pitcher_stats(
    rolling: pd.DataFrame,
    player_id: str,
    game_date: pd.Timestamp,
) -> list[float]:
    """Get the most recent EWMA pitcher stats before game_date."""
    if rolling.empty:
        return list(_PITCHER_DEFAULTS)

    mask = (rolling["player_id"] == player_id) & (rolling["date"] < game_date)
    rows = rolling.loc[mask]
    if rows.empty:
        return list(_PITCHER_DEFAULTS)

    latest = rows.iloc[-1]
    return [float(latest.get(k, d)) for k, d in zip(_PITCHER_STAT_KEYS, _PITCHER_DEFAULTS)]


def _lookup_bio(
    bio_lookup: dict[int, dict[str, float]],
    mlbam_id: int,
    game_date: pd.Timestamp,
) -> list[float]:
    """Get [bat_side, age_normalized, position] for a batter."""
    info = bio_lookup.get(mlbam_id, {})
    bat_side = info.get("bat_side", 0.0)

    bd_ord = info.get("birth_date_ordinal", float("nan"))
    if not np.isnan(bd_ord):
        age_years = (game_date.toordinal() - bd_ord) / 365.25
        age_norm = (age_years - 27.5) / 10.0
    else:
        age_norm = 0.0

    position = 0.5
    return [bat_side, age_norm, position]


def _lookup_pitcher_bio(
    bio_lookup: dict[int, dict[str, float]],
    mlbam_id: int,
    game_date: pd.Timestamp,
) -> list[float]:
    """Get [throw_side, age_normalized] for a pitcher."""
    info = bio_lookup.get(mlbam_id, {})
    throw_side = info.get("throw_side", 1.0)

    bd_ord = info.get("birth_date_ordinal", float("nan"))
    if not np.isnan(bd_ord):
        age_years = (game_date.toordinal() - bd_ord) / 365.25
        age_norm = (age_years - 27.5) / 10.0
    else:
        age_norm = 0.0

    return [throw_side, age_norm]


def _safe_num(val: Any) -> float:
    """Safe numeric conversion."""
    if val is None:
        return float("nan")
    try:
        return float(val)
    except (ValueError, TypeError):
        return float("nan")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_stage1(
    train_tensors: dict[str, torch.Tensor],
    vocab: PlayerVocab,
    *,
    val_tensors: dict[str, torch.Tensor] | None = None,
    lr: float = _DEFAULT_LR,
    weight_decay: float = _DEFAULT_WEIGHT_DECAY,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    max_epochs: int = _DEFAULT_MAX_EPOCHS,
    patience: int = _DEFAULT_PATIENCE,
    emb_reg_lambda: float = _DEFAULT_EMB_REG_LAMBDA,
) -> PlayerGameModel:
    """Train the Stage 1 player embedding model.

    Returns the trained model (in eval mode).
    """
    torch.manual_seed(_SEED)
    np.random.seed(_SEED)

    model = PlayerGameModel(vocab_size=vocab.size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    mask = train_tensors["valid_mask"]
    train_loader = _make_loader(train_tensors, mask, batch_size, shuffle=True)

    val_loader = None
    if val_tensors is not None:
        val_mask = val_tensors["valid_mask"]
        val_loader = _make_loader(val_tensors, val_mask, batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            features, logits = model(
                batch[0],
                batch[1],
                batch[2],
                batch[3],
                batch[4],
                batch[5],
                batch[6],
                batch[7],
                batch[8],
                batch[9],
                batch[10],
                batch[11],
            )
            targets = batch[12]
            loss = criterion(logits.squeeze(-1), targets)

            bat_emb_loss = model.embedding_regularization_loss(batch[0], batch[1], is_batter=True)
            pit_emb_loss = model.embedding_regularization_loss(
                batch[6], batch[7], is_batter=False
            ) + model.embedding_regularization_loss(batch[9], batch[10], is_batter=False)
            loss = loss + emb_reg_lambda * (bat_emb_loss + pit_emb_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        avg_train = train_loss / max(n_batches, 1)

        if val_loader is not None:
            val_loss = _eval_loss(model, val_loader, criterion)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                logger.info("Early stopping at epoch %d (val_loss=%.4f)", epoch + 1, best_val_loss)
                break

            if (epoch + 1) % 5 == 0:
                logger.info(
                    "Epoch %d: train_loss=%.4f val_loss=%.4f best=%.4f",
                    epoch + 1,
                    avg_train,
                    val_loss,
                    best_val_loss,
                )
        else:
            if (epoch + 1) % 5 == 0:
                logger.info("Epoch %d: train_loss=%.4f", epoch + 1, avg_train)

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model


def _make_loader(
    tensors: dict[str, torch.Tensor],
    mask: torch.Tensor,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a DataLoader from game tensors, filtered by valid mask."""
    inputs = [
        tensors["home_batter_ids"][mask],
        tensors["home_batter_stats"][mask],
        tensors["home_batter_bio"][mask],
        tensors["away_batter_ids"][mask],
        tensors["away_batter_stats"][mask],
        tensors["away_batter_bio"][mask],
        tensors["home_sp_id"][mask],
        tensors["home_sp_stats"][mask],
        tensors["home_sp_bio"][mask],
        tensors["away_sp_id"][mask],
        tensors["away_sp_stats"][mask],
        tensors["away_sp_bio"][mask],
        tensors["targets"][mask],
    ]
    dataset = TensorDataset(*inputs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


@torch.no_grad()
def _eval_loss(
    model: PlayerGameModel,
    loader: DataLoader,
    criterion: nn.Module,
) -> float:
    """Compute average loss on a validation set."""
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in loader:
        features, logits = model(
            batch[0],
            batch[1],
            batch[2],
            batch[3],
            batch[4],
            batch[5],
            batch[6],
            batch[7],
            batch[8],
            batch[9],
            batch[10],
            batch[11],
        )
        loss = criterion(logits.squeeze(-1), batch[12])
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


# ---------------------------------------------------------------------------
# Feature generation (inference)
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_stage1_features(
    model: PlayerGameModel,
    tensors: dict[str, torch.Tensor],
    batch_size: int = 1024,
) -> np.ndarray:
    """Run inference on all games and return (n_games, 17) feature array."""
    model.eval()
    n = tensors["home_batter_ids"].shape[0]
    all_features = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        features, _ = model(
            tensors["home_batter_ids"][start:end],
            tensors["home_batter_stats"][start:end],
            tensors["home_batter_bio"][start:end],
            tensors["away_batter_ids"][start:end],
            tensors["away_batter_stats"][start:end],
            tensors["away_batter_bio"][start:end],
            tensors["home_sp_id"][start:end],
            tensors["home_sp_stats"][start:end],
            tensors["home_sp_bio"][start:end],
            tensors["away_sp_id"][start:end],
            tensors["away_sp_stats"][start:end],
            tensors["away_sp_bio"][start:end],
        )
        all_features.append(features.numpy())

    return np.concatenate(all_features, axis=0) if all_features else np.zeros((0, 17))


def stage1_features_to_df(
    features: np.ndarray,
    game_pks: pd.Series | list[int] | None = None,
) -> pd.DataFrame:
    """Convert Stage 1 feature array to a DataFrame with named columns."""
    df = pd.DataFrame(features, columns=STAGE1_FEATURE_NAMES)
    if game_pks is not None:
        df.insert(0, "game_pk", list(game_pks))
    return df
