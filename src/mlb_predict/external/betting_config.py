"""Betting configuration: Kelly fraction, budget, and flat bet amount.

Persisted at ``data/processed/odds/betting_config.json``.  Env vars
``MLB_KELLY_PCT``, ``MLB_BUDGET``, ``MLB_BET_AMOUNT`` override the file.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BETTING_CONFIG_PATH = _REPO_ROOT / "data" / "processed" / "odds" / "betting_config.json"

_DEFAULT_KELLY_PCT: float = 25.0
_DEFAULT_BUDGET: float = 300.0
_DEFAULT_BET_AMOUNT: float = 2.0


@dataclass
class BettingConfig:
    """User-configurable betting parameters."""

    kelly_pct: float = _DEFAULT_KELLY_PCT
    budget: float = _DEFAULT_BUDGET
    bet_amount: float = _DEFAULT_BET_AMOUNT


def get_betting_config() -> BettingConfig:
    """Load betting config from env vars (highest priority) then config file.

    Falls back to defaults if neither source is available.
    """
    cfg = BettingConfig()

    if BETTING_CONFIG_PATH.exists():
        try:
            data = json.loads(BETTING_CONFIG_PATH.read_text())
            cfg.kelly_pct = float(data.get("kelly_pct", cfg.kelly_pct))
            cfg.budget = float(data.get("budget", cfg.budget))
            cfg.bet_amount = float(data.get("bet_amount", cfg.bet_amount))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    env_kelly = os.environ.get("MLB_KELLY_PCT", "").strip()
    if env_kelly:
        try:
            cfg.kelly_pct = float(env_kelly)
        except ValueError:
            pass
    env_budget = os.environ.get("MLB_BUDGET", "").strip()
    if env_budget:
        try:
            cfg.budget = float(env_budget)
        except ValueError:
            pass
    env_bet = os.environ.get("MLB_BET_AMOUNT", "").strip()
    if env_bet:
        try:
            cfg.bet_amount = float(env_bet)
        except ValueError:
            pass

    return cfg


def save_betting_config(cfg: BettingConfig) -> None:
    """Write betting config to disk."""
    BETTING_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    BETTING_CONFIG_PATH.write_text(json.dumps(asdict(cfg), indent=2))
