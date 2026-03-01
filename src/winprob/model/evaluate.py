"""Evaluation metrics for win-probability models.

All metrics operate on arrays of true binary outcomes and predicted
probabilities in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss


@dataclass(frozen=True)
class EvalResult:
    """Bundle of evaluation metrics for one model on one dataset."""

    n_games: int
    brier_score: float         # Lower is better; perfect calibration = 0
    log_loss: float            # Lower is better
    accuracy: float            # Fraction of games where predicted winner was correct
    mean_pred_prob: float      # Sanity check: should be ~0.54 (home-field bias)
    calibration_mean_err: float  # Mean absolute calibration error across 10 bins


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> EvalResult:
    """Compute a full suite of evaluation metrics.

    Parameters
    ----------
    y_true:
        Binary array of actual outcomes (1 = home win).
    y_prob:
        Predicted probability that the home team wins.
    """
    n = int(len(y_true))
    brier = float(brier_score_loss(y_true, y_prob))
    ll = float(log_loss(y_true, y_prob))
    acc = float(np.mean((y_prob >= 0.5).astype(float) == y_true))
    mean_p = float(np.mean(y_prob))

    fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    cal_err = float(np.mean(np.abs(fraction_pos - mean_pred)))

    return EvalResult(
        n_games=n,
        brier_score=brier,
        log_loss=ll,
        accuracy=acc,
        mean_pred_prob=mean_p,
        calibration_mean_err=cal_err,
    )
