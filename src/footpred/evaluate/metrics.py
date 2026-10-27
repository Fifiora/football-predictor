"""Probabilistic forecast metrics. Probabilities are ``(n, 3)`` in order [home, draw, away]."""
from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-15
_CODES = {"H": 0, "D": 1, "A": 2}


def encode_outcome(result: pd.Series) -> np.ndarray:
    """Map result labels H/D/A to 0/1/2."""
    return result.map(_CODES).to_numpy(dtype=int)


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    p = np.clip(np.asarray(probs, dtype=float), _EPS, 1.0)
    return float(-np.mean(np.log(p[np.arange(len(outcomes)), outcomes])))


def accuracy(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Share of matches where the most probable outcome is what happened."""
    return float(np.mean(np.argmax(np.asarray(probs, dtype=float), axis=1) == np.asarray(outcomes)))


def rps(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Ranked probability score: draw sits between a home win and an away win."""
    probs = np.asarray(probs, dtype=float)
    onehot = np.eye(3)[outcomes]
    cum_diff = np.cumsum(probs, axis=1)[:, :-1] - np.cumsum(onehot, axis=1)[:, :-1]
    return float(np.mean(np.sum(cum_diff**2, axis=1) / 2.0))


def expected_calibration_error(probs: np.ndarray, outcomes: np.ndarray, bins: int = 10) -> float:
    """Mean over the three outcomes of the one-vs-rest binned |frequency - mean probability|."""
    probs = np.asarray(probs, dtype=float)
    onehot = np.eye(3)[outcomes]
    n = len(outcomes)
    edges = np.linspace(0.0, 1.0, bins + 1)
    per_class = []
    for k in range(3):
        which = np.clip(np.digitize(probs[:, k], edges[1:-1]), 0, bins - 1)
        err = 0.0
        for b in range(bins):
            mask = which == b
            if mask.any():
                err += mask.sum() / n * abs(onehot[mask, k].mean() - probs[mask, k].mean())
        per_class.append(err)
    return float(np.mean(per_class))
