"""Baselines every model must beat, plus the bookmaker benchmark.

The first three have the ``fit_predict(train, test)`` shape that ``backtest`` expects. The
closing-odds benchmark is for comparison only and is never a model input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

_DC_COLS = ["dc_p_home", "dc_p_draw", "dc_p_away"]


def _class_rates(outcomes: np.ndarray) -> np.ndarray:
    counts = np.bincount(outcomes.astype(int), minlength=3).astype(float)
    return counts / counts.sum()


def baseline_base_rate(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Every match gets the training home/draw/away frequencies."""
    return np.tile(_class_rates(train["outcome"].to_numpy()), (len(test), 1))


def baseline_elo(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Multinomial logistic regression of the outcome on the Elo gap alone."""
    y = train["outcome"].to_numpy(dtype=int)
    model = LogisticRegression(max_iter=1000)
    model.fit(train[["elo_diff"]].to_numpy(), y)
    raw = model.predict_proba(test[["elo_diff"]].to_numpy())
    probs = np.zeros((len(test), 3))
    probs[:, model.classes_] = raw
    return probs


def baseline_dixon_coles(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Dixon-Coles probabilities as-is; rows without a model fall back to the base rate."""
    probs = test[_DC_COLS].to_numpy(dtype=float).copy()
    missing = np.isnan(probs).any(axis=1)
    probs[missing] = _class_rates(train["outcome"].to_numpy())
    return probs


def benchmark_closing_odds(matches: pd.DataFrame) -> np.ndarray:
    """De-vigged closing-odds probabilities (proportional normalization); NaN without odds."""
    implied = 1.0 / matches[["close_h", "close_d", "close_a"]].to_numpy(dtype=float)
    return implied / implied.sum(axis=1, keepdims=True)
