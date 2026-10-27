"""Walk-forward backtesting: train on the past, predict the next block, roll forward."""
from __future__ import annotations

from typing import Callable, Iterator

import numpy as np
import pandas as pd

from footpred.evaluate.metrics import encode_outcome

FitPredict = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


def walk_forward_splits(
    dates: pd.Series, first_test_date: pd.Timestamp, step_days: int = 30
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield positional ``(train_idx, test_idx)``; every train date precedes every test date."""
    values = pd.to_datetime(dates).reset_index(drop=True)
    start = pd.Timestamp(first_test_date)
    last = values.max()
    while start <= last:
        end = start + pd.Timedelta(days=step_days)
        train_idx = np.flatnonzero(values < start)
        test_idx = np.flatnonzero((values >= start) & (values < end))
        if len(train_idx) and len(test_idx):
            yield train_idx, test_idx
        start = end


def backtest(
    fit_predict: FitPredict,
    features: pd.DataFrame,
    matches: pd.DataFrame,
    first_test_date: pd.Timestamp,
    step_days: int = 30,
) -> pd.DataFrame:
    """Run ``fit_predict(train, test)`` over walk-forward blocks of played matches.

    ``train`` holds the feature columns plus ``outcome`` (0/1/2); ``test`` holds the feature
    columns only, so a model can never see the outcome it is predicting.
    """
    played = matches[matches["result"].isin(["H", "D", "A"])].reset_index(drop=True)
    frame = features.reindex(played["match_id"]).copy()
    outcome = encode_outcome(played["result"])

    pieces = []
    for train_idx, test_idx in walk_forward_splits(played["date"], first_test_date, step_days):
        train = frame.iloc[train_idx].copy()
        train["outcome"] = outcome[train_idx]
        test = frame.iloc[test_idx].copy()
        probs = np.asarray(fit_predict(train, test), dtype=float)
        pieces.append(
            pd.DataFrame(
                {
                    "match_id": played["match_id"].iloc[test_idx].to_numpy(),
                    "p_home": probs[:, 0],
                    "p_draw": probs[:, 1],
                    "p_away": probs[:, 2],
                    "outcome": outcome[test_idx],
                    "date": played["date"].iloc[test_idx].to_numpy(),
                    "league": played["league"].iloc[test_idx].to_numpy(),
                }
            )
        )
    if not pieces:
        return pd.DataFrame(columns=["match_id", "p_home", "p_draw", "p_away", "outcome", "date", "league"])
    return pd.concat(pieces, ignore_index=True)
