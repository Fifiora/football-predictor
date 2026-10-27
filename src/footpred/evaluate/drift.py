"""Drift monitoring over the graded prediction log."""
from __future__ import annotations

import pandas as pd

from footpred.evaluate.metrics import expected_calibration_error, log_loss

LOG_LOSS_MARGIN = 1.10  # alert when rolling loss exceeds 110% of the champion's backtest loss


def check_drift(
    graded_log: pd.DataFrame,
    window: int = 200,
    log_loss_ceiling: float | None = None,
    ece_ceiling: float = 0.10,
    baseline_log_loss: float | None = None,
    ece_bins: int = 5,
) -> list[str]:
    """Alert messages when the latest ``window`` graded predictions look worse than expected.

    The log-loss ceiling is ``log_loss_ceiling`` if given, else ``LOG_LOSS_MARGIN`` times
    ``baseline_log_loss`` (the champion's recorded backtest loss); with neither, only
    calibration is checked. Fewer than ``window`` graded rows is not enough evidence: no alerts.
    """
    graded = graded_log[graded_log["outcome"].notna()].sort_values("kickoff_date")
    if len(graded) < window:
        return []
    recent = graded.tail(window)
    probs = recent[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    outcomes = recent["outcome"].to_numpy(dtype=int)

    alerts: list[str] = []
    ceiling = log_loss_ceiling
    if ceiling is None and baseline_log_loss is not None:
        ceiling = LOG_LOSS_MARGIN * baseline_log_loss
    loss = log_loss(probs, outcomes)
    if ceiling is not None and loss > ceiling:
        alerts.append(f"rolling log loss {loss:.3f} over the last {window} matches exceeds {ceiling:.3f}")
    ece = expected_calibration_error(probs, outcomes, bins=ece_bins)
    if ece > ece_ceiling:
        alerts.append(f"rolling calibration error {ece:.3f} over the last {window} matches exceeds {ece_ceiling:.2f}")
    return alerts
