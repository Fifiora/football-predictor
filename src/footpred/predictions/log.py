"""Append-only log of pre-kickoff predictions, graded after full time."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

LOG_COLUMNS = [
    "match_id", "league", "home", "away", "kickoff_date", "model_version", "logged_at",
    "p_home", "p_draw", "p_away", "outcome",
]
_CODES = {"H": 0, "D": 1, "A": 2}


def _read(log_path: Path) -> pd.DataFrame:
    if not Path(log_path).exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.read_csv(log_path)


def _write_atomic(df: pd.DataFrame, log_path: Path) -> None:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = log_path.with_suffix(log_path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, log_path)


def log_predictions(log_path: Path, preds: pd.DataFrame, model_version: str, logged_at: pd.Timestamp) -> None:
    """Append predictions for matches that have not kicked off.

    The whole batch is refused if any ``kickoff_date`` is on or before ``logged_at``. A match
    that is already in the log keeps its original prediction (the first one before kickoff
    stands), so re-running a daily job never rewrites history.
    """
    kickoff = pd.to_datetime(preds["kickoff_date"])
    late = kickoff <= pd.Timestamp(logged_at)
    if late.any():
        raise ValueError(f"refusing to log predictions on or after kickoff: {preds.loc[late, 'match_id'].tolist()}")

    existing = _read(log_path)
    fresh = preds[~preds["match_id"].isin(existing["match_id"])].drop_duplicates("match_id")
    if fresh.empty:
        return
    new = fresh[["match_id", "league", "home", "away", "kickoff_date", "p_home", "p_draw", "p_away"]].copy()
    new["model_version"] = model_version
    new["logged_at"] = pd.Timestamp(logged_at)
    new["outcome"] = np.nan
    new = new[LOG_COLUMNS]
    combined = new if existing.empty else pd.concat([existing[LOG_COLUMNS], new], ignore_index=True)
    _write_atomic(combined, log_path)


def grade_predictions(log_path: Path, results: pd.DataFrame) -> pd.DataFrame:
    """Fill ``outcome`` (0/1/2 for H/D/A) for finished matches; probabilities are never touched."""
    log = _read(log_path)
    if log.empty:
        return log
    codes = results.drop_duplicates("match_id").set_index("match_id")["result"].map(_CODES)
    fill = log["outcome"].isna() & log["match_id"].isin(codes.index)
    if fill.any():
        log.loc[fill, "outcome"] = log.loc[fill, "match_id"].map(codes).astype(float)
        _write_atomic(log, log_path)
    return log
