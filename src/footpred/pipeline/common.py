"""Shared data loading and feature plumbing for the daily and retrain pipelines."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from footpred.evaluate.metrics import encode_outcome
from footpred.features.builder import build_features
from footpred.ingest.historical import load_historical
from footpred.ingest.schema import MATCH_COLUMNS, validate_matches
from footpred.ratings.dixon_coles import dc_pre_match
from footpred.ratings.elo import EloParams, run_elo

PLAYED = ["H", "D", "A"]
API_RESULTS = "api_results.csv"


def load_all_matches(data_dir: Path, aliases_path: Path | None = None) -> pd.DataFrame:
    """Historical CSVs plus results fetched from the API; raises on any bad or unmapped data."""
    data_dir = Path(data_dir)
    hist = load_historical(data_dir / "raw", aliases_path)
    api_path = data_dir / API_RESULTS
    if api_path.exists():
        api = pd.read_csv(api_path, parse_dates=["date"])
        api = validate_matches(api[~api["match_id"].isin(hist["match_id"])])
        hist = pd.concat([hist, api], ignore_index=True)
    return hist.sort_values(["date", "match_id"]).reset_index(drop=True)[MATCH_COLUMNS]


def append_api_results(data_dir: Path, new: pd.DataFrame) -> None:
    path = Path(data_dir) / API_RESULTS
    existing = pd.read_csv(path, parse_dates=["date"]) if path.exists() else pd.DataFrame(columns=MATCH_COLUMNS)
    frames = [f for f in (existing, new[MATCH_COLUMNS]) if not f.empty]
    combined = pd.concat(frames, ignore_index=True).drop_duplicates("match_id")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    combined.to_csv(tmp, index=False)
    os.replace(tmp, path)


def played_before(matches: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    keep = matches["result"].isin(PLAYED) & (matches["date"] < pd.Timestamp(today))
    return matches[keep].reset_index(drop=True)


def training_frame(played: pd.DataFrame, elo_params: EloParams | None = None, xi: float | None = None,
                   columns: list[str] | None = None):
    """Point-in-time features, Elo pre-ratings and encoded outcomes for played matches."""
    elo_pre, ratings = run_elo(played, elo_params) if elo_params else run_elo(played)
    dc_pre = dc_pre_match(played, xi=xi) if xi is not None else dc_pre_match(played)
    features = build_features(played, elo_pre, dc_pre, columns)
    outcome = pd.Series(encode_outcome(played["result"]), index=pd.Index(played["match_id"], name="match_id"))
    return features, outcome, elo_pre, ratings


def code_version() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10, check=True
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def per_league_log_loss(probs: np.ndarray, outcomes: np.ndarray, leagues: np.ndarray) -> dict[str, float]:
    from footpred.evaluate.metrics import log_loss

    return {str(lg): float(log_loss(probs[leagues == lg], outcomes[leagues == lg])) for lg in np.unique(leagues)}
