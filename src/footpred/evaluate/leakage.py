"""Automated look-ahead leakage check for feature builders."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from footpred.features.builder import FEATURE_COLUMNS

_OUTCOME_COLUMNS = [
    "home_goals", "away_goals", "home_shots", "away_shots", "home_sot", "away_sot",
    "close_h", "close_d", "close_a",
]


class LeakageError(AssertionError):
    """A feature changed when information from the match date onward was removed."""


def assert_no_leakage(
    build_fn: Callable[[pd.DataFrame], pd.DataFrame],
    matches: pd.DataFrame,
    n_probe: int = 20,
    seed: int = 0,
) -> None:
    """Check that features for random probe matches depend only on strictly earlier data.

    For each probe match dated ``d``: drop every match after ``d`` and blank the results of
    every match on ``d`` (including the probe itself), rebuild, and require the probe's
    features to be unchanged. A builder that uses the match's own outcome, or anything
    that happens later, fails.
    """
    full = build_fn(matches)
    rng = np.random.default_rng(seed)
    ids = matches["match_id"].to_numpy()
    probes = rng.choice(ids, size=min(n_probe, len(ids)), replace=False)

    for probe in probes:
        date = matches.loc[matches["match_id"] == probe, "date"].iloc[0]
        cut = matches[matches["date"] <= date].copy()
        same_day = cut["date"] == date
        cut.loc[same_day, [c for c in _OUTCOME_COLUMNS if c in cut.columns]] = np.nan
        if "result" in cut.columns:
            cut.loc[same_day, "result"] = None

        rebuilt = build_fn(cut)
        a = full.loc[probe, FEATURE_COLUMNS].to_numpy(dtype=float)
        b = rebuilt.loc[probe, FEATURE_COLUMNS].to_numpy(dtype=float)
        if not np.allclose(a, b, equal_nan=True):
            changed = [c for c, x, y in zip(FEATURE_COLUMNS, a, b) if not (x == y or (np.isnan(x) and np.isnan(y)))]
            raise LeakageError(f"features for {probe} changed without future data: {changed}")
