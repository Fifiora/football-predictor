"""Canonical match schema and validation."""
from __future__ import annotations

import pandas as pd

MATCH_COLUMNS: list[str] = [
    "match_id",
    "league",
    "season",
    "date",
    "home",
    "away",
    "home_goals",
    "away_goals",
    "result",
    "home_shots",
    "away_shots",
    "home_sot",
    "away_sot",
    "close_h",
    "close_d",
    "close_a",
]

VALID_RESULTS = {"H", "D", "A"}


class SchemaError(ValueError):
    """Raised when match data does not satisfy the canonical schema."""


def validate_matches(df: pd.DataFrame, allow_unplayed: bool = False) -> pd.DataFrame:
    """Return a validated copy of ``df`` restricted to ``MATCH_COLUMNS``.

    Rows without a full-time score (postponed, abandoned, or upcoming) are
    dropped unless ``allow_unplayed`` is True, in which case they are kept with
    NaN goals and a missing result.
    """
    missing = [c for c in MATCH_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")

    out = df[MATCH_COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"])

    unplayed = out["home_goals"].isna() | out["away_goals"].isna()
    if not allow_unplayed:
        out = out[~unplayed]

    played = out[out["home_goals"].notna() & out["away_goals"].notna()]
    if (played[["home_goals", "away_goals"]] < 0).any().any():
        raise SchemaError("negative goals")
    bad_results = ~played["result"].isin(VALID_RESULTS)
    if bad_results.any():
        raise SchemaError(f"invalid result labels: {sorted(played.loc[bad_results, 'result'].astype(str).unique())}")
    if (out["home"] == out["away"]).any():
        raise SchemaError("a team cannot play itself")
    if out["match_id"].duplicated().any():
        dupes = out.loc[out["match_id"].duplicated(), "match_id"].tolist()
        raise SchemaError(f"duplicate match_id values: {dupes[:5]}")

    return out.reset_index(drop=True)
