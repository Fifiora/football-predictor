"""Point-in-time feature construction.

Every feature for a match uses only information dated strictly before that match's date.
Matches without a score (postponed or upcoming) get features but never feed anyone's history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = 5

DC_FEATURES = ["dc_exp_home", "dc_exp_away", "dc_p_home", "dc_p_draw", "dc_p_away"]

FEATURE_COLUMNS: list[str] = [
    "elo_diff",
    *DC_FEATURES,
    "form_goals_for_home",
    "form_goals_against_home",
    "form_goals_for_away",
    "form_goals_against_away",
    "form_shots_diff_home",
    "form_shots_diff_away",
    "rest_days_home",
    "rest_days_away",
]

# Candidate inputs beyond the production set; used by the tuning run, not by the shipped model.
EXTRA_COLUMNS: list[str] = [
    "form_ppg_home",
    "form_ppg_away",
    "form_gd10_home",
    "form_gd10_away",
    "venue_gd_home",
    "venue_gd_away",
]
ALL_COLUMNS: list[str] = FEATURE_COLUMNS + EXTRA_COLUMNS
PPG_WINDOW = 5
GD_WINDOW = 10
VENUE_WINDOW = 6


def _team_rows(m: pd.DataFrame) -> pd.DataFrame:
    """Long format: one row per team per match, from that team's perspective."""
    home = pd.DataFrame(
        {
            "match_id": m["match_id"],
            "side": "home",
            "team": m["home"],
            "date": m["date"],
            "gf": m["home_goals"],
            "ga": m["away_goals"],
            "sd": m["home_shots"] - m["away_shots"],
        }
    )
    away = pd.DataFrame(
        {
            "match_id": m["match_id"],
            "side": "away",
            "team": m["away"],
            "date": m["date"],
            "gf": m["away_goals"],
            "ga": m["home_goals"],
            "sd": m["away_shots"] - m["home_shots"],
        }
    )
    return pd.concat([home, away], ignore_index=True)


def _window_mean(cs: np.ndarray, cnt_cs: np.ndarray, idx: np.ndarray, lo: np.ndarray) -> np.ndarray:
    n = cnt_cs[idx] - cnt_cs[lo]
    total = cs[idx] - cs[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, total / np.where(n > 0, n, 1), np.nan)


def _team_state(long: pd.DataFrame) -> pd.DataFrame:
    """Rolling form and rest days for every team-match row, using strictly earlier played matches."""
    long = long.copy()
    long["form_gf"] = np.nan
    long["form_ga"] = np.nan
    long["form_sd"] = np.nan
    long["rest"] = np.nan
    played_mask = long["gf"].notna() & long["ga"].notna()

    for _, group in long.groupby("team"):
        played = group[played_mask.loc[group.index]].sort_values(["date", "match_id"], kind="stable")
        p_dates = played["date"].to_numpy(dtype="datetime64[D]")
        targets = group["date"].to_numpy(dtype="datetime64[D]")
        idx = np.searchsorted(p_dates, targets, side="left")
        lo = np.maximum(idx - WINDOW, 0)

        ones = np.arange(len(p_dates) + 1, dtype=float)  # count of played matches up to i
        cs_gf = np.concatenate([[0.0], np.cumsum(played["gf"].to_numpy(dtype=float))])
        cs_ga = np.concatenate([[0.0], np.cumsum(played["ga"].to_numpy(dtype=float))])
        sd = played["sd"].to_numpy(dtype=float)
        cs_sd = np.concatenate([[0.0], np.cumsum(np.nan_to_num(sd))])
        cs_sd_n = np.concatenate([[0.0], np.cumsum(~np.isnan(sd))])

        long.loc[group.index, "form_gf"] = _window_mean(cs_gf, ones, idx, lo)
        long.loc[group.index, "form_ga"] = _window_mean(cs_ga, ones, idx, lo)
        long.loc[group.index, "form_sd"] = _window_mean(cs_sd, cs_sd_n, idx, lo)
        if len(p_dates):
            prev = p_dates[np.maximum(idx - 1, 0)]
            rest = (targets - prev).astype(float)
            long.loc[group.index, "rest"] = np.where(idx > 0, rest, np.nan)
    return long


def _prior_mean(values: np.ndarray, played_dates: np.ndarray, targets: np.ndarray, window: int) -> np.ndarray:
    """Mean of the last ``window`` values dated strictly before each target date (NaN if none)."""
    idx = np.searchsorted(played_dates, targets, side="left")
    lo = np.maximum(idx - window, 0)
    cs = np.concatenate([[0.0], np.cumsum(values)])
    n = (idx - lo).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (cs[idx] - cs[lo]) / np.where(n > 0, n, 1), np.nan)


def _extra_state(long: pd.DataFrame) -> pd.DataFrame:
    """Points per game, longer goal difference and venue-specific goal difference, strictly prior."""
    long = long.copy()
    for col in ("form_ppg", "form_gd10", "venue_gd"):
        long[col] = np.nan
    played_mask = long["gf"].notna() & long["ga"].notna()

    for _, group in long.groupby("team"):
        played = group[played_mask.loc[group.index]].sort_values(["date", "match_id"], kind="stable")
        if played.empty:
            continue
        p_dates = played["date"].to_numpy(dtype="datetime64[D]")
        targets = group["date"].to_numpy(dtype="datetime64[D]")
        gd = (played["gf"] - played["ga"]).to_numpy(dtype=float)
        pts = np.where(gd > 0, 3.0, np.where(gd == 0, 1.0, 0.0))
        long.loc[group.index, "form_ppg"] = _prior_mean(pts, p_dates, targets, PPG_WINDOW)
        long.loc[group.index, "form_gd10"] = _prior_mean(gd, p_dates, targets, GD_WINDOW)

    for _, group in long.groupby(["team", "side"]):
        played = group[played_mask.loc[group.index]].sort_values(["date", "match_id"], kind="stable")
        if played.empty:
            continue
        p_dates = played["date"].to_numpy(dtype="datetime64[D]")
        targets = group["date"].to_numpy(dtype="datetime64[D]")
        gd = (played["gf"] - played["ga"]).to_numpy(dtype=float)
        long.loc[group.index, "venue_gd"] = _prior_mean(gd, p_dates, targets, VENUE_WINDOW)
    return long


def build_features(
    matches: pd.DataFrame, elo_pre: pd.DataFrame, dc_pre: pd.DataFrame, columns: list[str] | None = None
) -> pd.DataFrame:
    """Feature frame indexed by ``match_id`` with exactly ``columns`` (default ``FEATURE_COLUMNS``)."""
    m = matches.reset_index(drop=True)
    ids = pd.Index(m["match_id"], name="match_id")
    out = pd.DataFrame(index=ids)

    elo = elo_pre.reindex(ids)
    out["elo_diff"] = (elo["elo_home_pre"] - elo["elo_away_pre"]).to_numpy()
    dc = dc_pre.reindex(ids)
    for col in DC_FEATURES:
        out[col] = dc[col].to_numpy()

    state = _team_state(_team_rows(m))
    for side in ("home", "away"):
        part = state[state["side"] == side].set_index("match_id").reindex(ids)
        out[f"form_goals_for_{side}"] = part["form_gf"].to_numpy()
        out[f"form_goals_against_{side}"] = part["form_ga"].to_numpy()
        out[f"form_shots_diff_{side}"] = part["form_sd"].to_numpy()
        out[f"rest_days_{side}"] = part["rest"].to_numpy()

    extra = _extra_state(_team_rows(m))
    for side in ("home", "away"):
        part = extra[extra["side"] == side].set_index("match_id").reindex(ids)
        out[f"form_ppg_{side}"] = part["form_ppg"].to_numpy()
        out[f"form_gd10_{side}"] = part["form_gd10"].to_numpy()
        out[f"venue_gd_{side}"] = part["venue_gd"].to_numpy()

    return out[columns or FEATURE_COLUMNS]
