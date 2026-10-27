"""One Elo system across the big five leagues and European competitions.

Cross-league games are what make ratings comparable across leagues. Each club starts at the
league-group offset of its country; the offsets are fitted on cross-group results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from footpred.european.names import GROUPS
from footpred.ratings.elo import _margin_multiplier, expected_home_score


@dataclass(frozen=True)
class PooledParams:
    k: float = 20.0
    home_adv: float = 60.0
    season_regress: float = 0.25
    initial: float = 1500.0
    offsets: Mapping[str, float] = field(default_factory=lambda: {g: 0.0 for g in GROUPS})


def run_pooled_elo(
    matches: pd.DataFrame, team_group: dict[str, str], params: PooledParams = PooledParams()
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run Elo forward in date order over ``matches``.

    ``matches`` needs ``match_id, date, season, home, away, home_goals, away_goals, neutral``.
    Returns a frame indexed by ``match_id`` with ``elo_home_pre``/``elo_away_pre`` (ratings known
    before that match, never reflecting its own day) and every club's final rating. Rows with
    missing goals get pre-match ratings but never update a rating. At each season boundary the
    clubs seen last season regress toward their group's mean.
    """
    ordered = matches.sort_values(["date", "match_id"], kind="stable")
    ratings: dict[str, float] = {}
    seen: set[str] = set()
    season: str | None = None
    out_ids: list[str] = []
    home_pre: list[float] = []
    away_pre: list[float] = []

    def start(team: str) -> float:
        return params.initial + params.offsets.get(team_group.get(team, "OTHER"), 0.0)

    for _, day in ordered.groupby("date", sort=True):
        day_season = day["season"].iloc[0]
        if season is not None and day_season != season and seen:
            by_group: dict[str, list[str]] = {}
            for team in seen:
                by_group.setdefault(team_group.get(team, "OTHER"), []).append(team)
            for teams in by_group.values():
                mean = float(np.mean([ratings[t] for t in teams]))
                for t in teams:
                    ratings[t] -= params.season_regress * (ratings[t] - mean)
            seen = set()
        season = day_season

        pending: list[tuple[str, str, float]] = []
        for row in day.itertuples(index=False):
            for team in (row.home, row.away):
                ratings.setdefault(team, start(team))
                seen.add(team)
            r_home, r_away = ratings[row.home], ratings[row.away]
            out_ids.append(row.match_id)
            home_pre.append(r_home)
            away_pre.append(r_away)
            if pd.isna(row.home_goals) or pd.isna(row.away_goals):
                continue
            goal_diff = int(row.home_goals) - int(row.away_goals)
            score = 1.0 if goal_diff > 0 else 0.5 if goal_diff == 0 else 0.0
            adv = 0.0 if bool(row.neutral) else params.home_adv
            delta = params.k * _margin_multiplier(goal_diff) * (score - expected_home_score(r_home, r_away, adv))
            pending.append((row.home, row.away, delta))
        for home, away, delta in pending:
            ratings[home] += delta
            ratings[away] -= delta

    pre = pd.DataFrame(
        {"elo_home_pre": home_pre, "elo_away_pre": away_pre}, index=pd.Index(out_ids, name="match_id")
    )
    return pre, dict(ratings)


def _cross_group_error(
    matches: pd.DataFrame, team_group: dict[str, str], params: PooledParams, train_end: pd.Timestamp
) -> float:
    pre, _ = run_pooled_elo(matches, team_group, params)
    played = matches[matches["home_goals"].notna() & matches["away_goals"].notna() & (matches["date"] < train_end)]
    cross = played[
        played["home"].map(lambda t: team_group.get(t, "OTHER")) != played["away"].map(lambda t: team_group.get(t, "OTHER"))
    ]
    if cross.empty:
        return 0.0
    ratings = pre.loc[cross["match_id"]]
    adv = np.where(cross["neutral"].astype(bool), 0.0, params.home_adv)
    expected = 1.0 / (1.0 + 10.0 ** (-(ratings["elo_home_pre"].to_numpy() + adv - ratings["elo_away_pre"].to_numpy()) / 400.0))
    diff = cross["home_goals"].to_numpy() - cross["away_goals"].to_numpy()
    score = np.where(diff > 0, 1.0, np.where(diff == 0, 0.5, 0.0))
    return float(np.mean((expected - score) ** 2))


def fit_offsets(
    matches: pd.DataFrame,
    team_group: dict[str, str],
    train_end: pd.Timestamp,
    base: PooledParams = PooledParams(),
    grid: tuple[float, ...] = tuple(float(x) for x in range(-300, 301, 25)),
    passes: int = 2,
) -> dict[str, float]:
    """Fit one starting-rating offset per group by coordinate search; E0 is the anchor at 0.

    The objective is the mean squared error between expected and actual score (a draw counts 0.5)
    on matches before ``train_end`` whose clubs are in different groups. Elo has no draw
    probability, so log loss is not defined for it.
    """
    train_end = pd.Timestamp(train_end)
    offsets = {g: 0.0 for g in GROUPS}
    offsets.update({g: float(v) for g, v in dict(base.offsets).items()})
    offsets["E0"] = 0.0
    present = {team_group.get(t, "OTHER") for t in pd.concat([matches["home"], matches["away"]]).unique()}
    movable = [g for g in GROUPS if g != "E0" and g in present]
    for _ in range(passes):
        for group in movable:
            best_value, best_error = offsets[group], None
            for value in grid:
                trial = PooledParams(base.k, base.home_adv, base.season_regress, base.initial, {**offsets, group: value})
                error = _cross_group_error(matches, team_group, trial, train_end)
                if best_error is None or error < best_error - 1e-12:
                    best_value, best_error = value, error
            offsets[group] = best_value
    return offsets
