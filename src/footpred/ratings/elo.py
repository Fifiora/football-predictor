"""Elo ratings computed forward in time, recording each match's pre-kickoff ratings."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EloParams:
    k: float = 20.0
    home_adv: float = 60.0
    season_regress: float = 0.25
    initial: float = 1500.0


def expected_home_score(home_rating: float, away_rating: float, home_adv: float) -> float:
    diff = home_rating + home_adv - away_rating
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def _margin_multiplier(goal_diff: int) -> float:
    return math.log(abs(goal_diff) + 1.0) + 1.0 if goal_diff != 0 else 1.0


def run_elo(
    matches: pd.DataFrame, params: EloParams = EloParams()
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run Elo over ``matches`` in date order.

    Returns a frame indexed by ``match_id`` with ``elo_home_pre``/``elo_away_pre`` (ratings
    known before that match) and the final rating of every team. Rows with missing goals
    (postponed or upcoming) get pre-match ratings but never update any rating.

    At each league's season boundary, ratings regress toward that league's mean. Teams that
    were not in the league the previous season (promoted or returning) restart at the
    previous season's league mean.
    """
    ordered = matches.sort_values(["date", "match_id"], kind="stable")

    ratings: dict[str, float] = {}
    season_of: dict[str, str] = {}  # league -> season currently being processed
    season_teams: dict[str, set[str]] = {}  # league -> teams seen in the current season
    league_prior: dict[str, float] = {}  # league -> start rating for newcomers
    team_league: dict[str, str] = {}

    out_ids: list[str] = []
    home_pre: list[float] = []
    away_pre: list[float] = []

    for _, day in ordered.groupby("date", sort=True):
        # Pre-ratings for the whole day are recorded before any of the day's results are
        # applied, so a rating only ever reflects matches from strictly earlier dates.
        pending: list[tuple[str, str, float]] = []
        for row in day.itertuples(index=False):
            league = row.league
            if season_of.get(league) != row.season:
                if league in season_of:  # a season ended: regress toward the league mean
                    prev = season_teams[league]
                    mean = float(np.mean([ratings[t] for t in prev]))
                    for team in prev:
                        ratings[team] -= params.season_regress * (ratings[team] - mean)
                    league_prior[league] = mean
                    for team in [t for t in ratings if team_league[t] == league and t not in prev]:
                        del ratings[team]  # absent last season: restarts at the league prior if it returns
                season_of[league] = row.season
                season_teams[league] = set()

            start = league_prior.get(league, params.initial)
            for team in (row.home, row.away):
                ratings.setdefault(team, start)
                team_league[team] = league
                season_teams[league].add(team)

            r_home, r_away = ratings[row.home], ratings[row.away]
            out_ids.append(row.match_id)
            home_pre.append(r_home)
            away_pre.append(r_away)

            if pd.isna(row.home_goals) or pd.isna(row.away_goals):
                continue

            goal_diff = int(row.home_goals) - int(row.away_goals)
            score = 1.0 if goal_diff > 0 else 0.5 if goal_diff == 0 else 0.0
            expected = expected_home_score(r_home, r_away, params.home_adv)
            delta = params.k * _margin_multiplier(goal_diff) * (score - expected)
            pending.append((row.home, row.away, delta))

        for home, away, delta in pending:
            ratings[home] += delta
            ratings[away] -= delta

    pre = pd.DataFrame(
        {"elo_home_pre": home_pre, "elo_away_pre": away_pre},
        index=pd.Index(out_ids, name="match_id"),
    )
    return pre, dict(ratings)
