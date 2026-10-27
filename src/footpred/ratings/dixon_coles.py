"""Dixon-Coles goals model: time-weighted Poisson attack/defense strengths per league.

Log-rates for a match are ``mu + hadv + att[home] - dfn[away]`` (home goals) and
``mu + att[away] - dfn[home]`` (away goals). ``att`` and ``dfn`` each sum to zero within a
league, so an unseen team (att = dfn = 0) is a league-average team. The Dixon-Coles
correction ``tau`` with parameter ``rho`` adjusts the 0-0, 0-1, 1-0 and 1-1 scorelines.
Leagues are fitted independently because the big five never play each other, so their
strengths are not comparable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

GRID_SIZE = 9  # scorelines 0..8 for each side
RHO_BOUND = 0.2
RIDGE = 0.05


class DCUnavailable(ValueError):
    """Raised when no fitted league model can price the requested match."""


@dataclass(frozen=True)
class DCPrediction:
    exp_home_goals: float
    exp_away_goals: float
    p_home: float
    p_draw: float
    p_away: float
    grid: np.ndarray


@dataclass(frozen=True)
class _LeagueFit:
    mu: float
    hadv: float
    rho: float
    att: dict[str, float]
    dfn: dict[str, float]


@dataclass(frozen=True)
class _Data:
    teams: list[str]
    hi: np.ndarray
    ai: np.ndarray
    hg: np.ndarray
    ag: np.ndarray
    w: np.ndarray


def _prepare(league_matches: pd.DataFrame, as_of: pd.Timestamp, xi: float) -> _Data:
    played = league_matches[
        (league_matches["date"] < as_of)
        & league_matches["home_goals"].notna()
        & league_matches["away_goals"].notna()
    ]
    teams = sorted(set(played["home"]) | set(played["away"]))
    index = {t: i for i, t in enumerate(teams)}
    days_ago = (as_of - played["date"]).dt.days.to_numpy(dtype=float)
    return _Data(
        teams=teams,
        hi=played["home"].map(index).to_numpy(dtype=int),
        ai=played["away"].map(index).to_numpy(dtype=int),
        hg=played["home_goals"].to_numpy(dtype=int),
        ag=played["away_goals"].to_numpy(dtype=int),
        w=np.exp(-xi * days_ago),
    )


def _neg_ll_and_grad(x: np.ndarray, data: _Data) -> tuple[float, np.ndarray]:
    n = len(data.teams)
    mu, hadv, rho = x[0], x[1], x[2 + 2 * n]
    a, d = x[2 : 2 + n], x[2 + n : 2 + 2 * n]
    att, dfn = a - a.mean(), d - d.mean()
    hi, ai, hg, ag, w = data.hi, data.ai, data.hg, data.ag, data.w

    eta_l = mu + hadv + att[hi] - dfn[ai]
    eta_m = mu + att[ai] - dfn[hi]
    lam, mrate = np.exp(eta_l), np.exp(eta_m)

    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)
    tau = np.ones_like(lam)
    tau[m00] = 1.0 - lam[m00] * mrate[m00] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m10] = 1.0 + mrate[m10] * rho
    tau[m11] = 1.0 - rho
    tau = np.maximum(tau, 1e-10)

    ll = w * (np.log(tau) + hg * eta_l - lam + ag * eta_m - mrate)
    f = -ll.sum() + 0.5 * RIDGE * (a @ a + d @ d)

    gl = (hg - lam).astype(float)
    gm = (ag - mrate).astype(float)
    gl[m00] += lam[m00] * (-mrate[m00] * rho / tau[m00])
    gl[m01] += lam[m01] * (rho / tau[m01])
    gm[m00] += mrate[m00] * (-lam[m00] * rho / tau[m00])
    gm[m10] += mrate[m10] * (rho / tau[m10])

    g_rho = np.zeros_like(lam)
    g_rho[m00] = -lam[m00] * mrate[m00] / tau[m00]
    g_rho[m01] = lam[m01] / tau[m01]
    g_rho[m10] = mrate[m10] / tau[m10]
    g_rho[m11] = -1.0 / tau[m11]

    wl, wm = w * gl, w * gm
    g_att = np.bincount(hi, wl, n) + np.bincount(ai, wm, n)
    g_dfn = -np.bincount(ai, wl, n) - np.bincount(hi, wm, n)

    grad = np.empty_like(x)
    grad[0] = -(wl + wm).sum()
    grad[1] = -wl.sum()
    grad[2 : 2 + n] = -(g_att - g_att.mean()) + RIDGE * a
    grad[2 + n : 2 + 2 * n] = -(g_dfn - g_dfn.mean()) + RIDGE * d
    grad[2 + 2 * n] = -(w * g_rho).sum()
    return float(f), grad


def _fit_league(data: _Data) -> _LeagueFit:
    n = len(data.teams)
    mean_goals = max(float((data.w * (data.hg + data.ag)).sum() / (2 * data.w.sum())), 0.1)
    x0 = np.zeros(3 + 2 * n)
    x0[0] = np.log(mean_goals)
    bounds = [(None, None)] * (2 + 2 * n) + [(-RHO_BOUND, RHO_BOUND)]
    res = minimize(_neg_ll_and_grad, x0, args=(data,), jac=True, method="L-BFGS-B", bounds=bounds)
    x = res.x
    a, d = x[2 : 2 + n], x[2 + n : 2 + 2 * n]
    att, dfn = a - a.mean(), d - d.mean()
    return _LeagueFit(
        mu=float(x[0]),
        hadv=float(x[1]),
        rho=float(x[2 + 2 * n]),
        att=dict(zip(data.teams, map(float, att))),
        dfn=dict(zip(data.teams, map(float, dfn))),
    )


def _scoreline_grid(lam: float, mrate: float, rho: float) -> np.ndarray:
    goals = np.arange(GRID_SIZE)
    grid = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mrate))
    grid[0, 0] *= 1.0 - lam * mrate * rho
    grid[0, 1] *= 1.0 + lam * rho
    grid[1, 0] *= 1.0 + mrate * rho
    grid[1, 1] *= 1.0 - rho
    grid = np.maximum(grid, 0.0)
    return grid / grid.sum()


class DCModel:
    def __init__(self, fits: dict[str, _LeagueFit]):
        self._fits = fits
        self._team_league = {team: lg for lg, fit in fits.items() for team in fit.att}

    def has_league(self, league: str) -> bool:
        return league in self._fits

    def predict(self, home: str, away: str, league: str | None = None) -> DCPrediction:
        league = league or self._team_league.get(home) or self._team_league.get(away)
        if league is None or league not in self._fits:
            raise DCUnavailable(f"no fitted league for {home!r} vs {away!r} (league={league!r})")
        fit = self._fits[league]
        lam = np.exp(fit.mu + fit.hadv + fit.att.get(home, 0.0) - fit.dfn.get(away, 0.0))
        mrate = np.exp(fit.mu + fit.att.get(away, 0.0) - fit.dfn.get(home, 0.0))
        grid = _scoreline_grid(float(lam), float(mrate), fit.rho)
        return DCPrediction(
            exp_home_goals=float(lam),
            exp_away_goals=float(mrate),
            p_home=float(np.tril(grid, -1).sum()),
            p_draw=float(np.trace(grid)),
            p_away=float(np.triu(grid, 1).sum()),
            grid=grid,
        )


def fit_dixon_coles(
    matches: pd.DataFrame, as_of: pd.Timestamp, xi: float = 0.0018, min_matches: int = 50
) -> DCModel:
    """Fit every league using only played matches dated strictly before ``as_of``."""
    fits: dict[str, _LeagueFit] = {}
    for league, group in matches.groupby("league"):
        data = _prepare(group, as_of, xi)
        if len(data.hg) >= min_matches:
            fits[str(league)] = _fit_league(data)
    return DCModel(fits)


DC_COLUMNS = ["dc_exp_home", "dc_exp_away", "dc_p_home", "dc_p_draw", "dc_p_away"]


def dc_pre_match(
    matches: pd.DataFrame, refit_days: int = 7, xi: float = 0.0018, min_matches: int = 50
) -> pd.DataFrame:
    """Pre-kickoff Dixon-Coles predictions for every match, safe against look-ahead.

    The model is refitted at most every ``refit_days`` days, always with ``as_of`` set to a
    match date, so each row only ever sees matches from strictly earlier dates. Rows whose
    league has fewer than ``min_matches`` earlier played matches are NaN.
    """
    ordered = matches.sort_values(["date", "match_id"], kind="stable")
    out = pd.DataFrame(np.nan, index=pd.Index(ordered["match_id"], name="match_id"), columns=DC_COLUMNS)

    model: DCModel | None = None
    last_fit: pd.Timestamp | None = None
    for date, day in ordered.groupby("date", sort=True):
        if last_fit is None or (date - last_fit).days >= refit_days:
            model = fit_dixon_coles(ordered, date, xi=xi, min_matches=min_matches)
            last_fit = date
        for row in day.itertuples(index=False):
            try:
                pred = model.predict(row.home, row.away, league=row.league)
            except DCUnavailable:
                continue
            out.loc[row.match_id] = [
                pred.exp_home_goals,
                pred.exp_away_goals,
                pred.p_home,
                pred.p_draw,
                pred.p_away,
            ]
    return out
