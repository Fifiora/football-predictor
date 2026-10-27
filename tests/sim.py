"""Synthetic league simulation shared by tests."""
import numpy as np
import pandas as pd

TEAMS = ["A", "B", "C", "D", "E", "F"]


def simulate_league(att, dfn, rounds, start="2025-08-01", seed=0, mu=0.15, hadv=0.25, league="E0"):
    """Round-robin of ordered pairs; one round per week. att/dfn: dict team -> strength."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(rounds):
        date = pd.Timestamp(start) + pd.Timedelta(days=7 * r)
        for h in TEAMS:
            for a in TEAMS:
                if h == a:
                    continue
                lam = np.exp(mu + hadv + att[h] - dfn[a])
                mu_away = np.exp(mu + att[a] - dfn[h])
                hg, ag = int(rng.poisson(lam)), int(rng.poisson(mu_away))
                rows.append(
                    (date, "2025-26", league, h, a, hg, ag,
                     hg * 3 + int(rng.integers(4, 10)), ag * 3 + int(rng.integers(4, 10)))
                )
    df = pd.DataFrame(
        rows,
        columns=["date", "season", "league", "home", "away", "home_goals", "away_goals",
                 "home_shots", "away_shots"],
    )
    df["result"] = np.where(df.home_goals > df.away_goals, "H", np.where(df.home_goals == df.away_goals, "D", "A"))
    df["home_sot"] = np.nan
    df["away_sot"] = np.nan
    df["close_h"] = np.nan
    df["close_d"] = np.nan
    df["close_a"] = np.nan
    df["match_id"] = [
        f"{lg}-{d:%Y%m%d}-{h}-{a}"
        for lg, d, h, a in zip(df["league"], df["date"], df["home"], df["away"])
    ]
    return df
