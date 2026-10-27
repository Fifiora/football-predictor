"""Synthetic league CSV world shared by pipeline and European tests."""
import numpy as np
import pandas as pd

from sim import TEAMS, simulate_league


def _strengths(seed):
    rng = np.random.default_rng(seed)
    att = {t: float(rng.normal(0, 0.35)) for t in TEAMS}
    dfn = {t: float(rng.normal(0, 0.35)) for t in TEAMS}
    return att, dfn


def write_world(root):
    raw = root / "raw"
    raw.mkdir(parents=True)
    for league, seed in (("E0", 1), ("SP1", 2)):
        att, dfn = _strengths(seed)  # same strengths across seasons so there is signal to learn
        for k, start in enumerate(("2024-08-01", "2025-08-01")):
            sim = simulate_league(att, dfn, rounds=10, start=start, seed=seed * 10 + k, league=league)
            csv = pd.DataFrame(
                {
                    "Div": league,
                    "Date": sim["date"].dt.strftime("%d/%m/%Y"),
                    "HomeTeam": sim["home"], "AwayTeam": sim["away"],
                    "FTHG": sim["home_goals"], "FTAG": sim["away_goals"], "FTR": sim["result"],
                    "HS": sim["home_shots"], "AS": sim["away_shots"],
                }
            )
            csv.to_csv(raw / f"{league}_{k}.csv", index=False)
    rows = ["league,alias,canonical"]
    for league in ("E0", "SP1"):
        for t in TEAMS:
            rows += [f"{league},{t},{t}", f"{league},{t} FC,{t}"]
    aliases = root / "aliases.csv"
    aliases.write_text("\n".join(rows) + "\n")
    return raw, aliases
