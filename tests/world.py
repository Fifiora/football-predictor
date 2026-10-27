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


STRONG = ["A", "B", "C", "D", "E", "F"]
OTHERS = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
SEASONS = [2021, 2022, 2023, 2024, 2025]


def write_openfootball(root, spread=1.2, seed=3, seasons=SEASONS):
    """Five seasons of synthetic Champions League results with real strength differences."""
    rng = np.random.default_rng(seed)
    strength = {t: float(rng.normal(0, spread)) for t in STRONG + OTHERS}
    country = {**{t: "ENG" for t in STRONG}, **{t: "POR" for t in OTHERS}}
    for year in seasons:
        lines = [f"= UEFA Champions League {year}/{str(year + 1)[-2:]}", "", "▪ League, Matchday 1"]
        def play(count, first_day, step, home_edge):
            for i in range(count):
                home, away = rng.choice(STRONG + OTHERS, size=2, replace=False)
                gap = strength[home] - strength[away] + home_edge
                hg = int(rng.poisson(np.exp(0.2 + 0.4 * gap)))
                ag = int(rng.poisson(np.exp(0.2 - 0.4 * gap)))
                date = first_day + pd.Timedelta(days=int(i * step))
                lines.append(f"  {date:%a %b} {date.day} {date.year}")
                lines.append(f"    21:00  {home} ({country[home]})  v {away} ({country[away]})  {hg}-{ag} (0-0)")

        play(160, pd.Timestamp(f"{year}-09-10"), 250 / 160, 0.3)
        lines.append("▪ Finals, Round of 16")  # knockout home edge is much larger: a signal only the stage flag can carry
        play(60, pd.Timestamp(f"{year + 1}-03-01"), 1.0, 1.6)
        folder = root / "openfootball" / f"{year}-{str(year + 1)[-2:]}"
        folder.mkdir(parents=True)
        (folder / "cl.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
