"""A small synthetic world: 2 leagues x 2 seasons x 6 teams, as football-data.co.uk style CSVs."""
import numpy as np
import pandas as pd
import pytest

from sim import TEAMS, simulate_league

TODAY = pd.Timestamp("2025-10-27")
NO_SLEEP = lambda seconds: None  # noqa: E731


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


def fx(day, home, away, status="SCHEDULED", hg=None, ag=None):
    return {"date": day, "home": home, "away": away, "status": status, "home_goals": hg, "away_goals": ag}


class FakeClient:
    def __init__(self, fixtures):
        self.fixtures = fixtures

    def get_fixtures(self, league, date_from, date_to):
        return [f for f in self.fixtures if league == "E0"]


@pytest.fixture
def world(tmp_path):
    data_dir = tmp_path / "data"
    raw, aliases = write_world(data_dir)
    return {
        "data_dir": data_dir,
        "aliases": aliases,
        "registry": tmp_path / "registry",
        "site": tmp_path / "site_data",
        "client": FakeClient(
            [
                fx("2025-10-30", "A FC", "B FC"),
                fx("2025-10-31", "C FC", "D FC"),
                fx("2025-10-25", "E FC", "F FC", status="FINISHED", hg=2, ag=1),
            ]
        ),
        "empty_client": FakeClient([]),
    }
