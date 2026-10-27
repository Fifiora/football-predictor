"""A small synthetic world: 2 leagues x 2 seasons x 6 teams, as football-data.co.uk style CSVs."""
import numpy as np
import pandas as pd
import pytest

from sim import TEAMS, simulate_league  # noqa: F401
from world import write_world

TODAY = pd.Timestamp("2025-10-27")
NO_SLEEP = lambda seconds: None  # noqa: E731


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
