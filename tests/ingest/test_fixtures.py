import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from footpred.ingest.fixtures import (
    FixturesUnavailable,
    FootballDataOrgClient,
    fetch_latest_results,
    fetch_upcoming,
)
from footpred.ingest.schema import MATCH_COLUMNS

SAMPLE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "api_fixtures_sample.json").read_text())
TODAY = pd.Timestamp("2025-10-27 06:00")
NO_SLEEP = lambda seconds: None  # noqa: E731


def fx(day, home, away, status="SCHEDULED", hg=None, ag=None):
    return {"date": day, "home": home, "away": away, "status": status, "home_goals": hg, "away_goals": ag}


class FakeClient:
    def __init__(self, fixtures=None, fail_times=0):
        self.fixtures = fixtures or []
        self.fail_times = fail_times
        self.calls = 0

    def get_fixtures(self, league, date_from, date_to):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionError("temporary failure")
        return [f for f in self.fixtures if league == "E0"]


UPCOMING = [
    fx("2025-10-30", "Arsenal FC", "Chelsea FC"),
    fx("2025-10-31", "Ghost FC", "Everton FC"),
    fx("2025-11-01", "Liverpool FC", "Fulham FC"),
    fx("2025-11-02", "Burnley FC", "Leeds United FC", status="POSTPONED"),
]


def test_unmapped_team_blocks_only_that_fixture(tmp_path):
    df, warnings = fetch_upcoming(FakeClient(UPCOMING), ["E0"], tmp_path, today=TODAY, sleep=NO_SLEEP)
    assert sorted(df["home"]) == ["Arsenal", "Liverpool"]
    assert any("Ghost FC" in w for w in warnings)


def test_only_scheduled_fixtures_are_upcoming(tmp_path):
    df, _ = fetch_upcoming(FakeClient(UPCOMING), ["E0"], tmp_path, today=TODAY, sleep=NO_SLEEP)
    assert "Burnley" not in set(df["home"])
    assert list(df.columns) == ["match_id", "league", "date", "home", "away"]
    assert df.loc[df["home"] == "Arsenal", "match_id"].iloc[0] == "E0-20251030-Arsenal-Chelsea"


def test_empty_response_returns_empty_frame_without_error(tmp_path):
    df, warnings = fetch_upcoming(FakeClient([]), ["E0", "SP1"], tmp_path, today=TODAY, sleep=NO_SLEEP)
    assert df.empty and list(df.columns) == ["match_id", "league", "date", "home", "away"]
    assert warnings == []


def test_cache_avoids_repeat_calls_within_ttl(tmp_path):
    client = FakeClient(UPCOMING)
    fetch_upcoming(client, ["E0"], tmp_path, today=TODAY, sleep=NO_SLEEP)
    first = client.calls
    fetch_upcoming(client, ["E0"], tmp_path, today=TODAY + pd.Timedelta(minutes=10), sleep=NO_SLEEP)
    assert client.calls == first


def test_cache_expires_so_matchday_night_run_sees_new_data(tmp_path):
    client = FakeClient(UPCOMING)
    fetch_upcoming(client, ["E0"], tmp_path, today=TODAY, sleep=NO_SLEEP)
    first = client.calls
    fetch_upcoming(client, ["E0"], tmp_path, today=TODAY + pd.Timedelta(hours=8), sleep=NO_SLEEP)
    assert client.calls > first


def test_retries_with_backoff_on_failure(tmp_path):
    delays = []
    client = FakeClient(UPCOMING, fail_times=2)
    df, _ = fetch_upcoming(client, ["E0"], tmp_path, today=TODAY, sleep=delays.append, attempts=3)
    assert len(df) == 2
    assert client.calls == 3
    assert len(delays) == 2 and delays[1] > delays[0]


def test_always_failing_client_raises_fixtures_unavailable(tmp_path):
    with pytest.raises(FixturesUnavailable):
        fetch_upcoming(FakeClient(UPCOMING, fail_times=99), ["E0"], tmp_path, today=TODAY, sleep=NO_SLEEP, attempts=3)


def test_latest_results_match_schema(tmp_path):
    finished = [
        fx("2025-10-25", "Liverpool FC", "Manchester United FC", status="FINISHED", hg=3, ag=1),
        fx("2025-10-26", "Arsenal FC", "Chelsea FC", status="FINISHED", hg=0, ag=0),
        fx("2025-10-26", "Ghost FC", "Everton FC", status="FINISHED", hg=1, ag=0),
        fx("2025-10-27", "Burnley FC", "Fulham FC", status="IN_PLAY", hg=1, ag=0),
    ]
    df, warnings = fetch_latest_results(
        FakeClient(finished), ["E0"], since=pd.Timestamp("2025-10-20"), cache_dir=tmp_path, today=TODAY, sleep=NO_SLEEP
    )
    assert list(df.columns) == MATCH_COLUMNS
    assert sorted(df["result"]) == ["D", "H"]
    liverpool = df[df["home"] == "Liverpool"].iloc[0]
    assert liverpool["home_goals"] == 3 and liverpool["away"] == "Manchester United"
    assert liverpool["season"] == "2025-26" and liverpool["match_id"] == "E0-20251025-Liverpool-Manchester United"
    assert any("Ghost FC" in w for w in warnings)


def test_latest_results_empty_is_fine(tmp_path):
    df, warnings = fetch_latest_results(
        FakeClient([]), ["E0"], since=pd.Timestamp("2025-10-20"), cache_dir=tmp_path, today=TODAY, sleep=NO_SLEEP
    )
    assert df.empty and list(df.columns) == MATCH_COLUMNS and warnings == []


# --- real-provider adapter, exercised with an injected HTTP function -------------------------


def test_football_data_org_adapter_normalizes_and_authenticates():
    seen = {}

    def http_get(url, headers, params):
        seen.update(url=url, headers=headers, params=params)
        return 200, SAMPLE

    client = FootballDataOrgClient("secret-token", http_get=http_get)
    rows = client.get_fixtures("E0", date(2025, 10, 27), date(2025, 11, 10))

    assert seen["url"].endswith("/v4/competitions/PL/matches")
    assert seen["headers"]["X-Auth-Token"] == "secret-token"
    assert seen["params"] == {"dateFrom": "2025-10-27", "dateTo": "2025-11-10"}
    assert rows[0] == fx("2025-10-30", "Arsenal FC", "Chelsea FC", status="TIMED")
    assert rows[1]["home_goals"] == 3 and rows[1]["away_goals"] == 1 and rows[1]["status"] == "FINISHED"


def test_football_data_org_adapter_rejects_unknown_league_and_http_errors():
    client = FootballDataOrgClient("t", http_get=lambda *a: (429, {"message": "slow down"}))
    with pytest.raises(FixturesUnavailable):
        client.get_fixtures("E0", date(2025, 10, 27), date(2025, 11, 10))
    with pytest.raises(ValueError):
        FootballDataOrgClient("t", http_get=lambda *a: (200, SAMPLE)).get_fixtures("ZZ9", date(2025, 10, 27), date(2025, 11, 10))
