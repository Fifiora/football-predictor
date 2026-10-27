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
    assert list(df.columns) == ["match_id", "league", "date", "home", "away", "stage"]
    assert df.loc[df["home"] == "Arsenal", "match_id"].iloc[0] == "E0-20251030-Arsenal-Chelsea"


def test_empty_response_returns_empty_frame_without_error(tmp_path):
    df, warnings = fetch_upcoming(FakeClient([]), ["E0", "SP1"], tmp_path, today=TODAY, sleep=NO_SLEEP)
    assert df.empty and list(df.columns) == ["match_id", "league", "date", "home", "away", "stage"]
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
    assert rows[0] == {**fx("2025-10-30", "Arsenal FC", "Chelsea FC", status="TIMED"), "stage": None}
    assert rows[1]["home_goals"] == 3 and rows[1]["away_goals"] == 1 and rows[1]["status"] == "FINISHED"


def test_football_data_org_adapter_rejects_unknown_league_and_http_errors():
    client = FootballDataOrgClient("t", http_get=lambda *a: (429, {"message": "slow down"}))
    with pytest.raises(FixturesUnavailable):
        client.get_fixtures("E0", date(2025, 10, 27), date(2025, 11, 10))
    with pytest.raises(ValueError):
        FootballDataOrgClient("t", http_get=lambda *a: (200, SAMPLE)).get_fixtures("ZZ9", date(2025, 10, 27), date(2025, 11, 10))


def test_football_data_org_lists_competition_team_names():
    seen = {}

    def http_get(url, headers, params):
        seen["url"] = url
        return 200, {"teams": [{"name": "FC Barcelona"}, {"name": "Real Betis Balompié"}]}

    client = FootballDataOrgClient("t", http_get=http_get)
    assert client.get_team_names("SP1") == ["FC Barcelona", "Real Betis Balompié"]
    assert seen["url"].endswith("/v4/competitions/PD/teams")


def cl_match(day, home, away, status="TIMED", stage="LEAGUE_STAGE", full=(None, None), duration="REGULAR", regular=None):
    score = {"duration": duration, "fullTime": {"home": full[0], "away": full[1]}}
    if regular:
        score["regularTime"] = {"home": regular[0], "away": regular[1]}
    return {"utcDate": f"{day}T19:00:00Z", "status": status, "stage": stage,
            "homeTeam": {"name": home}, "awayTeam": {"name": away}, "score": score}


def cl_client(matches):
    seen = {}

    def http_get(url, headers, params):
        seen["url"] = url
        return 200, {"matches": matches}

    return FootballDataOrgClient("token", http_get=http_get), seen


def test_champions_league_fixtures_are_fetched_and_staged():
    client, seen = cl_client([cl_match("2026-11-04", "Arsenal FC", "Sport Lisboa e Benfica", stage="LAST_16")])
    rows = client.get_fixtures("CL", date(2026, 11, 1), date(2026, 11, 10))
    assert seen["url"].endswith("/v4/competitions/CL/matches")
    assert rows[0]["stage"] == "LAST_16"


@pytest.mark.parametrize("raw,mapped", [
    ("LEAGUE_STAGE", "LEAGUE_STAGE"), ("GROUP_STAGE", "LEAGUE_STAGE"), ("PLAYOFFS", "PLAYOFFS"),
    ("LAST_32", "PLAYOFFS"), ("LAST_16", "LAST_16"), ("QUARTER_FINALS", "QUARTER_FINALS"),
    ("SEMI_FINALS", "SEMI_FINALS"), ("FINAL", "FINAL"), ("THIRD_PLACE", None), (None, None),
])
def test_stage_is_mapped_to_our_stage_names(raw, mapped):
    client, _ = cl_client([cl_match("2026-11-04", "A", "B", stage=raw)])
    assert client.get_fixtures("CL", date(2026, 11, 1), date(2026, 11, 10))[0]["stage"] == mapped


def test_results_use_the_ninety_minute_score_when_the_match_went_long():
    matches = [
        cl_match("2026-11-04", "A", "B", status="FINISHED", full=(5, 4), duration="PENALTY_SHOOTOUT", regular=(1, 1)),
        cl_match("2026-11-05", "C", "D", status="FINISHED", full=(3, 2), duration="EXTRA_TIME", regular=(2, 2)),
        cl_match("2026-11-06", "E", "F", status="FINISHED", full=(2, 0)),
    ]
    client, _ = cl_client(matches)
    rows = client.get_fixtures("CL", date(2026, 11, 1), date(2026, 11, 10))
    assert [(r["home_goals"], r["away_goals"]) for r in rows] == [(1, 1), (2, 2), (2, 0)]


class CLClient:
    def __init__(self, fixtures):
        self.fixtures = fixtures

    def get_fixtures(self, league, date_from, date_to):
        return [f for f in self.fixtures if league == "CL"]


def test_canonical_hook_is_used_and_never_raises_for_unknown_clubs(tmp_path):
    from footpred.european.names import EuropeanNames

    names = EuropeanNames()
    rows = [dict(fx("2025-10-30", "Arsenal FC", "Qarabağ FK"), stage="LEAGUE_STAGE")]
    df, warnings = fetch_upcoming(CLClient(rows), ["CL"], tmp_path, today=TODAY, sleep=NO_SLEEP, canonical=names.canonical)
    assert df["match_id"].tolist() == ["CL-20251030-Arsenal-Qarabağ FK"] and df["stage"].tolist() == ["LEAGUE_STAGE"]
    assert warnings == [] and "Qarabağ FK" in names.unmatched


def test_without_a_hook_an_unknown_club_in_a_league_still_skips_the_fixture(tmp_path):
    df, warnings = fetch_upcoming(CLClient([fx("2025-10-30", "Arsenal FC", "Qarabağ FK")]), ["CL"], tmp_path, today=TODAY, sleep=NO_SLEEP)
    assert df.empty and any("Qarabağ FK" in w for w in warnings)


def test_latest_results_also_use_the_hook(tmp_path):
    from footpred.european.names import EuropeanNames

    rows = [fx("2025-10-25", "Arsenal FC", "Qarabağ FK", status="FINISHED", hg=2, ag=0)]
    df, _ = fetch_latest_results(
        CLClient(rows), ["CL"], pd.Timestamp("2025-10-20"), tmp_path, today=TODAY, sleep=NO_SLEEP,
        canonical=EuropeanNames().canonical,
    )
    assert df["home"].tolist() == ["Arsenal"] and df["away"].tolist() == ["Qarabağ FK"] and df["league"].tolist() == ["CL"]


def test_league_fixtures_have_no_stage(tmp_path):
    df, _ = fetch_upcoming(FakeClient(UPCOMING), ["E0"], tmp_path, today=TODAY, sleep=NO_SLEEP)
    assert df["stage"].isna().all()
