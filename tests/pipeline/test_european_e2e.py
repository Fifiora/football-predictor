"""Champions League track through the daily and retrain jobs, on synthetic leagues plus synthetic European history."""
import json

import pandas as pd
import pytest
from conftest import NO_SLEEP, TODAY, FakeClient, fx
from world import write_openfootball

from footpred.models.registry import load_champion
from footpred.pipeline import european as euro_mod
from footpred.pipeline import daily as daily_mod
from footpred.pipeline.daily import run_daily
from footpred.pipeline.european import run_european_retrain
from footpred.pipeline.retrain import run_retrain


def cl(day, home, away, status="SCHEDULED", hg=None, ag=None, stage="LEAGUE_STAGE"):
    return {**fx(day, home, away, status, hg, ag), "stage": stage}


class ClClient(FakeClient):
    """Serves league rows like the shared FakeClient, plus Champions League rows."""

    def __init__(self, fixtures, cl_rows):
        super().__init__(fixtures)
        self.cl_rows = cl_rows

    def get_fixtures(self, league, date_from, date_to):
        return self.cl_rows if league == "CL" else super().get_fixtures(league, date_from, date_to)


@pytest.fixture
def w(world):
    write_openfootball(world["data_dir"], seasons=[2021, 2022, 2023, 2024, 2025])
    world["registry_euro"] = world["registry"] / "european"
    return world


def client_with(w, *cl_rows):
    return ClClient(w["client"].fixtures, list(cl_rows))


def daily(w, client, european=True, today=TODAY, site=None):
    return run_daily(
        w["data_dir"], site or w["site"], client, today, registry_dir=w["registry"],
        aliases_path=w["aliases"], sleep=NO_SLEEP, european=european,
    )


def seed(w):
    run_retrain(w["data_dir"], w["registry"], TODAY, aliases_path=w["aliases"])
    return run_european_retrain(w["data_dir"], w["registry"], TODAY, aliases_path=w["aliases"])


def fixtures_json(w, site=None):
    return json.loads(((site or w["site"]) / "fixtures.json").read_text())


def test_retrain_promotes_first_model_only_when_gate_passes(w, monkeypatch):
    monkeypatch.setattr(euro_mod, "european_publish_gate", lambda dev: (False, "forced failure"))
    blocked = run_european_retrain(w["data_dir"], w["registry"], TODAY, aliases_path=w["aliases"])
    assert not blocked.promote and "forced failure" in blocked.reason
    with pytest.raises(FileNotFoundError):
        load_champion(w["registry_euro"])

    monkeypatch.undo()
    passed = run_european_retrain(w["data_dir"], w["registry"], TODAY, aliases_path=w["aliases"])
    assert passed.promote
    version, _, _ = load_champion(w["registry_euro"])
    meta = json.loads((w["registry_euro"] / version / "metadata.json").read_text())
    assert meta["publish_gate"]["ok"] is True and set(meta["offsets"]) >= {"E0", "OTHER"}


def test_daily_publishes_a_champions_league_fixture_with_reasons(w):
    seed(w)
    result = daily(w, client_with(w, cl("2025-10-30", "A FC", "Q1")))
    assert result.published
    rows = {f["match_id"]: f for f in fixtures_json(w)}
    row = rows["CL-20251030-A-Q1"]
    assert row["league"] == "CL" and abs(row["p_home"] + row["p_draw"] + row["p_away"] - 1) < 1e-6
    assert 1 <= len(row["drivers"]) <= 3
    assert "E0-20251030-A-B" in rows  # league fixtures are still there
    log = pd.read_csv(w["data_dir"] / "prediction_log.csv")
    assert "CL-20251030-A-Q1" in set(log["match_id"])


def test_unmatched_club_does_not_stop_the_run_and_is_warned(w):
    seed(w)
    result = daily(w, client_with(w, cl("2025-10-30", "A FC", "Q1"), cl("2025-10-31", "Mystery United", "Q2")))
    assert result.published
    ids = {f["match_id"] for f in fixtures_json(w)}
    assert {"CL-20251030-A-Q1", "CL-20251031-Mystery United-Q2"} <= ids  # kept at the default rating
    assert any("Mystery United" in text for text in result.warnings)
    assert not any("Q1" in text or "Q2" in text for text in result.warnings)  # known history clubs are not flagged
    rows = {f["match_id"]: f for f in fixtures_json(w)}
    assert rows["CL-20251031-Mystery United-Q2"]["low_confidence"] is True
    assert "low_confidence" not in rows["CL-20251030-A-Q1"] and "low_confidence" not in rows["E0-20251030-A-B"]


def test_failed_publish_gate_hides_european_fixtures_only(w, monkeypatch):
    monkeypatch.setattr(euro_mod, "european_publish_gate", lambda dev: (False, "forced failure"))
    run_retrain(w["data_dir"], w["registry"], TODAY, aliases_path=w["aliases"])
    run_european_retrain(w["data_dir"], w["registry"], TODAY, aliases_path=w["aliases"])
    result = daily(w, client_with(w, cl("2025-10-30", "A FC", "Q1")))
    assert result.published
    leagues = {f["league"] for f in fixtures_json(w)}
    assert leagues == {"E0"} and any("gate" in text.lower() for text in result.warnings)


def test_league_predictions_are_unchanged_by_european_data(w, tmp_path):
    seed(w)
    daily(w, client_with(w, cl("2025-10-30", "A FC", "Q1")), european=True, site=tmp_path / "with")
    daily(w, w["client"], european=False, site=tmp_path / "without")
    with_euro = {f["match_id"]: f for f in fixtures_json(w, tmp_path / "with") if f["league"] != "CL"}
    without = {f["match_id"]: f for f in fixtures_json(w, tmp_path / "without")}
    assert with_euro == without
    assert json.loads((tmp_path / "with" / "teams.json").read_text()) == json.loads((tmp_path / "without" / "teams.json").read_text())


def test_european_results_are_graded_in_the_track_record(w):
    seed(w)
    daily(w, client_with(w, cl("2025-10-30", "A FC", "Q1")))
    later = TODAY + pd.Timedelta(days=5)
    result = daily(w, client_with(w, cl("2025-10-30", "A FC", "Q1", status="FINISHED", hg=2, ag=0)), today=later)
    assert result.new_results >= 1
    record = json.loads((w["site"] / "track_record.json").read_text())
    assert record["by_league"]["CL"]["n"] == 1
    feed = pd.read_csv(w["data_dir"] / "european_results.csv")
    assert feed["match_id"].tolist() == ["CL-20251030-A-Q1"] and feed["result"].tolist() == ["H"]


def test_european_off_is_the_old_behaviour(w):
    run_retrain(w["data_dir"], w["registry"], TODAY, aliases_path=w["aliases"])
    result = daily(w, client_with(w, cl("2025-10-30", "A FC", "Q1")), european=False)
    assert {f["league"] for f in fixtures_json(w)} == {"E0"} and result.warnings == []


def test_european_feed_outage_does_not_stop_league_run(w):
    seed(w)

    class Broken(ClClient):
        def get_fixtures(self, league, date_from, date_to):
            if league == "CL":
                from footpred.ingest.fixtures import FixturesUnavailable
                raise FixturesUnavailable("CL is down")
            return FakeClient.get_fixtures(self, league, date_from, date_to)

    result = daily(w, Broken(w["client"].fixtures, []))
    assert result.published and {f["league"] for f in fixtures_json(w)} == {"E0"}
    assert any("Champions League" in text for text in result.warnings)


def test_any_european_failure_leaves_the_league_run_intact(w):
    seed(w)
    bad = w["data_dir"] / "openfootball" / "2025-26" / "cl.txt"
    bad.write_text(bad.read_text(encoding="utf-8") + "\n▪ Mystery Phase, Matchday 9\n", encoding="utf-8")
    result = daily(w, client_with(w, cl("2025-10-30", "A FC", "Q1")))
    assert result.published and {f["league"] for f in fixtures_json(w)} == {"E0"}
    assert any("Champions League" in text and "Mystery Phase" in text for text in result.warnings)


def test_a_feed_spelling_that_differs_only_in_form_matches_the_history_club(w):
    seed(w)
    result = daily(w, client_with(w, cl("2025-10-30", "A FC", "Q1 FC")))  # history has "Q1"
    ids = {f["match_id"] for f in fixtures_json(w)}
    assert "CL-20251030-A-Q1" in ids and not any("Q1" in text for text in result.warnings)


def test_stored_feed_rows_follow_later_alias_fixes(w):
    from footpred.european.history import append_european_results
    from footpred.european.names import EuropeanNames
    from footpred.european.history import load_european_matches

    row = {"match_id": "CL-20250918-Q1 FC-A", "competition": "CL", "season": "2025-26", "stage": None,
           "date": pd.Timestamp("2025-09-18"), "home": "Q1 FC", "away": "A", "home_country": None, "away_country": None,
           "home_goals": 1.0, "away_goals": 1.0, "result": "D"}
    append_european_results(w["data_dir"], pd.DataFrame([row]))
    df = load_european_matches(w["data_dir"], EuropeanNames(league_aliases_path=w["aliases"]))
    assert "Q1 FC" not in set(df["home"]) and "Q1" in set(df["home"])


def test_champions_league_predictions_are_logged_under_the_european_model_version(w):
    seed(w)
    daily(w, client_with(w, cl("2025-10-30", "A FC", "Q1")))
    log = pd.read_csv(w["data_dir"] / "prediction_log.csv").set_index("match_id")
    euro_version = load_champion(w["registry_euro"])[0]
    league_version = load_champion(w["registry"])[0]
    assert log.loc["CL-20251030-A-Q1", "model_version"] == f"european/{euro_version}"
    assert log.loc["E0-20251030-A-B", "model_version"] == league_version


def test_league_drift_check_ignores_champions_league_grades(w, monkeypatch):
    seen = []
    graded = pd.DataFrame({"match_id": ["a", "b"], "league": ["E0", "CL"], "home": ["x", "y"], "away": ["p", "q"], "outcome": [0.0, 1.0], "kickoff_date": ["2025-10-01"] * 2,
                           "p_home": [.5, .4], "p_draw": [.3, .3], "p_away": [.2, .3]})
    monkeypatch.setattr(daily_mod, "grade_predictions", lambda *a, **k: graded)
    monkeypatch.setattr(daily_mod, "check_drift", lambda g, **kw: seen.append(set(g["league"])) or [])
    seed(w)
    daily(w, client_with(w, cl("2025-10-30", "A FC", "Q1")))
    assert seen == [{"E0"}]
