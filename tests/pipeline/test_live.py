import pandas as pd
import pytest

from footpred.ingest.fixtures import FixturesUnavailable
from footpred.pipeline.live import build_live, recorded_match_ids, still_busy, unrecorded_results, write_outputs
from footpred.predictions.log import LOG_COLUMNS

NOW = pd.Timestamp("2025-10-27 20:30")


def row(home, away, status, hg=None, ag=None, kickoff="2025-10-27T19:45:00Z"):
    return {"date": kickoff[:10], "kickoff": kickoff, "home": home, "away": away, "status": status,
            "home_goals": hg, "away_goals": ag, "stage": None}


class Client:
    def __init__(self, rows_by_league):
        self.rows_by_league = rows_by_league

    def get_fixtures(self, league, date_from, date_to):
        got = self.rows_by_league.get(league, [])
        if isinstance(got, Exception):
            raise got
        return got


def log_with(*picks):
    rows = [{"match_id": m, "league": m.split("-")[0], "home": "x", "away": "y", "kickoff_date": "2025-10-27", "model_version": "v1",
             "logged_at": "2025-10-27", "p_home": h, "p_draw": d, "p_away": a, "outcome": None} for m, h, d, a in picks]
    return pd.DataFrame(rows, columns=LOG_COLUMNS)


def live(world, rows_by_league, log=None, leagues=("E0",)):
    return build_live(Client(rows_by_league), list(leagues), NOW, log if log is not None else log_with(), aliases_path=world["aliases"])


def test_a_live_match_carries_its_score_and_whether_the_pick_is_ahead(world):
    log = log_with(("E0-20251027-A-B", 0.6, 0.25, 0.15))
    feed, _ = live(world, {"E0": [row("A FC", "B FC", "IN_PLAY", 1, 0)]}, log)
    (m,) = feed["matches"]
    assert m["state"] == "live" and (m["home_goals"], m["away_goals"]) == (1, 0)
    assert m["pick"] == "H" and m["pick_ok"] is True and m["p_home"] == pytest.approx(0.6)
    assert feed["complete"] is True and feed["updated_at"] == "2025-10-27T20:30:00Z"


def test_a_finished_match_says_whether_the_pick_was_right(world):
    log = log_with(("E0-20251027-A-B", 0.6, 0.25, 0.15), ("E0-20251027-C-D", 0.2, 0.3, 0.5))
    feed, _ = live(world, {"E0": [row("A FC", "B FC", "FINISHED", 0, 2), row("C FC", "D FC", "FINISHED", 1, 3, "2025-10-27T17:00:00Z")]}, log)
    by_id = {m["match_id"]: m for m in feed["matches"]}
    assert by_id["E0-20251027-A-B"]["pick_ok"] is False
    assert by_id["E0-20251027-C-D"]["pick_ok"] is True
    assert [m["match_id"] for m in feed["matches"]] == ["E0-20251027-C-D", "E0-20251027-A-B"]  # by kickoff


def test_an_upcoming_match_has_no_score_and_a_match_without_a_logged_prediction_has_no_pick(world):
    feed, _ = live(world, {"E0": [row("A FC", "B FC", "TIMED", None, None, "2025-10-27T21:00:00Z")]})
    (m,) = feed["matches"]
    assert m["state"] == "upcoming" and m["home_goals"] is None
    assert "pick" not in m and "pick_ok" not in m


def test_postponed_matches_are_left_out(world):
    feed, _ = live(world, {"E0": [row("A FC", "B FC", "POSTPONED")]})
    assert feed["matches"] == []


def test_a_competition_that_cannot_be_read_marks_the_feed_incomplete(world):
    feed, warnings = live(world, {"E0": [row("A FC", "B FC", "IN_PLAY", 0, 0)], "SP1": FixturesUnavailable("HTTP 429")}, leagues=("E0", "SP1"))
    assert feed["complete"] is False and len(feed["matches"]) == 1
    assert any("SP1" in w for w in warnings)


def test_an_unknown_team_is_skipped_with_a_warning(world):
    feed, warnings = live(world, {"E0": [row("Nobody FC", "B FC", "IN_PLAY", 0, 0)]})
    assert feed["matches"] == [] and any("Nobody" in w for w in warnings)


def test_finished_matches_not_yet_stored_are_reported(world):
    feed, _ = live(world, {"E0": [row("A FC", "B FC", "FINISHED", 1, 1), row("C FC", "D FC", "FINISHED", 2, 0, "2025-10-27T17:00:00Z")]})
    assert unrecorded_results(feed, {"E0-20251027-C-D"}) == ["E0-20251027-A-B"]
    assert unrecorded_results(feed, {"E0-20251027-C-D", "E0-20251027-A-B"}) == []


def test_the_workflow_keeps_polling_while_a_match_is_live_or_about_to_start(world):
    playing, _ = live(world, {"E0": [row("A FC", "B FC", "IN_PLAY", 0, 0)]})
    soon, _ = live(world, {"E0": [row("A FC", "B FC", "TIMED", kickoff="2025-10-27T22:00:00Z")]})
    later, _ = live(world, {"E0": [row("A FC", "B FC", "TIMED", kickoff="2025-10-28T19:45:00Z")]})
    over, _ = live(world, {"E0": [row("A FC", "B FC", "FINISHED", 1, 0)]})
    assert still_busy(playing, NOW) and still_busy(soon, NOW)
    assert not still_busy(later, NOW) and not still_busy(over, NOW)


def test_recorded_ids_come_from_stored_results_and_graded_predictions(tmp_path):
    pd.DataFrame({"match_id": ["E0-1"]}).to_csv(tmp_path / "api_results.csv", index=False)
    pd.DataFrame({"match_id": ["CL-1"]}).to_csv(tmp_path / "european_results.csv", index=False)
    pd.DataFrame({"match_id": ["E0-2", "E0-3"], "outcome": [1.0, None]}).to_csv(tmp_path / "prediction_log.csv", index=False)
    assert recorded_match_ids(tmp_path) == {"E0-1", "CL-1", "E0-2"}
    assert recorded_match_ids(tmp_path / "nothing") == set()


def test_outputs_are_written_as_json(tmp_path, world):
    import json

    feed, _ = live(world, {"E0": [row("A FC", "B FC", "IN_PLAY", 0, 0)]})
    write_outputs(tmp_path / "out", feed, ["E0-x"], True)
    assert json.loads((tmp_path / "out" / "live.json").read_text())["matches"][0]["state"] == "live"
    assert json.loads((tmp_path / "out" / "status.json").read_text()) == {"unrecorded": ["E0-x"], "busy": True, "live": 1}


def test_a_match_stuck_as_live_for_days_does_not_keep_the_poller_running(world):
    stuck, _ = live(world, {"E0": [row("A FC", "B FC", "PAUSED", 0, 0, "2025-10-25T19:45:00Z")]})
    assert not still_busy(stuck, NOW)
