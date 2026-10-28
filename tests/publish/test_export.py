import json

import numpy as np
import pandas as pd
import pytest

from footpred.explain.reasons import Driver
from footpred.publish import export as export_mod
from footpred.publish.export import export_site_data

NOW = pd.Timestamp("2025-10-27 06:00")


def make_inputs():
    fixtures = pd.DataFrame(
        {
            "match_id": ["E0-20251030-Arsenal-Chelsea", "E0-20251031-Fulham-Everton"],
            "league": ["E0", "E0"],
            "date": pd.to_datetime(["2025-10-30", "2025-10-31"]),
            "home": ["Arsenal", "Fulham"],
            "away": ["Chelsea", "Everton"],
            "p_home": [0.5, 0.3],
            "p_draw": [0.3, 0.3],
            "p_away": [0.2, 0.4],
        }
    )
    grid = np.zeros((9, 9))
    grid[0, 0] = 0.2   # 0-0
    grid[1, 1] = 0.3   # 1-1  (both score, 2 goals)
    grid[2, 1] = 0.2   # 2-1  (both score, 3 goals)
    grid[3, 0] = 0.2   # 3-0  (over 2.5, clean sheet)
    grid[0, 1] = 0.1   # 0-1
    grids = {fixtures.match_id[0]: grid, fixtures.match_id[1]: grid}
    drivers = {fixtures.match_id[0]: [Driver("elo_diff", "home", 0.3, "The model weighted a big Elo gap which points to Arsenal.")]}
    graded = pd.DataFrame(
        {
            "match_id": ["a", "b", "c", "d"],
            "league": ["E0", "E0", "SP1", "E0"],
            "home": "x", "away": "y",
            "kickoff_date": "2025-10-01",
            "model_version": "v1",
            "logged_at": "2025-09-30",
            "p_home": [0.6, 0.2, 0.5, 0.9],
            "p_draw": [0.2, 0.3, 0.3, 0.05],
            "p_away": [0.2, 0.5, 0.2, 0.05],
            "outcome": [0.0, 2.0, 1.0, np.nan],
        }
    )
    return fixtures, drivers, grids, graded, {"Arsenal": 1600.0, "Chelsea": 1550.0}


def run(out_dir, **overrides):
    fixtures, drivers, grids, graded, ratings = make_inputs()
    export_site_data(out_dir, fixtures, drivers, grids, graded, ratings, NOW, model_version="v1")
    return out_dir


def load(out_dir, name):
    return json.loads((out_dir / name).read_text())


def test_exports_expected_files_and_valid_json(tmp_path):
    out = run(tmp_path / "data")
    for name in ("fixtures.json", "track_record.json", "teams.json", "meta.json"):
        assert (out / name).exists()
        load(out, name)
    meta = load(out, "meta.json")
    assert meta["model_version"] == "v1" and meta["updated_at"].startswith("2025-10-27")


def test_fixture_probabilities_sum_to_one(tmp_path):
    fixtures = load(run(tmp_path / "data"), "fixtures.json")
    assert len(fixtures) == 2
    for f in fixtures:
        assert abs(f["p_home"] + f["p_draw"] + f["p_away"] - 1) < 1e-6
    assert fixtures[0]["drivers"][0]["text"].startswith("The model weighted")
    assert fixtures[1]["drivers"] == []


def test_scoreline_derived_fields(tmp_path):
    f = load(run(tmp_path / "data"), "fixtures.json")[0]
    assert f["btts"] == pytest.approx(0.5)       # 1-1 and 2-1
    assert f["over_2_5"] == pytest.approx(0.4)   # 2-1 and 3-0
    assert f["top_scorelines"][0]["score"] == "1-1"


def test_track_record_ignores_ungraded_rows(tmp_path):
    tr = load(run(tmp_path / "data"), "track_record.json")
    assert tr["n_graded"] == 3
    assert tr["accuracy"] == pytest.approx(2 / 3)
    expected_ll = -np.mean(np.log([0.6, 0.5, 0.3]))
    assert tr["log_loss"] == pytest.approx(expected_ll)
    assert set(tr["by_league"]) == {"E0", "SP1"} and tr["by_league"]["E0"]["n"] == 2
    assert len(tr["calibration"]) > 0


def test_track_record_with_no_graded_rows_is_empty_not_error(tmp_path):
    fixtures, drivers, grids, graded, ratings = make_inputs()
    export_site_data(tmp_path / "data", fixtures, drivers, grids, graded.iloc[0:0], ratings, NOW, model_version="v1")
    assert load(tmp_path / "data", "track_record.json")["n_graded"] == 0


def test_export_is_atomic(tmp_path, monkeypatch):
    out = run(tmp_path / "data")
    before = {p.name: p.read_text() for p in out.iterdir()}

    real = export_mod._write_json
    calls = {"n": 0}

    def flaky(path, obj):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("disk full")
        real(path, obj)

    monkeypatch.setattr(export_mod, "_write_json", flaky)
    with pytest.raises(OSError):
        run(out)
    assert {p.name: p.read_text() for p in out.iterdir()} == before


def test_bookmaker_odds_are_exported_only_for_fixtures_that_have_them(tmp_path):
    fixtures, drivers, grids, graded, ratings = make_inputs()
    odds = {fixtures.match_id[0]: {"average": (0.5, 0.3, 0.2), "bet365": (0.52, 0.28, 0.2)}}
    export_site_data(tmp_path / "s", fixtures, drivers, grids, graded, ratings, NOW, model_version="v1", odds=odds)
    first, second = load(tmp_path / "s", "fixtures.json")
    assert first["bookies"]["average"] == {"p_home": 0.5, "p_draw": 0.3, "p_away": 0.2}
    assert set(first["bookies"]) == {"average", "bet365"}
    assert "bookies" not in second


def test_track_record_lists_recent_graded_picks_newest_first(tmp_path):
    out = run(tmp_path / "s")
    recent = load(out, "track_record.json")["recent"]
    assert len(recent) == 3  # the ungraded row is left out
    assert set(recent[0]) >= {"match_id", "league", "home", "away", "date", "pick", "result", "correct", "p_pick"}
    by_id = {r["match_id"]: r for r in recent}
    assert by_id["a"]["pick"] == "H" and by_id["a"]["result"] == "H" and by_id["a"]["correct"] is True
    assert by_id["b"]["pick"] == "A" and by_id["b"]["correct"] is True
    assert by_id["c"]["pick"] == "H" and by_id["c"]["result"] == "D" and by_id["c"]["correct"] is False


def test_benchmark_reference_is_passed_through_to_the_track_record(tmp_path):
    fixtures, drivers, grids, graded, ratings = make_inputs()
    ref = {"holdout_season": "2025-26", "holdout": {"n": 10}, "development": None}
    export_site_data(tmp_path / "s", fixtures, drivers, grids, graded, ratings, NOW, reference=ref)
    assert load(tmp_path / "s", "track_record.json")["benchmark"] == ref
    export_site_data(tmp_path / "t", fixtures, drivers, grids, graded, ratings, NOW)
    assert "benchmark" not in load(tmp_path / "t", "track_record.json")


def test_kickoff_time_is_exported_when_known(tmp_path):
    fixtures, drivers, grids, graded, ratings = make_inputs()
    fixtures = fixtures.assign(kickoff=["2025-10-30T19:45:00Z", None])
    export_site_data(tmp_path / "s", fixtures, drivers, grids, graded, ratings, NOW)
    first, second = load(tmp_path / "s", "fixtures.json")
    assert first["kickoff"] == "2025-10-30T19:45:00Z"
    assert "kickoff" not in second
