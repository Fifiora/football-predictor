import numpy as np
import pandas as pd
import pytest

from footpred.predictions.log import grade_predictions, log_predictions


def preds(ids, kickoff, p=(0.5, 0.3, 0.2)):
    return pd.DataFrame(
        {
            "match_id": ids,
            "league": "E0",
            "home": "Arsenal",
            "away": "Chelsea",
            "kickoff_date": pd.Timestamp(kickoff),
            "p_home": p[0],
            "p_draw": p[1],
            "p_away": p[2],
        }
    )


LOGGED_AT = pd.Timestamp("2026-10-27 06:00")


def test_log_is_append_only_and_rejects_late_rows(tmp_path):
    path = tmp_path / "log.csv"
    log_predictions(path, preds(["a", "b"], "2026-10-30"), "v1", LOGGED_AT)
    log_predictions(path, preds(["c"], "2026-10-31"), "v1", LOGGED_AT)
    assert pd.read_csv(path)["match_id"].tolist() == ["a", "b", "c"]

    with pytest.raises(ValueError):
        log_predictions(path, preds(["d", "late"], "2026-10-27"), "v1", LOGGED_AT)
    assert pd.read_csv(path)["match_id"].tolist() == ["a", "b", "c"]  # nothing from the failed batch


def test_late_row_in_mixed_batch_rejects_whole_batch(tmp_path):
    path = tmp_path / "log.csv"
    batch = pd.concat([preds(["ok"], "2026-10-30"), preds(["late"], "2026-10-26")], ignore_index=True)
    with pytest.raises(ValueError):
        log_predictions(path, batch, "v1", LOGGED_AT)
    assert not path.exists()


def test_relogging_a_match_keeps_the_original_prediction(tmp_path):
    path = tmp_path / "log.csv"
    log_predictions(path, preds(["a"], "2026-10-30", p=(0.5, 0.3, 0.2)), "v1", LOGGED_AT)
    log_predictions(path, preds(["a", "b"], "2026-10-30", p=(0.1, 0.1, 0.8)), "v2", LOGGED_AT + pd.Timedelta(days=1))
    log = pd.read_csv(path)
    assert log["match_id"].tolist() == ["a", "b"]
    assert log.loc[0, "p_home"] == 0.5 and log.loc[0, "model_version"] == "v1"


def test_log_records_version_and_time(tmp_path):
    path = tmp_path / "log.csv"
    log_predictions(path, preds(["a"], "2026-10-30"), "v7", LOGGED_AT)
    row = pd.read_csv(path).iloc[0]
    assert row["model_version"] == "v7"
    assert pd.Timestamp(row["logged_at"]) == LOGGED_AT


def test_grade_fills_outcomes_without_mutating_probabilities(tmp_path):
    path = tmp_path / "log.csv"
    log_predictions(path, preds(["a", "b", "c"], "2026-10-30"), "v1", LOGGED_AT)
    results = pd.DataFrame({"match_id": ["a", "b", "zzz"], "result": ["A", "H", "D"]})

    graded = grade_predictions(path, results)
    assert graded.set_index("match_id").loc["a", "outcome"] == 2
    assert graded.set_index("match_id").loc["b", "outcome"] == 0
    assert np.isnan(graded.set_index("match_id").loc["c", "outcome"])
    assert (graded["p_home"] == 0.5).all() and (graded["p_draw"] == 0.3).all()

    on_disk = pd.read_csv(path)
    assert on_disk["outcome"].notna().sum() == 2
    assert len(on_disk) == 3


def test_grading_never_rewrites_an_existing_outcome(tmp_path):
    path = tmp_path / "log.csv"
    log_predictions(path, preds(["a"], "2026-10-30"), "v1", LOGGED_AT)
    grade_predictions(path, pd.DataFrame({"match_id": ["a"], "result": ["H"]}))
    graded = grade_predictions(path, pd.DataFrame({"match_id": ["a"], "result": ["A"]}))
    assert graded.loc[0, "outcome"] == 0


def test_grade_on_missing_log_returns_empty_frame(tmp_path):
    graded = grade_predictions(tmp_path / "none.csv", pd.DataFrame({"match_id": ["a"], "result": ["H"]}))
    assert graded.empty
