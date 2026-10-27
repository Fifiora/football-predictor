import pandas as pd
import pytest

from footpred.ingest.schema import MATCH_COLUMNS, SchemaError, validate_matches


def _row(**overrides):
    row = {
        "match_id": "E0-20250815-Liverpool-Bournemouth",
        "league": "E0",
        "season": "2025-26",
        "date": pd.Timestamp("2025-08-15"),
        "home": "Liverpool",
        "away": "Bournemouth",
        "home_goals": 4.0,
        "away_goals": 2.0,
        "result": "H",
        "home_shots": 22.0,
        "away_shots": 10.0,
        "home_sot": 9.0,
        "away_sot": 4.0,
        "close_h": 1.3,
        "close_d": 6.0,
        "close_a": 10.0,
    }
    row.update(overrides)
    return row


def _frame(*rows):
    return pd.DataFrame(list(rows), columns=MATCH_COLUMNS)


def test_validate_rejects_missing_column():
    df = _frame(_row()).drop(columns=["home_goals"])
    with pytest.raises(SchemaError):
        validate_matches(df)


def test_validate_drops_unplayed_rows():
    unplayed = _row(
        match_id="E0-20250920-Fulham-Chelsea",
        home="Fulham",
        away="Chelsea",
        home_goals=float("nan"),
        away_goals=float("nan"),
        result=None,
    )
    out = validate_matches(_frame(_row(), unplayed))
    assert list(out["match_id"]) == ["E0-20250815-Liverpool-Bournemouth"]


def test_validate_keeps_unplayed_rows_when_allowed():
    unplayed = _row(
        match_id="E0-20250920-Fulham-Chelsea",
        home="Fulham",
        away="Chelsea",
        home_goals=float("nan"),
        away_goals=float("nan"),
        result=None,
    )
    out = validate_matches(_frame(_row(), unplayed), allow_unplayed=True)
    assert len(out) == 2


def test_validate_rejects_duplicate_match_ids():
    with pytest.raises(SchemaError):
        validate_matches(_frame(_row(), _row()))


def test_validate_rejects_bad_result_label():
    with pytest.raises(SchemaError):
        validate_matches(_frame(_row(result="X")))
