import pandas as pd
import pytest

from footpred.ingest.odds import devig, parse_fixture_odds

CSV = (
    "﻿Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A,AvgH,AvgD,AvgA\n"
    "E0,03/11/2026,20:00,Arsenal,Chelsea,2.0,3.5,4.0,2.1,3.4,3.9\n"
    "E0,03/11/2026,20:00,Nowhere FC,Chelsea,2.0,3.5,4.0,2.1,3.4,3.9\n"
    "E0,04/11/2026,20:00,Chelsea,Arsenal,,,,,,\n"
)


def test_devig_removes_the_margin_and_sums_to_one():
    p = devig(2.0, 3.5, 4.0)
    assert sum(p) == pytest.approx(1.0)
    assert p[0] > p[1] > p[2]
    assert devig(None, 3.0, 3.0) is None
    assert devig(0.9, 3.0, 3.0) is None  # odds at or below 1 are not prices


def test_parse_maps_names_handles_bom_and_skips_unknown_clubs(tmp_path):
    aliases = tmp_path / "a.csv"
    aliases.write_text("league,alias,canonical\nE0,Arsenal,Arsenal\nE0,Chelsea,Chelsea\n")
    frame, warnings = parse_fixture_odds(CSV, aliases_path=aliases)
    assert list(frame["match_id"]) == ["E0-20261103-Arsenal-Chelsea"]  # unknown club and odds-less row dropped
    row = frame.iloc[0]
    assert row["average"] == pytest.approx(devig(2.1, 3.4, 3.9))
    assert row["bet365"] == pytest.approx(devig(2.0, 3.5, 4.0))
    assert any("Nowhere FC" in w for w in warnings)


def test_average_falls_back_to_bet365_when_missing(tmp_path):
    aliases = tmp_path / "a.csv"
    aliases.write_text("league,alias,canonical\nE0,Arsenal,Arsenal\nE0,Chelsea,Chelsea\n")
    text = "Div,Date,HomeTeam,AwayTeam,B365H,B365D,B365A\nE0,03/11/2026,Arsenal,Chelsea,2.0,3.5,4.0\n"
    frame, _ = parse_fixture_odds(text, aliases_path=aliases)
    assert frame.iloc[0]["average"] is None and frame.iloc[0]["bet365"] is not None


def test_divisions_without_aliases_are_ignored_quietly(tmp_path):
    aliases = tmp_path / "a.csv"
    aliases.write_text("league,alias,canonical\nE0,Arsenal,Arsenal\nE0,Chelsea,Chelsea\n")
    text = ("Div,Date,HomeTeam,AwayTeam,B365H,B365D,B365A\nSC0,03/11/2026,Celtic,Dundee,2.0,3.5,4.0\n"
            "E0,03/11/2026,Arsenal,Chelsea,2.0,3.5,4.0\n")
    frame, warnings = parse_fixture_odds(text, aliases_path=aliases)
    assert len(frame) == 1 and warnings == []
