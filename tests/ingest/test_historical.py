from pathlib import Path

import pandas as pd
import pytest

from footpred.ingest.historical import load_historical
from footpred.ingest.schema import MATCH_COLUMNS
from footpred.ingest.teams import UnknownTeamError

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture
def raw_dir(tmp_path):
    (tmp_path / "E0_sample.csv").write_text((FIXTURES / "E0_sample.csv").read_text())
    return tmp_path


def test_load_historical_sample(raw_dir):
    df = load_historical(raw_dir)
    assert list(df.columns) == MATCH_COLUMNS
    assert set(df["result"]) <= {"H", "D", "A"}
    assert set(df["league"]) == {"E0"}
    assert df["match_id"].is_unique


def test_load_historical_drops_postponed_and_sorts(raw_dir):
    df = load_historical(raw_dir)
    assert len(df) == 10  # 11 rows in the sample, 1 has no score
    assert df["date"].is_monotonic_increasing
    assert not ((df["home"] == "Fulham") & (df["away"] == "Chelsea")).any()


def test_load_historical_values_and_season(raw_dir):
    df = load_historical(raw_dir)
    row = df[df["match_id"] == "E0-20250815-Liverpool-Bournemouth"].iloc[0]
    assert row["home_goals"] == 4 and row["away_goals"] == 2
    assert row["season"] == "2025-26"
    assert row["home_shots"] == 22 and row["away_sot"] == 4
    assert row["close_h"] == pytest.approx(1.30)
    assert pd.Timestamp("2025-08-15") == row["date"]


def test_load_historical_canonicalizes_names(raw_dir):
    df = load_historical(raw_dir)
    assert "Man United" not in set(df["home"]) | set(df["away"])
    assert "Manchester United" in set(df["home"]) | set(df["away"])


def test_load_historical_lists_all_unknown_teams(tmp_path):
    (tmp_path / "bad.csv").write_text(
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
        "E0,15/08/2025,Ghost FC,Arsenal,1,0,H\n"
        "E0,16/08/2025,Arsenal,Phantom Utd,2,2,D\n"
    )
    with pytest.raises(UnknownTeamError) as exc:
        load_historical(tmp_path)
    assert "Ghost FC" in str(exc.value) and "Phantom Utd" in str(exc.value)


def test_loader_handles_bom_and_latin1_files(tmp_path):
    from footpred.ingest.historical import load_historical

    raw = tmp_path / "raw"
    raw.mkdir()
    aliases = tmp_path / "aliases.csv"
    aliases.write_text("league,alias,canonical\nSP1,Málaga,Málaga\nSP1,Betis,Betis\n", encoding="utf-8")
    header = "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
    (raw / "SP1_bom.csv").write_bytes(("﻿" + header + "SP1,17/08/2014,Málaga,Betis,1,0,H\n").encode("utf-8"))
    (raw / "SP1_latin.csv").write_bytes((header + "SP1,24/08/2014,Betis,Málaga,2,2,D\n").encode("latin-1"))
    out = load_historical(raw, aliases)
    assert sorted(out["result"]) == ["D", "H"] and set(out["home"]) == {"Málaga", "Betis"}
