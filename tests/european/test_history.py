import subprocess

import pandas as pd

from footpred.european.history import (
    EURO_COLUMNS,
    append_european_results,
    download_openfootball,
    load_european_history,
    load_european_matches,
)
from footpred.european.names import EuropeanNames
from footpred.ingest.teams import canonical_team

CL = """= UEFA Champions League 2024/25

▪ League, Matchday 1
  Tue Sep 17 2024
    18:45  BSC Young Boys (SUI)    v Aston Villa FC (ENG)     0-3 (0-2)
           Juventus FC (ITA)       v PSV (NED)                3-1 (2-0)
▪ Finals, Round of 16
  Tue Mar 11 2025
    21:00  Liverpool FC (ENG)      v Paris Saint-Germain FC (FRA)  1-4 pen. 0-1 a.e.t. (0-1, 0-1)
"""
EL = """= UEFA Europa League 2024/25

▪ League, Matchday 1
  Thu Sep 26 2024
    18:45  AS Roma (ITA)           v Athletic Club (ESP)      1-1 (0-0)
"""


def world(tmp_path):
    root = tmp_path / "openfootball"
    (root / "2024-25").mkdir(parents=True)
    (root / "2024-25" / "cl.txt").write_text(CL, encoding="utf-8")
    (root / "2024-25" / "el.txt").write_text(EL, encoding="utf-8")
    (root / "README.md").write_text("ignored")
    return root


def row(match_id, competition, day, home, away, hg, ag):
    return {"match_id": match_id, "competition": competition, "season": "2024-25", "stage": None, "date": pd.Timestamp(day),
            "home": home, "away": away, "home_country": None, "away_country": None, "home_goals": hg, "away_goals": ag,
            "result": "H" if hg > ag else "A" if hg < ag else "D"}


def test_history_loads_every_competition_file_and_canonicalises(tmp_path):
    df = load_european_history(world(tmp_path), EuropeanNames())
    assert list(df.columns) == EURO_COLUMNS
    assert set(df["competition"]) == {"CL", "EL"} and len(df) == 4
    juve = df[df["home"] == canonical_team("Juventus FC", "I1")].iloc[0]
    assert juve["match_id"] == f"CL-20240917-{juve['home']}-PSV" and (juve["home_goals"], juve["away_goals"]) == (3, 1)
    pens = df[df["stage"] == "LAST_16"].iloc[0]
    assert (pens["home_goals"], pens["away_goals"], pens["result"]) == (0, 1, "A")


def test_missing_history_directory_gives_an_empty_frame(tmp_path):
    df = load_european_history(tmp_path / "nope", EuropeanNames())
    assert df.empty and list(df.columns) == EURO_COLUMNS


def test_feed_row_one_day_off_is_not_double_counted(tmp_path):
    root = world(tmp_path)
    names = EuropeanNames()
    juve = canonical_team("Juventus FC", "I1")
    append_european_results(tmp_path, pd.DataFrame([row(f"CL-20240918-{juve}-PSV", "CL", "2024-09-18", juve, "PSV", 3, 1)]))
    df = load_european_matches(tmp_path, names)
    assert len(df[(df["home"] == juve) & (df["away"] == "PSV")]) == 1
    assert df[(df["home"] == juve)].iloc[0]["match_id"].startswith("CL-20240917")  # the history row wins


def test_feed_row_far_apart_is_kept(tmp_path):
    world(tmp_path)
    juve = canonical_team("Juventus FC", "I1")
    append_european_results(tmp_path, pd.DataFrame([row(f"CL-20241008-{juve}-PSV", "CL", "2024-10-08", juve, "PSV", 0, 0)]))
    df = load_european_matches(tmp_path, EuropeanNames())
    assert len(df[(df["home"] == juve) & (df["away"] == "PSV")]) == 2


def test_different_competition_same_pair_is_not_a_duplicate(tmp_path):
    world(tmp_path)
    juve = canonical_team("Juventus FC", "I1")
    append_european_results(tmp_path, pd.DataFrame([row(f"EL-20240917-{juve}-PSV", "EL", "2024-09-17", juve, "PSV", 1, 0)]))
    assert len(load_european_matches(tmp_path, EuropeanNames())) == 5


def test_append_is_idempotent_and_atomic(tmp_path):
    new = pd.DataFrame([row("CL-20261104-A-B", "CL", "2026-11-04", "A", "B", 2, 1)])
    append_european_results(tmp_path, new)
    append_european_results(tmp_path, new)
    saved = pd.read_csv(tmp_path / "european_results.csv")
    assert len(saved) == 1 and not list(tmp_path.glob("*.tmp"))


def test_download_clones_then_pulls_and_reports_failure_without_raising(tmp_path):
    calls = []

    def ok(cmd, **kwargs):
        calls.append(cmd)
        if cmd[1] == "clone":
            (tmp_path / "of" / ".git").mkdir(parents=True)
        return subprocess.CompletedProcess(cmd, 0)

    assert download_openfootball(tmp_path / "of", run=ok) is True
    assert download_openfootball(tmp_path / "of", run=ok) is True
    assert calls[0][1] == "clone" and "pull" in calls[1]

    def broken(cmd, **kwargs):
        raise OSError("no git")

    assert download_openfootball(tmp_path / "other", run=broken) is False
    assert download_openfootball(tmp_path / "again", run=lambda c, **k: (_ for _ in ()).throw(subprocess.CalledProcessError(128, c))) is False
