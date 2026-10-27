import numpy as np
import pandas as pd
import pytest

from footpred.european.pooled import PooledParams, fit_offsets, run_pooled_elo

GROUPS = {"A": "E0", "B": "OTHER", "C": "E0", "D": "OTHER"}


def frame(rows):
    cols = ["match_id", "date", "season", "home", "away", "home_goals", "away_goals", "neutral"]
    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_pre_ratings_ignore_same_day_and_future_results():
    base = [
        ("m1", "2024-09-01", "2024-25", "A", "B", 2, 0, False),
        ("m2", "2024-09-01", "2024-25", "C", "D", 1, 0, False),
        ("m3", "2024-09-08", "2024-25", "A", "C", 0, 0, False),
        ("m4", "2024-09-15", "2024-25", "B", "D", 1, 1, False),
    ]
    changed = [list(r) for r in base]
    changed[0][5], changed[0][6] = 0, 3  # flip m1's result
    pre_a, _ = run_pooled_elo(frame(base), GROUPS)
    pre_b, _ = run_pooled_elo(frame(changed), GROUPS)
    assert pre_a.loc["m1"].equals(pre_b.loc["m1"])
    assert pre_a.loc["m2"].equals(pre_b.loc["m2"])      # same day as the changed result
    assert not pre_a.loc["m3"].equals(pre_b.loc["m3"])  # later: reflects it


def test_cross_group_result_moves_both_clubs_the_right_way():
    _, ratings = run_pooled_elo(frame([("m1", "2024-09-01", "2024-25", "A", "B", 0, 1, True)]), GROUPS)
    assert ratings["B"] > 1500 > ratings["A"]
    assert ratings["B"] - 1500 == pytest.approx(1500 - ratings["A"])


def test_new_club_starts_at_its_group_offset():
    params = PooledParams(offsets={"E0": 0.0, "SP1": 0.0, "I1": 0.0, "D1": 0.0, "F1": 0.0, "OTHER": -100.0})
    pre, _ = run_pooled_elo(frame([("m1", "2024-09-01", "2024-25", "A", "B", 1, 1, True)]), GROUPS, params)
    assert pre.loc["m1", "elo_home_pre"] == 1500 and pre.loc["m1", "elo_away_pre"] == 1400


def test_neutral_match_has_no_home_advantage():
    draw = lambda neutral: frame([("m1", "2024-09-01", "2024-25", "A", "C", 0, 0, neutral)])
    _, neutral = run_pooled_elo(draw(True), GROUPS)
    _, home = run_pooled_elo(draw(False), GROUPS)
    assert neutral["A"] == neutral["C"] == 1500
    assert home["A"] < 1500 < home["C"]


def test_unplayed_rows_get_pre_ratings_but_never_update():
    rows = [
        ("m1", "2024-09-01", "2024-25", "A", "C", 1, 0, False),
        ("m2", "2024-09-08", "2024-25", "A", "C", np.nan, np.nan, False),
    ]
    pre, ratings = run_pooled_elo(frame(rows), GROUPS)
    _, after_m1 = run_pooled_elo(frame(rows[:1]), GROUPS)
    assert "m2" in pre.index and ratings == after_m1


def test_season_boundary_pulls_ratings_toward_the_group_mean():
    rows = [
        ("m1", "2024-09-01", "2024-25", "A", "C", 4, 0, False),
        ("m2", "2025-09-01", "2025-26", "A", "C", 0, 0, False),
    ]
    pre, _ = run_pooled_elo(frame(rows), GROUPS)
    gap_before = abs(pre.loc["m2", "elo_home_pre"] - pre.loc["m2", "elo_away_pre"])
    _, after = run_pooled_elo(frame(rows[:1]), GROUPS)
    assert gap_before < abs(after["A"] - after["C"])


def test_fit_offsets_recovers_a_weaker_group():
    rows = []
    for i in range(60):  # E0 clubs keep beating OTHER clubs at neutral venues
        home, away = ("A", "B") if i % 2 == 0 else ("C", "D")
        rows.append((f"x{i}", f"2024-0{1 + i // 20}-{1 + i % 20:02d}", "2023-24", home, away, 3, 0, True))
    offsets = fit_offsets(frame(rows), GROUPS, train_end=pd.Timestamp("2024-12-31"))
    assert offsets["E0"] == 0.0 and offsets["OTHER"] < 0
