import numpy as np
import pandas as pd
import pytest

from footpred.ratings.elo import EloParams, run_elo


def make_matches(rows):
    """rows: (date, season, league, home, away, home_goals, away_goals)"""
    df = pd.DataFrame(
        rows,
        columns=["date", "season", "league", "home", "away", "home_goals", "away_goals"],
    )
    df["date"] = pd.to_datetime(df["date"])
    df["match_id"] = [
        f"{lg}-{d:%Y%m%d}-{h}-{a}"
        for lg, d, h, a in zip(df["league"], df["date"], df["home"], df["away"])
    ]
    return df


SEASON_1 = [
    ("2025-08-16", "2025-26", "E0", "A", "B", 2, 0),
    ("2025-08-16", "2025-26", "E0", "C", "D", 1, 1),
    ("2025-08-23", "2025-26", "E0", "A", "C", 3, 1),
    ("2025-08-23", "2025-26", "E0", "B", "D", 0, 2),
    ("2025-08-30", "2025-26", "E0", "D", "A", 1, 2),
    ("2025-08-30", "2025-26", "E0", "B", "C", 0, 0),
]


def test_pre_ratings_exclude_own_result():
    matches = make_matches([("2025-08-16", "2025-26", "E0", "A", "B", 5, 0)])
    pre, _ = run_elo(matches)
    row = pre.loc[matches["match_id"][0]]
    assert row["elo_home_pre"] == EloParams().initial
    assert row["elo_away_pre"] == EloParams().initial


def test_winner_gains_loser_loses():
    matches = make_matches([("2025-08-16", "2025-26", "E0", "A", "B", 2, 0)])
    _, final = run_elo(matches)
    initial = EloParams().initial
    assert final["A"] > initial > final["B"]
    assert final["A"] - initial == pytest.approx(initial - final["B"])


def test_season_regression():
    s1 = make_matches(SEASON_1)
    _, final_s1 = run_elo(s1)
    mean_s1 = np.mean([final_s1[t] for t in "ABCD"])
    params = EloParams(season_regress=0.25)

    s2_match = ("2026-08-15", "2026-27", "E0", "A", "B", 1, 0)
    both = make_matches(SEASON_1 + [s2_match])
    pre, _ = run_elo(both, params)
    row = pre.loc[both["match_id"].iloc[-1]]

    expected_a = final_s1["A"] - 0.25 * (final_s1["A"] - mean_s1)
    expected_b = final_s1["B"] - 0.25 * (final_s1["B"] - mean_s1)
    assert row["elo_home_pre"] == pytest.approx(expected_a)
    assert row["elo_away_pre"] == pytest.approx(expected_b)


# Season 2: D is relegated and E is promoted, so the league's mean rating drifts from 1500.
SEASON_2 = [
    ("2026-08-15", "2026-27", "E0", "A", "E", 2, 1),
    ("2026-08-15", "2026-27", "E0", "B", "C", 0, 1),
    ("2026-08-22", "2026-27", "E0", "E", "B", 1, 1),
    ("2026-08-22", "2026-27", "E0", "C", "A", 0, 2),
    ("2026-08-29", "2026-27", "E0", "A", "B", 3, 0),
    ("2026-08-29", "2026-27", "E0", "C", "E", 2, 0),
]


def test_new_team_gets_league_prior():
    _, final_s2 = run_elo(make_matches(SEASON_1 + SEASON_2))
    mean_s2 = np.mean([final_s2[t] for t in "ABCE"])
    assert abs(mean_s2 - EloParams().initial) > 1.0  # the prior must differ from the default

    both = make_matches(SEASON_1 + SEASON_2 + [("2027-08-14", "2027-28", "E0", "F", "A", 0, 1)])
    pre, _ = run_elo(both)
    row = pre.loc[both["match_id"].iloc[-1]]
    assert row["elo_home_pre"] == pytest.approx(mean_s2)


def test_returning_team_restarts_at_league_prior():
    s1_s2 = make_matches(SEASON_1 + SEASON_2)
    _, final_s2 = run_elo(s1_s2)
    mean_s2 = np.mean([final_s2[t] for t in "ABCE"])
    stale_d = final_s2["D"]
    assert abs(stale_d - mean_s2) > 1.0

    both = make_matches(SEASON_1 + SEASON_2 + [("2027-08-14", "2027-28", "E0", "D", "A", 0, 1)])
    pre, _ = run_elo(both)
    row = pre.loc[both["match_id"].iloc[-1]]
    assert row["elo_home_pre"] == pytest.approx(mean_s2)


def test_no_future_leak():
    full = make_matches(SEASON_1)
    pre_full, _ = run_elo(full)

    altered = full.copy()
    altered.loc[altered.index[-1], ["home_goals", "away_goals"]] = [9, 0]
    pre_altered, _ = run_elo(altered)

    earlier = full["match_id"].iloc[:-1]
    pd.testing.assert_frame_equal(pre_full.loc[earlier], pre_altered.loc[earlier])


def test_unplayed_rows_get_pre_ratings_but_do_not_update():
    played = make_matches(SEASON_1)
    _, final_played = run_elo(played)

    with_upcoming = make_matches(
        SEASON_1 + [("2025-09-06", "2025-26", "E0", "A", "B", float("nan"), float("nan"))]
    )
    pre, final = run_elo(with_upcoming)

    upcoming_id = with_upcoming["match_id"].iloc[-1]
    assert pre.loc[upcoming_id, "elo_home_pre"] == pytest.approx(final_played["A"])
    assert pre.loc[upcoming_id, "elo_away_pre"] == pytest.approx(final_played["B"])
    assert final == pytest.approx(final_played)


def test_same_date_matches_do_not_see_each_other():
    # A plays twice on one date: the second match's pre-rating must not reflect the first result.
    both = make_matches([
        ("2025-08-16", "2025-26", "E0", "A", "B", 4, 0),
        ("2025-08-16", "2025-26", "E0", "A", "C", 0, 4),
    ])
    pre, final = run_elo(both)
    initial = EloParams().initial
    assert pre["elo_home_pre"].tolist() == [initial, initial]
    assert pre["elo_away_pre"].tolist() == [initial, initial]
    assert final["A"] != initial  # both results are still applied after the day


def test_margin_flag_changes_update_size():
    import pandas as pd
    from footpred.ratings.elo import EloParams, run_elo

    df = pd.DataFrame({"match_id": ["a", "b"], "date": pd.to_datetime(["2025-08-01", "2025-08-08"]),
                       "league": "E0", "season": "2025-26", "home": ["A", "A"], "away": ["B", "C"],
                       "home_goals": [4.0, 0.0], "away_goals": [0.0, 0.0]})
    with_margin = run_elo(df, EloParams(margin=True))[0].loc["b", "elo_home_pre"]
    without = run_elo(df, EloParams(margin=False))[0].loc["b", "elo_home_pre"]
    assert with_margin > without > 1500
