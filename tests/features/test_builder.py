import numpy as np
import pandas as pd
import pytest

from footpred.features.builder import FEATURE_COLUMNS, build_features
from footpred.ratings.dixon_coles import dc_pre_match
from footpred.ratings.elo import run_elo


def mk(rows):
    """rows: (date, home, away, home_goals, away_goals); shots fixed at 10 for home, 6 for away."""
    df = pd.DataFrame(rows, columns=["date", "home", "away", "home_goals", "away_goals"])
    df["date"] = pd.to_datetime(df["date"])
    df["league"] = "E0"
    df["season"] = "2025-26"
    played = df["home_goals"].notna()
    df["result"] = np.where(
        ~played, None,
        np.where(df["home_goals"] > df["away_goals"], "H", np.where(df["home_goals"] == df["away_goals"], "D", "A")),
    )
    df["home_shots"] = np.where(played, 10.0, np.nan)
    df["away_shots"] = np.where(played, 6.0, np.nan)
    for c in ("home_sot", "away_sot", "close_h", "close_d", "close_a"):
        df[c] = np.nan
    df["match_id"] = [f"E0-{d:%Y%m%d}-{h}-{a}" for d, h, a in zip(df["date"], df["home"], df["away"])]
    return df


def features_for(matches):
    elo_pre, _ = run_elo(matches)
    dc_pre = dc_pre_match(matches)
    return build_features(matches, elo_pre, dc_pre)


def test_columns_and_index():
    m = mk([("2025-08-16", "A", "B", 1, 0)])
    feats = features_for(m)
    assert list(feats.columns) == FEATURE_COLUMNS
    assert list(feats.index) == list(m["match_id"])


def test_form_uses_only_prior_matches():
    m = mk([("2025-08-16", "A", "B", 3, 1)])
    feats = features_for(m).iloc[0]
    for col in FEATURE_COLUMNS:
        if col.startswith("form_") or col.startswith("rest_"):
            assert np.isnan(feats[col]), col


def test_form_is_mean_of_previous_five_matches():
    opponents = ["B", "C", "D", "E", "F", "G", "H"]
    rows = [(pd.Timestamp("2025-08-16") + pd.Timedelta(days=7 * i), "A", opp, i + 1, 0)
            for i, opp in enumerate(opponents)]
    m = mk(rows)
    feats = features_for(m)
    seventh = feats.loc[m["match_id"].iloc[6]]
    # A's previous five matches before the seventh are matches 2..6 with goals 2,3,4,5,6
    assert seventh["form_goals_for_home"] == pytest.approx(4.0)
    assert seventh["form_goals_against_home"] == pytest.approx(0.0)
    assert seventh["form_shots_diff_home"] == pytest.approx(4.0)  # 10 shots for, 6 against


def test_away_team_form_uses_its_own_perspective():
    m = mk([
        ("2025-08-16", "B", "A", 2, 3),  # A scores 3 away
        ("2025-08-23", "C", "A", 0, 1),  # A scores 1 away
        ("2025-08-30", "D", "A", 0, 0),
    ])
    feats = features_for(m).loc[m["match_id"].iloc[2]]
    assert feats["form_goals_for_away"] == pytest.approx(2.0)
    assert feats["form_goals_against_away"] == pytest.approx(1.0)
    assert feats["form_shots_diff_away"] == pytest.approx(-4.0)  # away team: 6 for, 10 against


def test_rest_days():
    m = mk([("2025-08-16", "A", "B", 1, 0), ("2025-08-19", "A", "C", 1, 1)])
    feats = features_for(m).loc[m["match_id"].iloc[1]]
    assert feats["rest_days_home"] == 3
    assert feats["rest_days_away"] != feats["rest_days_away"]  # C has no previous match: NaN


def test_elo_diff_matches_pre_ratings():
    m = mk([("2025-08-16", "A", "B", 3, 0), ("2025-08-23", "A", "C", 1, 0)])
    elo_pre, _ = run_elo(m)
    feats = build_features(m, elo_pre, dc_pre_match(m))
    second = m["match_id"].iloc[1]
    assert feats.loc[second, "elo_diff"] == pytest.approx(
        elo_pre.loc[second, "elo_home_pre"] - elo_pre.loc[second, "elo_away_pre"]
    )
    assert feats.loc[second, "elo_diff"] > 0  # A beat B, so A is rated above fresh C


def test_same_day_result_not_used():
    m = mk([("2025-08-16", "A", "B", 5, 0), ("2025-08-16", "C", "D", 0, 0)])
    feats = features_for(m)
    assert feats[[c for c in FEATURE_COLUMNS if c.startswith("form_")]].isna().all().all()


def test_unplayed_matches_excluded_from_form_history():
    m = mk([
        ("2025-08-16", "A", "B", 2, 0),
        ("2025-08-23", "A", "C", float("nan"), float("nan")),  # postponed, never played
        ("2025-08-30", "A", "D", 1, 1),
    ])
    feats = features_for(m).loc[m["match_id"].iloc[2]]
    assert feats["form_goals_for_home"] == pytest.approx(2.0)  # only the 2-0 counts
    assert feats["rest_days_home"] == 14  # days since the last match actually played


def test_upcoming_fixture_gets_features_without_changing_earlier_rows():
    played = mk([("2025-08-16", "A", "B", 2, 0), ("2025-08-23", "B", "C", 0, 1)])
    with_upcoming = mk([
        ("2025-08-16", "A", "B", 2, 0),
        ("2025-08-23", "B", "C", 0, 1),
        ("2025-08-30", "A", "C", float("nan"), float("nan")),
    ])
    base = features_for(played)
    ext = features_for(with_upcoming)
    pd.testing.assert_frame_equal(base, ext.loc[base.index])
    upcoming = ext.loc[with_upcoming["match_id"].iloc[2]]
    assert upcoming["form_goals_for_home"] == pytest.approx(2.0)  # A's one played match
    assert upcoming["rest_days_home"] == 14


def test_extra_columns_are_prior_only_and_venue_specific():
    from footpred.features.builder import ALL_COLUMNS

    # A plays home to B (2-0), away at C (0-3), then home to D (the match under test)
    m = mk([("2025-08-16", "A", "B", 2, 0), ("2025-08-23", "C", "A", 3, 0), ("2025-08-30", "A", "D", 1, 1)])
    elo_pre, _ = run_elo(m)
    feats = build_features(m, elo_pre, dc_pre_match(m), ALL_COLUMNS)
    assert list(feats.columns) == ALL_COLUMNS
    first, last = feats.iloc[0], feats.iloc[2]
    assert np.isnan(first["form_ppg_home"]) and np.isnan(first["venue_gd_home"])
    assert last["form_ppg_home"] == pytest.approx(1.5)  # a win then a loss
    assert last["form_gd10_home"] == pytest.approx(-0.5)  # +2 and -3
    assert last["venue_gd_home"] == pytest.approx(2.0)  # only the earlier home game counts


def test_extra_columns_do_not_see_the_match_itself():
    from footpred.features.builder import ALL_COLUMNS

    base = [("2025-08-16", "A", "B", 2, 0), ("2025-08-30", "A", "D", 1, 1)]
    a = mk(base)
    b = mk([base[0], ("2025-08-30", "A", "D", 9, 0)])
    fa = build_features(a, run_elo(a)[0], dc_pre_match(a), ALL_COLUMNS).iloc[1]
    fb = build_features(b, run_elo(b)[0], dc_pre_match(b), ALL_COLUMNS).iloc[1]
    pd.testing.assert_series_equal(fa, fb)
