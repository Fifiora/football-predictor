import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad

from footpred.ratings import dixon_coles as dc

from sim import TEAMS, simulate_league


NEUTRAL = {t: 0.0 for t in TEAMS}
STRONG_A = {**NEUTRAL, "A": 0.35}
STRONG_A_DEF = {**NEUTRAL, "A": 0.2}


@pytest.fixture(scope="module")
def strong_a_matches():
    return simulate_league(STRONG_A, STRONG_A_DEF, rounds=8, seed=1)


def test_probabilities_sum_to_one(strong_a_matches):
    model = dc.fit_dixon_coles(strong_a_matches, as_of=pd.Timestamp("2026-06-01"))
    pred = model.predict("A", "B")
    assert pred.p_home + pred.p_draw + pred.p_away == pytest.approx(1.0, abs=1e-9)
    assert pred.grid.sum() == pytest.approx(1.0, abs=1e-9)
    assert pred.grid.shape == (9, 9)


def test_strong_attack_team_favored(strong_a_matches):
    model = dc.fit_dixon_coles(strong_a_matches, as_of=pd.Timestamp("2026-06-01"))
    pred = model.predict("A", "C")
    assert pred.p_home > 0.6
    assert pred.p_home > pred.p_away


def test_time_decay_weights_recent():
    weak_then_strong = {t: 0.0 for t in TEAMS}
    early = simulate_league({**weak_then_strong, "A": -0.7}, NEUTRAL, rounds=10, seed=2)
    late = simulate_league(
        {**weak_then_strong, "A": 0.7}, NEUTRAL, rounds=10, start="2026-01-01", seed=3
    )
    matches = pd.concat([early, late], ignore_index=True)
    as_of = pd.Timestamp("2026-06-01")

    flat = dc.fit_dixon_coles(matches, as_of, xi=0.0).predict("A", "B").exp_home_goals
    decayed = dc.fit_dixon_coles(matches, as_of, xi=0.03).predict("A", "B").exp_home_goals
    assert decayed > flat + 0.15


def test_unseen_team_does_not_crash(strong_a_matches):
    model = dc.fit_dixon_coles(strong_a_matches, as_of=pd.Timestamp("2026-06-01"))
    for pred in (model.predict("A", "NewTeam"), model.predict("NewTeam", "A")):
        assert pred.p_home + pred.p_draw + pred.p_away == pytest.approx(1.0, abs=1e-9)
        assert 0 < pred.p_home < 1


def test_two_unseen_teams_use_league_hint(strong_a_matches):
    model = dc.fit_dixon_coles(strong_a_matches, as_of=pd.Timestamp("2026-06-01"))
    pred = model.predict("New1", "New2", league="E0")
    assert pred.p_home + pred.p_draw + pred.p_away == pytest.approx(1.0, abs=1e-9)
    with pytest.raises(dc.DCUnavailable):
        model.predict("New1", "New2")


def test_fit_ignores_matches_on_or_after_as_of(strong_a_matches):
    as_of = pd.Timestamp("2025-10-01")
    a = dc.fit_dixon_coles(strong_a_matches, as_of)
    truncated = strong_a_matches[strong_a_matches["date"] < as_of]
    b = dc.fit_dixon_coles(truncated, as_of)
    assert a.predict("A", "B").p_home == pytest.approx(b.predict("A", "B").p_home, abs=1e-9)


def test_gradient_matches_numeric(strong_a_matches):
    data = dc._prepare(strong_a_matches.iloc[:120], pd.Timestamp("2026-06-01"), xi=0.002)
    n = len(data.teams)
    rng = np.random.default_rng(5)
    x0 = np.concatenate([[0.1, 0.2], rng.normal(0, 0.1, 2 * n), [-0.05]])

    def f(x):
        return dc._neg_ll_and_grad(x, data)[0]

    def g(x):
        return dc._neg_ll_and_grad(x, data)[1]

    assert check_grad(f, g, x0, epsilon=1e-6) < 1e-3 * max(1.0, np.linalg.norm(g(x0)))


def test_pre_match_excludes_future(strong_a_matches):
    cut = pd.Timestamp("2025-10-05")
    full = strong_a_matches
    altered = full.copy()
    later = altered["date"] > cut
    altered.loc[later, ["home_goals", "away_goals"]] = [9, 0]

    pre_full = dc.dc_pre_match(full)
    pre_altered = dc.dc_pre_match(altered)

    keep = full.loc[full["date"] <= cut, "match_id"]
    pd.testing.assert_frame_equal(pre_full.loc[keep], pre_altered.loc[keep])


def test_pre_match_columns_and_early_rows_are_nan(strong_a_matches):
    pre = dc.dc_pre_match(strong_a_matches)
    assert list(pre.columns) == ["dc_exp_home", "dc_exp_away", "dc_p_home", "dc_p_draw", "dc_p_away"]
    first_day = strong_a_matches["date"].min()
    first_ids = strong_a_matches.loc[strong_a_matches["date"] == first_day, "match_id"]
    assert pre.loc[first_ids].isna().all().all()  # nothing to learn from yet
    last = pre.dropna()
    assert len(last) > 0
    sums = last[["dc_p_home", "dc_p_draw", "dc_p_away"]].sum(axis=1)
    assert np.allclose(sums, 1.0)
