import pytest
from sim import simulate_league, TEAMS

from footpred.evaluate.leakage import LeakageError, assert_no_leakage
from footpred.features.builder import build_features
from footpred.ratings.dixon_coles import dc_pre_match
from footpred.ratings.elo import run_elo

NEUTRAL = {t: 0.0 for t in TEAMS}


def real_builder(matches):
    elo_pre, _ = run_elo(matches)
    return build_features(matches, elo_pre, dc_pre_match(matches, min_matches=30))


def leaky_builder(matches):
    feats = real_builder(matches)
    feats["elo_diff"] = matches.set_index("match_id")["home_goals"].reindex(feats.index)  # uses own result
    return feats


@pytest.fixture(scope="module")
def league():
    return simulate_league(NEUTRAL, NEUTRAL, rounds=6, seed=4)


def test_leakage_check_passes_on_builder(league):
    assert_no_leakage(real_builder, league, n_probe=5, seed=1)


def test_leakage_check_catches_leak(league):
    with pytest.raises(LeakageError):
        assert_no_leakage(leaky_builder, league, n_probe=5, seed=1)


def test_leakage_check_catches_future_peeking(league):
    def peeks_at_future(matches):
        feats = real_builder(matches)
        feats["rest_days_home"] = float(len(matches))  # depends on how many rows exist after the match
        return feats

    with pytest.raises(LeakageError):
        assert_no_leakage(peeks_at_future, league, n_probe=5, seed=1)
