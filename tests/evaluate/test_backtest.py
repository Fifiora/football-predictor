import numpy as np
import pandas as pd
import pytest

from footpred.evaluate.backtest import backtest, walk_forward_splits
from footpred.evaluate.baselines import baseline_base_rate, baseline_dixon_coles, baseline_elo
from footpred.features.builder import FEATURE_COLUMNS


def make_data(n_days=120, per_day=2, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for day in range(n_days):
        for k in range(per_day):
            rows.append((pd.Timestamp("2025-08-01") + pd.Timedelta(days=day), f"H{day}{k}", f"A{day}{k}"))
    m = pd.DataFrame(rows, columns=["date", "home", "away"])
    m["league"] = "E0"
    m["match_id"] = [f"{d:%Y%m%d}-{h}-{a}" for d, h, a in zip(m["date"], m["home"], m["away"])]
    m["result"] = rng.choice(["H", "D", "A"], size=len(m), p=[0.45, 0.27, 0.28])
    feats = pd.DataFrame(
        rng.normal(size=(len(m), len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS, index=pd.Index(m["match_id"], name="match_id")
    )
    return m, feats


def test_splits_are_strictly_time_ordered():
    m, _ = make_data()
    splits = list(walk_forward_splits(m["date"], pd.Timestamp("2025-10-01"), step_days=30))
    assert len(splits) >= 2
    for train_idx, test_idx in splits:
        assert m["date"].iloc[train_idx].max() < m["date"].iloc[test_idx].min()
        assert len(train_idx) > 0 and len(test_idx) > 0


def test_splits_cover_each_test_match_once():
    m, _ = make_data()
    first = pd.Timestamp("2025-10-01")
    tested = np.concatenate([t for _, t in walk_forward_splits(m["date"], first, step_days=30)])
    assert len(tested) == len(set(tested))
    assert set(tested) == set(np.flatnonzero(m["date"] >= first))


def test_backtest_plumbing_hides_test_outcomes():
    m, feats = make_data()
    seen = []

    def fit_predict(train, test):
        assert "outcome" in train.columns and "outcome" not in test.columns
        assert train["outcome"].isin([0, 1, 2]).all()
        seen.append(len(test))
        return np.full((len(test), 3), 1 / 3)

    out = backtest(fit_predict, feats, m, pd.Timestamp("2025-10-01"), step_days=30)
    assert list(out.columns[:5]) == ["match_id", "p_home", "p_draw", "p_away", "outcome"]
    assert len(out) == sum(seen) == (m["date"] >= pd.Timestamp("2025-10-01")).sum()
    assert out[["p_home", "p_draw", "p_away"]].sum(axis=1).tolist() == pytest.approx([1.0] * len(out))


def test_backtest_excludes_unplayed_matches():
    m, feats = make_data()
    m.loc[m.index[-5:], "result"] = None
    out = backtest(lambda tr, te: np.full((len(te), 3), 1 / 3), feats, m, pd.Timestamp("2025-10-01"))
    assert not out["match_id"].isin(m["match_id"].iloc[-5:]).any()


def test_base_rate_matches_training_frequencies():
    m, feats = make_data()
    train = feats.iloc[:100].copy()
    train["outcome"] = [0] * 50 + [1] * 30 + [2] * 20
    probs = baseline_base_rate(train, feats.iloc[100:110])
    assert probs.shape == (10, 3)
    assert probs[0] == pytest.approx([0.5, 0.3, 0.2], abs=0.02)
    assert probs.sum(axis=1) == pytest.approx([1.0] * 10)


def test_elo_baseline_is_monotone_in_elo_diff():
    rng = np.random.default_rng(1)
    elo = rng.normal(0, 100, 600)
    p_home = 1 / (1 + np.exp(-elo / 150))
    outcome = np.where(rng.random(600) < p_home, 0, rng.choice([1, 2], 600))
    train = pd.DataFrame({c: np.nan for c in FEATURE_COLUMNS}, index=range(600))
    train["elo_diff"] = elo
    train["outcome"] = outcome
    test = pd.DataFrame({c: np.nan for c in FEATURE_COLUMNS}, index=range(2))
    test["elo_diff"] = [-150.0, 150.0]
    probs = baseline_elo(train, test)
    assert probs.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert probs[1, 0] > probs[0, 0]


def test_dc_baseline_falls_back_to_base_rate_when_missing():
    m, feats = make_data()
    train = feats.iloc[:100].copy()
    train["outcome"] = [0] * 50 + [1] * 30 + [2] * 20
    test = feats.iloc[100:103].copy()
    test[["dc_p_home", "dc_p_draw", "dc_p_away"]] = [[0.6, 0.25, 0.15], [np.nan] * 3, [0.2, 0.3, 0.5]]
    probs = baseline_dixon_coles(train, test)
    assert probs[0] == pytest.approx([0.6, 0.25, 0.15])
    assert probs[1] == pytest.approx([0.5, 0.3, 0.2], abs=0.02)
    assert probs[2] == pytest.approx([0.2, 0.3, 0.5])
