import numpy as np
import pandas as pd
import pytest

from footpred.evaluate.baselines import baseline_base_rate
from footpred.evaluate.metrics import log_loss
from footpred.models.meta_model import MetaModel, fit_hybrid

COLS = ["elo_diff", "dc_p_home", "rest_days_home"]


def synthetic(n, seed=0):
    rng = np.random.default_rng(seed)
    elo = rng.normal(0, 100, n)
    X = pd.DataFrame({"elo_diff": elo, "dc_p_home": rng.uniform(0.2, 0.7, n), "rest_days_home": rng.integers(2, 9, n).astype(float)})
    p_home = 1 / (1 + np.exp(-elo / 60))
    u = rng.random(n)
    y = np.where(u < p_home * 0.8, 0, np.where(u < p_home * 0.8 + 0.2, 1, 2))
    return X, y


def test_predict_proba_shape_and_sum():
    X, y = synthetic(600)
    probs = MetaModel().fit(X, y).predict_proba(X.iloc[:50])
    assert probs.shape == (50, 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_meta_model_beats_base_rate_on_synthetic_signal():
    X, y = synthetic(2500, seed=1)
    train, test = slice(0, 1800), slice(1800, 2500)
    probs = MetaModel().fit(X.iloc[train], y[train]).predict_proba(X.iloc[test])
    base = baseline_base_rate(X.iloc[train].assign(outcome=y[train]), X.iloc[test])
    assert log_loss(probs, y[test]) < log_loss(base, y[test])


def test_supports_missing_feature_values():
    X, y = synthetic(600, seed=2)
    X.loc[X.index[::3], "dc_p_home"] = np.nan
    X.loc[X.index[::7], "rest_days_home"] = np.nan
    probs = MetaModel().fit(X, y).predict_proba(X.iloc[:20])
    assert np.isfinite(probs).all()


def test_fit_is_deterministic():
    X, y = synthetic(600, seed=3)
    a = MetaModel().fit(X, y).predict_proba(X.iloc[:30])
    b = MetaModel().fit(X, y).predict_proba(X.iloc[:30])
    assert np.allclose(a, b)


def test_predict_uses_training_columns_by_name():
    X, y = synthetic(600, seed=4)
    model = MetaModel().fit(X, y)
    shuffled = X[["rest_days_home", "elo_diff", "dc_p_home"]].iloc[:30]
    assert np.allclose(model.predict_proba(X.iloc[:30]), model.predict_proba(shuffled))


def test_backend_is_reported():
    X, y = synthetic(300, seed=5)
    backend = MetaModel().fit(X, y).backend
    try:
        import lightgbm  # noqa: F401
        assert backend == "lightgbm"
    except ImportError:
        assert backend == "sklearn"


def test_fit_hybrid_returns_calibrated_pipeline():
    X, y = synthetic(3000, seed=6)
    model, cal = fit_hybrid(X.iloc[:2000], y[:2000], X.iloc[2000:2800], y[2000:2800])
    out = cal.transform(model.predict_proba(X.iloc[2800:]))
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-9)


def test_fit_hybrid_rejects_calibration_data_older_than_training():
    X, y = synthetic(600, seed=7)
    dates = pd.Series(pd.date_range("2025-01-01", periods=600))
    with pytest.raises(ValueError):
        fit_hybrid(X.iloc[:300], y[:300], X.iloc[300:], y[300:],
                   train_dates=dates.iloc[300:].reset_index(drop=True),
                   cal_dates=dates.iloc[:300].reset_index(drop=True))
