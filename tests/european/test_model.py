import numpy as np
import pandas as pd
import pytest

from footpred.european.model import (
    EURO_FEATURES,
    european_features,
    european_publish_gate,
    fit_european,
    predict_european,
)


def frame(stages):
    m = pd.DataFrame({"match_id": [f"m{i}" for i in range(len(stages))], "stage": stages,
                      "home": "A", "away": "B"})
    pre = pd.DataFrame({"elo_home_pre": 1500.0, "elo_away_pre": 1500.0}, index=pd.Index(m["match_id"], name="match_id"))
    return m, pre


def test_features_final_is_neutral_and_has_no_home_advantage():
    m, pre = frame(["FINAL", "LEAGUE_STAGE"])
    x = european_features(m, pre)
    assert list(x.columns) == EURO_FEATURES and x.index.name == "match_id"
    assert x.loc["m0", "elo_diff"] == 0.0 and x.loc["m0", "neutral"] == 1.0
    assert x.loc["m1", "elo_diff"] == 50.0 and x.loc["m1", "neutral"] == 0.0


def test_knockout_flag_by_stage():
    stages = ["LEAGUE_STAGE", "QUALIFYING", "PLAYOFFS", "LAST_16", "QUARTER_FINALS", "SEMI_FINALS", "FINAL"]
    x = european_features(*frame(stages))
    assert list(x["knockout"]) == [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def test_missing_stage_is_treated_as_an_ordinary_match():
    x = european_features(*frame([None, np.nan]))
    assert list(x["neutral"]) == [0.0, 0.0] and list(x["knockout"]) == [0.0, 0.0]


def test_fit_and_predict_return_probabilities_that_sum_to_one():
    rng = np.random.default_rng(0)
    n = 400
    diff = rng.normal(0, 150, n)
    y = np.where(diff + rng.normal(0, 200, n) > 80, 0, np.where(diff + rng.normal(0, 200, n) < -80, 2, 1))
    X = pd.DataFrame({"elo_diff": diff, "neutral": 0.0, "knockout": (rng.random(n) < 0.2).astype(float)})
    model, calibrator = fit_european(X.iloc[:300], y[:300], X.iloc[300:], y[300:])
    probs = predict_european(model, calibrator, X.iloc[300:])
    assert probs.shape == (100, 3) and np.allclose(probs.sum(axis=1), 1.0) and (probs > 0).all()
    assert probs[np.argmax(diff[300:]), 0] > probs[np.argmin(diff[300:]), 0]  # a bigger Elo gap favours the home side


def dev(base, elo, model):
    return {"base_rate": {"log_loss": base}, "elo": {"log_loss": elo}, "model": {"log_loss": model}}


def test_gate_requires_beating_both_baselines():
    ok, reason = european_publish_gate(dev(1.07, 1.00, 0.98))
    assert ok and "0.980" in reason
    assert european_publish_gate(dev(0.97, 1.00, 0.98))[0] is False  # loses to the base rate
    assert european_publish_gate(dev(1.07, 0.97, 0.98))[0] is False  # loses to the Elo baseline
    assert european_publish_gate(dev(1.07, 1.00, 1.001))[0] is True   # a tie with plain Elo passes
    assert european_publish_gate(dev(1.07, 1.00, 1.01))[0] is False   # clearly worse does not
