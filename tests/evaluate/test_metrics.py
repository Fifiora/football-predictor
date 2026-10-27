import numpy as np
import pandas as pd
import pytest

from footpred.evaluate.baselines import benchmark_closing_odds
from footpred.evaluate.metrics import (
    encode_outcome,
    expected_calibration_error,
    log_loss,
    rps,
)


def test_encode_outcome():
    assert encode_outcome(pd.Series(["H", "D", "A", "H"])).tolist() == [0, 1, 2, 0]


def test_log_loss_perfect_and_uniform():
    outcomes = np.array([0, 1, 2])
    perfect = np.eye(3)
    uniform = np.full((3, 3), 1 / 3)
    assert log_loss(perfect, outcomes) == pytest.approx(0.0, abs=1e-9)
    assert log_loss(uniform, outcomes) == pytest.approx(np.log(3))


def test_log_loss_stays_finite_for_zero_probability():
    probs = np.array([[0.0, 0.5, 0.5]])
    assert np.isfinite(log_loss(probs, np.array([0])))


def test_rps_known_value():
    probs = np.array([[0.5, 0.3, 0.2]])
    assert rps(probs, np.array([0])) == pytest.approx(0.145)


def test_rps_penalizes_distance():
    # Predicting a home win when the away team wins is worse than predicting a draw.
    outcome = np.array([2])
    assert rps(np.array([[1.0, 0.0, 0.0]]), outcome) > rps(np.array([[0.0, 1.0, 0.0]]), outcome)


def test_ece_zero_for_perfectly_calibrated():
    probs = np.tile([0.5, 0.3, 0.2], (1000, 1))
    outcomes = np.array([0] * 500 + [1] * 300 + [2] * 200)
    assert expected_calibration_error(probs, outcomes) < 0.02


def test_ece_large_for_overconfident():
    probs = np.tile([0.9, 0.05, 0.05], (1000, 1))
    outcomes = np.array([0] * 400 + [1] * 300 + [2] * 300)
    assert expected_calibration_error(probs, outcomes) > 0.2


def test_devig_odds_sum_to_one():
    matches = pd.DataFrame({"close_h": [1.5, 2.0, np.nan], "close_d": [4.0, 3.5, np.nan], "close_a": [7.0, 3.8, np.nan]})
    probs = benchmark_closing_odds(matches)
    assert probs.shape == (3, 3)
    assert probs[:2].sum(axis=1) == pytest.approx([1.0, 1.0])
    assert probs[0, 0] > probs[0, 2]  # the favorite has the higher probability
    assert np.isnan(probs[2]).all()  # no odds, no benchmark
