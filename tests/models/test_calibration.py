import numpy as np
import pandas as pd
import pytest

from footpred.evaluate.metrics import expected_calibration_error
from footpred.models.calibration import Calibrator


def overconfident_data(n, seed=0):
    rng = np.random.default_rng(seed)
    true = rng.dirichlet([2.0, 1.2, 1.6], size=n)
    outcomes = np.array([rng.choice(3, p=p) for p in true])
    sharp = true**2.5
    return sharp / sharp.sum(axis=1, keepdims=True), outcomes


def test_calibrated_rows_sum_to_one_and_bounded():
    probs, y = overconfident_data(3000)
    cal = Calibrator().fit(probs, y)
    extremes = np.array([[0.99, 0.005, 0.005], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.34, 0.33, 0.33]])
    out = cal.transform(np.vstack([probs[:100], extremes]))
    assert out.shape == (104, 3)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-9)
    assert (out >= 0).all() and (out <= 1).all()


def test_uses_platt_below_threshold_and_isotonic_above():
    small_p, small_y = overconfident_data(200, seed=1)
    big_p, big_y = overconfident_data(2000, seed=2)
    assert Calibrator().fit(small_p, small_y).method == "platt"
    assert Calibrator().fit(big_p, big_y).method == "isotonic"


def test_threshold_is_configurable():
    p, y = overconfident_data(300, seed=3)
    assert Calibrator(min_samples_isotonic=100).fit(p, y).method == "isotonic"


@pytest.mark.parametrize("n_cal", [250, 3000])
def test_calibration_improves_ece_on_miscalibrated_input(n_cal):
    probs, y = overconfident_data(n_cal + 6000, seed=4)
    cal = Calibrator().fit(probs[:n_cal], y[:n_cal])
    held_p, held_y = probs[n_cal:], y[n_cal:]
    before = expected_calibration_error(held_p, held_y)
    after = expected_calibration_error(cal.transform(held_p), held_y)
    assert after < before


def test_handles_class_never_observed_in_calibration_data():
    probs, y = overconfident_data(500, seed=5)
    y = np.where(y == 1, 0, y)  # no draws at all
    out = Calibrator().fit(probs, y).transform(probs[:10])
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-9)
    assert np.isfinite(out).all()
