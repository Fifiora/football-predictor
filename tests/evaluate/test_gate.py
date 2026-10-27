import numpy as np
import pytest

from footpred.evaluate.gate import GateResult, should_promote
from footpred.evaluate.metrics import expected_calibration_error, log_loss, rps


def world(n=300, seed=0):
    rng = np.random.default_rng(seed)
    true = rng.dirichlet([2.0, 1.2, 1.6], size=n)
    y = np.array([rng.choice(3, p=p) for p in true])
    return true, y


def norm(p):
    p = np.clip(p, 1e-3, None)
    return p / p.sum(axis=1, keepdims=True)


def base_rate(y):
    return np.tile(np.bincount(y, minlength=3) / len(y), (len(y), 1))


def test_gate_blocks_below_min_matches():
    true, y = world(100)
    result = should_promote(base_rate(y), true, y, min_matches=200)
    assert isinstance(result, GateResult)
    assert result.promote is False
    assert "200" in result.reason and "minimum" in result.reason.lower()


def test_gate_blocks_worse_calibration():
    true, y = world()
    champion = base_rate(y)
    challenger = norm(true + np.array([0.12, 0.0, -0.06]))  # informative but biased toward home
    # preconditions: better on log loss and RPS, clearly worse on calibration
    assert log_loss(challenger, y) < log_loss(champion, y)
    assert rps(challenger, y) < rps(champion, y)
    assert expected_calibration_error(challenger, y) - expected_calibration_error(champion, y) > 0.02

    result = should_promote(champion, challenger, y)
    assert result.promote is False
    assert "calibration" in result.reason.lower()


def test_gate_promotes_clear_winner():
    true, y = world()
    champion = norm(true**0.4)  # underconfident
    challenger = true
    result = should_promote(champion, challenger, y)
    assert result.promote is True
    assert result.challenger_log_loss < result.champion_log_loss
    assert result.challenger_rps < result.champion_rps


def test_gate_blocks_when_not_better_on_log_loss():
    true, y = world()
    result = should_promote(true, norm(true**0.4), y)
    assert result.promote is False
    assert "log loss" in result.reason.lower()


def test_gate_blocks_identical_models():
    true, y = world()
    assert should_promote(true, true.copy(), y).promote is False
