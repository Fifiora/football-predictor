import numpy as np
import pandas as pd

from footpred.evaluate.drift import check_drift
from footpred.evaluate.metrics import log_loss


def graded_log(n, seed, sharpen=1.0):
    rng = np.random.default_rng(seed)
    true = rng.dirichlet([2.0, 1.2, 1.6], size=n)
    y = np.array([rng.choice(3, p=p) for p in true])
    shown = true**sharpen
    shown = shown / shown.sum(axis=1, keepdims=True)
    return pd.DataFrame(
        {
            "match_id": [f"m{i}" for i in range(n)],
            "kickoff_date": pd.date_range("2026-01-01", periods=n, freq="D"),
            "p_home": shown[:, 0],
            "p_draw": shown[:, 1],
            "p_away": shown[:, 2],
            "outcome": y,
        }
    ), true, y


def test_healthy_log_returns_no_alerts():
    log, true, y = graded_log(400, seed=1)
    baseline = log_loss(true[:200], y[:200])
    assert check_drift(log, window=200, baseline_log_loss=baseline) == []


def test_drift_alerts_when_rolling_loss_rises():
    healthy, _, _ = graded_log(300, seed=2)
    broken, _, _ = graded_log(200, seed=3, sharpen=3.0)  # overconfident recent predictions
    broken["kickoff_date"] = pd.date_range("2026-11-01", periods=200, freq="D")
    broken["match_id"] = [f"late{i}" for i in range(200)]
    log = pd.concat([healthy, broken], ignore_index=True)
    alerts = check_drift(log, window=200, baseline_log_loss=0.90)
    assert alerts
    assert any("log loss" in a.lower() for a in alerts)
    assert any("calibration" in a.lower() for a in alerts)


def test_too_few_graded_rows_is_not_evidence():
    log, _, _ = graded_log(50, seed=4, sharpen=3.0)
    assert check_drift(log, window=200, baseline_log_loss=0.90) == []


def test_ungraded_rows_are_ignored():
    log, true, y = graded_log(400, seed=1)
    ungraded = log.iloc[:5].copy()
    ungraded["match_id"] = [f"u{i}" for i in range(5)]
    ungraded["outcome"] = np.nan
    ungraded["kickoff_date"] = pd.date_range("2027-01-01", periods=5)
    combined = pd.concat([log, ungraded], ignore_index=True)
    assert check_drift(combined, window=200, baseline_log_loss=log_loss(true[:200], y[:200])) == []


def test_explicit_ceiling_overrides_baseline():
    log, _, _ = graded_log(400, seed=1)
    alerts = check_drift(log, window=200, log_loss_ceiling=0.1)
    assert any("log loss" in a.lower() for a in alerts)
