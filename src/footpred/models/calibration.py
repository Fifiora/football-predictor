"""Probability calibration: Platt scaling by default, isotonic only with very large data."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p)).reshape(-1, 1)


class _Constant:
    """Used when a class never (or always) occurs in the calibration data."""

    def __init__(self, value: float):
        self.value = value

    def predict(self, p: np.ndarray) -> np.ndarray:
        return np.full(len(p), self.value)


class _Platt:
    def __init__(self, model: LogisticRegression):
        self.model = model

    def predict(self, p: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(_logit(p))[:, 1]


class Calibrator:
    """One-vs-rest calibration of home/draw/away probabilities, renormalized to sum to 1."""

    def __init__(self, min_samples_isotonic: int = 20000):
        self.min_samples_isotonic = min_samples_isotonic
        self.method: str | None = None
        self._maps: list = []

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "Calibrator":
        probs = np.asarray(probs, dtype=float)
        y = np.asarray(y, dtype=int)
        self.method = "isotonic" if len(y) >= self.min_samples_isotonic else "platt"
        self._maps = []
        for k in range(3):
            target = (y == k).astype(int)
            if target.min() == target.max():  # class absent (or universal): nothing to learn
                self._maps.append(_Constant(float(np.clip(target[0], _EPS, 1.0 - _EPS))))
            elif self.method == "isotonic":
                iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                iso.fit(probs[:, k], target)
                self._maps.append(iso)
            else:
                lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(probs[:, k]), target)
                self._maps.append(_Platt(lr))
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if not self._maps:
            raise RuntimeError("Calibrator.fit must be called before transform")
        probs = np.asarray(probs, dtype=float)
        cols = [np.clip(m.predict(probs[:, k]), _EPS, 1.0) for k, m in enumerate(self._maps)]
        out = np.column_stack(cols)
        return out / out.sum(axis=1, keepdims=True)
