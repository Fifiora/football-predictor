"""Gradient-boosted meta-model over ratings and form features.

Uses LightGBM (the spec's choice) when installed and falls back to scikit-learn's histogram
gradient boosting otherwise; both handle missing feature values natively.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from footpred.models.calibration import Calibrator

DEFAULT_LIGHTGBM_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=8,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=0,
    verbose=-1,
)

DEFAULT_SKLEARN_PARAMS = dict(
    max_iter=200,
    learning_rate=0.05,
    max_depth=3,
    min_samples_leaf=40,
    l2_regularization=1.0,
    random_state=0,
)


def _lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        return False
    return True


class MetaModel:
    """Predicts [home, draw, away] probabilities from a feature frame."""

    def __init__(self, params: dict | None = None):
        self.params = params
        self.backend = "lightgbm" if _lightgbm_available() else "sklearn"
        self.estimator = None
        self.feature_names: list[str] = []
        self.baseline: pd.Series | None = None  # per-feature training median, used by explanations

    def _new_estimator(self):
        if self.backend == "lightgbm":
            import lightgbm as lgb

            return lgb.LGBMClassifier(**{**DEFAULT_LIGHTGBM_PARAMS, **(self.params or {})})
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(**{**DEFAULT_SKLEARN_PARAMS, **(self.params or {})})

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MetaModel":
        self.feature_names = list(X.columns)
        self.baseline = X[self.feature_names].median()
        self.estimator = self._new_estimator()
        self.estimator.fit(X[self.feature_names], np.asarray(y, dtype=int))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("MetaModel.fit must be called before predict_proba")
        raw = self.estimator.predict_proba(X[self.feature_names])
        probs = np.zeros((len(X), 3))
        probs[:, np.asarray(self.estimator.classes_, dtype=int)] = raw
        return probs / probs.sum(axis=1, keepdims=True)


def fit_hybrid(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_cal: pd.DataFrame,
    y_cal: np.ndarray,
    train_dates: pd.Series | None = None,
    cal_dates: pd.Series | None = None,
) -> tuple[MetaModel, Calibrator]:
    """Fit the meta-model on the training block, then calibrate on a strictly later block."""
    if train_dates is not None and cal_dates is not None and cal_dates.min() < train_dates.max():
        raise ValueError("calibration data must not be older than the newest training data")
    model = MetaModel().fit(X_train, y_train)
    calibrator = Calibrator().fit(model.predict_proba(X_cal), y_cal)
    return model, calibrator
