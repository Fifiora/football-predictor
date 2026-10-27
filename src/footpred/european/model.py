"""Champions League model: a small calibrated logistic regression on the pooled-Elo gap."""
from __future__ import annotations

import numpy as np
import pandas as pd

from footpred.models.calibration import Calibrator
from footpred.models.meta_model import MetaModel

EURO_FEATURES = ["elo_diff", "neutral", "knockout"]
MODEL_PARAMS = {"backend": "logistic", "C": 0.1}
KNOCKOUT_STAGES = {"PLAYOFFS", "LAST_16", "QUARTER_FINALS", "SEMI_FINALS", "FINAL"}


def european_features(matches: pd.DataFrame, pre: pd.DataFrame, home_adv: float = 60.0) -> pd.DataFrame:
    """Point-in-time inputs per match, indexed by ``match_id``.

    ``elo_diff`` is home minus away pooled rating plus home advantage, except in the neutral-venue
    final. ``knockout`` is 1 from the play-offs onward; a missing stage counts as an ordinary match.
    """
    ids = matches["match_id"].to_numpy()
    stage = matches["stage"].fillna("").to_numpy()
    neutral = (stage == "FINAL").astype(float)
    ratings = pre.reindex(ids)
    diff = ratings["elo_home_pre"].to_numpy() - ratings["elo_away_pre"].to_numpy() + home_adv * (1.0 - neutral)
    knockout = np.isin(stage, list(KNOCKOUT_STAGES)).astype(float)
    return pd.DataFrame({"elo_diff": diff, "neutral": neutral, "knockout": knockout},
                        index=pd.Index(ids, name="match_id"))[EURO_FEATURES]


class IdentityCalibrator:
    """Leaves probabilities as they are (for models that are not calibrated separately)."""

    def transform(self, probs: np.ndarray) -> np.ndarray:
        return np.asarray(probs, dtype=float)


def fit_european(
    X_train: pd.DataFrame, y_train, X_cal: pd.DataFrame, y_cal,
    params: dict | None = None, calibrate: bool = True, features: list[str] | None = None,
):
    """Fit on the older block and Platt-calibrate on the later block.

    With ``calibrate=False`` both blocks train the model and probabilities are used as they come.
    """
    cols = list(features or EURO_FEATURES)
    model_params = dict(params or MODEL_PARAMS)
    y_train, y_cal = np.asarray(y_train, dtype=int), np.asarray(y_cal, dtype=int)
    if not calibrate:
        X_all = pd.concat([X_train[cols], X_cal[cols]])
        return MetaModel(model_params).fit(X_all, np.concatenate([y_train, y_cal])), IdentityCalibrator()
    model = MetaModel(model_params).fit(X_train[cols], y_train)
    return model, Calibrator().fit(model.predict_proba(X_cal[cols]), y_cal)


def predict_european(model: MetaModel, calibrator, X: pd.DataFrame) -> np.ndarray:
    return calibrator.transform(model.predict_proba(X[model.feature_names]))


def european_publish_gate(development: dict) -> tuple[bool, str]:
    """Publish only if the model's log loss is strictly below both the base rate's and pooled Elo's."""
    model = development["model"]["log_loss"]
    base = development["base_rate"]["log_loss"]
    elo = development["elo"]["log_loss"]
    ok = model < base and model < elo
    verdict = "beats both baselines" if ok else "does not beat both baselines"
    return ok, f"model log loss {model:.3f} vs base rate {base:.3f} and pooled Elo {elo:.3f}: {verdict}"
