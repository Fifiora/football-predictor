"""Turn feature attributions into short, plain-language reasons for a prediction."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from footpred.explain.templates import TEMPLATES
from footpred.models.meta_model import MetaModel


@dataclass(frozen=True)
class Driver:
    feature: str
    direction: str  # "home" or "away": which side the feature pushes the prediction toward
    strength: float
    text: str


def driver_text(feature: str, direction: str, value: float, home: str, away: str) -> str:
    phrase = TEMPLATES[feature][0](value, home, away)
    team = home if direction == "home" else away
    return f"The model weighted {phrase}, which points to {team}."


def _shap_scores(model: MetaModel, x_row: pd.Series) -> np.ndarray | None:
    """SHAP attribution (home-win minus away-win, log-odds) when SHAP and a tree model exist."""
    if model.backend != "lightgbm":
        return None
    try:
        import shap
    except ImportError:
        return None
    values = shap.TreeExplainer(model.estimator).shap_values(x_row.to_frame().T[model.feature_names])
    arr = np.stack(values, axis=-1)[0] if isinstance(values, list) else np.asarray(values)[0]
    return arr[:, 0] - arr[:, 2]


def _occlusion_scores(model: MetaModel, x_row: pd.Series) -> np.ndarray:
    """Model-agnostic fallback: change in P(home) - P(away) when one feature is reset to its
    training median. Positive means the feature's actual value favours the home side."""
    actual = x_row[model.feature_names].astype(float)
    rows = [actual]
    for name in model.feature_names:
        occluded = actual.copy()
        occluded[name] = model.baseline[name]
        rows.append(occluded)
    probs = model.predict_proba(pd.DataFrame(rows).reset_index(drop=True))
    margin = probs[:, 0] - probs[:, 2]
    return margin[0] - margin[1:]


def top_drivers(
    model: MetaModel,
    x_row: pd.Series,
    k: int = 3,
    min_strength: float = 0.02,
    home: str = "the home side",
    away: str = "the away side",
) -> list[Driver]:
    """The ``k`` strongest drivers, one per feature family, skipping noise and missing values."""
    scores = _shap_scores(model, x_row)
    if scores is None:
        scores = _occlusion_scores(model, x_row)

    ranked = sorted(zip(model.feature_names, scores), key=lambda fs: -abs(fs[1]))
    drivers: list[Driver] = []
    seen_families: set[str] = set()
    for feature, score in ranked:
        if len(drivers) == k or abs(score) < min_strength:
            break
        value = x_row[feature]
        if feature not in TEMPLATES or pd.isna(value):
            continue
        family = TEMPLATES[feature][1]
        if family in seen_families:
            continue
        seen_families.add(family)
        direction = "home" if score > 0 else "away"
        drivers.append(
            Driver(feature, direction, float(abs(score)), driver_text(feature, direction, float(value), home, away))
        )
    return drivers
