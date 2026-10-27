"""Per-version model card: where the model is strong and where it is weak."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from footpred.evaluate.metrics import log_loss, rps
from footpred.models.meta_model import MetaModel

_PROB_COLS = ["p_home", "p_draw", "p_away"]
_CLASSES = ["home win", "draw", "away win"]


def _md_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def _permutation_importance(model: MetaModel, X: pd.DataFrame, y: np.ndarray, repeats: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    base = log_loss(model.predict_proba(X), y)
    rows = []
    for name in model.feature_names:
        increases = []
        for _ in range(repeats):
            shuffled = X.copy()
            shuffled[name] = rng.permutation(shuffled[name].to_numpy())
            increases.append(log_loss(model.predict_proba(shuffled), y) - base)
        rows.append((name, round(float(np.mean(increases)), 5)))
    return pd.DataFrame(rows, columns=["feature", "log loss increase when shuffled"]).sort_values(
        "log loss increase when shuffled", ascending=False
    )


def _feature_importance_section(model: MetaModel, X_eval, y_eval) -> str:
    if X_eval is not None and y_eval is not None:
        return _md_table(_permutation_importance(model, X_eval, np.asarray(y_eval, dtype=int)))
    importances = getattr(model.estimator, "feature_importances_", None)
    if importances is not None:
        table = pd.DataFrame({"feature": model.feature_names, "importance": np.round(importances, 4)})
        return _md_table(table.sort_values("importance", ascending=False))
    return "Not available (no evaluation data supplied)."


def _calibration_section(preds: pd.DataFrame, bins: int = 5) -> str:
    edges = np.linspace(0, 1, bins + 1)
    outcomes = preds["outcome"].to_numpy(dtype=int)
    rows = []
    for k, label in enumerate(_CLASSES):
        p = preds[_PROB_COLS[k]].to_numpy(dtype=float)
        which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
        for b in range(bins):
            mask = which == b
            if mask.any():
                rows.append((label, f"{edges[b]:.1f}-{edges[b + 1]:.1f}", int(mask.sum()),
                             round(float(p[mask].mean()), 3), round(float((outcomes[mask] == k).mean()), 3)))
    return _md_table(pd.DataFrame(rows, columns=["outcome", "predicted range", "matches", "mean predicted", "observed"]))


def _performance(group: pd.DataFrame) -> pd.Series:
    probs = group[_PROB_COLS].to_numpy(dtype=float)
    outcomes = group["outcome"].to_numpy(dtype=int)
    return pd.Series(
        {
            "matches": len(group),
            "log loss": round(log_loss(probs, outcomes), 4),
            "RPS": round(rps(probs, outcomes), 4),
            "top-pick accuracy": round(float((probs.argmax(axis=1) == outcomes).mean()), 3),
        }
    )


def _by(preds: pd.DataFrame, key: str) -> str:
    table = preds.groupby(key, observed=True).apply(_performance, include_groups=False).reset_index()
    return _md_table(table)


def write_model_card(
    out_path: Path,
    model: MetaModel,
    backtest_preds: pd.DataFrame,
    matches: pd.DataFrame,
    X_eval: pd.DataFrame | None = None,
    y_eval: np.ndarray | None = None,
    elo_pre: pd.DataFrame | None = None,
) -> None:
    """Write feature importance, calibration, and performance by league and team-strength tier.

    Team-strength tier splits matches into terciles of the two teams' mean pre-match Elo
    (``elo_pre`` from ``run_elo``); without it that section is marked not available.
    """
    preds = backtest_preds.copy()
    sections = [
        "# Model card",
        f"Backend: {model.backend}. Evaluated on {len(preds)} walk-forward predictions.",
        "## Feature importance\n" + _feature_importance_section(model, X_eval, y_eval),
        "## Calibration\n" + _calibration_section(preds),
        "## Performance by league\n" + _by(preds, "league"),
    ]
    if elo_pre is not None:
        strength = ((elo_pre["elo_home_pre"] + elo_pre["elo_away_pre"]) / 2).reindex(preds["match_id"]).to_numpy()
        ranks = pd.Series(strength).rank(method="first")  # unique ranks, so ties can never break the split
        preds["tier"] = pd.qcut(ranks, 3, labels=["low", "mid", "high"]).to_numpy()
        tier_text = _by(preds.dropna(subset=["tier"]), "tier")
    else:
        tier_text = "Not available (no Elo ratings supplied)."
    sections.append("## Performance by team-strength tier\n" + tier_text)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n".join(sections) + "\n")
