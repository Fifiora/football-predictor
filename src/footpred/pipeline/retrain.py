"""Weekly retrain: train a challenger, gate it against the champion, always keep its model card."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from footpred.evaluate.baselines import baseline_dixon_coles, baseline_elo
from footpred.evaluate.gate import GateResult, should_promote
from footpred.evaluate.metrics import expected_calibration_error, log_loss, rps
from footpred.features.builder import FEATURE_COLUMNS
from footpred.models.calibration import Calibrator
from footpred.models.meta_model import MetaModel
from footpred.models.model_card import write_model_card
from footpred.models.registry import load_champion, save_model, set_champion
from footpred.pipeline.common import code_version, load_all_matches, per_league_log_loss, played_before, training_frame

# Production model: chosen on development seasons of the real five-league benchmark (docs/baseline-results.md).
PRODUCTION_PARAMS = {"backend": "logistic", "C": 0.1}
EVAL_FRACTION = 0.20  # newest matches, used only to compare champion and challenger
CAL_FRACTION = 0.15   # block just before it, used only to calibrate


def _next_version(registry_dir: Path, today: pd.Timestamp) -> str:
    base = f"v{pd.Timestamp(today):%Y%m%d}"
    version, n = base, 1
    while (registry_dir / version).exists():
        n += 1
        version = f"{base}-{n}"
    return version


def run_retrain(
    data_dir: Path,
    registry_dir: Path,
    today: pd.Timestamp,
    challenger_params: dict | None = None,
    aliases_path: Path | None = None,
    gate_min_matches: int = 200,
) -> GateResult:
    registry_dir = Path(registry_dir)
    matches = load_all_matches(data_dir, aliases_path)
    played = played_before(matches, today)
    features, outcome, elo_pre, _ = training_frame(played)

    dates = played["date"].sort_values().reset_index(drop=True)
    cal_start = dates.iloc[int(len(dates) * (1 - EVAL_FRACTION - CAL_FRACTION))]
    eval_start = dates.iloc[int(len(dates) * (1 - EVAL_FRACTION))]
    d = played.set_index("match_id")["date"].reindex(features.index)
    train_m, cal_m, eval_m = d < cal_start, (d >= cal_start) & (d < eval_start), d >= eval_start
    if not (train_m.any() and cal_m.any() and eval_m.any()):
        raise ValueError("not enough history to split into train, calibration and evaluation blocks")

    X = features[FEATURE_COLUMNS]
    y = outcome.reindex(features.index).to_numpy()
    params = PRODUCTION_PARAMS if challenger_params is None else challenger_params
    model = MetaModel(params).fit(X[train_m], y[train_m])
    calibrator = Calibrator().fit(model.predict_proba(X[cal_m]), y[cal_m])

    X_eval, y_eval = X[eval_m], y[eval_m.to_numpy()]
    challenger = calibrator.transform(model.predict_proba(X_eval))

    train_df = X[train_m].assign(outcome=y[train_m.to_numpy()])
    leagues = played.set_index("match_id")["league"].reindex(X_eval.index).to_numpy()
    metrics = {
        "n_eval": int(len(y_eval)),
        "log_loss": float(log_loss(challenger, y_eval)),
        "rps": float(rps(challenger, y_eval)),
        "ece": float(expected_calibration_error(challenger, y_eval)),
        "baseline_elo_log_loss": float(log_loss(baseline_elo(train_df, X_eval), y_eval)),
        "baseline_dixon_coles_log_loss": float(log_loss(baseline_dixon_coles(train_df, X_eval), y_eval)),
        "by_league_log_loss": per_league_log_loss(challenger, y_eval, leagues),
    }
    version = _next_version(registry_dir, today)
    target = save_model(
        registry_dir, version, model, calibrator,
        {"data_snapshot": str(played["date"].max().date()), "code_version": code_version(),
         "params": params, "metrics": metrics, "trained_on": int(train_m.sum())},
    )

    try:
        _, champ_model, champ_cal = load_champion(registry_dir)
    except FileNotFoundError:
        champ_model = None
    if champ_model is None:
        result = GateResult(True, "first model: there is no champion to beat", float("nan"),
                            metrics["log_loss"], float("nan"), metrics["rps"])
    else:
        champion = champ_cal.transform(champ_model.predict_proba(X_eval))
        result = should_promote(champion, challenger, y_eval, min_matches=gate_min_matches)

    preds = pd.DataFrame(
        {"match_id": X_eval.index, "p_home": challenger[:, 0], "p_draw": challenger[:, 1],
         "p_away": challenger[:, 2], "outcome": y_eval,
         "date": d[eval_m].to_numpy(), "league": leagues}
    )
    write_model_card(target / "model_card.md", model, preds, played, X_eval=X_eval, y_eval=y_eval, elo_pre=elo_pre)
    if result.promote:
        set_champion(registry_dir, version)
    return result
