"""Baseline-versus-hybrid benchmark with a protected hold-out season.

Development seasons (everything before the most recent full season) are used for all
comparison and tuning. The most recent full season is evaluated once, and only if the hybrid
beats both Elo-only and Dixon-Coles-only on the development seasons.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from footpred.evaluate.backtest import backtest
from footpred.evaluate.baselines import baseline_base_rate, baseline_dixon_coles, baseline_elo
from footpred.evaluate.metrics import expected_calibration_error, log_loss, rps
from footpred.features.builder import FEATURE_COLUMNS
from footpred.models.calibration import Calibrator
from footpred.models.meta_model import MetaModel
from footpred.pipeline.common import load_all_matches, per_league_log_loss, played_before, training_frame

CAL_FRACTION = 0.15
MODELS = {"base_rate": "Base rate", "elo": "Elo only", "dixon_coles": "Dixon-Coles only", "hybrid": "Hybrid"}


def _holdout_season(played: pd.DataFrame) -> str:
    """Most recent season that is complete: at least 90% of the typical earlier season's matches."""
    counts = played.groupby("season").size().sort_index()
    seasons = list(counts.index)
    if len(seasons) < 2:
        raise ValueError("need at least two seasons: one to develop on, one to hold out")
    for i in range(len(seasons) - 1, 0, -1):
        if counts.iloc[i] >= 0.9 * counts.iloc[:i].median():
            return seasons[i]
    raise ValueError("no complete season to hold out")


def _hybrid_fit_predict(dates: pd.Series):
    def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        order = dates.reindex(train.index).sort_values(kind="stable").index
        train = train.loc[order]
        cut = int(len(train) * (1 - CAL_FRACTION))
        fit, cal = train.iloc[:cut], train.iloc[cut:]
        model = MetaModel().fit(fit[FEATURE_COLUMNS], fit["outcome"].to_numpy())
        calibrator = Calibrator().fit(model.predict_proba(cal[FEATURE_COLUMNS]), cal["outcome"].to_numpy())
        return calibrator.transform(model.predict_proba(test[FEATURE_COLUMNS]))

    return fit_predict


def _summarize(preds: pd.DataFrame) -> dict:
    probs = preds[["p_home", "p_draw", "p_away"]].to_numpy()
    y = preds["outcome"].to_numpy(dtype=int)
    return {
        "n": int(len(preds)),
        "log_loss": float(log_loss(probs, y)),
        "rps": float(rps(probs, y)),
        "ece": float(expected_calibration_error(probs, y)),
        "by_league_log_loss": per_league_log_loss(probs, y, preds["league"].to_numpy()),
    }


def _table(results: dict) -> str:
    leagues = sorted({lg for r in results.values() for lg in r["by_league_log_loss"]})
    lines = ["| Model | Matches | Log loss | RPS | ECE | " + " | ".join(leagues) + " |",
             "|---|---|---|---|---|" + "---|" * len(leagues)]
    for key, label in MODELS.items():
        if key in results:
            r = results[key]
            per = " | ".join(f"{r['by_league_log_loss'].get(lg, float('nan')):.3f}" for lg in leagues)
            lines.append(f"| {label} | {r['n']} | {r['log_loss']:.3f} | {r['rps']:.3f} | {r['ece']:.3f} | {per} |")
    return "\n".join(lines)


def run_benchmark(
    data_dir: Path,
    out_path: Path,
    aliases_path: Path | None = None,
    today: pd.Timestamp | None = None,
    step_days: int = 60,
) -> dict:
    today = pd.Timestamp.now().normalize() if today is None else pd.Timestamp(today)
    played = played_before(load_all_matches(data_dir, aliases_path), today + pd.Timedelta(days=1))
    holdout = _holdout_season(played)
    features, outcome, _, _ = training_frame(played)

    meta = played.set_index("match_id")
    is_holdout = meta["season"].reindex(features.index) >= holdout
    dev_ids = features.index[~is_holdout.to_numpy()]
    dev_matches = played[played["match_id"].isin(dev_ids)].reset_index(drop=True)
    dates = meta["date"]

    seasons = sorted(dev_matches["season"].unique())
    first_test = dev_matches.loc[dev_matches["season"] == seasons[-1], "date"].min() if len(seasons) > 1 \
        else dev_matches["date"].quantile(0.5)

    fit_predicts = {"base_rate": baseline_base_rate, "elo": baseline_elo, "dixon_coles": baseline_dixon_coles,
                    "hybrid": _hybrid_fit_predict(dates)}
    dev_features = features.loc[dev_ids]
    development = {
        name: _summarize(backtest(fp, dev_features, dev_matches, first_test, step_days=step_days))
        for name, fp in fit_predicts.items()
    }

    beats_both = development["hybrid"]["log_loss"] < min(
        development["elo"]["log_loss"], development["dixon_coles"]["log_loss"]
    )
    held_out = None
    if beats_both:  # the one and only look at the held-out season
        train = features.loc[dev_ids].assign(outcome=outcome.reindex(dev_ids).to_numpy())
        test_ids = features.index[is_holdout.to_numpy()]
        test = features.loc[test_ids]
        y = outcome.reindex(test_ids).to_numpy()
        leagues = meta["league"].reindex(test_ids).to_numpy()
        held_out = {}
        for name, fp in fit_predicts.items():
            p = np.asarray(fp(train, test), dtype=float)
            held_out[name] = _summarize(pd.DataFrame(
                {"p_home": p[:, 0], "p_draw": p[:, 1], "p_away": p[:, 2], "outcome": y, "league": leagues}))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    text = [
        "# Baseline versus hybrid results", "",
        f"Generated {today:%Y-%m-%d}. Walk-forward backtest, no random splits; lower is better for log loss, RPS and ECE.", "",
        f"## Development seasons", "", f"Everything before {holdout}. All model comparison happens here.", "",
        _table(development), "",
        "## Held-out season", "",
    ]
    if held_out is None:
        text.append(f"Not evaluated: the hybrid did not beat both Elo-only and Dixon-Coles-only on the development seasons, so {holdout} stays untouched.")
    else:
        text += [f"{holdout} was evaluated once, after the hybrid beat Elo-only and Dixon-Coles-only on the development seasons. "
                 "It is not used for tuning; the hybrid is the first release candidate.", "", _table(held_out)]
    Path(out_path).write_text("\n".join(text) + "\n")
    return {"development": development, "holdout": held_out, "holdout_season": holdout}
