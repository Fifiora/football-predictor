"""Baseline-versus-hybrid benchmark with a protected hold-out season.

Development seasons (everything before the most recent full season) are used for all
comparison and tuning. The most recent full season is evaluated only after a model is chosen on the development seasons, and only if the hybrid
beats both Elo-only and Dixon-Coles-only on the development seasons.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from footpred.evaluate.backtest import backtest
from footpred.evaluate.baselines import benchmark_closing_odds, baseline_base_rate, baseline_blend, baseline_dixon_coles, baseline_elo
from footpred.evaluate.metrics import accuracy, expected_calibration_error, log_loss, rps
from footpred.features.builder import FEATURE_COLUMNS
from footpred.models.calibration import Calibrator
from footpred.models.meta_model import MetaModel
from footpred.pipeline.common import load_all_matches, per_league_log_loss, played_before, training_frame

CAL_FRACTION = 0.15
MODELS = {"base_rate": "Base rate", "elo": "Elo only", "dixon_coles": "Dixon-Coles only",
          "blend": "Blend (Elo + Dixon-Coles)", "hybrid": "Hybrid"}
VARIANT_LABELS = {"hybrid_blend": "Hybrid (blend backend, calibrated)", "hybrid_lean": "Hybrid (logistic, core inputs only)", "hybrid_logistic": "Hybrid (logistic)", "hybrid_regularized": "Hybrid (regularized trees)"}


def default_variants() -> dict[str, dict | None]:
    """Hybrid meta-model candidates. Tree parameter names depend on the available backend."""
    if MetaModel().backend == "lightgbm":
        regularized = {"n_estimators": 150, "learning_rate": 0.03, "num_leaves": 4, "min_child_samples": 100, "reg_lambda": 10.0}
    else:
        regularized = {"max_iter": 150, "learning_rate": 0.03, "max_depth": 2, "min_samples_leaf": 100, "l2_regularization": 10.0}
    return {
        "hybrid": None,
        "hybrid_regularized": regularized,
        "hybrid_logistic": {"backend": "logistic", "C": 0.1},
        "hybrid_blend": {"backend": "blend"},
        "hybrid_lean": {"backend": "logistic", "C": 0.1, "features": ["elo_diff", "dc_p_home", "dc_p_draw", "dc_p_away"]},
    }


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


def _hybrid_fit_predict(dates: pd.Series, params: dict | None = None):
    def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        order = dates.reindex(train.index).sort_values(kind="stable").index
        train = train.loc[order]
        cut = int(len(train) * (1 - CAL_FRACTION))
        fit, cal = train.iloc[:cut], train.iloc[cut:]
        columns = (params or {}).get("features") or FEATURE_COLUMNS  # optional restriction to core inputs
        model = MetaModel(params).fit(fit[columns], fit["outcome"].to_numpy())
        calibrator = Calibrator().fit(model.predict_proba(cal[columns]), cal["outcome"].to_numpy())
        return calibrator.transform(model.predict_proba(test[columns]))

    return fit_predict


def _summarize(preds: pd.DataFrame) -> dict:
    probs = preds[["p_home", "p_draw", "p_away"]].to_numpy()
    y = preds["outcome"].to_numpy(dtype=int)
    leagues = preds["league"].to_numpy()
    return {
        "n": int(len(preds)),
        "log_loss": float(log_loss(probs, y)),
        "rps": float(rps(probs, y)),
        "ece": float(expected_calibration_error(probs, y)),
        "accuracy": float(accuracy(probs, y)),
        "by_league_accuracy": {
            str(lg): float(accuracy(probs[leagues == lg], y[leagues == lg])) for lg in np.unique(leagues)
        },
        "by_league_log_loss": per_league_log_loss(probs, y, preds["league"].to_numpy()),
    }


def bookmaker_reference(preds: pd.DataFrame, matches: pd.DataFrame) -> dict | None:
    """Model versus de-vigged closing odds on the same matches. Comparison only: odds are never an input."""
    odds = matches.set_index("match_id")[["close_h", "close_d", "close_a"]]
    merged = preds.join(odds, on="match_id")
    usable = merged[["close_h", "close_d", "close_a"]].notna().all(axis=1) & (merged[["close_h", "close_d", "close_a"]] > 1).all(axis=1)
    merged = merged[usable]
    if merged.empty:
        return None
    y = merged["outcome"].to_numpy(dtype=int)
    book = benchmark_closing_odds(merged)
    model = merged[["p_home", "p_draw", "p_away"]].to_numpy()
    return {"n": int(len(y)),
            "model": {"log_loss": float(log_loss(model, y)), "rps": float(rps(model, y)), "accuracy": float(accuracy(model, y))},
            "bookmaker": {"log_loss": float(log_loss(book, y)), "rps": float(rps(book, y)), "accuracy": float(accuracy(book, y))}}


def _bookmaker_table(ref: dict | None) -> str:
    if ref is None:
        return "No closing odds were available for these matches."
    rows = ["| Source | Matches | Correct picks | Log loss | RPS |", "|---|---|---|---|---|"]
    for key, label in (("model", "Our model (Hybrid)"), ("bookmaker", "Bookmaker closing odds (reference only)")):
        r = ref[key]
        rows.append(f"| {label} | {ref['n']} | {r['accuracy']:.1%} | {r['log_loss']:.3f} | {r['rps']:.3f} |")
    return "\n".join(rows)


def _table(results: dict) -> str:
    leagues = sorted({lg for r in results.values() for lg in r["by_league_log_loss"]})
    lines = ["| Model | Matches | Correct picks | Log loss | RPS | ECE | " + " | ".join(leagues) + " |",
             "|---|---|---|---|---|---|" + "---|" * len(leagues)]
    labels = {**MODELS, **VARIANT_LABELS}
    for key, label in labels.items():
        if key in results:
            r = results[key]
            per = " | ".join(f"{r['by_league_log_loss'].get(lg, float('nan')):.3f}" for lg in leagues)
            lines.append(f"| {label} | {r['n']} | {r['accuracy']:.1%} | {r['log_loss']:.3f} | {r['rps']:.3f} | {r['ece']:.3f} | {per} |")
    return "\n".join(lines)


def _accuracy_table(results: dict) -> str:
    leagues = sorted({lg for r in results.values() for lg in r["by_league_accuracy"]})
    lines = ["Correct picks by league (the most probable outcome was what happened):", "",
             "| Model | " + " | ".join(leagues) + " |", "|---|" + "---|" * len(leagues)]
    for key, label in {**MODELS, **VARIANT_LABELS}.items():
        if key in results:
            per = " | ".join(f"{results[key]['by_league_accuracy'].get(lg, float('nan')):.1%}" for lg in leagues)
            lines.append(f"| {label} | {per} |")
    return "\n".join(lines)


def run_benchmark(
    data_dir: Path,
    out_path: Path,
    aliases_path: Path | None = None,
    today: pd.Timestamp | None = None,
    step_days: int = 60,
    variants: dict | None = None,
    test_seasons: int = 3,
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
    if len(seasons) > 1:  # test on the last `test_seasons` development seasons, always keeping one to train on
        first_test = dev_matches.loc[dev_matches["season"] == seasons[max(1, len(seasons) - test_seasons)], "date"].min()
    else:
        first_test = dev_matches["date"].quantile(0.5)

    variants = variants if variants is not None else default_variants()
    fit_predicts = {"base_rate": baseline_base_rate, "elo": baseline_elo, "dixon_coles": baseline_dixon_coles,
                    "blend": baseline_blend, **{name: _hybrid_fit_predict(dates, params) for name, params in variants.items()}}
    dev_features = features.loc[dev_ids]
    dev_preds = {name: backtest(fp, dev_features, dev_matches, first_test, step_days=step_days)
                 for name, fp in fit_predicts.items()}
    development = {name: _summarize(preds) for name, preds in dev_preds.items()}

    chosen = min(variants, key=lambda name: development[name]["log_loss"])  # chosen on development seasons only
    beats_both = development[chosen]["log_loss"] < min(
        development["elo"]["log_loss"], development["dixon_coles"]["log_loss"]
    )
    dev_book = bookmaker_reference(dev_preds[chosen], played)
    held_out = None
    book_held = None
    if beats_both:  # the one and only look at the held-out season
        train = features.loc[dev_ids].assign(outcome=outcome.reindex(dev_ids).to_numpy())
        test_ids = features.index[is_holdout.to_numpy()]
        test = features.loc[test_ids]
        y = outcome.reindex(test_ids).to_numpy()
        leagues = meta["league"].reindex(test_ids).to_numpy()
        held_out = {}
        final = {"base_rate": baseline_base_rate, "elo": baseline_elo, "dixon_coles": baseline_dixon_coles,
                 "blend": baseline_blend, "hybrid": fit_predicts[chosen]}  # only the chosen variant is evaluated on the held-out season
        for name, fp in final.items():
            p = np.asarray(fp(train, test), dtype=float)
            held_out[name] = _summarize(pd.DataFrame(
                {"p_home": p[:, 0], "p_draw": p[:, 1], "p_away": p[:, 2], "outcome": y, "league": leagues}))
            if name == "hybrid":
                book_held = bookmaker_reference(pd.DataFrame(
                    {"match_id": test_ids, "p_home": p[:, 0], "p_draw": p[:, 1], "p_away": p[:, 2], "outcome": y}), played)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    text = [
        "# Baseline versus hybrid results", "",
        f"Generated {today:%Y-%m-%d}. Walk-forward backtest, no random splits; lower is better for log loss, RPS and ECE.", "",
        f"## Development seasons", "",
        f"Everything before {holdout}; tested on the last {min(test_seasons, max(1, len(seasons) - 1))} of them. "
        f"All model comparison happens here. Best hybrid variant on these seasons: {chosen}.", "",
        _table(development), "", _accuracy_table(development), "",
        "### Against the bookmakers", "",
        "Closing odds with the margin removed, on the same development matches as the chosen hybrid. "
        "A reference point only: odds are never a model input.", "",
        _bookmaker_table(dev_book), "",
        "## Held-out season", "",
    ]
    if held_out is None:
        text.append(f"Not evaluated: the hybrid did not beat both Elo-only and Dixon-Coles-only on the development seasons, so {holdout} stays untouched.")
    else:
        text += [f"{holdout} was evaluated after the hybrid beat Elo-only and Dixon-Coles-only on the development seasons. "
                 f"Only the chosen variant ({chosen}) was evaluated, shown as Hybrid. It is not used for tuning, though it has been looked at on more than one benchmark run while the project was built.", "", _table(held_out), "", _accuracy_table(held_out), "",
                 "Against the bookmakers on the held-out season (reference only):", "", _bookmaker_table(book_held)]
    Path(out_path).write_text("\n".join(text) + "\n")
    (Path(out_path).parent / "benchmark-reference.json").write_text(json.dumps(
        {"generated": f"{today:%Y-%m-%d}", "holdout_season": holdout, "development": dev_book, "holdout": book_held}, indent=2))
    return {"development": development, "holdout": held_out, "bookmaker_dev": dev_book, "bookmaker_holdout": book_held, "holdout_season": holdout, "chosen_hybrid": chosen}
