"""Development-season tuning of the league model: Elo settings, goal-model decay and inputs.

Reads only seasons before the hold-out season, which is never evaluated here. Stages run in
order so each search stays small: Elo settings (judged by the Elo-only logistic), then the
goal-model decay (judged by Dixon-Coles alone), then the hybrid's inputs and strength.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd

from footpred.evaluate.backtest import backtest
from footpred.evaluate.baselines import baseline_dixon_coles, baseline_elo
from footpred.evaluate.metrics import accuracy, encode_outcome, log_loss
from footpred.features.builder import ALL_COLUMNS, FEATURE_COLUMNS, build_features
from footpred.pipeline.benchmark import CAL_FRACTION, _hybrid_fit_predict, _holdout_season
from footpred.pipeline.common import load_all_matches, played_before
from footpred.ratings.dixon_coles import dc_pre_match
from footpred.ratings.elo import EloParams, run_elo

ELO_GRID = {"k": (15.0, 20.0, 30.0), "home_adv": (50.0, 70.0), "season_regress": (0.15, 0.25, 0.35), "margin": (True, False)}
XI_GRID = (0.0008, 0.0013, 0.0018, 0.0025, 0.0035)
NO_SHOTS = [c for c in FEATURE_COLUMNS if "shots" not in c]
EXTRAS = [c for c in ALL_COLUMNS if c not in FEATURE_COLUMNS]
INPUT_SETS = {
    "current inputs": FEATURE_COLUMNS,
    "without shot form": NO_SHOTS,
    "current + new form inputs": ALL_COLUMNS,
    "without shot form + new form inputs": NO_SHOTS + EXTRAS,
}
STRENGTHS = (0.03, 0.1, 0.3)


def _score(preds: pd.DataFrame) -> tuple[float, float]:
    probs = preds[["p_home", "p_draw", "p_away"]].to_numpy()
    y = preds["outcome"].to_numpy(dtype=int)
    return float(log_loss(probs, y)), float(accuracy(probs, y))


def tune_league(data_dir: Path, out_path: Path, aliases_path: Path | None = None,
                today: pd.Timestamp | None = None, test_seasons: int = 3, step_days: int = 60,
                elo_grid: dict | None = None, xi_grid: tuple | None = None) -> dict:
    today = pd.Timestamp.now().normalize() if today is None else pd.Timestamp(today)
    played = played_before(load_all_matches(data_dir, aliases_path), today + pd.Timedelta(days=1))
    holdout = _holdout_season(played)
    dev = played[played["season"] < holdout].reset_index(drop=True)
    seasons = sorted(dev["season"].unique())
    if len(seasons) > 1:
        first_test = dev.loc[dev["season"] == seasons[max(1, len(seasons) - test_seasons)], "date"].min()
    else:
        first_test = dev["date"].quantile(0.5)
    outcome = pd.Series(encode_outcome(dev["result"]), index=pd.Index(dev["match_id"], name="match_id"))
    dates = dev.set_index("match_id")["date"]

    # Stage 1: Elo settings
    elo_rows = []
    grid = elo_grid or ELO_GRID
    for k, adv, reg, margin in itertools.product(grid["k"], grid["home_adv"], grid["season_regress"], grid["margin"]):
        params = EloParams(k=k, home_adv=adv, season_regress=reg, margin=margin)
        pre, _ = run_elo(dev, params)
        frame = pd.DataFrame({"elo_diff": pre["elo_home_pre"] - pre["elo_away_pre"]})
        ll, acc = _score(backtest(baseline_elo, frame, dev, first_test, step_days=step_days))
        elo_rows.append({"k": k, "home_adv": adv, "season_regress": reg, "margin": margin, "log_loss": ll, "accuracy": acc})
    best_elo = min(elo_rows, key=lambda r: r["log_loss"])
    elo_params = EloParams(k=best_elo["k"], home_adv=best_elo["home_adv"],
                           season_regress=best_elo["season_regress"], margin=best_elo["margin"])

    # Stage 2: goal-model time decay
    xi_rows = []
    for xi in (xi_grid or XI_GRID):
        dc = dc_pre_match(dev, xi=xi)
        ll, acc = _score(backtest(baseline_dixon_coles, dc, dev, first_test, step_days=step_days))
        xi_rows.append({"xi": xi, "log_loss": ll, "accuracy": acc})
    best_xi = min(xi_rows, key=lambda r: r["log_loss"])["xi"]

    # Stage 3: hybrid inputs and strength, on the chosen rating settings
    pre, _ = run_elo(dev, elo_params)
    features = build_features(dev, pre, dc_pre_match(dev, xi=best_xi), ALL_COLUMNS)
    hybrid_rows = []
    for (label, columns), c in itertools.product(INPUT_SETS.items(), STRENGTHS):
        params = {"backend": "logistic", "C": c, "features": columns}
        preds = backtest(_hybrid_fit_predict(dates, params), features, dev, first_test, step_days=step_days)
        ll, acc = _score(preds)
        hybrid_rows.append({"inputs": label, "C": c, "log_loss": ll, "accuracy": acc})
    best_hybrid = min(hybrid_rows, key=lambda r: r["log_loss"])

    lines = ["# League tuning (development seasons only)", "",
             f"Seasons before {holdout}; walk-forward from {first_test:%Y-%m-%d}. Lower log loss is better. "
             "The hold-out season is not read here.", "",
             "## Stage 1: Elo settings (Elo-only logistic)", "",
             "| k | home adv | regress | margin | correct picks | log loss |", "|---|---|---|---|---|---|"]
    for r in sorted(elo_rows, key=lambda r: r["log_loss"]):
        lines.append(f"| {r['k']:g} | {r['home_adv']:g} | {r['season_regress']:g} | {r['margin']} | {r['accuracy']:.1%} | {r['log_loss']:.4f} |")
    lines += ["", "## Stage 2: goal-model decay (Dixon-Coles only)", "", "| xi | correct picks | log loss |", "|---|---|---|"]
    for r in sorted(xi_rows, key=lambda r: r["log_loss"]):
        lines.append(f"| {r['xi']:g} | {r['accuracy']:.1%} | {r['log_loss']:.4f} |")
    lines += ["", f"## Stage 3: hybrid inputs (Elo {elo_params}, xi {best_xi:g})", "",
              "| inputs | C | correct picks | log loss |", "|---|---|---|---|"]
    for r in sorted(hybrid_rows, key=lambda r: r["log_loss"]):
        lines.append(f"| {r['inputs']} | {r['C']:g} | {r['accuracy']:.1%} | {r['log_loss']:.4f} |")
    lines += ["", f"Chosen: Elo k={elo_params.k:g}, home adv {elo_params.home_adv:g}, regress {elo_params.season_regress:g}, "
              f"margin {elo_params.margin}; xi {best_xi:g}; inputs '{best_hybrid['inputs']}' at C={best_hybrid['C']:g}."]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines) + "\n")
    return {"elo": best_elo, "xi": best_xi, "hybrid": best_hybrid, "holdout_season": holdout,
            "elo_rows": elo_rows, "xi_rows": xi_rows, "hybrid_rows": hybrid_rows}
