"""Development-season tuning of the Champions League model and its rating settings.

Reads only seasons before ``HOLDOUT_SEASON``; the hold-out season is never evaluated here.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd

from footpred.european.benchmark import (
    HOLDOUT_SEASON,
    PUBLISHED_COMPETITION,
    _model_fit_predict,
    pooled_matches,
    team_groups,
)
from footpred.european.history import load_european_matches
from footpred.european.model import european_features
from footpred.european.names import EuropeanNames
from footpred.european.pooled import PooledParams, fit_offsets, run_pooled_elo
from footpred.evaluate.backtest import backtest
from footpred.evaluate.baselines import baseline_base_rate, baseline_elo
from footpred.evaluate.metrics import accuracy, encode_outcome, log_loss
from footpred.pipeline.common import PLAYED, load_all_matches

RATING_GRID = {"k": (15.0, 25.0), "home_adv": (50.0, 80.0), "season_regress": (0.15, 0.3)}
VARIANTS = {
    "all3 C=0.1 calibrated (current)": ({"backend": "logistic", "C": 0.1}, True, None),
    "all3 C=1 calibrated": ({"backend": "logistic", "C": 1.0}, True, None),
    "all3 C=1 uncalibrated": ({"backend": "logistic", "C": 1.0}, False, None),
    "all3 C=10 uncalibrated": ({"backend": "logistic", "C": 10.0}, False, None),
    "elo_diff only C=10 uncalibrated": ({"backend": "logistic", "C": 10.0}, False, ["elo_diff"]),
}


def tune_european(data_dir: Path, out_path: Path, aliases_path: Path | None = None,
                  today: pd.Timestamp | None = None, test_seasons: int = 3, step_days: int = 60) -> list[dict]:
    today = pd.Timestamp.now().normalize() if today is None else pd.Timestamp(today)
    cutoff = today + pd.Timedelta(days=1)
    names = EuropeanNames(league_aliases_path=aliases_path)
    league = load_all_matches(data_dir, aliases_path)
    league = league[league["result"].isin(PLAYED) & (league["date"] < cutoff) & (league["season"] < HOLDOUT_SEASON)]
    euro = load_european_matches(data_dir, names)
    euro = euro[euro["result"].isin(PLAYED) & (euro["date"] < cutoff) & (euro["season"] < HOLDOUT_SEASON)].reset_index(drop=True)
    seasons = sorted(euro["season"].unique())
    first_test = euro.loc[euro["season"] == seasons[max(1, len(seasons) - test_seasons)], "date"].min()
    groups = team_groups(league, euro)
    pool = pooled_matches(league, euro)
    outcome = pd.Series(encode_outcome(euro["result"]), index=pd.Index(euro["match_id"], name="match_id"))
    dates = euro.set_index("match_id")["date"]
    matches = euro.assign(league=euro["competition"])
    coarse = tuple(float(x) for x in range(-300, 301, 50))

    rows: list[dict] = []
    for k, adv, reg in itertools.product(*RATING_GRID.values()):
        base = PooledParams(k=k, home_adv=adv, season_regress=reg)
        offsets = fit_offsets(pool, groups, first_test, base=base, grid=coarse)
        pre, _ = run_pooled_elo(pool, groups, PooledParams(k, adv, reg, 1500.0, offsets))
        features = european_features(euro, pre, home_adv=adv)
        candidates = {"base rate": baseline_base_rate, "pooled Elo baseline": baseline_elo,
                      **{name: _model_fit_predict(dates, p, cal, feats) for name, (p, cal, feats) in VARIANTS.items()}}
        for name, fp in candidates.items():
            preds = backtest(fp, features, matches, first_test, step_days=step_days)
            preds = preds[preds["league"] == PUBLISHED_COMPETITION]
            probs = preds[["p_home", "p_draw", "p_away"]].to_numpy()
            y = preds["outcome"].to_numpy(dtype=int)
            rows.append({"k": k, "home_adv": adv, "season_regress": reg, "variant": name, "n": len(y),
                         "log_loss": float(log_loss(probs, y)), "accuracy": float(accuracy(probs, y))})

    table = pd.DataFrame(rows)
    lines = ["# Champions League tuning (development seasons only)", "",
             f"Seasons before {HOLDOUT_SEASON}; walk-forward from {first_test:%Y-%m-%d}. Lower log loss is better.", "",
             "| k | home adv | regress | variant | matches | correct picks | log loss |", "|---|---|---|---|---|---|---|"]
    for r in table.sort_values("log_loss").itertuples():
        lines.append(f"| {r.k:g} | {r.home_adv:g} | {r.season_regress:g} | {r.variant} | {r.n} | {r.accuracy:.1%} | {r.log_loss:.4f} |")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines) + "\n")
    return rows
