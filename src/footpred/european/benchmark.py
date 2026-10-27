"""Walk-forward benchmark for the Champions League model, with a protected hold-out season.

Every choice is made on development seasons (before ``HOLDOUT_SEASON``). The hold-out season is
evaluated once, and only if the publish gate passes on the development seasons. Models are
trained on every European competition but scored on Champions League matches, the only
competition that is published.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from footpred.european.history import load_european_matches
from footpred.european.model import (
    EURO_FEATURES,
    european_features,
    european_publish_gate,
    fit_european,
    predict_european,
)
from footpred.european.names import EuropeanNames, group_for_country
from footpred.european.pooled import PooledParams, fit_offsets, run_pooled_elo
from footpred.evaluate.backtest import backtest
from footpred.evaluate.baselines import baseline_base_rate, baseline_elo
from footpred.evaluate.metrics import encode_outcome
from footpred.pipeline.benchmark import _summarize
from footpred.pipeline.common import PLAYED, load_all_matches

HOLDOUT_SEASON = "2025-26"
PUBLISHED_COMPETITION = "CL"
CAL_FRACTION = 0.15
LABELS = {"base_rate": "Base rate", "elo": "Pooled Elo only", "model": "Model"}


def team_groups(league: pd.DataFrame, euro: pd.DataFrame) -> dict[str, str]:
    """League membership decides a club's group; otherwise the country tag in the European files."""
    groups: dict[str, str] = {}
    for code, df in league.groupby("league"):
        for team in set(df["home"]) | set(df["away"]):
            groups[team] = str(code)
    for side, country in (("home", "home_country"), ("away", "away_country")):
        for team, tag in zip(euro[side], euro[country]):
            if isinstance(tag, str):
                groups.setdefault(team, group_for_country(tag))
    return groups


def pooled_matches(league: pd.DataFrame, euro: pd.DataFrame) -> pd.DataFrame:
    """League and European matches in one frame for the pooled Elo (the final is on neutral ground)."""
    columns = ["match_id", "date", "season", "home", "away", "home_goals", "away_goals", "neutral", "competition"]
    lg = league.assign(neutral=False, competition=league["league"])
    eu = euro.assign(neutral=euro["stage"].fillna("") == "FINAL")
    return pd.concat([lg[columns], eu[columns]], ignore_index=True)


def _model_fit_predict(dates: pd.Series, params: dict | None = None, calibrate: bool = True, features: list[str] | None = None):
    def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        order = dates.reindex(train.index).sort_values(kind="stable").index
        train = train.loc[order]
        cut = int(len(train) * (1 - CAL_FRACTION))
        fit, cal = train.iloc[:cut], train.iloc[cut:]
        model, calibrator = fit_european(fit[EURO_FEATURES], fit["outcome"].to_numpy(), cal[EURO_FEATURES], cal["outcome"].to_numpy(),
                                         params=params, calibrate=calibrate, features=features)
        return predict_european(model, calibrator, test)

    return fit_predict


def _table(results: dict) -> str:
    lines = ["| Model | Matches | Correct picks | Log loss | RPS | ECE |", "|---|---|---|---|---|---|"]
    for key, label in LABELS.items():
        r = results[key]
        lines.append(f"| {label} | {r['n']} | {r['accuracy']:.1%} | {r['log_loss']:.3f} | {r['rps']:.3f} | {r['ece']:.3f} |")
    return "\n".join(lines)


def run_european_benchmark(
    data_dir: Path,
    out_path: Path,
    aliases_path: Path | None = None,
    today: pd.Timestamp | None = None,
    test_seasons: int = 3,
    step_days: int = 60,
) -> dict:
    today = pd.Timestamp.now().normalize() if today is None else pd.Timestamp(today)
    cutoff = today + pd.Timedelta(days=1)
    names = EuropeanNames(league_aliases_path=aliases_path)

    league = load_all_matches(data_dir, aliases_path)
    league = league[league["result"].isin(PLAYED) & (league["date"] < cutoff) & (league["season"] <= HOLDOUT_SEASON)]
    euro = load_european_matches(data_dir, names)
    euro = euro[euro["result"].isin(PLAYED) & (euro["date"] < cutoff) & (euro["season"] <= HOLDOUT_SEASON)]
    euro = euro.reset_index(drop=True)
    dev_euro = euro[euro["season"] < HOLDOUT_SEASON]
    if dev_euro.empty:
        raise ValueError(f"no European results before {HOLDOUT_SEASON} to develop on")

    seasons = sorted(dev_euro["season"].unique())
    if len(seasons) > 1:  # test on the last `test_seasons` development seasons, always keeping one to train on
        first_test = dev_euro.loc[dev_euro["season"] == seasons[max(1, len(seasons) - test_seasons)], "date"].min()
    else:
        first_test = dev_euro["date"].quantile(0.5)

    groups = team_groups(league, euro)
    pool = pooled_matches(league, euro)
    offsets = fit_offsets(pool, groups, first_test)  # development matches only: nothing from first_test on
    pre, _ = run_pooled_elo(pool, groups, PooledParams(offsets=offsets))
    features = european_features(euro, pre)
    outcome = pd.Series(encode_outcome(euro["result"]), index=pd.Index(euro["match_id"], name="match_id"))
    dates = euro.set_index("match_id")["date"]
    matches = euro.assign(league=euro["competition"])
    dev_matches = matches[matches["season"] < HOLDOUT_SEASON].reset_index(drop=True)

    fit_predicts = {"base_rate": baseline_base_rate, "elo": baseline_elo, "model": _model_fit_predict(dates)}
    development = {}
    for name, fp in fit_predicts.items():
        preds = backtest(fp, features.loc[dev_matches["match_id"]], dev_matches, first_test, step_days=step_days)
        preds = preds[preds["league"] == PUBLISHED_COMPETITION]
        if preds.empty:
            raise ValueError("no Champions League matches in the test window")
        development[name] = _summarize(preds)

    publish_ok, reason = european_publish_gate(development)
    held_out = None
    if publish_ok:  # the one and only look at the held-out season
        test = matches[(matches["season"] == HOLDOUT_SEASON) & (matches["competition"] == PUBLISHED_COMPETITION)]
        if not test.empty:
            ids = dev_matches["match_id"]
            train = features.loc[ids].assign(outcome=outcome.reindex(ids).to_numpy())
            y = outcome.reindex(test["match_id"]).to_numpy()
            held_out = {}
            for name, fp in fit_predicts.items():
                p = np.asarray(fp(train, features.loc[test["match_id"]]), dtype=float)
                held_out[name] = _summarize(pd.DataFrame(
                    {"p_home": p[:, 0], "p_draw": p[:, 1], "p_away": p[:, 2], "outcome": y,
                     "league": test["competition"].to_numpy()}))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    text = [
        "# Champions League model results", "",
        f"Generated {today:%Y-%m-%d}. Predictions are for the 90-minute result. Walk-forward backtest, no random "
        "splits; lower is better for log loss, RPS and ECE. Models learn from every European competition and are "
        "scored on Champions League matches only.", "",
        "## Development seasons", "",
        f"Everything before {HOLDOUT_SEASON}; tested on the last {min(test_seasons, max(1, len(seasons) - 1))} of them. "
        f"Publish gate: {reason}.", "",
        _table(development), "",
        "Starting-rating offsets fitted on development seasons: " + ", ".join(f"{g} {v:+.0f}" for g, v in offsets.items()) + ".", "",
        "## Held-out season", "",
    ]
    if not publish_ok:
        text.append(f"Not evaluated: the model did not pass the publish gate on the development seasons, so {HOLDOUT_SEASON} stays untouched.")
    elif held_out is None:
        text.append(f"The gate passed, but there are no Champions League results for {HOLDOUT_SEASON} yet.")
    else:
        text += [f"{HOLDOUT_SEASON} was evaluated once, after the model passed the publish gate. It is not used for tuning.", "", _table(held_out)]
    Path(out_path).write_text("\n".join(text) + "\n")
    return {"development": development, "holdout": held_out, "holdout_season": HOLDOUT_SEASON,
            "publish_ok": publish_ok, "reason": reason, "offsets": offsets}
