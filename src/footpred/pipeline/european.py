"""Champions League track of the weekly retrain and the daily refresh.

Its registry lives in ``<registry_dir>/european`` so it can never replace or disturb the league champion.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from footpred.european.benchmark import PUBLISHED_COMPETITION, pooled_matches, team_groups
from footpred.european.history import (
    EURO_COLUMNS,
    append_european_results,
    load_european_history,
    load_european_matches,
    read_european_results,
)
from footpred.european.model import (
    MODEL_PARAMS,
    european_features,
    european_publish_gate,
    fit_european,
    predict_european,
)
from footpred.european.names import EuropeanNames
from footpred.european.pooled import PooledParams, fit_offsets, run_pooled_elo
from footpred.evaluate.baselines import baseline_base_rate, baseline_elo
from footpred.evaluate.gate import GateResult, should_promote
from footpred.evaluate.metrics import encode_outcome, expected_calibration_error, log_loss, rps
from footpred.explain.reasons import top_drivers
from footpred.ingest.fixtures import FixturesClient, FixturesUnavailable, fetch_latest_results, fetch_upcoming
from footpred.ingest.historical import season_label
from footpred.models.registry import load_champion, save_model, set_champion
from footpred.pipeline.common import PLAYED, code_version, load_all_matches, played_before
from footpred.pipeline.retrain import CAL_FRACTION, EVAL_FRACTION, _next_version

REGISTRY_SUBDIR = "european"
MIN_GATE_ROWS = 50        # the publish gate looks at Champions League matches when the eval block has this many
RESULTS_LOOKBACK_DAYS = 60  # longest feed window requested; the history files usually lag the feed
PROB_COLUMNS = ["p_home", "p_draw", "p_away"]


@dataclass
class EuropeanScores:
    fixtures: pd.DataFrame
    drivers: dict
    grids: dict
    results: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    published: bool = False
    new_results: int = 0
    version: str = ""  # the European champion that priced these fixtures

    @property
    def nothing_new(self) -> bool:
        return self.fixtures.empty and self.new_results == 0


def _empty_fixtures() -> pd.DataFrame:
    return pd.DataFrame(columns=["match_id", "league", "date", "home", "away", "stage", *PROB_COLUMNS])


def run_european_retrain(
    data_dir: Path,
    registry_dir: Path,
    today: pd.Timestamp,
    aliases_path: Path | None = None,
    gate_min_matches: int = 100,
) -> GateResult:
    """Train on every played European match before ``today`` (70/15/15 time blocks) and gate it."""
    registry = Path(registry_dir) / REGISTRY_SUBDIR
    today = pd.Timestamp(today)
    names = EuropeanNames(league_aliases_path=aliases_path)
    league = played_before(load_all_matches(data_dir, aliases_path), today)
    euro = load_european_matches(data_dir, names)
    euro = euro[euro["result"].isin(PLAYED) & (euro["date"] < today)].reset_index(drop=True)

    dates = euro["date"].sort_values().reset_index(drop=True)
    if len(dates) < 30:
        raise ValueError("not enough European history to train on")
    cal_start = dates.iloc[int(len(dates) * (1 - EVAL_FRACTION - CAL_FRACTION))]
    eval_start = dates.iloc[int(len(dates) * (1 - EVAL_FRACTION))]
    train_m = (euro["date"] < cal_start).to_numpy()
    cal_m = ((euro["date"] >= cal_start) & (euro["date"] < eval_start)).to_numpy()
    eval_m = (euro["date"] >= eval_start).to_numpy()
    if not (train_m.any() and cal_m.any() and eval_m.any()):
        raise ValueError("not enough European history to split into train, calibration and evaluation blocks")

    groups = team_groups(league, euro)
    pool = pooled_matches(league, euro)
    offsets = fit_offsets(pool, groups, cal_start)  # nothing from the calibration or evaluation blocks
    pre, _ = run_pooled_elo(pool, groups, PooledParams(offsets=offsets))
    X = european_features(euro, pre)
    y = encode_outcome(euro["result"])

    model, calibrator = fit_european(X[train_m], y[train_m], X[cal_m], y[cal_m])
    X_eval, y_eval = X[eval_m], y[eval_m]
    challenger = predict_european(model, calibrator, X_eval)

    train_df = X[train_m].assign(outcome=y[train_m])
    focus = (euro.loc[eval_m, "competition"] == PUBLISHED_COMPETITION).to_numpy()
    if focus.sum() < MIN_GATE_ROWS:
        focus = np.ones(len(y_eval), dtype=bool)
    gate_ok, gate_reason = european_publish_gate({
        "base_rate": {"log_loss": float(log_loss(baseline_base_rate(train_df, X_eval)[focus], y_eval[focus]))},
        "elo": {"log_loss": float(log_loss(baseline_elo(train_df, X_eval)[focus], y_eval[focus]))},
        "model": {"log_loss": float(log_loss(challenger[focus], y_eval[focus]))},
    })
    metrics = {
        "n_eval": int(len(y_eval)),
        "log_loss": float(log_loss(challenger, y_eval)),
        "rps": float(rps(challenger, y_eval)),
        "ece": float(expected_calibration_error(challenger, y_eval)),
    }
    version = _next_version(registry, today)
    save_model(
        registry, version, model, calibrator,
        {"data_snapshot": str(euro["date"].max().date()), "code_version": code_version(), "params": MODEL_PARAMS,
         "metrics": metrics, "trained_on": int(train_m.sum()), "offsets": offsets,
         "publish_gate": {"ok": bool(gate_ok), "reason": gate_reason}},
    )

    try:
        _, champ_model, champ_cal = load_champion(registry)
    except FileNotFoundError:
        champ_model = None
    if champ_model is None:
        result = GateResult(True, "first model: there is no champion to beat", float("nan"),
                            metrics["log_loss"], float("nan"), metrics["rps"])
    else:
        champion = champ_cal.transform(champ_model.predict_proba(X_eval))
        result = should_promote(champion, challenger, y_eval, min_matches=gate_min_matches)
    if not gate_ok:  # no model, first or later, goes live without passing the publish gate
        result = GateResult(False, f"publish gate failed: {gate_reason}", result.champion_log_loss,
                            result.challenger_log_loss, result.champion_rps, result.challenger_rps)
    if result.promote:
        set_champion(registry, version)
    return result


def _feed_rows(got: pd.DataFrame) -> pd.DataFrame:
    """Feed results in the European history schema (the feed carries neither stage nor country)."""
    rows = pd.DataFrame({
        "match_id": got["match_id"], "competition": PUBLISHED_COMPETITION, "season": got["season"], "stage": None,
        "date": got["date"], "home": got["home"], "away": got["away"], "home_country": None, "away_country": None,
        "home_goals": got["home_goals"], "away_goals": got["away_goals"], "result": got["result"],
    })
    return rows[EURO_COLUMNS]


def score_european(
    data_dir: Path,
    client: FixturesClient,
    today: pd.Timestamp,
    registry_dir: Path,
    aliases_path: Path | None = None,
    sleep=time.sleep,
    league_matches: pd.DataFrame | None = None,
) -> EuropeanScores:
    """Fetch Champions League fixtures and results, store the results and score the fixtures.

    Problems here never stop the league run: they come back as warnings and an empty fixture list.
    ``league_matches`` lets the caller pass league results that are not saved yet.
    """
    data_dir = Path(data_dir)
    today = pd.Timestamp(today)
    registry = Path(registry_dir) / REGISTRY_SUBDIR
    warnings: list[str] = []

    live = EuropeanNames(league_aliases_path=aliases_path)  # spells the clubs of the live feed
    names = EuropeanNames(league_aliases_path=aliases_path)  # spells the clubs of the history files
    history = load_european_history(data_dir / "openfootball", names)
    live.learn(set(history["home"]) | set(history["away"]))
    before = load_european_matches(data_dir, names)
    latest = before["date"].max() if len(before) else today
    since = max(latest - pd.Timedelta(days=7), today - pd.Timedelta(days=RESULTS_LOOKBACK_DAYS))
    cache_dir = data_dir / "cache"
    try:
        upcoming, w_up = fetch_upcoming(client, [PUBLISHED_COMPETITION], cache_dir, today=today, sleep=sleep,
                                        aliases_path=aliases_path, canonical=live.canonical)
        got, w_res = fetch_latest_results(client, [PUBLISHED_COMPETITION], since, cache_dir, today=today, sleep=sleep,
                                          aliases_path=aliases_path, canonical=live.canonical)
    except FixturesUnavailable as err:
        warnings.append(f"Champions League feed unavailable, skipped this run: {err}")
        return EuropeanScores(_empty_fixtures(), {}, {}, read_european_results(data_dir), warnings)
    warnings += w_up + w_res
    warnings.append(f"Champions League feed: {len(upcoming)} upcoming, {len(got)} finished since {since:%Y-%m-%d}")

    if not got.empty:
        append_european_results(data_dir, _feed_rows(got))
    euro = load_european_matches(data_dir, names)
    new_results = len(euro) - len(before)
    results = read_european_results(data_dir)
    base = EuropeanScores(_empty_fixtures(), {}, {}, results, warnings, False, new_results)

    league = load_all_matches(data_dir, aliases_path) if league_matches is None else league_matches
    league = played_before(league, today + pd.Timedelta(days=1))
    known = set(history["home"]) | set(history["away"]) | set(league["home"]) | set(league["away"])
    live_clubs = set(upcoming["home"]) | set(upcoming["away"]) | set(got["home"]) | set(got["away"])
    warnings += [f"Champions League club not recognised, kept at the default rating: {club}" for club in sorted(live_clubs - known)]

    try:
        version, model, calibrator = load_champion(registry)
    except FileNotFoundError:
        warnings.append("Champions League predictions are not published: no model has passed the publish gate yet")
        return base
    meta_path = registry / version / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    gate = meta.get("publish_gate", {})
    if gate.get("ok") is False:
        warnings.append(f"Champions League predictions are not published: publish gate failed ({gate.get('reason', '')})")
        return base
    if upcoming.empty:
        base.published = True
        return base

    rows = upcoming.assign(
        competition=PUBLISHED_COMPETITION, season=upcoming["date"].map(season_label), home_country=None, away_country=None,
        home_goals=np.nan, away_goals=np.nan, result=None,
    )[EURO_COLUMNS]
    played = euro[euro["result"].isin(PLAYED) & (euro["date"] < today + pd.Timedelta(days=1))]
    with_fixtures = pd.concat([played, rows], ignore_index=True)
    groups = team_groups(league, with_fixtures)
    params = PooledParams(offsets=meta.get("offsets") or PooledParams().offsets)
    pre, _ = run_pooled_elo(pooled_matches(league, with_fixtures), groups, params)
    X = european_features(rows, pre)
    probs = predict_european(model, calibrator, X)
    unknown = live_clubs - known
    fixtures = upcoming.assign(p_home=probs[:, 0], p_draw=probs[:, 1], p_away=probs[:, 2],
                               low_confidence=[h in unknown or a in unknown for h, a in zip(upcoming['home'], upcoming['away'])])
    drivers = {r.match_id: top_drivers(model, X.loc[r.match_id], home=r.home, away=r.away) for r in fixtures.itertuples(index=False)}
    return EuropeanScores(fixtures, drivers, {}, results, warnings, True, new_results, version)
