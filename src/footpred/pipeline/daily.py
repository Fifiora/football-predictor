"""Daily refresh: ingest, score upcoming fixtures with the champion, log, grade, export."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from footpred.evaluate.drift import check_drift
from footpred.explain.reasons import top_drivers
from footpred.features.builder import FEATURE_COLUMNS, build_features
from footpred.ingest.fixtures import FixturesClient, fetch_latest_results, fetch_upcoming
from footpred.ingest.historical import season_label
from footpred.ingest.schema import MATCH_COLUMNS, validate_matches
from footpred.models.registry import load_champion
from footpred.pipeline.european import EuropeanScores, score_european
from footpred.pipeline.common import PLAYED, append_api_results, load_all_matches, played_before
from footpred.predictions.log import grade_predictions, log_predictions
from footpred.publish.export import export_site_data
from footpred.ratings.dixon_coles import DC_COLUMNS, DCUnavailable, fit_dixon_coles
from footpred.ratings.elo import run_elo


@dataclass
class DailyResult:
    new_results: int
    scored_fixtures: int
    published: bool
    warnings: list[str] = field(default_factory=list)


def _fixture_rows(upcoming: pd.DataFrame) -> pd.DataFrame:
    rows = upcoming.copy()
    rows["season"] = rows["date"].map(season_label)
    for col in MATCH_COLUMNS:
        if col not in rows.columns:
            rows[col] = np.nan
    return rows[MATCH_COLUMNS]


def run_daily(
    data_dir: Path,
    site_data_dir: Path,
    client: FixturesClient,
    today: pd.Timestamp,
    registry_dir: Path | None = None,
    aliases_path: Path | None = None,
    sleep: Callable[[float], None] = time.sleep,
    updated_at: pd.Timestamp | None = None,
    european: bool = False,
    fetch_odds: Callable[[], tuple[dict, list[str]]] | None = None,
    reference_path: Path | None = None,
) -> DailyResult:
    """Refresh predictions. Any failure before the final export leaves ``site_data_dir`` untouched."""
    data_dir = Path(data_dir)
    registry_dir = Path(registry_dir) if registry_dir else data_dir / "registry"
    today = pd.Timestamp(today)

    matches = load_all_matches(data_dir, aliases_path)  # raises on bad data, before anything is written
    leagues = sorted(matches["league"].unique())
    cache_dir = data_dir / "cache"

    upcoming, warnings = fetch_upcoming(client, leagues, cache_dir, today=today, sleep=sleep, aliases_path=aliases_path)
    since = (matches["date"].max() if len(matches) else today) - pd.Timedelta(days=7)
    results, result_warnings = fetch_latest_results(
        client, leagues, since, cache_dir, today=today, sleep=sleep, aliases_path=aliases_path
    )
    warnings = warnings + result_warnings

    new = results[~results["match_id"].isin(matches["match_id"])]
    euro = None
    if european:  # never raises for feed or model problems: those come back as warnings
        try:
            euro = score_european(
                data_dir, client, today, registry_dir, aliases_path=aliases_path, sleep=sleep,
                league_matches=matches if new.empty else pd.concat([matches, new], ignore_index=True),
            )
        except Exception as err:  # noqa: BLE001  the Champions League must never stop the league run
            euro = EuropeanScores(pd.DataFrame(columns=["match_id", "league", "date", "home", "away", "stage"]), {}, {},
                                  pd.DataFrame(columns=["match_id", "result"]),
                                  [f"Champions League skipped this run: {type(err).__name__}: {err}"], False)
        warnings = warnings + euro.warnings
    if new.empty and upcoming.empty and (euro is None or euro.nothing_new):
        return DailyResult(0, 0, False, warnings)  # nothing changed: keep the last good site
    if not new.empty:
        append_api_results(data_dir, new)
        matches = pd.concat([matches, new], ignore_index=True).sort_values(["date", "match_id"]).reset_index(drop=True)

    version, model, calibrator = load_champion(registry_dir)
    played = played_before(matches, today + pd.Timedelta(days=1))  # results up to and including today
    scored = upcoming.reset_index(drop=True)

    drivers: dict[str, list] = {}
    grids: dict[str, np.ndarray] = {}
    ratings: dict[str, float]
    if scored.empty:
        _, ratings = run_elo(played)
    else:
        combined = validate_matches(pd.concat([played, _fixture_rows(scored)], ignore_index=True), allow_unplayed=True)
        elo_pre, ratings = run_elo(combined)
        dc_model = fit_dixon_coles(played, as_of=today)
        dc_rows = {}
        for row in scored.itertuples(index=False):
            try:
                pred = dc_model.predict(row.home, row.away, league=row.league)
            except DCUnavailable:
                continue
            grids[row.match_id] = pred.grid
            dc_rows[row.match_id] = [pred.exp_home_goals, pred.exp_away_goals, pred.p_home, pred.p_draw, pred.p_away]
        dc_pre = pd.DataFrame.from_dict(dc_rows, orient="index", columns=DC_COLUMNS)
        dc_pre.index.name = "match_id"

        features = build_features(combined, elo_pre, dc_pre)
        X = features.loc[scored["match_id"], FEATURE_COLUMNS]
        probs = calibrator.transform(model.predict_proba(X))
        scored = scored.assign(p_home=probs[:, 0], p_draw=probs[:, 1], p_away=probs[:, 2])
        for row in scored.itertuples(index=False):
            drivers[row.match_id] = top_drivers(model, X.loc[row.match_id], home=row.home, away=row.away)

    european_results = None
    if euro is not None:
        european_results = euro.results
        if not euro.fixtures.empty:  # the league columns already in `scored` are untouched
            scored = pd.concat([scored, euro.fixtures], ignore_index=True)
            drivers.update(euro.drivers)
            grids.update(euro.grids)
    if not scored.empty:
        future = scored[scored["date"] > today].assign(kickoff_date=lambda df: df["date"])
        if len(future) < len(scored):
            warnings.append(f"{len(scored) - len(future)} same-day fixtures shown but not logged (no kickoff times)")
        log_predictions(data_dir / "prediction_log.csv", future, version, logged_at=today)

    graded_inputs = played[played["result"].isin(PLAYED)]
    if european_results is not None and len(european_results):
        graded_inputs = pd.concat([graded_inputs[["match_id", "result"]], european_results[["match_id", "result"]]], ignore_index=True)
    graded = grade_predictions(data_dir / "prediction_log.csv", graded_inputs)
    metadata_path = registry_dir / version / "metadata.json"
    baseline = json.loads(metadata_path.read_text()).get("metrics", {}).get("log_loss") if metadata_path.exists() else None
    if len(graded):
        warnings += check_drift(graded, baseline_log_loss=baseline)

    odds: dict = {}
    if fetch_odds is not None:  # bookmaker odds are display-only extras; a failure only adds a warning
        try:
            odds, odds_warnings = fetch_odds()
            warnings += odds_warnings
            # the odds file only lists the coming round, so only near-term league fixtures are expected in it
            league_fixtures = scored[(scored["league"] != "CL") & (scored["date"] <= today + pd.Timedelta(days=3))] if len(scored) else scored
            have = sum(1 for m in league_fixtures["match_id"] if m in odds) if len(league_fixtures) else 0
            if len(league_fixtures) and have < len(league_fixtures):
                missing = league_fixtures[~league_fixtures["match_id"].isin(odds)]
                names = "; ".join(f"{d:%d %b} {h} v {a}" for d, h, a in zip(missing["date"], missing["home"], missing["away"]))
                warnings.append(f"bookmaker odds found for {have} of {len(league_fixtures)} league fixtures; none for: {names}")
        except Exception as err:  # noqa: BLE001
            warnings.append(f"bookmaker odds unavailable: {type(err).__name__}: {err}")

    reference = None
    if reference_path is not None and Path(reference_path).exists():
        try:
            reference = json.loads(Path(reference_path).read_text())
        except (OSError, ValueError):
            warnings.append("benchmark reference file could not be read")

    active = set(played.loc[played["date"] >= today - pd.Timedelta(days=400), ["home", "away"]].to_numpy().ravel())
    export_site_data(
        site_data_dir, scored, drivers, grids, graded,
        {t: r for t, r in ratings.items() if t in active}, updated_at if updated_at is not None else today,
        model_version=version, odds=odds, reference=reference,
    )
    return DailyResult(len(new) + (euro.new_results if euro is not None else 0), len(scored), True, warnings)
