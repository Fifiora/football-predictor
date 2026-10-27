"""Export site data as JSON, atomically.

Everything is written to a sibling temp directory and swapped into place, so a failed run never
leaves the site with partial or mixed-version data.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from footpred.evaluate.metrics import expected_calibration_error, log_loss, rps
from footpred.explain.reasons import Driver

_OUTCOME_COLS = ["p_home", "p_draw", "p_away"]


def _write_json(path: Path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False))


def _scoreline_fields(grid: np.ndarray) -> dict:
    grid = np.asarray(grid, dtype=float)
    grid = grid / grid.sum()
    home_goals, away_goals = np.indices(grid.shape)
    top = np.argsort(grid, axis=None)[::-1][:5]
    return {
        "btts": float(grid[(home_goals > 0) & (away_goals > 0)].sum()),
        "over_2_5": float(grid[(home_goals + away_goals) > 2].sum()),
        "top_scorelines": [
            {"score": f"{i}-{j}", "p": float(grid[i, j])} for i, j in (np.unravel_index(k, grid.shape) for k in top)
        ],
    }


def _fixtures_payload(fixtures, drivers, dc_grids, odds=None) -> list[dict]:
    out = []
    for row in fixtures.itertuples(index=False):
        probs = np.array([row.p_home, row.p_draw, row.p_away], dtype=float)
        probs = probs / probs.sum()
        item = {
            "match_id": row.match_id,
            "league": row.league,
            "date": pd.Timestamp(row.date).strftime("%Y-%m-%d"),
            "home": row.home,
            "away": row.away,
            "p_home": float(probs[0]),
            "p_draw": float(probs[1]),
            "p_away": float(probs[2]),
            "drivers": [
                {"feature": d.feature, "direction": d.direction, "strength": float(d.strength), "text": d.text}
                for d in drivers.get(row.match_id, [])
            ],
        }
        if getattr(row, "low_confidence", False) is True:  # a club the ratings do not know yet
            item["low_confidence"] = True
        book = (odds or {}).get(row.match_id)
        if book:  # bookmaker view, shown beside the model and never fed into it
            item["bookies"] = {name: {"p_home": float(p[0]), "p_draw": float(p[1]), "p_away": float(p[2])}
                               for name, p in book.items()}
        if row.match_id in dc_grids:
            item.update(_scoreline_fields(dc_grids[row.match_id]))
        out.append(item)
    return out


def _metrics(df: pd.DataFrame) -> dict:
    probs = df[_OUTCOME_COLS].to_numpy(dtype=float)
    outcomes = df["outcome"].to_numpy(dtype=int)
    return {
        "n": int(len(df)),
        "log_loss": float(log_loss(probs, outcomes)),
        "rps": float(rps(probs, outcomes)),
        "accuracy": float((probs.argmax(axis=1) == outcomes).mean()),
    }


def _calibration_table(df: pd.DataFrame, bins: int = 5) -> list[dict]:
    """Predicted-vs-observed frequency of the home/draw/away outcome probabilities, pooled."""
    probs = df[_OUTCOME_COLS].to_numpy(dtype=float).ravel()
    hits = (np.eye(3)[df["outcome"].to_numpy(dtype=int)]).ravel()
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1]), 0, bins - 1)
    table = []
    for b in range(bins):
        mask = idx == b
        if mask.any():
            table.append(
                {"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "predicted": float(probs[mask].mean()),
                 "observed": float(hits[mask].mean()), "n": int(mask.sum())}
            )
    return table


RECENT_LIMIT = 30
_LETTERS = np.array(["H", "D", "A"])


def _recent_picks(graded: pd.DataFrame) -> list[dict]:
    """The latest graded predictions, newest first, each marked right or wrong."""
    if graded.empty:
        return []
    probs = graded[_OUTCOME_COLS].to_numpy(dtype=float)
    pick = probs.argmax(axis=1)
    frame = graded.assign(pick_idx=pick, p_pick=probs.max(axis=1), date_ts=pd.to_datetime(graded["kickoff_date"]))
    frame = frame.sort_values("date_ts", ascending=False, kind="stable").head(RECENT_LIMIT)
    return [
        {"match_id": r["match_id"], "league": r["league"], "home": r["home"], "away": r["away"],
         "date": r["date_ts"].strftime("%Y-%m-%d"), "pick": str(_LETTERS[int(r["pick_idx"])]),
         "result": str(_LETTERS[int(r["outcome"])]), "correct": bool(int(r["pick_idx"]) == int(r["outcome"])),
         "p_pick": float(r["p_pick"])}
        for r in frame.to_dict("records")
    ]


def _track_record_payload(graded_log: pd.DataFrame, reference: dict | None = None) -> dict:
    payload = _track_record_core(graded_log)
    if reference:
        payload["benchmark"] = reference
    return payload


def _track_record_core(graded_log: pd.DataFrame) -> dict:
    graded = graded_log.dropna(subset=["outcome"]) if len(graded_log) else graded_log
    if graded.empty:
        return {"n_graded": 0, "log_loss": None, "rps": None, "accuracy": None, "ece": None,
                "calibration": [], "by_league": {}, "recent": []}
    graded = graded.assign(outcome=graded["outcome"].astype(int))
    overall = _metrics(graded)
    probs = graded[_OUTCOME_COLS].to_numpy(dtype=float)
    return {
        "n_graded": overall["n"],
        "log_loss": overall["log_loss"],
        "rps": overall["rps"],
        "accuracy": overall["accuracy"],
        "ece": float(expected_calibration_error(probs, graded["outcome"].to_numpy())),
        "calibration": _calibration_table(graded),
        "by_league": {league: _metrics(g) for league, g in graded.groupby("league")},
        "recent": _recent_picks(graded),
    }


def export_site_data(
    out_dir: Path,
    fixtures: pd.DataFrame,
    drivers: dict[str, list[Driver]],
    dc_grids: dict[str, np.ndarray],
    graded_log: pd.DataFrame,
    ratings: dict[str, float],
    updated_at: pd.Timestamp,
    model_version: str = "unknown",
    odds: dict | None = None,
    reference: dict | None = None,
) -> None:
    """Write fixtures/track_record/teams/meta JSON into ``out_dir``, replacing it atomically."""
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    old = out_dir.with_name(out_dir.name + ".old")
    for stale in (tmp, old):
        shutil.rmtree(stale, ignore_errors=True)
    tmp.mkdir()
    try:
        _write_json(tmp / "fixtures.json", _fixtures_payload(fixtures, drivers, dc_grids, odds))
        _write_json(tmp / "track_record.json", _track_record_payload(graded_log, reference))
        _write_json(tmp / "teams.json", {team: round(float(r), 1) for team, r in sorted(ratings.items())})
        _write_json(tmp / "meta.json", {"updated_at": pd.Timestamp(updated_at).isoformat(), "model_version": model_version})
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if out_dir.exists():
        out_dir.rename(old)
    tmp.rename(out_dir)
    shutil.rmtree(old, ignore_errors=True)
