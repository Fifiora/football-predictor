"""Live scores: what is being played now and what just finished, each with the pick made before kickoff.

The feed is rebuilt every couple of minutes on match days by the ``live`` workflow and published
to a branch the site reads in the browser. It never changes a prediction: picks come from the
append-only prediction log, and a match with no logged prediction simply carries no pick.
"""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pandas as pd

from footpred.european.names import EuropeanNames
from footpred.ingest.fixtures import FixturesUnavailable, _canonical_pair
from footpred.ingest.teams import UnknownTeamError
from footpred.predictions.log import LOG_COLUMNS

_STATE = {"SCHEDULED": "upcoming", "TIMED": "upcoming", "IN_PLAY": "live", "PAUSED": "live", "FINISHED": "finished"}
_LETTERS = ("H", "D", "A")


def _leader(home_goals: int, away_goals: int) -> str:
    return "H" if home_goals > away_goals else "A" if away_goals > home_goals else "D"


def _int_or_none(x):
    try:
        return None if x is None or pd.isna(x) else int(x)
    except (TypeError, ValueError):
        return None


def build_live(client, leagues: list[str], now: pd.Timestamp, log: pd.DataFrame, aliases_path: Path | None = None) -> tuple[dict, list[str]]:
    """Matches from yesterday to tomorrow in the given competitions, with scores and the logged pick.

    ``now`` is a naive UTC timestamp. The feed says ``complete: false`` when any competition could not be read,
    so the caller can keep the last good feed instead of publishing one with matches missing.
    """
    now = pd.Timestamp(now)
    first, last = (now - pd.Timedelta(days=1)).date(), (now + pd.Timedelta(days=1)).date()
    preds = log.drop_duplicates("match_id").set_index("match_id") if len(log) else pd.DataFrame(columns=LOG_COLUMNS[1:])
    names = None
    if "CL" in leagues:  # Champions League clubs are spelled as the prediction log already spells them
        names = EuropeanNames(league_aliases_path=aliases_path)
        cl = log[log["league"] == "CL"] if len(log) else log
        names.learn(set(cl["home"]) | set(cl["away"]) if len(cl) else set())

    warnings: list[str] = []
    matches: list[dict] = []
    complete = True
    for league in leagues:
        try:
            rows = client.get_fixtures(league, first, last)
        except (FixturesUnavailable, OSError) as err:
            complete = False
            warnings.append(f"{league}: live scores unavailable ({err})")
            continue
        for row in rows:
            state = _STATE.get(row["status"])
            if state is None:  # postponed, suspended, cancelled
                continue
            try:
                home, away = _canonical_pair(row, league, aliases_path, names.canonical if league == "CL" and names else None)
            except UnknownTeamError as err:
                warnings.append(f"{err}; left out of live scores")
                continue
            day = pd.Timestamp(row["date"])
            match_id = f"{league}-{day:%Y%m%d}-{home}-{away}"
            hg, ag = _int_or_none(row.get("home_goals")), _int_or_none(row.get("away_goals"))
            entry = {"match_id": match_id, "league": league, "home": home, "away": away, "kickoff": row.get("kickoff"),
                     "state": state, "home_goals": hg if state != "upcoming" else None, "away_goals": ag if state != "upcoming" else None}
            if match_id in preds.index:
                p = preds.loc[match_id]
                probs = [float(p["p_home"]), float(p["p_draw"]), float(p["p_away"])]
                pick = _LETTERS[max(range(3), key=lambda i: probs[i])]
                entry.update(p_home=probs[0], p_draw=probs[1], p_away=probs[2], pick=pick)
                if state != "upcoming" and hg is not None and ag is not None:
                    entry["pick_ok"] = pick == _leader(hg, ag)  # for a live match: is the pick ahead right now
            matches.append(entry)
    matches.sort(key=lambda m: (m["kickoff"] or "", m["match_id"]))
    return {"updated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "complete": complete, "matches": matches}, warnings


def unrecorded_results(feed: dict, recorded_ids: set[str]) -> list[str]:
    """Finished matches whose result the daily refresh has not stored yet; seeing any means it should run now."""
    return sorted(m["match_id"] for m in feed["matches"] if m["state"] == "finished" and m["match_id"] not in recorded_ids)


def still_busy(feed: dict, now: pd.Timestamp, horizon: timedelta = timedelta(hours=3)) -> bool:
    """True while a match is being played or kicks off soon, so the live workflow should keep polling."""
    now = pd.Timestamp(now)
    for m in feed["matches"]:
        if not m["kickoff"]:
            continue
        kick = pd.to_datetime(m["kickoff"], utc=True).tz_localize(None)
        if m["state"] == "live":
            if now - kick < timedelta(hours=3.5):  # a match stuck as live in the feed must not keep the poller running
                return True
        elif m["state"] == "upcoming":
            if now - timedelta(hours=1) <= kick <= now + horizon:
                return True
    return False


def recorded_match_ids(data_dir: Path) -> set[str]:
    """Match ids the state already holds a result for: stored API and Champions League results, and graded predictions."""
    ids: set[str] = set()
    for name in ("api_results.csv", "european_results.csv"):
        path = Path(data_dir) / name
        if path.exists():
            ids |= set(pd.read_csv(path, usecols=["match_id"])["match_id"])
    log_path = Path(data_dir) / "prediction_log.csv"
    if log_path.exists():
        log = pd.read_csv(log_path, usecols=["match_id", "outcome"])
        ids |= set(log.loc[log["outcome"].notna(), "match_id"])
    return ids


def read_log(data_dir: Path) -> pd.DataFrame:
    path = Path(data_dir) / "prediction_log.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=LOG_COLUMNS)


def write_outputs(out_dir: Path, feed: dict, unrecorded: list[str], busy: bool) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "live.json").write_text(json.dumps(feed, separators=(",", ":")))
    (out_dir / "status.json").write_text(json.dumps({"unrecorded": unrecorded, "busy": busy,
                                                     "live": sum(m["state"] == "live" for m in feed["matches"])}))
