"""Upcoming bookmaker odds from football-data.co.uk, for display only.

These numbers are never a model input. They are shown next to the model's own chances so a
fan can see what the popular bookmakers think.
"""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path
from typing import Callable

import pandas as pd

from footpred.ingest.teams import UnknownTeamError, canonical_team

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
_SOURCES = {"bet365": ("B365H", "B365D", "B365A"), "average": ("AvgH", "AvgD", "AvgA")}


def devig(home, draw, away) -> tuple[float, float, float] | None:
    """Implied probabilities with the bookmaker margin removed proportionally; None if unusable."""
    try:
        odds = [float(x) for x in (home, draw, away)]
    except (TypeError, ValueError):
        return None
    if any(o != o or o <= 1.0 for o in odds):  # NaN or not a real price
        return None
    implied = [1.0 / o for o in odds]
    total = sum(implied)
    return tuple(i / total for i in implied)


def parse_fixture_odds(text: str, aliases_path: Path | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Rows with canonical names, ``match_id`` and de-vigged ``average`` / ``bet365`` probabilities."""
    raw = pd.read_csv(io.StringIO(text.lstrip("﻿")))
    raw.columns = [c.strip() for c in raw.columns]
    warnings: list[str] = []
    unknown: set[str] = set()
    rows = []
    for r in raw.dropna(subset=["Div", "Date", "HomeTeam", "AwayTeam"]).to_dict("records"):
        league = str(r["Div"]).strip()
        try:
            home = canonical_team(r["HomeTeam"], league, aliases_path)
            away = canonical_team(r["AwayTeam"], league, aliases_path)
        except UnknownTeamError:
            unknown.add(f"{r['HomeTeam']} / {r['AwayTeam']} ({league})")
            continue
        probs = {name: devig(*(r.get(c) for c in cols)) for name, cols in _SOURCES.items()}
        if all(p is None for p in probs.values()):
            continue
        date = pd.to_datetime(r["Date"], dayfirst=True)
        rows.append({"match_id": f"{league}-{date:%Y%m%d}-{home}-{away}", **probs})
    if unknown:
        warnings.append("bookmaker odds skipped for unmapped clubs: " + "; ".join(sorted(unknown)))
    return pd.DataFrame(rows, columns=["match_id", "average", "bet365"]), warnings


def _download(url: str) -> str:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "footpred"}), timeout=30) as resp:
        data = resp.read()
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def fetch_fixture_odds(aliases_path: Path | None = None, download: Callable[[str], str] = _download,
                       url: str = FIXTURES_URL) -> tuple[dict[str, dict], list[str]]:
    """``{match_id: {"average": (h, d, a), "bet365": (h, d, a)}}``; failures become warnings."""
    try:
        frame, warnings = parse_fixture_odds(download(url), aliases_path)
    except Exception as err:  # noqa: BLE001  odds are optional extras; never stop the daily run
        return {}, [f"bookmaker odds unavailable: {type(err).__name__}: {err}"]
    out = {}
    for row in frame.itertuples(index=False):
        out[row.match_id] = {k: v for k, v in (("average", row.average), ("bet365", row.bet365)) if v is not None}
    return out, warnings
