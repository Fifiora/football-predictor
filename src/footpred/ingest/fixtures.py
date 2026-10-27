"""Upcoming fixtures and latest results from a free-tier football API.

Provider-independent: ``FixturesClient`` returns normalized dicts, and ``FootballDataOrgClient``
is the one small adapter for football-data.org (see docs/api-findings.md for why). Training
never depends on this module, only the daily refresh does.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from footpred.ingest.historical import season_label
from footpred.ingest.schema import MATCH_COLUMNS, validate_matches
from footpred.ingest.teams import UnknownTeamError, canonical_team

UPCOMING_COLUMNS = ["match_id", "league", "date", "home", "away"]
_UPCOMING_STATUSES = {"SCHEDULED", "TIMED"}
_COMPETITION_CODES = {"E0": "PL", "SP1": "PD", "I1": "SA", "D1": "BL1", "F1": "FL1"}


class FixturesUnavailable(RuntimeError):
    """The fixtures provider could not be reached or answered with an error."""


class FixturesClient(Protocol):
    def get_fixtures(self, league: str, date_from: date, date_to: date) -> list[dict]:
        """Normalized rows: ``{date: 'YYYY-MM-DD', home, away, status, home_goals, away_goals}``."""


HttpGet = Callable[[str, dict, dict], "tuple[int, dict]"]


def _urllib_get(url: str, headers: dict, params: dict) -> tuple[int, dict]:
    request = urllib.request.Request(f"{url}?{urllib.parse.urlencode(params)}", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as err:
        return err.code, {}


class FootballDataOrgClient:
    BASE_URL = "https://api.football-data.org"

    def __init__(self, token: str, http_get: HttpGet | None = None, base_url: str | None = None):
        self._token = token
        self._http_get = http_get or _urllib_get
        self._base_url = base_url or self.BASE_URL

    def get_fixtures(self, league: str, date_from: date, date_to: date) -> list[dict]:
        if league not in _COMPETITION_CODES:
            raise ValueError(f"unsupported league {league!r}")
        status, payload = self._http_get(
            f"{self._base_url}/v4/competitions/{_COMPETITION_CODES[league]}/matches",
            {"X-Auth-Token": self._token},
            {"dateFrom": date_from.isoformat(), "dateTo": date_to.isoformat()},
        )
        if status != 200:
            raise FixturesUnavailable(f"football-data.org answered HTTP {status} for {league}")
        rows = []
        for match in payload.get("matches", []):
            full_time = match.get("score", {}).get("fullTime", {})
            rows.append(
                {
                    "date": match["utcDate"][:10],
                    "home": match["homeTeam"]["name"],
                    "away": match["awayTeam"]["name"],
                    "status": match["status"],
                    "home_goals": full_time.get("home"),
                    "away_goals": full_time.get("away"),
                }
            )
        return rows


def _get_rows(
    client: FixturesClient,
    league: str,
    date_from: date,
    date_to: date,
    cache_dir: Path,
    now: pd.Timestamp,
    cache_ttl: timedelta,
    attempts: int,
    sleep: Callable[[float], None],
    base_delay: float = 1.0,
) -> list[dict]:
    cache_file = Path(cache_dir) / f"{league}_{date_from.isoformat()}_{date_to.isoformat()}.json"
    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        if now - pd.Timestamp(cached["fetched_at"]) < cache_ttl:
            return cached["rows"]

    for attempt in range(attempts):
        try:
            rows = client.get_fixtures(league, date_from, date_to)
            break
        except (FixturesUnavailable, OSError) as err:
            if attempt == attempts - 1:
                raise FixturesUnavailable(f"{league}: gave up after {attempts} attempts ({err})") from err
            sleep(base_delay * 2**attempt)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({"fetched_at": now.isoformat(), "rows": rows}))
    return rows


def _canonical_pair(row: dict, league: str, aliases_path: Path | None = None) -> tuple[str, str]:
    return canonical_team(row["home"], league, aliases_path), canonical_team(row["away"], league, aliases_path)


def fetch_upcoming(
    client: FixturesClient,
    leagues: list[str],
    cache_dir: Path,
    today: pd.Timestamp | None = None,
    horizon_days: int = 14,
    cache_ttl: timedelta = timedelta(hours=1),
    attempts: int = 3,
    sleep: Callable[[float], None] = time.sleep,
    aliases_path: Path | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Scheduled fixtures from today through ``horizon_days`` ahead, plus warnings.

    A fixture with a team that cannot be mapped to a canonical name is skipped and reported;
    the rest are unaffected. An empty provider response gives an empty frame, not an error.
    """
    now = pd.Timestamp.now() if today is None else pd.Timestamp(today)
    start = now.date()
    rows_out: list[tuple] = []
    warnings: list[str] = []

    for league in leagues:
        rows = _get_rows(client, league, start, start + timedelta(days=horizon_days), cache_dir, now, cache_ttl, attempts, sleep)
        for row in rows:
            if row["status"] not in _UPCOMING_STATUSES:
                continue
            try:
                home, away = _canonical_pair(row, league, aliases_path)
            except UnknownTeamError as err:
                warnings.append(f"{err}; skipped fixture {row['home']} v {row['away']} on {row['date']}")
                continue
            day = pd.Timestamp(row["date"])
            rows_out.append((f"{league}-{day:%Y%m%d}-{home}-{away}", league, day, home, away))

    df = pd.DataFrame(rows_out, columns=UPCOMING_COLUMNS)
    return df.sort_values(["date", "match_id"]).reset_index(drop=True), warnings


def fetch_latest_results(
    client: FixturesClient,
    leagues: list[str],
    since: pd.Timestamp,
    cache_dir: Path,
    today: pd.Timestamp | None = None,
    cache_ttl: timedelta = timedelta(hours=1),
    attempts: int = 3,
    sleep: Callable[[float], None] = time.sleep,
    aliases_path: Path | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Finished matches from ``since`` to today in the canonical match schema, plus warnings.

    Shots and closing odds are not provided by the free tier and are left missing.
    """
    now = pd.Timestamp.now() if today is None else pd.Timestamp(today)
    records: list[dict] = []
    warnings: list[str] = []

    for league in leagues:
        rows = _get_rows(client, league, since.date(), now.date(), cache_dir, now, cache_ttl, attempts, sleep)
        for row in rows:
            if row["status"] != "FINISHED" or row["home_goals"] is None or row["away_goals"] is None:
                continue
            try:
                home, away = _canonical_pair(row, league, aliases_path)
            except UnknownTeamError as err:
                warnings.append(f"{err}; skipped result {row['home']} v {row['away']} on {row['date']}")
                continue
            day = pd.Timestamp(row["date"])
            hg, ag = float(row["home_goals"]), float(row["away_goals"])
            records.append(
                {
                    "match_id": f"{league}-{day:%Y%m%d}-{home}-{away}",
                    "league": league,
                    "season": season_label(day),
                    "date": day,
                    "home": home,
                    "away": away,
                    "home_goals": hg,
                    "away_goals": ag,
                    "result": "H" if hg > ag else "D" if hg == ag else "A",
                }
            )

    if not records:
        return pd.DataFrame(columns=MATCH_COLUMNS), warnings
    df = pd.DataFrame(records)
    for col in ("home_shots", "away_shots", "home_sot", "away_sot", "close_h", "close_d", "close_a"):
        df[col] = np.nan
    return validate_matches(df).sort_values(["date", "match_id"]).reset_index(drop=True), warnings
