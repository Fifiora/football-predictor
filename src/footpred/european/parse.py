"""Parse openfootball/champions-league text files into 90-minute match rows."""
from __future__ import annotations

import re

import pandas as pd

STAGES = ("LEAGUE_STAGE", "QUALIFYING", "PLAYOFFS", "LAST_16", "QUARTER_FINALS", "SEMI_FINALS", "FINAL")
COMPETITIONS = {"cl": "CL", "clq": "CLQ", "el": "EL", "elq": "ELQ", "conf": "CONF", "confq": "CONFQ"}
QUALIFIER_COMPETITIONS = {"CLQ", "ELQ", "CONFQ"}

_MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
_DAY = re.compile(r"^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Z][a-z]{2})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$")
_MATCH = re.compile(
    r"^\s*(?:\d{1,2}:\d{2}\s+)?(?P<home>.+?)\s+\((?P<hc>[A-Z]{2,4})\)\s+v\s+"
    r"(?P<away>.+?)\s+\((?P<ac>[A-Z]{2,4})\)\s+(?P<tail>.*?)\s*$"
)
_RANGE = re.compile(
    r"^#\s*Date\s+\w{3}\s+([A-Z][a-z]{2})\s+(\d{1,2})(?:\s+(\d{4}))?\s*-\s*\w{3}\s+([A-Z][a-z]{2})\s+(\d{1,2})\s+(\d{4})", re.M
)
_SCORE = re.compile(r"(\d+)-(\d+)")


def _stage(header: str, competition: str) -> str:
    if competition in QUALIFIER_COMPETITIONS:
        return "QUALIFYING"
    h = header.split(",")[-1].strip().lower() if header.lower().startswith("finals,") else header.strip().lower()
    if h.startswith(("league", "group", "gruppe")):
        return "LEAGUE_STAGE"
    if h.startswith(("playoffs", "play-offs", "round of 32", "sechzehntelfinale")):
        return "PLAYOFFS"
    if h.startswith(("round of 16", "achtelfinale")):
        return "LAST_16"
    if h.startswith("quarterfinals"):
        return "QUARTER_FINALS"
    if h.startswith("semifinals"):
        return "SEMI_FINALS"
    if h == "final":
        return "FINAL"
    raise ValueError(f"unknown stage header {header!r} in a {competition} file")


def _ninety_minute_score(tail: str) -> tuple[int, int] | None:
    """The 90-minute score from the text after the second club, or None when there is no usable score."""
    if "[" in tail:  # [awarded] / [cancelled]: not a played result
        return None
    if "a.e.t." in tail:
        brackets = re.search(r"\(([^)]*)\)", tail)
        if not brackets:
            return None
        scores = _SCORE.findall(brackets.group(1))
        if not scores:
            return None
        goals = int(scores[0][0]), int(scores[0][1])
        if len(scores) >= 2:  # "(90-minute score, half-time score)"
            return goals
        # A lone bracketed score is either the 90-minute score or the half-time score. In a
        # one-off final the 90-minute score is level, so only a level lone score is trusted.
        return goals if goals[0] == goals[1] else None
    full_time = _SCORE.match(tail)
    return (int(full_time.group(1)), int(full_time.group(2))) if full_time else None


def _date_window(text: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """The first and last day of the file, from its ``# Date`` line (with a week of slack)."""
    m = _RANGE.search(text)
    if not m:
        return None
    end_year = int(m.group(6))
    start_month, end_month = _MONTHS[m.group(1)], _MONTHS[m.group(4)]
    start_year = int(m.group(3)) if m.group(3) else (end_year if start_month <= end_month else end_year - 1)
    slack = pd.Timedelta(days=7)
    return (pd.Timestamp(year=start_year, month=start_month, day=int(m.group(2))) - slack,
            pd.Timestamp(year=end_year, month=end_month, day=int(m.group(5))) + slack)


def _year_for(month: int, day: int, season: str, window: tuple[pd.Timestamp, pd.Timestamp] | None) -> int:
    """Year of a date written without one: the calendar year that puts it inside the file's date range,
    else the season's convention (July to December is its first year, January to June the next)."""
    start = int(season[:4])
    if window is not None:
        fits = [y for y in (window[0].year, window[1].year)
                if window[0] <= pd.Timestamp(year=y, month=month, day=day) <= window[1]]
        if len(set(fits)) == 1:
            return fits[0]
    return start if month >= 7 else start + 1


def parse_openfootball(text: str, competition: str, season: str) -> pd.DataFrame:
    rows: list[dict] = []
    stage: str | None = None
    date: pd.Timestamp | None = None
    window = _date_window(text)
    for line in text.splitlines():
        if line.startswith("▪"):
            stage = _stage(line.lstrip("▪").strip(), competition)
            continue
        day = _DAY.match(line)
        if day:
            mon = _MONTHS[day.group(1)]
            if day.group(3):
                year = int(day.group(3))
            else:
                year = _year_for(mon, int(day.group(2)), season, window)
            date = pd.Timestamp(year=year, month=mon, day=int(day.group(2)))
            continue
        m = _MATCH.match(line)
        if not m or stage is None or date is None:
            continue
        goals = _ninety_minute_score(m.group("tail"))
        if goals is None:
            continue
        rows.append({
            "competition": competition, "season": season, "stage": stage, "date": date,
            "home": m.group("home"), "away": m.group("away"),
            "home_country": m.group("hc"), "away_country": m.group("ac"),
            "home_goals": goals[0], "away_goals": goals[1],
            "result": "H" if goals[0] > goals[1] else "A" if goals[0] < goals[1] else "D",
        })
    columns = ["competition", "season", "stage", "date", "home", "away", "home_country", "away_country",
               "home_goals", "away_goals", "result"]
    return pd.DataFrame(rows, columns=columns)
