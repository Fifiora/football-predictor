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


def parse_openfootball(text: str, competition: str, season: str) -> pd.DataFrame:
    rows: list[dict] = []
    stage: str | None = None
    year: int | None = None
    month = 0
    date: pd.Timestamp | None = None
    for line in text.splitlines():
        if line.startswith("▪"):
            stage = _stage(line.lstrip("▪").strip(), competition)
            continue
        day = _DAY.match(line)
        if day:
            mon = _MONTHS[day.group(1)]
            if day.group(3):
                year = int(day.group(3))
            elif year is None:
                raise ValueError(f"date without a year before any dated line: {line!r}")
            elif mon < month:
                year += 1
            month = mon
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
