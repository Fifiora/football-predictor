"""Resolve club names from the European sources to the canonical names the league model uses."""
from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path

from footpred.ingest.teams import DEFAULT_ALIASES_PATH, UnknownTeamError, _load, canonical_team, covered_leagues

DEFAULT_EURO_ALIASES_PATH = Path(__file__).resolve().parents[3] / "data" / "european_aliases.csv"
COUNTRY_TO_LEAGUE = {"ENG": "E0", "ESP": "SP1", "ITA": "I1", "GER": "D1", "FRA": "F1"}
GROUPS = ["E0", "SP1", "I1", "D1", "F1", "OTHER"]
_NOISE = {
    "fc", "cf", "ac", "sc", "afc", "fk", "sk", "ss", "ssc", "us", "rc", "rcd", "sv", "vfb", "vfl", "tsg",
    "club", "de", "del", "di", "da", "calcio", "1909", "1899", "1900", "04", "05", "96", "29",
}


def group_for_country(country: str | None) -> str:
    return COUNTRY_TO_LEAGUE.get(country or "", "OTHER")


def _clean(name: str) -> str:
    return " ".join(name.split())


def normalize_key(name: str) -> str:
    """Lower-case, accent-free name without club-type tokens, for matching spelling variants."""
    text = unicodedata.normalize("NFD", name)
    text = "".join(c for c in text if not unicodedata.combining(c)).casefold()
    text = re.sub(r"[.,'’\-/]", " ", text)
    tokens = [t for t in text.split() if t not in _NOISE]
    return " ".join(tokens)


class EuropeanNames:
    """Maps a source spelling (with an optional country tag) to a canonical club name.

    Never raises: a club it cannot place is returned with cleaned whitespace and recorded in
    ``unmatched`` so the run can report it.
    """

    def __init__(self, euro_aliases_path: Path | None = None, league_aliases_path: Path | None = None):
        self._league_path = Path(league_aliases_path) if league_aliases_path else DEFAULT_ALIASES_PATH
        euro_path = Path(euro_aliases_path) if euro_aliases_path else DEFAULT_EURO_ALIASES_PATH
        self._euro: dict[str, str] = {}
        if euro_path.exists():
            with open(euro_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self._euro[_clean(row["alias"]).casefold()] = _clean(row["canonical"])
        table = _load(self._league_path)
        self._by_key: dict[str, dict[str, str]] = {}  # league -> normalized key -> canonical
        for (league, raw_key), canonical in table.items():
            self._by_key.setdefault(league, {}).setdefault(normalize_key(raw_key), canonical)
        self._canonical = {c for per in self._by_key.values() for c in per.values()}
        self._learned: dict[str, str] = {}  # normalized key -> club name already used by the history files
        self.unmatched: set[str] = set()

    def learn(self, clubs) -> None:
        """Remember club names that are already in use, so a differently written feed spelling joins them."""
        for club in clubs:
            self._canonical.add(club)
            self._learned.setdefault(normalize_key(club), club)

    def is_canonical(self, name: str) -> bool:
        return name in self._canonical

    def _leagues_for(self, country: str | None) -> list[str]:
        if country in COUNTRY_TO_LEAGUE:
            return [COUNTRY_TO_LEAGUE[country]]
        if country:  # a known country outside the big five cannot be a league club
            return []
        return covered_leagues(self._league_path)

    def resolve(self, name: str, country: str | None = None) -> str:
        cleaned = _clean(name)
        exact = self._euro.get(cleaned.casefold())
        if exact:
            return exact
        leagues = self._leagues_for(country)
        for league in leagues:
            try:
                return canonical_team(cleaned, league, self._league_path)
            except UnknownTeamError:
                pass
        key = normalize_key(cleaned)
        for league in leagues:
            found = self._by_key.get(league, {}).get(key)
            if found:
                return found
        learned = self._learned.get(key)
        if learned:
            return learned
        self.unmatched.add(cleaned)
        return cleaned

    def canonical(self, name: str, league: str) -> str:
        """Same as ``resolve``; the signature the fixtures fetcher expects of a name hook."""
        return self.resolve(name)
