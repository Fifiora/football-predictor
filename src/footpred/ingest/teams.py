"""Team-name normalization to canonical IDs."""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

DEFAULT_ALIASES_PATH = Path(__file__).resolve().parents[3] / "data" / "team_aliases.csv"


class UnknownTeamError(KeyError):
    """Raised when a team name cannot be mapped to a canonical team."""

    def __str__(self) -> str:  # KeyError wraps its message in repr(); keep it readable
        return str(self.args[0]) if self.args else ""


def _key(name: str) -> str:
    return " ".join(name.split()).casefold()


@lru_cache(maxsize=8)
def _load(path: Path) -> dict[tuple[str, str], str]:
    table: dict[tuple[str, str], str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            league, canonical = row["league"].strip(), row["canonical"].strip()
            table[(league, _key(canonical))] = canonical
            table[(league, _key(row["alias"]))] = canonical
    return table


def canonical_team(name: str, league: str, aliases_path: Path | None = None) -> str:
    """Map a source team name to its canonical name for ``league``."""
    table = _load(Path(aliases_path) if aliases_path else DEFAULT_ALIASES_PATH)
    try:
        return table[(league, _key(name))]
    except KeyError:
        raise UnknownTeamError(f"unknown team {name!r} in league {league}") from None


def covered_leagues(aliases_path: Path | None = None) -> list[str]:
    """Leagues that have team aliases; the pipelines only handle these."""
    return sorted({league for league, _ in _load(Path(aliases_path) if aliases_path else DEFAULT_ALIASES_PATH)})
