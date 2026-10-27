"""Load historical football-data.co.uk style CSVs into the canonical schema."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from footpred.ingest.schema import MATCH_COLUMNS, validate_matches
from footpred.ingest.teams import UnknownTeamError, canonical_team

_RENAME = {
    "HomeTeam": "home",
    "AwayTeam": "away",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_sot",
    "AST": "away_sot",
    "PSCH": "close_h",
    "PSCD": "close_d",
    "PSCA": "close_a",
}
_OPTIONAL = ["home_shots", "away_shots", "home_sot", "away_sot", "close_h", "close_d", "close_a"]


def read_csv_text(path: Path, **kwargs) -> pd.DataFrame:
    """Read a football-data.co.uk CSV: UTF-8 (with or without a BOM) first, then Latin-1."""
    try:
        return pd.read_csv(path, encoding="utf-8-sig", **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", **kwargs)


def season_label(date: pd.Timestamp) -> str:
    """European season label: July onward belongs to the season starting that year."""
    start = date.year if date.month >= 7 else date.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def load_historical(raw_dir: Path, aliases_path: Path | None = None) -> pd.DataFrame:
    """Read every ``*.csv`` in ``raw_dir`` and return validated ``MATCH_COLUMNS`` rows."""
    frames: list[pd.DataFrame] = []
    unknown: set[str] = set()

    for path in sorted(Path(raw_dir).glob("*.csv")):
        raw = read_csv_text(path)
        raw = raw.dropna(subset=["HomeTeam", "AwayTeam", "Date"])
        df = raw.rename(columns=_RENAME)
        df["league"] = raw["Div"].astype(str).str.strip()
        df["date"] = pd.to_datetime(raw["Date"], dayfirst=True, format="mixed")
        for col in _OPTIONAL:
            if col not in df.columns:
                df[col] = np.nan

        for side in ("home", "away"):
            mapped = []
            for name, league in zip(df[side], df["league"]):
                try:
                    mapped.append(canonical_team(name, league, aliases_path))
                except UnknownTeamError:
                    unknown.add(f"{name} ({league})")
                    mapped.append(name)
            df[side] = mapped
        frames.append(df)

    if unknown:
        raise UnknownTeamError(
            "unmapped teams, add them to data/team_aliases.csv: " + ", ".join(sorted(unknown))
        )
    if not frames:
        return pd.DataFrame(columns=MATCH_COLUMNS)

    out = pd.concat(frames, ignore_index=True)
    out["season"] = out["date"].map(season_label)
    out["match_id"] = [
        f"{lg}-{d:%Y%m%d}-{h}-{a}"
        for lg, d, h, a in zip(out["league"], out["date"], out["home"], out["away"])
    ]
    out = validate_matches(out)
    return out.sort_values(["date", "match_id"]).reset_index(drop=True)[MATCH_COLUMNS]
