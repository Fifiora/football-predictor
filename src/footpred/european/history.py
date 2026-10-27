"""European match history: openfootball files plus results accumulated from the live feed."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd

from footpred.european.names import EuropeanNames
from footpred.european.parse import COMPETITIONS, parse_openfootball

EURO_COLUMNS = [
    "match_id", "competition", "season", "stage", "date", "home", "away",
    "home_country", "away_country", "home_goals", "away_goals", "result",
]
OPENFOOTBALL_URL = "https://github.com/openfootball/champions-league.git"
RESULTS_FILE = "european_results.csv"
DUPLICATE_WINDOW_DAYS = 3


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=EURO_COLUMNS)


def _with_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["match_id"] = [
        f"{c}-{d:%Y%m%d}-{h}-{a}" for c, d, h, a in zip(df["competition"], df["date"], df["home"], df["away"])
    ]
    return df[EURO_COLUMNS]


def load_european_history(openfootball_dir: Path, names: EuropeanNames) -> pd.DataFrame:
    """Every competition file under ``openfootball_dir/<season>/``, with canonical club names."""
    root = Path(openfootball_dir)
    frames: list[pd.DataFrame] = []
    if root.is_dir():
        for season_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
            for stem, competition in COMPETITIONS.items():
                path = season_dir / f"{stem}.txt"
                if not path.exists():
                    continue
                df = parse_openfootball(path.read_text(encoding="utf-8"), competition, season_dir.name)
                if df.empty:
                    continue
                df["home"] = [names.resolve(n, c) for n, c in zip(df["home"], df["home_country"])]
                df["away"] = [names.resolve(n, c) for n, c in zip(df["away"], df["away_country"])]
                frames.append(df)
    if not frames:
        return _empty()
    return _with_ids(pd.concat(frames, ignore_index=True)).sort_values(["date", "match_id"]).reset_index(drop=True)


def read_european_results(data_dir: Path) -> pd.DataFrame:
    path = Path(data_dir) / RESULTS_FILE
    if not path.exists():
        return _empty()
    df = pd.read_csv(path, parse_dates=["date"])
    return df.reindex(columns=EURO_COLUMNS)


def append_european_results(data_dir: Path, new: pd.DataFrame) -> None:
    """Add rows to ``european_results.csv`` (atomic; rows with a known ``match_id`` are ignored)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    existing = read_european_results(data_dir)
    merged = pd.concat([existing, new.reindex(columns=EURO_COLUMNS)], ignore_index=True)
    merged = merged.drop_duplicates("match_id", keep="first").sort_values(["date", "match_id"])
    path = data_dir / RESULTS_FILE
    tmp = path.with_name(path.name + ".tmp")
    merged.to_csv(tmp, index=False)
    os.replace(tmp, path)


def load_european_matches(data_dir: Path, names: EuropeanNames) -> pd.DataFrame:
    """History plus feed results; a feed row that the history already holds (same competition and
    clubs within a few days; feed dates are UTC) is dropped."""
    history = load_european_history(Path(data_dir) / "openfootball", names)
    feed = read_european_results(data_dir)
    if feed.empty:
        return history
    if not history.empty:
        window = pd.Timedelta(days=DUPLICATE_WINDOW_DAYS)
        by_key = {k: g["date"].to_numpy() for k, g in history.groupby(["competition", "home", "away"])}

        def duplicate(r) -> bool:
            dates = by_key.get((r.competition, r.home, r.away))
            return dates is not None and bool((abs(pd.to_datetime(dates) - r.date) <= window).any())

        feed = feed[[not duplicate(r) for r in feed.itertuples(index=False)]]
    if feed.empty:
        return history
    return pd.concat([history, feed], ignore_index=True).sort_values(["date", "match_id"]).reset_index(drop=True)


def download_openfootball(dest: Path, run=subprocess.run) -> bool:
    """Shallow-clone (or update) the openfootball repository; False when git fails."""
    dest = Path(dest)
    try:
        if (dest / ".git").exists():
            run(["git", "-C", str(dest), "pull", "--ff-only", "--depth", "1"], check=True, capture_output=True, timeout=300)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            run(["git", "clone", "--depth", "1", OPENFOOTBALL_URL, str(dest)], check=True, capture_output=True, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return False
    return True
