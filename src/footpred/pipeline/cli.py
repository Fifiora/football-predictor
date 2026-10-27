"""Command line entry points used by the GitHub Actions workflows."""
from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

import pandas as pd

from footpred.european.benchmark import run_european_benchmark
from footpred.european.tune import tune_european
from footpred.european.history import download_openfootball
from footpred.ingest.fixtures import FootballDataOrgClient
from footpred.ingest.historical import read_csv_text
from footpred.ingest.teams import covered_leagues
from footpred.pipeline.benchmark import run_benchmark
from footpred.pipeline.daily import run_daily
from footpred.pipeline.european import run_european_retrain
from footpred.pipeline.retrain import run_retrain

LEAGUES = ["E0", "SP1", "I1", "D1", "F1"]
HISTORY_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"


def _urlopen_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def download_history(
    raw_dir: Path,
    first_start_year: int,
    last_start_year: int,
    leagues: list[str] | None = None,
    fetch: Callable[[str], bytes] = _urlopen_bytes,
) -> list[str]:
    """Fetch season CSVs that are not on disk yet; returns the names that failed (e.g. not published yet)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for year in range(first_start_year, last_start_year + 1):
        season = f"{year % 100:02d}{(year + 1) % 100:02d}"
        for league in leagues or LEAGUES:
            target = raw_dir / f"{league}_{season}.csv"
            current = year == last_start_year
            if target.exists() and not current:
                continue
            try:
                target.write_bytes(fetch(HISTORY_URL.format(season=season, league=league)))
            except (urllib.error.URLError, OSError):
                failed.append(target.name)
    return failed


def list_team_names(raw_dir: Path) -> dict[str, list[str]]:
    """Every team name that appears in the downloaded CSVs, per league (used to build the alias table)."""
    names: dict[str, set[str]] = {}
    for path in sorted(Path(raw_dir).glob("*.csv")):
        frame = read_csv_text(path, usecols=lambda c: c in {"Div", "HomeTeam", "AwayTeam"})
        frame = frame.dropna(subset=["Div", "HomeTeam", "AwayTeam"])
        for league, group in frame.groupby("Div"):
            names.setdefault(str(league).strip(), set()).update(group["HomeTeam"].str.strip(), group["AwayTeam"].str.strip())
    return {league: sorted(found) for league, found in sorted(names.items())}


def summarize_fixtures(site_data_dir: Path, limit: int = 8) -> list[str]:
    """A few published predictions with their reasons, for the run summary."""
    import json

    path = Path(site_data_dir) / "fixtures.json"
    if not path.exists():
        return []
    lines: list[str] = []
    for f in json.loads(path.read_text())[:limit]:
        lines.append(
            f"{f['date']} {f['league']} {f['home']} v {f['away']}: "
            f"{f['p_home']:.0%} / {f['p_draw']:.0%} / {f['p_away']:.0%}"
        )
        lines += [f"  - {d['text']}" for d in f.get("drivers", [])]
    return lines


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    except SystemExit:
        raise
    except Exception as err:  # surface the reason in the GitHub run annotations, then fail the step
        message = f"{type(err).__name__}: {err}".replace("\r", " ").replace("\n", " ")[:4000]
        print(f"::error title=footpred failed::{message}")
        return 1


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="footpred")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("download", "daily", "retrain", "benchmark", "benchmark-european", "tune-european", "teams"):
        p = sub.add_parser(name)
        p.add_argument("--data-dir", type=Path, default=Path("state/data"))
        p.add_argument("--registry-dir", type=Path, default=Path("state/registry"))
        p.add_argument("--site-data-dir", type=Path, default=Path("site/data"))
        p.add_argument("--out", type=Path, default=Path("docs/baseline-results.md"))
        p.add_argument("--first-year", type=int, default=2014)
    args = parser.parse_args(argv)
    today = pd.Timestamp.now().normalize()
    summary = os.environ.get("GITHUB_STEP_SUMMARY")

    def note(text: str) -> None:
        print(text)
        if summary:
            with open(summary, "a") as f:
                f.write(text + "\n")

    if args.command == "download":
        year = today.year if today.month >= 7 else today.year - 1
        covered = covered_leagues()
        skipped = [lg for lg in LEAGUES if lg not in covered]
        failed = download_history(args.data_dir / "raw", args.first_year, year, leagues=covered)
        note(f"history download for {covered}: {len(failed)} files unavailable {failed[:5]}")
        if skipped:
            note(f"- warning: no team aliases yet for {skipped}; add them to data/team_aliases.csv to enable these leagues")
        if not download_openfootball(args.data_dir / "openfootball"):
            print("::warning title=openfootball::European history could not be downloaded; Champions League data may be stale or missing")
        return 0
    if args.command == "teams":
        year = today.year if today.month >= 7 else today.year - 1
        download_history(args.data_dir / "raw", args.first_year, year, leagues=LEAGUES)
        client = FootballDataOrgClient(os.environ["FOOTBALL_DATA_TOKEN"])
        for league, names in list_team_names(args.data_dir / "raw").items():
            print(f"::notice title=csv names {league}::" + " | ".join(names))
        for league in LEAGUES:
            print(f"::notice title=api names {league}::" + " | ".join(client.get_team_names(league)))
        return 0
    if args.command == "daily":
        client = FootballDataOrgClient(os.environ["FOOTBALL_DATA_TOKEN"])
        result = run_daily(
            args.data_dir, args.site_data_dir, client, today, registry_dir=args.registry_dir,
            updated_at=pd.Timestamp.now().floor("s"), european=True,
        )
        note(f"daily: published={result.published} new_results={result.new_results} scored={result.scored_fixtures}")
        for w in result.warnings:
            note(f"- warning: {w}")
        if result.warnings:  # annotations are what CI shows when logs cannot be read
            print("::warning title=daily warnings::" + "%0A".join(result.warnings[:30]))
        sample = summarize_fixtures(args.site_data_dir)
        if sample:
            print("::notice title=predictions::" + "%0A".join(sample))
        return 0 if result.published else 3  # 3 = nothing new; the workflow skips deployment
    if args.command == "retrain":
        result = run_retrain(args.data_dir, args.registry_dir, today)
        note(f"retrain: promoted={result.promote} ({result.reason})")
        try:  # the Champions League track must never fail the league retrain that already succeeded
            euro = run_european_retrain(args.data_dir, args.registry_dir, today)
            note(f"european retrain: promoted={euro.promote} ({euro.reason})")
        except Exception as err:  # noqa: BLE001
            print(f"::warning title=european retrain::{type(err).__name__}: {err}")
        return 0
    if args.command == "tune-european":
        tune_european(args.data_dir, args.out, today=today)
        note(f"european tuning written to {args.out}")
        return 0
    if args.command == "benchmark-european":
        result = run_european_benchmark(args.data_dir, args.out, today=today)
        note(f"european benchmark written to {args.out}; publish gate: {'pass' if result['publish_ok'] else 'fail'} ({result['reason']})")
        return 0
    run_benchmark(args.data_dir, args.out, today=today)
    note(f"benchmark written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
