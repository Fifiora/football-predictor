import re

import numpy as np
import pandas as pd
import pytest

import footpred.european.benchmark as bench
from footpred.european.benchmark import HOLDOUT_SEASON, run_european_benchmark
from world import write_world

STRONG = ["A", "B", "C", "D", "E", "F"]
OTHERS = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
SEASONS = [2021, 2022, 2023, 2024, 2025]


def write_openfootball(root, spread=1.2, seed=3):
    """Five seasons of synthetic Champions League results with real strength differences."""
    rng = np.random.default_rng(seed)
    strength = {t: float(rng.normal(0, spread)) for t in STRONG + OTHERS}
    country = {**{t: "ENG" for t in STRONG}, **{t: "POR" for t in OTHERS}}
    for year in SEASONS:
        lines = [f"= UEFA Champions League {year}/{str(year + 1)[-2:]}", "", "▪ League, Matchday 1"]
        def play(count, first_day, step, home_edge):
            for i in range(count):
                home, away = rng.choice(STRONG + OTHERS, size=2, replace=False)
                gap = strength[home] - strength[away] + home_edge
                hg = int(rng.poisson(np.exp(0.2 + 0.4 * gap)))
                ag = int(rng.poisson(np.exp(0.2 - 0.4 * gap)))
                date = first_day + pd.Timedelta(days=int(i * step))
                lines.append(f"  {date:%a %b} {date.day} {date.year}")
                lines.append(f"    21:00  {home} ({country[home]})  v {away} ({country[away]})  {hg}-{ag} (0-0)")

        play(160, pd.Timestamp(f"{year}-09-10"), 250 / 160, 0.3)
        lines.append("▪ Finals, Round of 16")  # knockout home edge is much larger: a signal only the stage flag can carry
        play(60, pd.Timestamp(f"{year + 1}-03-01"), 1.0, 1.6)
        folder = root / "openfootball" / f"{year}-{str(year + 1)[-2:]}"
        folder.mkdir(parents=True)
        (folder / "cl.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def world(tmp_path):
    data_dir = tmp_path / "data"
    _, aliases = write_world(data_dir)
    write_openfootball(data_dir)
    return data_dir, aliases


def run(world, tmp_path, **kw):
    data_dir, aliases = world
    out = tmp_path / "out" / "european-results.md"
    result = run_european_benchmark(data_dir, out, aliases_path=aliases, today=pd.Timestamp("2026-07-01"), **kw)
    return result, out


def test_benchmark_writes_report_and_respects_holdout_rule(world, tmp_path):
    result, out = run(world, tmp_path)
    assert result["holdout_season"] == HOLDOUT_SEASON == "2025-26"
    assert set(result["development"]) == {"base_rate", "elo", "model"}
    for r in result["development"].values():
        assert r["n"] > 100 and 0 <= r["accuracy"] <= 1 and r["log_loss"] > 0
    assert result["publish_ok"] is True  # real strength differences: the model must beat both baselines
    assert set(result["holdout"]) == {"base_rate", "elo", "model"}
    text = out.read_text()
    for needle in ("Base rate", "Pooled Elo", "Model", "Correct picks", "Log loss", "2025-26", result["reason"]):
        assert needle in text
    assert "Not evaluated" not in text


def test_holdout_is_untouched_when_the_gate_fails(world, tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "european_publish_gate", lambda dev: (False, "forced failure"))
    result, out = run(world, tmp_path)
    assert result["publish_ok"] is False and result["holdout"] is None and result["reason"] == "forced failure"
    assert "Not evaluated" in out.read_text()


def test_development_ignores_the_holdout_season(world, tmp_path):
    data_dir, _ = world
    first, _ = run(world, tmp_path)
    path = data_dir / "openfootball" / "2025-26" / "cl.txt"  # make every hold-out result a 9-0 home win
    path.write_text(re.sub(r"\s\d+-\d+ \(0-0\)", " 9-0 (0-0)", path.read_text(encoding="utf-8")), encoding="utf-8")
    second, _ = run(world, tmp_path)
    assert first["development"] == second["development"]
    assert first["holdout"]["model"]["log_loss"] != second["holdout"]["model"]["log_loss"]
