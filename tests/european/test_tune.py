import pandas as pd
import pytest
from world import write_openfootball, write_world

import footpred.european.tune as tune


def test_tuning_reports_every_combination_and_ignores_the_holdout(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    _, aliases = write_world(data_dir)
    write_openfootball(data_dir)
    monkeypatch.setattr(tune, "RATING_GRID", {"k": (20.0,), "home_adv": (60.0,), "season_regress": (0.25,)})
    rows = tune.tune_european(data_dir, tmp_path / "t.md", aliases_path=aliases, today=pd.Timestamp("2026-07-01"))
    assert len(rows) == 1 * (2 + len(tune.VARIANTS)) and all(r["n"] > 100 for r in rows)
    text = (tmp_path / "t.md").read_text()
    assert "pooled Elo baseline" in text and "elo_diff only" in text
    before = {(r["variant"]): r["log_loss"] for r in rows}
    path = data_dir / "openfootball" / "2025-26" / "cl.txt"
    import re
    path.write_text(re.sub(r"\s\d+-\d+ \(0-0\)", " 9-0 (0-0)", path.read_text(encoding="utf-8")), encoding="utf-8")
    after = {r["variant"]: r["log_loss"] for r in tune.tune_european(data_dir, tmp_path / "t2.md", aliases_path=aliases, today=pd.Timestamp("2026-07-01"))}
    assert before == after
