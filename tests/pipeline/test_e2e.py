import hashlib
import json

import pandas as pd
import pytest

from footpred.models.meta_model import MetaModel
from footpred.models.registry import load_champion
from footpred.pipeline import daily as daily_mod
from footpred.pipeline.daily import run_daily
from footpred.pipeline.retrain import run_retrain
from conftest import NO_SLEEP, TODAY


def seed_champion(w):
    return run_retrain(w["data_dir"], w["registry"], TODAY, aliases_path=w["aliases"])


def daily(w, client=None, site=None, today=TODAY):
    return run_daily(
        w["data_dir"], site or w["site"], client or w["client"], today,
        registry_dir=w["registry"], aliases_path=w["aliases"], sleep=NO_SLEEP,
    )


def snapshot(path):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(path.iterdir())}


def test_first_retrain_creates_a_champion_and_model_card(world):
    result = seed_champion(world)
    version, model, _ = load_champion(world["registry"])
    assert result.promote and "first" in result.reason.lower()
    assert (world["registry"] / version / "model_card.md").exists()


def test_daily_end_to_end_produces_valid_site_data(world):
    seed_champion(world)
    result = daily(world)
    assert result.published and result.new_results == 1 and result.scored_fixtures == 2

    site = world["site"]
    fixtures = json.loads((site / "fixtures.json").read_text())
    assert {f["match_id"] for f in fixtures} == {"E0-20251030-A-B", "E0-20251031-C-D"}
    for f in fixtures:
        assert abs(f["p_home"] + f["p_draw"] + f["p_away"] - 1) < 1e-6
        assert len(f["drivers"]) <= 3
        assert 0 <= f["btts"] <= 1
    meta = json.loads((site / "meta.json").read_text())
    assert meta["model_version"] == load_champion(world["registry"])[0]
    assert "A" in json.loads((site / "teams.json").read_text())
    log = pd.read_csv(world["data_dir"] / "prediction_log.csv")
    assert set(log["match_id"]) == {f["match_id"] for f in fixtures}


def test_daily_no_new_results_exits_cheaply_and_keeps_site_data(world, monkeypatch):
    world["site"].mkdir()
    (world["site"] / "fixtures.json").write_text('["last good"]')
    before = snapshot(world["site"])

    def boom(*args, **kwargs):
        raise AssertionError("scoring must not run when there is nothing new")

    monkeypatch.setattr(daily_mod, "load_champion", boom)
    result = daily(world, client=world["empty_client"])
    assert not result.published and result.new_results == 0 and result.scored_fixtures == 0
    assert snapshot(world["site"]) == before


def test_daily_schema_failure_keeps_last_good_site(world):
    seed_champion(world)
    daily(world)
    before = snapshot(world["site"])

    bad = world["data_dir"] / "raw" / "E0_1.csv"
    bad.write_text(bad.read_text().replace(",H,", ",Z,", 1).replace(",A,", ",Z,", 1))
    with pytest.raises(Exception):
        daily(world, today=TODAY + pd.Timedelta(days=1))
    assert snapshot(world["site"]) == before


def test_retrain_keeps_champion_when_challenger_not_better(world):
    first = seed_champion(world)
    champion_before = load_champion(world["registry"])[0]
    weak = {"max_iter": 1, "learning_rate": 0.001} if MetaModel().backend == "sklearn" else {"n_estimators": 1, "learning_rate": 0.001}

    result = run_retrain(world["data_dir"], world["registry"], TODAY, challenger_params=weak, aliases_path=world["aliases"])
    assert not result.promote
    assert load_champion(world["registry"])[0] == champion_before
    versions = [p.name for p in world["registry"].iterdir() if p.is_dir()]
    assert len(versions) == 2  # the rejected challenger is kept, with its model card
    assert all((world["registry"] / v / "model_card.md").exists() for v in versions)
