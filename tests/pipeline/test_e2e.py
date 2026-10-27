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
    # Seed with the default tree model ({} = no overrides) so the degraded challenger is clearly worse.
    run_retrain(world["data_dir"], world["registry"], TODAY, challenger_params={}, aliases_path=world["aliases"])
    champion_before = load_champion(world["registry"])[0]
    weak = {"max_iter": 1, "learning_rate": 0.001} if MetaModel().backend == "sklearn" else {"n_estimators": 1, "learning_rate": 0.001}

    result = run_retrain(world["data_dir"], world["registry"], TODAY, challenger_params=weak, aliases_path=world["aliases"])
    assert not result.promote
    assert load_champion(world["registry"])[0] == champion_before
    versions = [p.name for p in world["registry"].iterdir() if p.is_dir()]
    assert len(versions) == 2  # the rejected challenger is kept, with its model card
    assert all((world["registry"] / v / "model_card.md").exists() for v in versions)


def test_meta_shows_the_real_refresh_time_not_midnight(world):
    seed_champion(world)
    run_daily(
        world["data_dir"], world["site"], world["client"], TODAY,
        registry_dir=world["registry"], aliases_path=world["aliases"], sleep=NO_SLEEP,
        updated_at=pd.Timestamp("2025-10-27 22:47:13"),
    )
    assert json.loads((world["site"] / "meta.json").read_text())["updated_at"] == "2025-10-27T22:47:13"


def test_daily_attaches_bookmaker_odds_and_survives_their_failure(world):
    seed_champion(world)
    odds = {"E0-20251030-A-B": {"average": (0.5, 0.3, 0.2)}}
    run_daily(world["data_dir"], world["site"], world["client"], TODAY, registry_dir=world["registry"],
              aliases_path=world["aliases"], sleep=NO_SLEEP, fetch_odds=lambda: (odds, ["odds note"]))
    fixtures = {f["match_id"]: f for f in json.loads((world["site"] / "fixtures.json").read_text())}
    assert fixtures["E0-20251030-A-B"]["bookies"]["average"]["p_home"] == 0.5
    assert "bookies" not in fixtures["E0-20251031-C-D"]



def test_daily_publishes_even_when_the_odds_feed_fails(world):
    seed_champion(world)

    def boom():
        raise OSError("down")

    result = run_daily(world["data_dir"], world["site"], world["client"], TODAY, registry_dir=world["registry"],
                       aliases_path=world["aliases"], sleep=NO_SLEEP, fetch_odds=boom)
    assert result.published and any("bookmaker odds unavailable" in w for w in result.warnings)
    assert all("bookies" not in f for f in json.loads((world["site"] / "fixtures.json").read_text()))


def test_daily_includes_the_benchmark_reference_when_the_file_exists(world, tmp_path):
    seed_champion(world)
    ref = tmp_path / "ref.json"
    ref.write_text('{"holdout_season": "2025-26", "holdout": null, "development": null}')
    run_daily(world["data_dir"], world["site"], world["client"], TODAY, registry_dir=world["registry"],
              aliases_path=world["aliases"], sleep=NO_SLEEP, reference_path=ref)
    assert json.loads((world["site"] / "track_record.json").read_text())["benchmark"]["holdout_season"] == "2025-26"
