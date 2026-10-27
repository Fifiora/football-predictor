import json

import numpy as np

from footpred.models.registry import load_champion
from footpred.pipeline.daily import run_daily
from footpred.pipeline.retrain import run_retrain
from conftest import NO_SLEEP, TODAY


def probs(site):
    return [(f["match_id"], f["p_home"], f["p_draw"], f["p_away"]) for f in json.loads((site / "fixtures.json").read_text())]


def test_reproducibility(world, tmp_path):
    run_retrain(world["data_dir"], world["registry"], TODAY, aliases_path=world["aliases"])

    def go(site):
        run_daily(world["data_dir"], site, world["client"], TODAY,
                  registry_dir=world["registry"], aliases_path=world["aliases"], sleep=NO_SLEEP)
        return probs(site)

    first = go(tmp_path / "site1")
    second = go(tmp_path / "site2")
    assert first == second and len(first) == 2


def test_retraining_twice_on_same_data_gives_identical_models(world, tmp_path):
    run_retrain(world["data_dir"], world["registry"], TODAY, aliases_path=world["aliases"])
    run_retrain(world["data_dir"], tmp_path / "registry_b", TODAY, aliases_path=world["aliases"])
    _, model_a, cal_a = load_champion(world["registry"])
    _, model_b, cal_b = load_champion(tmp_path / "registry_b")
    X = __import__("pandas").DataFrame({c: [0.0, 1.0] for c in model_a.feature_names})
    assert np.allclose(cal_a.transform(model_a.predict_proba(X)), cal_b.transform(model_b.predict_proba(X)))
