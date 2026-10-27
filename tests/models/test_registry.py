import json

import numpy as np
import pandas as pd
import pytest

from footpred.models.calibration import Calibrator
from footpred.models.meta_model import MetaModel
from footpred.models.registry import load_champion, save_model, set_champion

META = {"data_snapshot": "2026-10-27", "code_version": "abc1234", "params": {}, "metrics": {"log_loss": 0.95}}


def fitted(seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"elo_diff": rng.normal(0, 100, 800), "rest_days_home": rng.integers(2, 9, 800).astype(float)})
    y = np.where(X["elo_diff"] + rng.normal(0, 80, 800) > 20, 0, rng.choice([1, 2], 800))
    model = MetaModel().fit(X, y)
    cal = Calibrator(min_samples_isotonic=100).fit(model.predict_proba(X), y)
    return model, cal, X


def test_registry_roundtrip(tmp_path):
    model, cal, X = fitted()
    path = save_model(tmp_path, "v1", model, cal, META)
    assert (path / "model.joblib").exists() and (path / "calibrator.joblib").exists()
    assert json.loads((path / "metadata.json").read_text())["code_version"] == "abc1234"

    set_champion(tmp_path, "v1")
    version, loaded_model, loaded_cal = load_champion(tmp_path)
    assert version == "v1"
    sample = X.iloc[:25]
    assert np.allclose(loaded_cal.transform(loaded_model.predict_proba(sample)), cal.transform(model.predict_proba(sample)))


def test_metadata_must_be_complete(tmp_path):
    model, cal, _ = fitted()
    with pytest.raises(ValueError):
        save_model(tmp_path, "v1", model, cal, {"data_snapshot": "x"})


def test_cannot_overwrite_existing_version(tmp_path):
    model, cal, _ = fitted()
    save_model(tmp_path, "v1", model, cal, META)
    with pytest.raises(FileExistsError):
        save_model(tmp_path, "v1", model, cal, META)


def test_set_champion_requires_saved_version(tmp_path):
    with pytest.raises(ValueError):
        set_champion(tmp_path, "nope")


def test_no_champion_yet_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_champion(tmp_path)


def test_champion_pointer_moves_between_versions(tmp_path):
    model, cal, _ = fitted()
    save_model(tmp_path, "v1", model, cal, META)
    save_model(tmp_path, "v2", model, cal, META)
    set_champion(tmp_path, "v1")
    set_champion(tmp_path, "v2")
    assert load_champion(tmp_path)[0] == "v2"
