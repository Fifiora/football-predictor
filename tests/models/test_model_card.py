import numpy as np
import pandas as pd

from footpred.models.meta_model import MetaModel
from footpred.models.model_card import write_model_card


def test_model_card_contains_required_sections(tmp_path):
    rng = np.random.default_rng(0)
    n = 900
    X = pd.DataFrame({"elo_diff": rng.normal(0, 100, n), "rest_days_home": rng.integers(2, 9, n).astype(float)})
    y = np.where(X["elo_diff"] + rng.normal(0, 80, n) > 20, 0, rng.choice([1, 2], n))
    model = MetaModel().fit(X, y)

    ids = [f"m{i}" for i in range(n)]
    probs = model.predict_proba(X)
    preds = pd.DataFrame(
        {
            "match_id": ids,
            "p_home": probs[:, 0],
            "p_draw": probs[:, 1],
            "p_away": probs[:, 2],
            "outcome": y,
            "date": pd.date_range("2025-09-01", periods=n, freq="D"),
            "league": np.where(np.arange(n) % 2 == 0, "E0", "SP1"),
        }
    )
    matches = pd.DataFrame({"match_id": ids})
    level = rng.normal(0, 60, n)  # how strong the two teams are overall, so tiers genuinely differ
    elo_pre = pd.DataFrame(
        {
            "elo_home_pre": 1500 + level + X["elo_diff"].to_numpy() / 2,
            "elo_away_pre": 1500 + level - X["elo_diff"].to_numpy() / 2,
        },
        index=pd.Index(ids, name="match_id"),
    )

    out = tmp_path / "card.md"
    write_model_card(out, model, preds, matches, X_eval=X, y_eval=y, elo_pre=elo_pre)
    text = out.read_text()

    assert "## Feature importance" in text
    assert "elo_diff" in text and "rest_days_home" in text
    assert "## Calibration" in text
    assert "## Performance by league" in text and "E0" in text and "SP1" in text
    assert "## Performance by team-strength tier" in text
    for tier in ("low", "mid", "high"):
        assert tier in text


def test_model_card_degrades_gracefully_without_optional_inputs(tmp_path):
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"elo_diff": rng.normal(0, 100, 300)})
    y = rng.integers(0, 3, 300)
    model = MetaModel().fit(X, y)
    probs = model.predict_proba(X)
    preds = pd.DataFrame(
        {"match_id": [f"m{i}" for i in range(300)], "p_home": probs[:, 0], "p_draw": probs[:, 1], "p_away": probs[:, 2],
         "outcome": y, "date": pd.Timestamp("2025-09-01"), "league": "E0"}
    )
    out = tmp_path / "card.md"
    write_model_card(out, model, preds, pd.DataFrame({"match_id": preds["match_id"]}))
    text = out.read_text()
    assert "## Feature importance" in text and "## Performance by team-strength tier" in text
    assert "not available" in text.lower()


def test_tiers_survive_identical_strength_values(tmp_path):
    # Every match has exactly the same mean Elo: tiering must still work instead of crashing.
    n = 90
    rng = np.random.default_rng(2)
    X = pd.DataFrame({"elo_diff": rng.normal(0, 100, n)})
    y = rng.integers(0, 3, n)
    model = MetaModel().fit(X, y)
    probs = model.predict_proba(X)
    ids = [f"m{i}" for i in range(n)]
    preds = pd.DataFrame({"match_id": ids, "p_home": probs[:, 0], "p_draw": probs[:, 1], "p_away": probs[:, 2],
                          "outcome": y, "date": pd.Timestamp("2025-09-01"), "league": "E0"})
    elo_pre = pd.DataFrame({"elo_home_pre": 1500.0, "elo_away_pre": 1500.0}, index=pd.Index(ids, name="match_id"))
    out = tmp_path / "card.md"
    write_model_card(out, model, preds, pd.DataFrame({"match_id": ids}), elo_pre=elo_pre)
    assert "low" in out.read_text()
