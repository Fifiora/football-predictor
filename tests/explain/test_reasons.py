import numpy as np
import pandas as pd
import pytest

from footpred.explain.reasons import Driver, driver_text, top_drivers
from footpred.explain.templates import TEMPLATES
from footpred.features.builder import FEATURE_COLUMNS
from footpred.models.meta_model import MetaModel


def synthetic(n=2500, seed=0):
    rng = np.random.default_rng(seed)
    elo = rng.normal(0, 100, n)
    X = pd.DataFrame({
        "elo_diff": elo,
        "dc_p_home": rng.uniform(0.2, 0.7, n),
        "rest_days_home": rng.integers(2, 9, n).astype(float),
    })
    p_home = 1 / (1 + np.exp(-elo / 40))
    u = rng.random(n)
    y = np.where(u < p_home * 0.8, 0, np.where(u < p_home * 0.8 + 0.2, 1, 2))
    return X, y


@pytest.fixture(scope="module")
def model():
    X, y = synthetic()
    return MetaModel().fit(X, y)


def row(model, **overrides):
    base = model.baseline.copy()
    for k, v in overrides.items():
        base[k] = v
    return base


def test_returns_at_most_k_and_ranked(model):
    drivers = top_drivers(model, row(model, elo_diff=180.0, dc_p_home=0.65, rest_days_home=8.0), k=2)
    assert 1 <= len(drivers) <= 2
    strengths = [d.strength for d in drivers]
    assert strengths == sorted(strengths, reverse=True)
    assert all(isinstance(d, Driver) for d in drivers)


def test_direction_follows_sign(model):
    home = top_drivers(model, row(model, elo_diff=200.0), 3)
    away = top_drivers(model, row(model, elo_diff=-200.0), 3)
    assert home[0].feature == "elo_diff" and home[0].direction == "home"
    assert away[0].feature == "elo_diff" and away[0].direction == "away"


def test_min_strength_filters_noise(model):
    assert top_drivers(model, row(model), 3) == []  # a baseline row has nothing to explain


def test_missing_values_are_never_shown_as_drivers(model):
    drivers = top_drivers(model, row(model, elo_diff=200.0, dc_p_home=np.nan), 3)
    assert "dc_p_home" not in [d.feature for d in drivers]


def test_text_has_no_causal_wording():
    for feature in TEMPLATES:
        for direction in ("home", "away"):
            text = driver_text(feature, direction, 1.0, "Arsenal", "Chelsea").lower()
            assert "model" in text, feature
            for banned in ("because", "caused", "due to", "leads to"):
                assert banned not in text, (feature, text)


def test_text_mentions_the_favoured_team_and_value():
    text = driver_text("elo_diff", "home", 85.0, "Arsenal", "Chelsea")
    assert text.endswith("which points to Arsenal.")
    assert "85 Elo points" in text


@pytest.mark.parametrize(
    "feature, direction, value, expected",
    [
        ("elo_diff", "home", 76.0,
         "The model weighted Aston Villa being rated 76 Elo points above Fulham, which points to Aston Villa."),
        ("elo_diff", "away", -40.0,
         "The model weighted Aston Villa being rated 40 Elo points below Fulham, which points to Fulham."),
        ("form_shots_diff_away", "home", -1.4,
         "The model weighted Fulham being out-shot by 1.4 shots a match lately, which points to Aston Villa."),
        ("form_shots_diff_home", "home", 5.4,
         "The model weighted Aston Villa out-shooting opponents by 5.4 shots a match lately, which points to Aston Villa."),
        ("rest_days_home", "away", 6.0,
         "The model weighted Aston Villa having had 6 days of rest, which points to Fulham."),
        ("dc_p_away", "away", 0.28,
         "The model weighted the goals model giving Fulham a 28% chance of winning, which points to Fulham."),
        ("form_goals_for_home", "home", 1.84,
         "The model weighted Aston Villa scoring 1.8 goals a match lately, which points to Aston Villa."),
    ],
)
def test_text_reads_naturally(feature, direction, value, expected):
    assert driver_text(feature, direction, value, "Aston Villa", "Fulham") == expected


def test_no_signed_numbers_in_text():
    for feature in TEMPLATES:
        text = driver_text(feature, "home", -2.5, "Arsenal", "Chelsea")
        assert "-2.5" not in text and "+" not in text, (feature, text)


def test_every_published_feature_has_a_template():
    assert set(FEATURE_COLUMNS) <= set(TEMPLATES)


def test_goals_model_features_are_shown_at_most_once():
    X, y = synthetic(seed=3)
    X["dc_exp_home"] = X["dc_p_home"] * 3
    m = MetaModel().fit(X, y)
    extreme = m.baseline.copy()
    extreme["dc_p_home"] = 0.9
    extreme["dc_exp_home"] = 2.7
    drivers = top_drivers(m, extreme, 5)
    assert sum(d.feature.startswith("dc_") for d in drivers) <= 1
