

def test_blend_is_the_mean_of_elo_and_dixon_coles():
    import numpy as np
    import pandas as pd

    from footpred.evaluate.baselines import baseline_blend, baseline_dixon_coles, baseline_elo

    rng = np.random.default_rng(0)
    n = 300
    train = pd.DataFrame({"elo_diff": rng.normal(0, 100, n), "outcome": rng.integers(0, 3, n)})
    for col, v in zip(("dc_p_home", "dc_p_draw", "dc_p_away"), (0.45, 0.27, 0.28)):
        train[col] = v
    test = train.drop(columns="outcome").iloc[:20]
    expected = (baseline_elo(train, test) + baseline_dixon_coles(train, test)) / 2
    got = baseline_blend(train, test)
    assert np.allclose(got, expected) and np.allclose(got.sum(axis=1), 1)
