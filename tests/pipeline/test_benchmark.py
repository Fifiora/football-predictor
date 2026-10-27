from footpred.pipeline.benchmark import run_benchmark
from conftest import TODAY


def test_benchmark_compares_models_and_respects_holdout_rule(world, tmp_path):
    out = tmp_path / "baseline-results.md"
    res = run_benchmark(world["data_dir"], out, aliases_path=world["aliases"], today=TODAY)

    dev = res["development"]
    assert set(dev) >= {"base_rate", "elo", "dixon_coles", "hybrid"}
    for name in dev:
        assert dev[name]["log_loss"] > 0 and "rps" in dev[name] and "ece" in dev[name]
        assert "E0" in dev[name]["by_league_log_loss"]

    beats_both = dev[res["chosen_hybrid"]]["log_loss"] < min(dev["elo"]["log_loss"], dev["dixon_coles"]["log_loss"])
    assert (res["holdout"] is not None) == beats_both  # held-out season is touched only on a development win
    assert res["holdout_season"] == "2025-26"

    text = out.read_text()
    assert "## Development seasons" in text and "Hybrid" in text and "Elo" in text and "Dixon-Coles" in text
    assert "## Held-out season" in text


def test_benchmark_compares_hybrid_variants_and_reports_the_chosen_one(world, tmp_path):
    variants = {"hybrid": None, "hybrid_logistic": {"backend": "logistic", "C": 0.1}}
    res = run_benchmark(world["data_dir"], tmp_path / "out.md", aliases_path=world["aliases"], today=TODAY,
                        variants=variants, test_seasons=3)
    dev = res["development"]
    assert {"hybrid", "hybrid_logistic"} <= set(dev)
    best = min(variants, key=lambda v: dev[v]["log_loss"])
    assert res["chosen_hybrid"] == best
    text = (tmp_path / "out.md").read_text()
    assert "Hybrid (logistic)" in text or "hybrid_logistic" in text
    if res["holdout"] is not None:  # only the chosen variant is evaluated on the held-out season
        assert set(res["holdout"]) == {"base_rate", "elo", "dixon_coles", "hybrid"}
