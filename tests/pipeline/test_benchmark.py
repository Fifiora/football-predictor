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

    beats_both = dev["hybrid"]["log_loss"] < min(dev["elo"]["log_loss"], dev["dixon_coles"]["log_loss"])
    assert (res["holdout"] is not None) == beats_both  # held-out season is touched only on a development win
    assert res["holdout_season"] == "2025-26"

    text = out.read_text()
    assert "## Development seasons" in text and "Hybrid" in text and "Elo" in text and "Dixon-Coles" in text
    assert "## Held-out season" in text
