from conftest import TODAY
from footpred.pipeline.tune import tune_league


def test_tune_league_searches_every_stage_on_development_seasons_only(world, tmp_path):
    out = tmp_path / "league-tuning.md"
    res = tune_league(world["data_dir"], out, aliases_path=world["aliases"], today=TODAY,
                      elo_grid={"k": (15.0, 30.0), "home_adv": (60.0,), "season_regress": (0.25,), "margin": (True, False)},
                      xi_grid=(0.001, 0.003))
    assert res["holdout_season"] == "2025-26"
    assert len(res["elo_rows"]) == 4 and len(res["xi_rows"]) == 2
    assert len(res["hybrid_rows"]) == 12  # 4 input sets x 3 strengths
    assert res["hybrid"]["log_loss"] == min(r["log_loss"] for r in res["hybrid_rows"])
    text = out.read_text()
    assert "Stage 1" in text and "Stage 2" in text and "Stage 3" in text and "Chosen:" in text
    assert "hold-out season is not read" in text
