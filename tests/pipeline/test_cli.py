from footpred.pipeline.cli import download_history


def test_download_only_requests_given_leagues_and_reports_failures(tmp_path):
    asked = []

    def fetch(url):
        asked.append(url)
        if "2425" in url:
            raise OSError("not published")
        return b"Div,Date\n"

    failed = download_history(tmp_path, 2023, 2024, leagues=["E0"], fetch=fetch)
    assert all("/E0.csv" in u for u in asked) and len(asked) == 2
    assert failed == ["E0_2425.csv"]
    assert (tmp_path / "E0_2324.csv").exists()


def test_completed_seasons_are_not_downloaded_twice(tmp_path):
    (tmp_path / "E0_2324.csv").write_text("cached")
    asked = []
    download_history(tmp_path, 2023, 2024, leagues=["E0"], fetch=lambda url: asked.append(url) or b"x")
    assert len(asked) == 1 and "2425" in asked[0]  # only the current season is refreshed


def test_failures_become_github_error_annotations(tmp_path, capsys):
    from footpred.pipeline.cli import main

    code = main(["retrain", "--data-dir", str(tmp_path / "empty"), "--registry-dir", str(tmp_path / "reg")])
    out = capsys.readouterr().out
    assert code == 1
    assert out.startswith("::error") or "\n::error" in out


def test_list_team_names_reads_every_csv_per_league(tmp_path):
    from footpred.pipeline.cli import list_team_names

    (tmp_path / "SP1_2324.csv").write_text("Div,HomeTeam,AwayTeam\nSP1,Betis,Ath Madrid\nSP1,Ath Madrid,Sevilla\n")
    (tmp_path / "SP1_2425.csv").write_text("Div,HomeTeam,AwayTeam\nSP1,Girona,Betis\n")
    (tmp_path / "D1_2425.csv").write_text("Div,HomeTeam,AwayTeam\nD1,Bayern Munich,Dortmund\n")
    assert list_team_names(tmp_path) == {
        "D1": ["Bayern Munich", "Dortmund"],
        "SP1": ["Ath Madrid", "Betis", "Girona", "Sevilla"],
    }


def test_list_team_names_handles_bom_and_latin1(tmp_path):
    from footpred.pipeline.cli import list_team_names

    header = "Div,HomeTeam,AwayTeam\n"
    (tmp_path / "SP1_a.csv").write_bytes(("﻿" + header + "SP1,Málaga,Betis\n").encode("utf-8"))
    (tmp_path / "SP1_b.csv").write_bytes((header + "SP1,Betis,Sevilla\n").encode("latin-1"))
    assert list_team_names(tmp_path)["SP1"] == ["Betis", "Málaga", "Sevilla"]


def test_summarize_fixtures_lists_probabilities_and_reasons(tmp_path):
    import json

    from footpred.pipeline.cli import summarize_fixtures

    (tmp_path / "fixtures.json").write_text(json.dumps([
        {"date": "2026-10-31", "league": "E0", "home": "Aston Villa", "away": "Fulham",
         "p_home": 0.49, "p_draw": 0.24, "p_away": 0.27,
         "drivers": [{"text": "The model weighted X, which points to Aston Villa."}]},
        {"date": "2026-11-01", "league": "D1", "home": "A", "away": "B", "p_home": 0.3, "p_draw": 0.3, "p_away": 0.4, "drivers": []},
    ]))
    lines = summarize_fixtures(tmp_path, limit=1)
    assert lines[0] == "2026-10-31 E0 Aston Villa v Fulham: 49% / 24% / 27%"
    assert lines[1] == "  - The model weighted X, which points to Aston Villa."
    assert len(lines) == 2 and summarize_fixtures(tmp_path / "missing") == []


def test_download_command_survives_a_failed_openfootball_clone(tmp_path, monkeypatch, capsys):
    from footpred.pipeline import cli

    monkeypatch.setattr(cli, "download_history", lambda *a, **k: [])
    monkeypatch.setattr(cli, "download_openfootball", lambda dest, **k: False)
    code = cli.main(["download", "--data-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0 and "::warning title=openfootball" in out


def test_benchmark_european_command_writes_the_report_and_prints_the_gate(tmp_path, monkeypatch, capsys):
    from footpred.pipeline import cli

    seen = {}

    def fake(data_dir, out_path, **kw):
        seen.update(data_dir=data_dir, out=out_path)
        return {"publish_ok": False, "reason": "model 1.0 vs base 0.9"}

    monkeypatch.setattr(cli, "run_european_benchmark", fake)
    code = cli.main(["benchmark-european", "--data-dir", str(tmp_path), "--out", str(tmp_path / "e.md")])
    out = capsys.readouterr().out
    assert code == 0 and seen["out"] == tmp_path / "e.md"
    assert "publish gate" in out and "model 1.0 vs base 0.9" in out
