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
