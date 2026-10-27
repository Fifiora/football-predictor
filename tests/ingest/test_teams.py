import pytest

from footpred.ingest.teams import UnknownTeamError, canonical_team


def test_canonical_team_maps_variants():
    assert canonical_team("Man United", "E0") == canonical_team("Manchester United", "E0")


def test_canonical_name_maps_to_itself():
    assert canonical_team("Arsenal", "E0") == "Arsenal"


def test_unknown_team_raises():
    with pytest.raises(UnknownTeamError):
        canonical_team("Nonexistent FC", "E0")


def test_lookup_ignores_case_and_whitespace():
    assert canonical_team("  man city ", "E0") == canonical_team("Man City", "E0")


def test_covered_leagues_lists_only_leagues_with_aliases(tmp_path):
    from footpred.ingest.teams import covered_leagues

    aliases = tmp_path / "a.csv"
    aliases.write_text("league,alias,canonical\nE0,Arsenal,Arsenal\nSP1,Betis,Real Betis\nE0,Spurs,Tottenham\n")
    assert covered_leagues(aliases) == ["E0", "SP1"]
    assert covered_leagues() == ["E0"]  # the shipped table only seeds the Premier League so far
