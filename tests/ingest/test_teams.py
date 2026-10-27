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
