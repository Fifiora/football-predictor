import pytest

from footpred.european.names import GROUPS, EuropeanNames, group_for_country, normalize_key
from footpred.ingest.teams import canonical_team


@pytest.fixture
def names(tmp_path):
    euro = tmp_path / "european_aliases.csv"
    euro.write_text("alias,canonical\nSport Lisboa e Benfica,Benfica\n")
    return EuropeanNames(euro_aliases_path=euro)


def test_big_five_clubs_resolve_to_league_canonical_names(names):
    assert names.resolve("FC Bayern München", "GER") == canonical_team("FC Bayern München", "D1") == "Bayern Munich"
    assert names.resolve("Club Atlético de Madrid", "ESP") == "Ath Madrid"
    assert names.resolve("Paris Saint-Germain FC", "FRA") == "Paris SG"
    assert names.unmatched == set()


def test_country_restricts_the_league_tables(names):
    # "Paris FC" and "Paris SG" both exist in F1; an ESP tag must not resolve to a French club.
    assert names.resolve("Paris Saint-Germain FC", "ESP") == "Paris Saint-Germain FC"
    assert "Paris Saint-Germain FC" in names.unmatched


def test_without_a_country_every_league_is_tried(names):
    assert names.resolve("Manchester City FC") == "Manchester City"


def test_normalized_key_matches_suffix_variants(names):
    assert normalize_key("FC Barcelona") == normalize_key("Barcelona")
    assert normalize_key("Real Madrid CF") == normalize_key("Real Madrid")
    assert names.resolve("Sport Lisboa e Benfica", "POR") == "Benfica"
    assert names.resolve("SL Benfica", "POR") == "SL Benfica"  # not guessed: reported instead


def test_normalized_key_finds_league_clubs_spelled_differently(names):
    assert names.resolve("Barcelona FC", "ESP") == "Barcelona"


def test_unknown_club_is_kept_flagged_and_never_raises(names):
    assert names.resolve("Qarabağ FK", "AZE") == "Qarabağ FK"
    assert "Qarabağ FK" in names.unmatched


def test_same_unknown_name_resolves_the_same_each_time(names):
    assert names.resolve("  Qarabağ   FK ", "AZE") == names.resolve("Qarabağ FK", "AZE")


def test_canonical_hook_has_the_fetcher_signature(names):
    assert names.canonical("Arsenal FC", "CL") == "Arsenal"


def test_groups():
    assert GROUPS == ["E0", "SP1", "I1", "D1", "F1", "OTHER"]
    assert group_for_country("ENG") == "E0" and group_for_country("POR") == "OTHER" and group_for_country(None) == "OTHER"


@pytest.mark.parametrize("raw,country,canonical", [
    ("1. FC Heidenheim 1846", "GER", "Heidenheim"),
    ("Lazio Roma", "ITA", "Lazio"),
    ("Montpellier HSC", "FRA", "Montpellier"),
    ("Rayo Vallecano", "ESP", "Vallecano"),
    ("Real Betis", "ESP", "Betis"),
    ("Real Sociedad", "ESP", "Sociedad"),
    ("Stade Rennais", "FRA", "Rennes"),
])
def test_openfootball_spellings_that_needed_an_alias(raw, country, canonical):
    names = EuropeanNames()  # the shipped data/european_aliases.csv
    assert names.resolve(raw, country) == canonical and names.unmatched == set()


def test_live_feed_spellings_of_non_big_five_clubs_match_the_history_files():
    names = EuropeanNames()
    assert names.resolve("Fenerbahçe SK") == "Fenerbahçe"
    assert names.resolve("LASK Linz") == "LASK"
    assert names.resolve("PAE AEK") == "AEK Athen"
