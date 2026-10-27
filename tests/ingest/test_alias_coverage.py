"""Every club the football-data.org API returns must map to the name used in the history CSVs."""
import pytest

from footpred.ingest.teams import canonical_team, covered_leagues

API_TO_CSV = {
    "F1": {
        "Toulouse FC": "Toulouse", "Stade Brestois 29": "Brest", "Olympique de Marseille": "Marseille",
        "AJ Auxerre": "Auxerre", "Lille OSC": "Lille", "OGC Nice": "Nice", "Olympique Lyonnais": "Lyon",
        "Paris Saint-Germain FC": "Paris SG", "FC Lorient": "Lorient", "Stade Rennais FC 1901": "Rennes",
        "ES Troyes AC": "Troyes", "Angers SCO": "Angers", "Le Havre AC": "Le Havre", "Le Mans FC": "Le Mans",
        "Racing Club de Lens": "Lens", "AS Monaco FC": "Monaco", "RC Strasbourg Alsace": "Strasbourg",
        "Paris FC": "Paris FC",
    },
    "D1": {
        "1. FC Köln": "FC Koln", "TSG 1899 Hoffenheim": "Hoffenheim", "Bayer 04 Leverkusen": "Leverkusen",
        "Borussia Dortmund": "Dortmund", "FC Bayern München": "Bayern Munich", "FC Schalke 04": "Schalke 04",
        "Hamburger SV": "Hamburg", "VfB Stuttgart": "Stuttgart", "SV Werder Bremen": "Werder Bremen",
        "1. FSV Mainz 05": "Mainz", "FC Augsburg": "Augsburg", "SC Freiburg": "Freiburg",
        "Borussia Mönchengladbach": "M'gladbach", "Eintracht Frankfurt": "Ein Frankfurt",
        "1. FC Union Berlin": "Union Berlin", "SC Paderborn 07": "Paderborn", "SV 07 Elversberg": "Elversberg",
        "RB Leipzig": "RB Leipzig",
    },
    "I1": {
        "AC Milan": "Milan", "ACF Fiorentina": "Fiorentina", "AS Roma": "Roma", "Atalanta BC": "Atalanta",
        "Bologna FC 1909": "Bologna", "Cagliari Calcio": "Cagliari", "Genoa CFC": "Genoa",
        "FC Internazionale Milano": "Inter", "Juventus FC": "Juventus", "SS Lazio": "Lazio",
        "Parma Calcio 1913": "Parma", "SSC Napoli": "Napoli", "Udinese Calcio": "Udinese", "Venezia FC": "Venezia",
        "Frosinone Calcio": "Frosinone", "US Sassuolo Calcio": "Sassuolo", "Torino FC": "Torino",
        "US Lecce": "Lecce", "AC Monza": "Monza", "Como 1907": "Como",
    },
    "SP1": {
        "Athletic Club": "Ath Bilbao", "Club Atlético de Madrid": "Ath Madrid", "CA Osasuna": "Osasuna",
        "RCD Espanyol de Barcelona": "Espanol", "FC Barcelona": "Barcelona", "Getafe CF": "Getafe",
        "Málaga CF": "Malaga", "Real Madrid CF": "Real Madrid", "Rayo Vallecano de Madrid": "Vallecano",
        "Levante UD": "Levante", "Real Betis Balompié": "Betis", "Real Sociedad de Fútbol": "Sociedad",
        "Villarreal CF": "Villarreal", "Valencia CF": "Valencia", "Deportivo Alavés": "Alaves", "Elche CF": "Elche",
        "RC Celta de Vigo": "Celta", "Sevilla FC": "Sevilla", "RC Deportivo La Coruña": "La Coruna",
        "Real Racing Club de Santander": "Santander",
    },
}


def test_all_five_leagues_are_covered():
    assert covered_leagues() == ["D1", "E0", "F1", "I1", "SP1"]


@pytest.mark.parametrize("league", sorted(API_TO_CSV))
def test_every_api_club_maps_to_its_csv_name(league):
    for api_name, csv_name in API_TO_CSV[league].items():
        assert canonical_team(api_name, league) == csv_name, api_name
        assert canonical_team(csv_name, league) == csv_name, csv_name
