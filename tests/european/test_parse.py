import pandas as pd
import pytest

from footpred.european.parse import COMPETITIONS, STAGES, parse_openfootball

HEADER = "= UEFA Champions League 2024/25\n\n# Date       Tue Sep 17 2024 - Sat May 31 2025 (256d)\n\n"


def parse(body: str, competition: str = "CL", season: str = "2024-25") -> pd.DataFrame:
    return parse_openfootball(HEADER + body, competition, season)


LEAGUE = """▪ League, Matchday 1
  Tue Sep 17 2024
    18:45  BSC Young Boys (SUI)    v Aston Villa FC (ENG)     0-3 (0-2)
           Juventus FC (ITA)       v PSV (NED)                3-1 (2-0)
  Wed Sep 18
    18:45  Bologna FC 1909 (ITA)   v FK Shakhtar Donetsk (UKR)  0-0
"""


def test_regular_match_takes_full_time_score():
    df = parse(LEAGUE)
    row = df[df["home"] == "Juventus FC"].iloc[0]
    assert (row["home_goals"], row["away_goals"], row["result"]) == (3, 1, "H")
    assert (row["home_country"], row["away_country"], row["stage"]) == ("ITA", "NED", "LEAGUE_STAGE")
    assert row["competition"] == "CL" and row["season"] == "2024-25"


def test_score_without_halftime():
    row = parse(LEAGUE).query("home == 'Bologna FC 1909'").iloc[0]
    assert (row["home_goals"], row["away_goals"], row["result"]) == (0, 0, "D")


def test_penalties_use_the_ninety_minute_score():
    body = """▪ Finals, Round of 16
  Tue Mar 11 2025
    21:00  Liverpool FC (ENG)      v Paris Saint-Germain FC (FRA)  1-4 pen. 0-1 a.e.t. (0-1, 0-1)
"""
    row = parse(body).iloc[0]
    assert (row["home_goals"], row["away_goals"], row["result"], row["stage"]) == (0, 1, "A", "LAST_16")


def test_extra_time_draw_is_a_draw_after_ninety():
    body = """▪ Finals, Final
  Sat May 31 2025
    21:00  Paris Saint-Germain FC (FRA) v Arsenal FC (ENG)         4-3 pen. 1-1 a.e.t. (1-1, 0-1)
"""
    row = parse(body).iloc[0]
    assert (row["home_goals"], row["away_goals"], row["result"], row["stage"]) == (1, 1, "D", "FINAL")


def test_extra_time_without_penalties():
    body = """▪ Round of 16
  Tue Mar 11 2025
    21:00  A FC (ENG) v B FC (ESP)  2-2 a.e.t. (1-1, 0-0)
"""
    row = parse(body).iloc[0]
    assert (row["home_goals"], row["away_goals"], row["result"]) == (1, 1, "D")


def test_single_bracket_after_extra_time_is_used_when_it_is_a_draw_after_ninety():
    body = """▪ Final
  Tue Mar 11 2025
    21:00  A FC (ENG) v B FC (ESP)  8-7 pen. 0-0 a.e.t. (0-0)
"""
    row = parse(body).iloc[0]
    assert (row["home_goals"], row["away_goals"], row["result"]) == (0, 0, "D")


def test_awarded_and_cancelled_matches_are_dropped():
    body = """▪ Group A
  Tue Sep 17 2024
    18:45  A FC (ENG) v B FC (ESP)  0-3    [awarded]
           C FC (ITA) v D FC (FRA)  [cancelled]
           E FC (GER) v F FC (NED)  1-0 (0-0)
"""
    df = parse(body)
    assert df["home"].tolist() == ["E FC"]


def test_dates_carry_forward_and_roll_the_year():
    body = """▪ League, Matchday 7
  Wed Dec 11 2024
    21:00  A FC (ENG) v B FC (ESP)  1-0 (0-0)
  Wed Jan 29
    21:00  C FC (ITA) v D FC (FRA)  1-0 (0-0)
"""
    df = parse(body)
    assert df["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-12-11", "2025-01-29"]


def test_year_less_day_inherits_the_year():
    assert parse(LEAGUE).query("home == 'Bologna FC 1909'").iloc[0]["date"] == pd.Timestamp("2024-09-18")


@pytest.mark.parametrize("header,stage", [
    ("League, Matchday 1", "LEAGUE_STAGE"),
    ("Playoffs, Matchday 1", "PLAYOFFS"),
    ("Finals, Round of 16", "LAST_16"),
    ("Finals, Quarterfinals", "QUARTER_FINALS"),
    ("Finals, Semifinals", "SEMI_FINALS"),
    ("Finals, Final", "FINAL"),
    ("Group A, Matchday 1", "LEAGUE_STAGE"),
    ("Group A", "LEAGUE_STAGE"),
    ("Gruppe H", "LEAGUE_STAGE"),
    ("Round of 16", "LAST_16"),
    ("Quarterfinals", "QUARTER_FINALS"),
    ("Semifinals", "SEMI_FINALS"),
    ("Final", "FINAL"),
    ("Playoffs", "PLAYOFFS"),
    ("Sechzehntelfinale", "PLAYOFFS"),
])
def test_stage_headers_map(header, stage):
    body = f"▪ {header}\n  Tue Sep 17 2024\n    21:00  A FC (ENG) v B FC (ESP)  1-0 (0-0)\n"
    assert parse(body).iloc[0]["stage"] == stage and stage in STAGES


def test_qualifier_files_use_qualifying_stage():
    body = "▪ Round 2\n  Tue Jul 23 2024\n    21:00  A FC (SCO) v B FC (SUI)  1-0 (0-0)\n▪ Play-offs\n  Tue Aug 20\n    21:00  C FC (AUT) v D FC (CRO)  2-1 (1-0)\n"
    df = parse(body, competition="CLQ")
    assert set(df["stage"]) == {"QUALIFYING"} and set(df["competition"]) == {"CLQ"}


def test_unknown_header_in_a_main_competition_raises():
    body = "▪ Mystery Round\n  Tue Sep 17 2024\n    21:00  A FC (ENG) v B FC (ESP)  1-0 (0-0)\n"
    with pytest.raises(ValueError, match="Mystery Round"):
        parse(body)


def test_competition_table():
    assert COMPETITIONS == {"cl": "CL", "clq": "CLQ", "el": "EL", "elq": "ELQ", "conf": "CONF", "confq": "CONFQ"}


def test_year_comes_from_the_season_even_when_dates_are_listed_out_of_order():
    body = """▪ League, Matchday 7
  Wed Jan 29
    21:00  C FC (ITA) v D FC (FRA)  1-0 (0-0)
  Tue Sep 17
    21:00  A FC (ENG) v B FC (ESP)  2-0 (1-0)
  Wed May 28
    21:00  E FC (ENG) v F FC (ESP)  0-0 (0-0)
"""
    df = parse(body)  # season 2024-25: July to December is 2024, January to June is 2025
    assert df["date"].dt.strftime("%Y-%m-%d").tolist() == ["2025-01-29", "2024-09-17", "2025-05-28"]


def test_a_season_that_runs_into_august_keeps_its_second_year():
    header = "= UEFA Champions League 2019/20\n\n# Date       Tue Sep 17 2019 - Sun Aug 23 2020 (341d)\n\n"
    body = """▪ League, Matchday 1
  Wed Mar 11
    21:00  A FC (ENG) v B FC (ESP)  1-0 (0-0)
▪ Finals, Quarterfinals
  Fri Aug 7
    21:00  C FC (ITA) v D FC (FRA)  1-0 (0-0)
"""
    df = parse_openfootball(header + body, "CL", "2019-20")
    assert df["date"].dt.strftime("%Y-%m-%d").tolist() == ["2020-03-11", "2020-08-07"]
