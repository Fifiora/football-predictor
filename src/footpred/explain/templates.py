"""Plain-language phrases for each model feature.

Each template is a function ``(value, home, away) -> str`` that states the fact the model looked
at, never the direction it pushed (``driver_text`` adds that). Wording stays descriptive
("the model weighted ...") and never claims a real-world cause. Numbers are shown unsigned:
the words say above/below, out-shot/out-shooting.
"""
from __future__ import annotations

from typing import Callable

Template = Callable[[float, str, str], str]


def _elo(value: float, home: str, away: str) -> str:
    side = "above" if value >= 0 else "below"
    return f"{home} being rated {abs(value):.0f} Elo points {side} {away}"


def _shots(team_side: str) -> Template:
    def phrase(value: float, home: str, away: str) -> str:
        team = home if team_side == "home" else away
        if value >= 0:
            return f"{team} out-shooting opponents by {abs(value):.1f} shots a match lately"
        return f"{team} being out-shot by {abs(value):.1f} shots a match lately"

    return phrase


def _rest(team_side: str) -> Template:
    return lambda value, home, away: (
        f"{home if team_side == 'home' else away} having had {abs(value):.0f} days of rest"
    )


def _goals(team_side: str, verb: str) -> Template:
    return lambda value, home, away: (
        f"{home if team_side == 'home' else away} {verb} {abs(value):.1f} goals a match lately"
    )


def _dc_exp(team_side: str) -> Template:
    return lambda value, home, away: (
        f"the goals model expecting {home if team_side == 'home' else away} to score {abs(value):.1f} goals"
    )


def _dc_chance(what: str) -> Template:
    def phrase(value: float, home: str, away: str) -> str:
        team = {"home": home, "away": away}.get(what)
        shown = f"{abs(value):.0%}"
        return f"the goals model giving {team} a {shown} chance of winning" if team else f"the goals model giving the match a {shown} chance of a draw"

    return phrase


# feature -> (phrase function, family). Features in the same family are shown at most once.
TEMPLATES: dict[str, tuple[Template, str]] = {
    "elo_diff": (_elo, "rating"),
    "dc_exp_home": (_dc_exp("home"), "goals_model"),
    "dc_exp_away": (_dc_exp("away"), "goals_model"),
    "dc_p_home": (_dc_chance("home"), "goals_model"),
    "dc_p_draw": (_dc_chance("draw"), "goals_model"),
    "dc_p_away": (_dc_chance("away"), "goals_model"),
    "form_goals_for_home": (_goals("home", "scoring"), "form_goals_for_home"),
    "form_goals_against_home": (_goals("home", "conceding"), "form_goals_against_home"),
    "form_goals_for_away": (_goals("away", "scoring"), "form_goals_for_away"),
    "form_goals_against_away": (_goals("away", "conceding"), "form_goals_against_away"),
    "form_shots_diff_home": (_shots("home"), "form_shots_diff_home"),
    "form_shots_diff_away": (_shots("away"), "form_shots_diff_away"),
    "rest_days_home": (_rest("home"), "rest_days_home"),
    "rest_days_away": (_rest("away"), "rest_days_away"),
}
