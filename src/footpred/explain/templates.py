"""Plain-language phrases for each model feature.

Each template is a format string over ``value``, ``home`` and ``away`` and names what the model
weighted. Wording stays descriptive ("the model weighted ...") and never claims a real-world cause.
"""
from __future__ import annotations

# feature -> (phrase template, family). Features in the same family are shown at most once.
TEMPLATES: dict[str, tuple[str, str]] = {
    "elo_diff": ("the rating gap ({value:+.0f} Elo)", "rating"),
    "dc_exp_home": ("the goals model's expected goals for {home} ({value:.1f})", "goals_model"),
    "dc_exp_away": ("the goals model's expected goals for {away} ({value:.1f})", "goals_model"),
    "dc_p_home": ("the goals model's home-win chance ({value:.0%})", "goals_model"),
    "dc_p_draw": ("the goals model's draw chance ({value:.0%})", "goals_model"),
    "dc_p_away": ("the goals model's away-win chance ({value:.0%})", "goals_model"),
    "form_goals_for_home": ("{home}'s recent scoring ({value:.1f} goals per match)", "form_goals_for_home"),
    "form_goals_against_home": ("{home}'s recent goals conceded ({value:.1f} per match)", "form_goals_against_home"),
    "form_goals_for_away": ("{away}'s recent scoring ({value:.1f} goals per match)", "form_goals_for_away"),
    "form_goals_against_away": ("{away}'s recent goals conceded ({value:.1f} per match)", "form_goals_against_away"),
    "form_shots_diff_home": ("{home}'s recent shot balance ({value:+.1f} per match)", "form_shots_diff_home"),
    "form_shots_diff_away": ("{away}'s recent shot balance ({value:+.1f} per match)", "form_shots_diff_away"),
    "rest_days_home": ("{home}'s rest ({value:.0f} days since their last match)", "rest_days_home"),
    "rest_days_away": ("{away}'s rest ({value:.0f} days since their last match)", "rest_days_away"),
}
