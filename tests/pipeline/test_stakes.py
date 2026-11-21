import pandas as pd
import pytest

from footpred.pipeline.stakes import START, cash, place_bets, read_ledger, settle_bets, summarize

NOW = pd.Timestamp("2026-11-20 12:00")


def fx(mid, ph, pd_, pa, kickoff="2026-11-21", league="E0"):
    return {"match_id": mid, "league": league, "home": "A", "away": "B", "kickoff_date": pd.Timestamp(kickoff), "p_home": ph, "p_draw": pd_, "p_away": pa}


def odds(*ids, price=(2.0, 3.5, 4.0)):
    return {m: {"bet365_price": price} for m in ids}


def test_only_confident_picks_with_a_price_are_staked_five_percent_of_cash(tmp_path):
    path = tmp_path / "s.csv"
    frame = pd.DataFrame([fx("E0-1", 0.6, 0.25, 0.15), fx("E0-2", 0.5, 0.3, 0.2), fx("E0-3", 0.2, 0.2, 0.6)])
    placed = place_bets(path, frame, odds("E0-1", "E0-2"), NOW)  # E0-2 not confident, E0-3 has no price
    assert placed == 1
    ledger = read_ledger(path)
    assert list(ledger["match_id"]) == ["E0-1"]
    row = ledger.iloc[0]
    assert row["pick"] == "H" and row["odds"] == 2.0 and row["stake"] == 5.0
    assert cash(ledger) == 95.0


def test_each_bet_is_sized_from_the_cash_left_and_prices_are_locked(tmp_path):
    path = tmp_path / "s.csv"
    place_bets(path, pd.DataFrame([fx("E0-1", 0.6, 0.2, 0.2), fx("E0-2", 0.1, 0.2, 0.7, "2026-11-22")]), odds("E0-1", "E0-2"), NOW)
    assert list(read_ledger(path)["stake"]) == [5.0, 4.75]
    # running again, even with a new price, changes nothing
    assert place_bets(path, pd.DataFrame([fx("E0-1", 0.6, 0.2, 0.2)]), odds("E0-1", price=(9.0, 9.0, 9.0)), NOW) == 0
    assert list(read_ledger(path)["odds"]) == [2.0, 4.0]


def test_champions_league_and_started_matches_are_never_staked(tmp_path):
    path = tmp_path / "s.csv"
    frame = pd.DataFrame([fx("CL-1", 0.7, 0.2, 0.1, league="CL"), fx("E0-9", 0.7, 0.2, 0.1, kickoff="2026-11-20 11:00")])
    assert place_bets(path, frame, odds("CL-1", "E0-9"), NOW) == 0
    assert read_ledger(path).empty


def test_a_win_pays_stake_times_price_and_a_loss_pays_nothing(tmp_path):
    path = tmp_path / "s.csv"
    place_bets(path, pd.DataFrame([fx("E0-1", 0.6, 0.2, 0.2), fx("E0-2", 0.1, 0.2, 0.7, "2026-11-22")]), odds("E0-1", "E0-2"), NOW)
    graded = pd.DataFrame({"match_id": ["E0-1", "E0-2"], "outcome": [0.0, 1.0]})
    assert settle_bets(path, graded) == 2
    ledger = read_ledger(path)
    assert list(ledger["payout"]) == [10.0, 0.0]
    assert cash(ledger) == pytest.approx(START - 5.0 - 4.75 + 10.0)
    assert settle_bets(path, graded) == 0  # settled once


def test_summary_shows_bankroll_record_curve_and_open_bets(tmp_path):
    path = tmp_path / "s.csv"
    place_bets(path, pd.DataFrame([fx("E0-1", 0.6, 0.2, 0.2), fx("E0-2", 0.1, 0.2, 0.7, "2026-11-22"), fx("E0-3", 0.7, 0.2, 0.1, "2026-11-23")]), odds("E0-1", "E0-2", "E0-3"), NOW)
    settle_bets(path, pd.DataFrame({"match_id": ["E0-1", "E0-2"], "outcome": [0.0, 1.0]}))
    s = summarize(read_ledger(path))
    assert s["bets"] == 3 and s["open"] == 1 and s["won"] == 1 and s["lost"] == 1
    assert s["profit"] == pytest.approx(5.0 - 4.75)
    assert s["bankroll"] == pytest.approx(100.25)  # open stakes count at cost
    assert [p["bankroll"] for p in s["curve"]] == [pytest.approx(105.0), pytest.approx(100.25)]
    assert s["open_stake"] == pytest.approx(4.51)
    assert s["recent"][0]["status"] == "open" and s["since"] == "2026-11-20"
    assert s["roi"] == pytest.approx(0.25 / 9.75)


def test_an_empty_ledger_summarizes_to_a_fresh_bankroll(tmp_path):
    s = summarize(read_ledger(tmp_path / "none.csv"))
    assert s["bankroll"] == 100.0 and s["bets"] == 0 and s["curve"] == [] and s["roi"] is None


def _ledger(tmp_path, *bets):
    path = tmp_path / "live.csv"
    place_bets(path, pd.DataFrame([fx(m, 0.6, 0.25, 0.15) for m in bets]), odds(*bets), NOW)
    return read_ledger(path)


def _m(mid, state, hg=None, ag=None):
    return {"match_id": mid, "state": state, "home_goals": hg, "away_goals": ag}


def test_live_bankroll_values_open_bets_by_the_score_right_now(tmp_path):
    from footpred.pipeline.stakes import live_bankroll
    led = _ledger(tmp_path, "E0-1", "E0-2", "E0-3", "E0-4")  # stakes 5.0, 4.75, 4.51, 4.29 at price 2.0 on a home win
    out = live_bankroll(led, [_m("E0-1", "live", 1, 0), _m("E0-2", "live", 0, 1), _m("E0-3", "finished", 2, 0), _m("E0-4", "upcoming")])
    by = {b["match_id"]: b for b in out["bets"]}
    assert [by[k]["status"] for k in ("E0-1", "E0-2", "E0-3", "E0-4")] == ["winning", "losing", "won", "open"]
    assert by["E0-1"]["profit"] == 5.0 and by["E0-2"]["profit"] == -4.75 and by["E0-4"]["profit"] is None
    assert out["bankroll"] == 100.0  # settled view: cash plus stakes still out
    cash_now = cash(led)
    assert out["if_now"] == round(cash_now + 10.0 + 0.0 + 9.02 + 4.29, 2)
    assert out["profit_now"] == round(out["if_now"] - START, 2)


def test_live_bankroll_ignores_settled_bets_and_missing_matches(tmp_path):
    from footpred.pipeline.stakes import live_bankroll
    path = tmp_path / "live.csv"
    place_bets(path, pd.DataFrame([fx("E0-1", 0.6, 0.25, 0.15), fx("E0-2", 0.6, 0.25, 0.15)]), odds("E0-1", "E0-2"), NOW)
    settle_bets(path, pd.DataFrame({"match_id": ["E0-1"], "outcome": [0]}))
    out = live_bankroll(read_ledger(path), [_m("E0-1", "finished", 1, 0)])
    assert [b["match_id"] for b in out["bets"]] == ["E0-2"] and out["bets"][0]["status"] == "open"
    assert live_bankroll(pd.DataFrame(columns=read_ledger(tmp_path / "none.csv").columns), [])["if_now"] == START
