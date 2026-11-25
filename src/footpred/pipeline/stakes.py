"""A pretend bankroll that backs the model's confident picks with Bet365 prices, to show what following them would have done.

Paper money only: nothing here places a bet or feeds back into the model. Every bet is written once, before kickoff, with the
price seen at that moment, and is settled when the result is known, so the record cannot be reshaped afterwards.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

START = 100.0
STAKE_FRACTION = 0.05   # of the cash on hand when the bet is placed
MIN_STAKE = 1.0
CONFIDENT_AT = 0.55     # same cut-off as the "Confident" label on the match cards
COLUMNS = ["match_id", "league", "home", "away", "kickoff_date", "pick", "p_pick", "odds", "stake", "placed_at", "outcome", "payout"]
_LETTERS = ("H", "D", "A")


def read_ledger(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path)


def _write(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def cash(ledger: pd.DataFrame) -> float:
    """Money on hand: the start, less every stake, plus every payout already settled."""
    if ledger.empty:
        return START
    return round(START - float(ledger["stake"].sum()) + float(ledger["payout"].fillna(0).sum()), 2)


def place_bets(path: Path, fixtures: pd.DataFrame, odds: dict, placed_at: pd.Timestamp, logged: pd.DataFrame | None = None) -> int:
    """Stake on each confident league pick that has a Bet365 price and has not kicked off; returns how many were placed.

    A match already in the ledger keeps its first bet and price. Bets go in kickoff order, each sized from the cash left.
    With ``logged`` (the prediction log) the bet follows the prediction that was saved before kickoff, the one the track record
    grades, and a match that was never logged is not staked.
    """
    ledger = read_ledger(path)
    have = set(ledger["match_id"])
    balance = cash(ledger)
    new = []
    kept = None if logged is None else logged.drop_duplicates("match_id").set_index("match_id")
    frame = fixtures.sort_values(["kickoff_date", "match_id"]) if len(fixtures) else fixtures
    for r in frame.to_dict("records"):
        if r["match_id"] in have or r["league"] == "CL" or pd.Timestamp(r["kickoff_date"]) <= pd.Timestamp(placed_at):
            continue
        src = r
        if kept is not None:
            if r["match_id"] not in kept.index:
                continue
            src = kept.loc[r["match_id"]]
        probs = [float(src["p_home"]), float(src["p_draw"]), float(src["p_away"])]
        idx = int(np.argmax(probs))
        if probs[idx] < CONFIDENT_AT:
            continue
        prices = (odds.get(r["match_id"]) or {}).get("bet365_price")
        if not prices:
            continue
        stake = round(balance * STAKE_FRACTION, 2)
        if stake < MIN_STAKE or stake > balance:
            continue  # nearly out of money: a careful person stops
        balance = round(balance - stake, 2)
        new.append({"match_id": r["match_id"], "league": r["league"], "home": r["home"], "away": r["away"],
                    "kickoff_date": r["kickoff_date"], "pick": _LETTERS[idx], "p_pick": round(probs[idx], 4),
                    "odds": float(prices[idx]), "stake": stake, "placed_at": pd.Timestamp(placed_at),
                    "outcome": np.nan, "payout": np.nan})
    if new:
        add = pd.DataFrame(new, columns=COLUMNS)
        _write(add if ledger.empty else pd.concat([ledger[COLUMNS], add], ignore_index=True), path)
    return len(new)


def settle_bets(path: Path, graded: pd.DataFrame) -> int:
    """Pay out or close every open bet whose match has a result (``graded`` has ``match_id`` and ``outcome`` 0/1/2)."""
    ledger = read_ledger(path)
    if ledger.empty or graded is None or not len(graded):
        return 0
    done = graded.dropna(subset=["outcome"]).drop_duplicates("match_id").set_index("match_id")["outcome"]
    open_rows = ledger["outcome"].isna() & ledger["match_id"].isin(done.index)
    if not open_rows.any():
        return 0
    won_codes = ledger.loc[open_rows, "match_id"].map(done).astype(int)
    pick_codes = ledger.loc[open_rows, "pick"].map({"H": 0, "D": 1, "A": 2})
    won = won_codes == pick_codes
    ledger.loc[open_rows, "outcome"] = won_codes.astype(float)
    ledger.loc[open_rows, "payout"] = np.where(won, (ledger.loc[open_rows, "stake"] * ledger.loc[open_rows, "odds"]).round(2), 0.0)
    _write(ledger, path)
    return int(open_rows.sum())


VOID = 3.0  # outcome code for a bet whose match never got a result: the stake is returned


def void_stale(path: Path, now: pd.Timestamp, after_days: int = 10) -> int:
    """Return the stake on any open bet whose match has had no result ``after_days`` after kickoff (postponed or moved)."""
    ledger = read_ledger(path)
    if ledger.empty:
        return 0
    kick = pd.to_datetime(ledger["kickoff_date"], format="mixed")
    stale = ledger["outcome"].isna() & (kick < pd.Timestamp(now) - pd.Timedelta(days=after_days))
    if not stale.any():
        return 0
    ledger.loc[stale, "outcome"] = VOID
    ledger.loc[stale, "payout"] = ledger.loc[stale, "stake"]
    _write(ledger, path)
    return int(stale.sum())


def summarize(ledger: pd.DataFrame, recent: int = 30) -> dict:
    """What the page shows: money now, record, the line of profit over time and the latest bets."""
    if ledger.empty:
        return {"start": START, "cash": START, "bankroll": START, "profit": 0.0, "roi": None, "bets": 0, "open": 0, "open_stake": 0.0,
                "won": 0, "lost": 0, "staked": 0.0, "since": None, "curve": [], "recent": [],
                "stake_fraction": STAKE_FRACTION, "confident_at": CONFIDENT_AT}
    settled = ledger[ledger["outcome"].notna() & (ledger["outcome"] != VOID)].copy()
    open_ = ledger[ledger["outcome"].isna()]
    settled["profit"] = settled["payout"] - settled["stake"]
    settled = settled.assign(_k=pd.to_datetime(settled["kickoff_date"], format="mixed")).sort_values(["_k", "match_id"])
    running, curve = START, []
    for r in settled.to_dict("records"):
        running = round(running + float(r["profit"]), 2)
        curve.append({"date": r["_k"].strftime("%Y-%m-%d"), "bankroll": running})
    money = cash(ledger)
    staked = float(settled["stake"].sum())
    won = int((settled["payout"] > 0).sum())
    latest = ledger.assign(_k=pd.to_datetime(ledger["kickoff_date"], format="mixed")).sort_values(["_k", "match_id"], ascending=False).head(recent)
    bets = []
    for r in latest.to_dict("records"):
        status = "open" if pd.isna(r["outcome"]) else "void" if r["outcome"] == VOID else ("won" if r["payout"] > 0 else "lost")
        bets.append({"match_id": r["match_id"], "league": r["league"], "home": r["home"], "away": r["away"],
                     "date": r["_k"].strftime("%Y-%m-%d"), "pick": r["pick"], "odds": float(r["odds"]), "stake": float(r["stake"]),
                     "status": status, "profit": None if status == "open" else round(float(r["payout"]) - float(r["stake"]), 2)})
    return {"start": START, "cash": money, "bankroll": round(money + float(open_["stake"].sum()), 2),
            "profit": round(running - START, 2), "roi": (float(settled["profit"].sum()) / staked) if staked else None,
            "bets": int(len(ledger)), "open": int(len(open_)), "open_stake": round(float(open_["stake"].sum()), 2), "won": won, "lost": int(len(settled)) - won,
            "staked": round(staked, 2), "since": pd.to_datetime(ledger["placed_at"], format="mixed").min().strftime("%Y-%m-%d"),
            "curve": curve, "recent": bets, "stake_fraction": STAKE_FRACTION, "confident_at": CONFIDENT_AT}


def live_bankroll(ledger: pd.DataFrame, matches: list[dict]) -> dict:
    """The bankroll while matches are on: each open bet is valued by the score right now.

    ``bankroll`` is the settled view (cash plus stakes still out); ``if_now`` is what the money would be if every match
    ended at its current score. Bets already settled in the ledger are left out.
    """
    by_id = {m["match_id"]: m for m in matches or []}
    open_ = ledger[ledger["outcome"].isna()] if len(ledger) else ledger
    bets, in_play = [], 0.0
    if len(ledger):  # bets the daily run settled after the page was built, so the page can still show their result
        for r in ledger[ledger["outcome"].notna() & (ledger["outcome"] != VOID) & ledger["match_id"].isin(by_id)].to_dict("records"):
            m = by_id[r["match_id"]]
            if m.get("home_goals") is None or m.get("away_goals") is None:
                continue
            bets.append({"match_id": r["match_id"], "status": "won" if r["payout"] > 0 else "lost", "home_goals": int(m["home_goals"]),
                         "away_goals": int(m["away_goals"]), "profit": round(float(r["payout"]) - float(r["stake"]), 2)})
    for r in open_.to_dict("records"):
        m = by_id.get(r["match_id"])
        stake, win = float(r["stake"]), round(float(r["stake"]) * float(r["odds"]), 2)
        hg, ag = (m or {}).get("home_goals"), (m or {}).get("away_goals")
        state = (m or {}).get("state")
        if m is None or state == "upcoming" or hg is None or ag is None:
            bets.append({"match_id": r["match_id"], "status": "open", "home_goals": None, "away_goals": None, "profit": None})
            in_play += stake
            continue
        right = _LETTERS[0 if hg > ag else 2 if ag > hg else 1] == r["pick"]
        status = ("won" if right else "lost") if state == "finished" else ("winning" if right else "losing")
        in_play += win if right else 0.0
        bets.append({"match_id": r["match_id"], "status": status, "home_goals": int(hg), "away_goals": int(ag),
                     "profit": round(win - stake, 2) if right else round(-stake, 2)})
    money = cash(ledger)
    out_stakes = float(open_["stake"].sum()) if len(open_) else 0.0
    if_now = round(money + in_play, 2)
    return {"cash": money, "bankroll": round(money + out_stakes, 2), "if_now": if_now,
            "profit_now": round(if_now - START, 2), "bets": bets}
