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


def place_bets(path: Path, fixtures: pd.DataFrame, odds: dict, placed_at: pd.Timestamp) -> int:
    """Stake on each confident league pick that has a Bet365 price and has not kicked off; returns how many were placed.

    A match already in the ledger keeps its first bet and price. Bets go in kickoff order, each sized from the cash left.
    """
    ledger = read_ledger(path)
    have = set(ledger["match_id"])
    balance = cash(ledger)
    new = []
    frame = fixtures.sort_values(["kickoff_date", "match_id"]) if len(fixtures) else fixtures
    for r in frame.to_dict("records"):
        if r["match_id"] in have or r["league"] == "CL" or pd.Timestamp(r["kickoff_date"]) <= pd.Timestamp(placed_at):
            continue
        probs = [float(r["p_home"]), float(r["p_draw"]), float(r["p_away"])]
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


def summarize(ledger: pd.DataFrame, recent: int = 30) -> dict:
    """What the page shows: money now, record, the line of profit over time and the latest bets."""
    if ledger.empty:
        return {"start": START, "cash": START, "bankroll": START, "profit": 0.0, "roi": None, "bets": 0, "open": 0,
                "won": 0, "lost": 0, "staked": 0.0, "since": None, "curve": [], "recent": [],
                "stake_fraction": STAKE_FRACTION, "confident_at": CONFIDENT_AT}
    settled = ledger[ledger["outcome"].notna()].copy()
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
        status = "open" if pd.isna(r["outcome"]) else ("won" if r["payout"] > 0 else "lost")
        bets.append({"match_id": r["match_id"], "league": r["league"], "home": r["home"], "away": r["away"],
                     "date": r["_k"].strftime("%Y-%m-%d"), "pick": r["pick"], "odds": float(r["odds"]), "stake": float(r["stake"]),
                     "status": status, "profit": None if status == "open" else round(float(r["payout"]) - float(r["stake"]), 2)})
    return {"start": START, "cash": money, "bankroll": round(money + float(open_["stake"].sum()), 2),
            "profit": round(running - START, 2), "roi": (float(settled["profit"].sum()) / staked) if staked else None,
            "bets": int(len(ledger)), "open": int(len(open_)), "won": won, "lost": int(len(settled)) - won,
            "staked": round(staked, 2), "since": pd.to_datetime(ledger["placed_at"], format="mixed").min().strftime("%Y-%m-%d"),
            "curve": curve, "recent": bets, "stake_fraction": STAKE_FRACTION, "confident_at": CONFIDENT_AT}
