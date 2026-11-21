import test from "node:test";
import assert from "node:assert/strict";
import { liveBankroll, statusWord } from "../src/lib/bankroll.mjs";

const NOW = Date.parse("2026-11-21T20:30:00Z");
const feed = (bets, extra = {}) => ({ updated_at: "2026-11-21T20:28:00Z", matches: [], bankroll: { if_now: 103.5, profit_now: 3.5, bets, ...extra } });
const bet = (id, status, profit = null) => ({ match_id: id, status, home_goals: 1, away_goals: 0, profit });

test("a fresh feed with a scored bet gives the if-ended-now balance and the bets by match", () => {
  const out = liveBankroll(feed([bet("a", "winning", 4), bet("b", "open")]), NOW);
  assert.equal(out.ifNow, 103.5);
  assert.equal(out.profitNow, 3.5);
  assert.equal(out.bets.get("a").status, "winning");
  assert.equal(out.playing, true);
});

test("a feed older than half an hour, or with nothing scored, or malformed, is ignored", () => {
  assert.equal(liveBankroll({ ...feed([bet("a", "winning")]), updated_at: "2026-11-21T19:00:00Z" }, NOW), null);
  assert.equal(liveBankroll(feed([bet("a", "open")]), NOW), null);
  assert.equal(liveBankroll(feed([bet("a", "winning")], { if_now: "x" }), NOW), null);
  assert.equal(liveBankroll({ updated_at: "2026-11-21T20:28:00Z", matches: [] }, NOW), null);
  assert.equal(liveBankroll(null, NOW), null);
});

test("finished matches awaiting the daily run are scored but not counted as playing", () => {
  const out = liveBankroll(feed([bet("a", "won", 4)]), NOW);
  assert.equal(out.playing, false);
  assert.equal(statusWord("losing"), "Losing now");
});
