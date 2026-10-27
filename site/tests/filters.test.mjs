import test from "node:test";
import assert from "node:assert/strict";
import {
  applyFilters, confidenceLabel, defaultState, leagueCounts, parseState, pick, serializeState,
} from "../src/lib/filters.mjs";

const mk = (id, league, date, home, away, p) => ({
  match_id: id, league, date, home, away, p_home: p[0], p_draw: p[1], p_away: p[2],
});
const F = [
  mk("a", "E0", "2026-10-27", "Arsenal", "Chelsea", [0.62, 0.22, 0.16]),
  mk("b", "I1", "2026-10-28", "Inter", "Venezia", [0.8, 0.13, 0.07]),
  mk("c", "I1", "2026-10-28", "Parma", "Udinese", [0.36, 0.29, 0.35]),
  mk("d", "D1", "2026-11-03", "Köln", "Mainz", [0.4, 0.3, 0.3]),
];
const TODAY = "2026-10-27";

test("pick names the most likely side and its team", () => {
  assert.deepEqual(pick(F[1]), { side: "home", team: "Inter", p: 0.8 });
  assert.equal(pick(mk("x", "E0", "2026-10-27", "A", "B", [0.2, 0.3, 0.5])).team, "B");
  assert.equal(pick(mk("x", "E0", "2026-10-27", "A", "B", [0.3, 0.4, 0.3])).team, "Draw");
});

test("confidence label has three levels", () => {
  assert.equal(confidenceLabel(F[1]), "Confident");
  assert.equal(confidenceLabel(mk("x", "E0", "d", "A", "B", [0.48, 0.27, 0.25])), "Lean");
  assert.equal(confidenceLabel(F[2]), "Toss-up");
});

test("default state shows everything in date order", () => {
  const out = applyFilters(F, defaultState(), TODAY);
  assert.deepEqual(out.map((f) => f.match_id), ["a", "b", "c", "d"]);
});

test("league filter keeps only chosen leagues", () => {
  const out = applyFilters(F, { ...defaultState(), leagues: ["I1", "D1"] }, TODAY);
  assert.deepEqual(out.map((f) => f.match_id), ["b", "c", "d"]);
});

test("date ranges are relative to today", () => {
  const ids = (range) => applyFilters(F, { ...defaultState(), range }, TODAY).map((f) => f.match_id);
  assert.deepEqual(ids("today"), ["a"]);
  assert.deepEqual(ids("3d"), ["a", "b", "c"]);
  assert.deepEqual(ids("week"), ["a", "b", "c"]);
  assert.deepEqual(ids("all"), ["a", "b", "c", "d"]);
});

test("team search ignores case and accents", () => {
  const ids = (q) => applyFilters(F, { ...defaultState(), q }, TODAY).map((f) => f.match_id);
  assert.deepEqual(ids("koln"), ["d"]);
  assert.deepEqual(ids("CHEL"), ["a"]);
  assert.deepEqual(ids("  "), ["a", "b", "c", "d"]);
});

test("confident-only keeps only Confident picks", () => {
  const out = applyFilters(F, { ...defaultState(), confidentOnly: true }, TODAY);
  assert.deepEqual(out.map((f) => f.match_id), ["a", "b"]);
});

test("sorting by confidence and by closeness", () => {
  const conf = applyFilters(F, { ...defaultState(), sort: "confidence" }, TODAY).map((f) => f.match_id);
  assert.deepEqual(conf, ["b", "a", "d", "c"]);
  const close = applyFilters(F, { ...defaultState(), sort: "close" }, TODAY).map((f) => f.match_id);
  assert.deepEqual(close, ["c", "d", "a", "b"]);
});

test("league counts feed the chips", () => {
  assert.deepEqual(leagueCounts(F), { E0: 1, I1: 2, D1: 1 });
});

test("state round-trips through the URL and ignores junk", () => {
  const s = { leagues: ["E0", "I1"], range: "3d", q: "inter", confidentOnly: true, sort: "close" };
  assert.deepEqual(parseState(serializeState(s)), s);
  assert.equal(serializeState(defaultState()), "");
  assert.deepEqual(parseState("?range=bogus&sort=nope&league=ZZ,E0"), { ...defaultState(), leagues: ["E0", "ZZ"] });
});
