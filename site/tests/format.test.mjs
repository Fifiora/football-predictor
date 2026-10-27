import test from "node:test";
import assert from "node:assert/strict";
import { isStale, topDrivers, pct, slug } from "../src/lib/format.mjs";

test("stale after 48 hours, fresh before", () => {
  const updated = "2025-10-27T06:00:00Z";
  assert.equal(isStale(updated, new Date("2025-10-29T05:00:00Z")), false);
  assert.equal(isStale(updated, new Date("2025-10-29T07:00:00Z")), true);
});

test("unparseable update time counts as stale", () => {
  assert.equal(isStale("nonsense"), true);
});

test("at most three drivers, strongest first", () => {
  const ds = [1, 5, 3, 4, 2].map((s) => ({ strength: s }));
  assert.deepEqual(topDrivers(ds).map((d) => d.strength), [5, 4, 3]);
  assert.deepEqual(topDrivers(undefined), []);
});

test("pct and slug", () => {
  assert.equal(pct(0.456), "46%");
  assert.equal(slug("Atlético Madrid"), "atletico-madrid");
});

import { isExperimental, leagueName } from "../src/lib/format.mjs";

test("league codes show as names, unknown codes pass through", () => {
  assert.equal(leagueName("E0"), "Premier League");
  assert.equal(leagueName("SP1"), "La Liga");
  assert.equal(leagueName("XX9"), "XX9");
  assert.equal(leagueName("CL"), "Champions League");
});

test("the Champions League is experimental until 30 of its predictions are graded", () => {
  const record = (n) => ({ by_league: { CL: { n } } });
  assert.equal(isExperimental(record(29), "CL"), true);
  assert.equal(isExperimental(record(30), "CL"), false);
  assert.equal(isExperimental(record(500), "CL"), false);
  assert.equal(isExperimental({ by_league: {} }, "CL"), true);
  assert.equal(isExperimental({}, "CL"), true);
  assert.equal(isExperimental(undefined, "CL"), true);
  assert.equal(isExperimental(record(0), "E0"), false);
});

test("day labels read like a fixture list", async () => {
  const { formatDay } = await import("../src/lib/format.mjs");
  assert.equal(formatDay("2026-10-28"), "Wed 28 Oct");
  assert.equal(formatDay("2026-11-03"), "Tue 3 Nov");
  assert.equal(formatDay("nonsense"), "nonsense");
});
