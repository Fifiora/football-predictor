import test from "node:test";
import assert from "node:assert/strict";
import { extraPicks, fixtureHidden, liveView, feedAgeMinutes, scoreLine } from "../src/lib/live.mjs";

const HOUR = 3600 * 1000;
const NOW = Date.parse("2026-11-01T20:30:00Z");
const m = (id, state, kickoff, extra = {}) => ({ match_id: id, league: "E0", home: "A", away: "B", kickoff, state, home_goals: 1, away_goals: 0, ...extra });
const feed = (...matches) => ({ updated_at: "2026-11-01T20:28:00Z", complete: true, matches });

test("live matches are listed first and finished ones from today follow, newest first", () => {
  const view = liveView(feed(
    m("a", "finished", "2026-11-01T14:00:00Z"),
    m("b", "live", "2026-11-01T19:45:00Z"),
    m("c", "finished", "2026-11-01T17:00:00Z"),
    m("d", "upcoming", "2026-11-01T22:00:00Z"),
  ), NOW, "UTC");
  assert.deepEqual(view.live.map((x) => x.match_id), ["b"]);
  assert.deepEqual(view.results.map((x) => x.match_id), ["c", "a"]);
  assert.deepEqual([...view.ids].sort(), ["a", "b", "c"]);
});

test("a result from the previous local day is dropped, except a late match that has just ended", () => {
  const old = m("old", "finished", "2026-10-31T19:45:00Z");
  const late = m("late", "finished", "2026-10-31T23:30:00Z");
  const view = liveView(feed(old, late), Date.parse("2026-11-01T01:30:00Z"), "UTC");
  assert.deepEqual(view.results.map((x) => x.match_id), ["late"]);
});

test("a match marked live long after it could still be playing is not shown as live", () => {
  const view = liveView(feed(m("x", "live", "2026-11-01T14:00:00Z")), NOW, "UTC");
  assert.deepEqual(view.live, []);
});

test("an unreadable feed shows nothing instead of failing", () => {
  for (const bad of [null, undefined, {}, { matches: "no" }, { matches: [null, 3] }]) {
    const view = liveView(bad, NOW, "UTC");
    assert.deepEqual([view.live, view.results, [...view.ids]], [[], [], []]);
  }
});

test("a fixture is hidden once its match is over, but not while it may still be playing", () => {
  const f = (kickoff, date = "2026-11-01") => ({ match_id: "k", date, kickoff });
  const none = new Set();
  assert.equal(fixtureHidden(f("2026-11-01T22:00:00Z"), NOW, "2026-11-01", none), false);   // not started
  assert.equal(fixtureHidden(f("2026-11-01T19:45:00Z"), NOW, "2026-11-01", none), false);   // started under 3 hours ago
  assert.equal(fixtureHidden(f("2026-11-01T17:00:00Z"), NOW, "2026-11-01", none), true);    // over
});

test("a fixture without a kickoff time is hidden once its day has passed", () => {
  const f = (date) => ({ match_id: "k", date });
  assert.equal(fixtureHidden(f("2026-10-31"), NOW, "2026-11-01", new Set()), true);
  assert.equal(fixtureHidden(f("2026-11-01"), NOW, "2026-11-01", new Set()), false);
});

test("a fixture that is in the live feed is hidden because the live section shows it", () => {
  assert.equal(fixtureHidden({ match_id: "k", date: "2026-11-01", kickoff: "2026-11-01T22:00:00Z" }, NOW, "2026-11-01", new Set(["k"])), true);
});

test("feed age in minutes, unknown when unreadable", () => {
  assert.equal(feedAgeMinutes(feed(), NOW), 2);
  assert.equal(feedAgeMinutes({ updated_at: "nope" }, NOW), null);
});

test("score line", () => {
  assert.equal(scoreLine({ home_goals: 2, away_goals: 1 }), "2 – 1");
  assert.equal(scoreLine({ home_goals: null, away_goals: null }), "–");
});

test("finished matches missing from the saved record become extra picks, and recorded or unpicked ones do not", () => {
  const f = feed(
    m("new-right", "finished", "2026-11-01T14:00:00Z", { pick: "H", pick_ok: true, p_home: 0.6, p_draw: 0.25, p_away: 0.15 }),
    m("new-wrong", "finished", "2026-11-01T15:00:00Z", { pick: "A", pick_ok: false, p_home: 0.2, p_draw: 0.3, p_away: 0.5 }),
    m("saved", "finished", "2026-11-01T16:00:00Z", { pick: "H", pick_ok: true, p_home: 0.5, p_draw: 0.3, p_away: 0.2 }),
    m("no-pick", "finished", "2026-11-01T17:00:00Z"),
    m("playing", "live", "2026-11-01T19:00:00Z", { pick: "H", pick_ok: true, p_home: 0.5, p_draw: 0.3, p_away: 0.2 }),
  );
  assert.deepEqual(extraPicks(f, new Set(["saved"])), [["E0", 0.6, 1, "new-right"], ["E0", 0.5, 0, "new-wrong"]]);
  assert.deepEqual(extraPicks(null, new Set()), []);
});

test("extra picks are ignored when the feed is more than half an hour old", () => {
  const f = feed(m("x", "finished", "2026-11-01T14:00:00Z", { pick: "H", pick_ok: true, p_home: 0.6, p_draw: 0.25, p_away: 0.15 }));
  assert.equal(extraPicks(f, new Set(), NOW).length, 1);
  assert.equal(extraPicks(f, new Set(), NOW + 40 * 60000).length, 0);
});
