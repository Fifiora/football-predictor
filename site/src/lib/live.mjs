// Pure helpers for the live scores feed and for hiding fixtures that are already over. Runs in tests (node) and in the browser.
import { localParts } from "./format.mjs";

const HOUR = 3600 * 1000;
const STILL_PLAYING_FOR = 3.5 * HOUR;  // 90 minutes plus break, stoppage and extra time
const MATCH_OVER_AFTER = 3 * HOUR;
const KEEP_RESULT_FOR = 4 * HOUR;      // a late match that ended after midnight stays visible a while

const kick = (m) => Date.parse(m.kickoff);

/** Split the feed into matches being played and today's results; ``ids`` is every match the live section will show. */
export function liveView(feed, nowMs, timeZone) {
  const matches = Array.isArray(feed?.matches) ? feed.matches.filter((m) => m && typeof m.match_id === "string") : [];
  const today = localParts(new Date(nowMs).toISOString(), timeZone)?.date;
  const live = matches
    .filter((m) => m.state === "live" && nowMs - kick(m) < STILL_PLAYING_FOR)
    .sort((a, b) => kick(a) - kick(b));
  const results = matches
    .filter((m) => {
      if (m.state !== "finished" || Number.isNaN(kick(m))) return false;
      return localParts(m.kickoff, timeZone)?.date === today || nowMs - kick(m) < KEEP_RESULT_FOR;
    })
    .sort((a, b) => kick(b) - kick(a));
  return { live, results, ids: new Set([...live, ...results].map((m) => m.match_id)) };
}

/** Whether a fixture card should be taken off the list: shown in the live section already, or the match is over. */
export function fixtureHidden(f, nowMs, today, liveIds) {
  if (liveIds?.has(f.match_id)) return true;
  const start = Date.parse(f.kickoff);
  if (Number.isNaN(start)) return f.date < today;  // no kickoff time: keep it for its own day only
  return nowMs - start > MATCH_OVER_AFTER;
}

export function feedAgeMinutes(feed, nowMs) {
  const updated = Date.parse(feed?.updated_at);
  return Number.isNaN(updated) ? null : Math.max(0, Math.round((nowMs - updated) / 60000));
}

export function scoreLine(m) {
  if (m.home_goals == null || m.away_goals == null) return "–";
  return `${m.home_goals} – ${m.away_goals}`;
}
