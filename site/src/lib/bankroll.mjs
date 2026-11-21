// Turns the live feed's bankroll block into what the Bankroll page overlays on its saved numbers. Pure: runs in node tests and the browser.
import { feedAgeMinutes } from "./live.mjs";

const MAX_AGE_MIN = 30;  // an older feed is a leftover from an earlier match day, so it is ignored
const WORD = { winning: "Winning now", losing: "Losing now", won: "Won", lost: "Lost", open: "Open" };

/** ``null`` unless the feed is fresh and at least one open bet has a score; otherwise what the page should show. */
export function liveBankroll(feed, nowMs) {
  const b = feed?.bankroll;
  const age = feedAgeMinutes(feed, nowMs);
  if (!b || age == null || age > MAX_AGE_MIN || !Array.isArray(b.bets)) return null;
  const bets = new Map();
  for (const x of b.bets) if (x && typeof x.match_id === "string" && x.status in WORD) bets.set(x.match_id, x);
  const scored = [...bets.values()].filter((x) => x.status !== "open");
  if (!scored.length || !Number.isFinite(b.if_now) || !Number.isFinite(b.profit_now)) return null;
  return { ifNow: b.if_now, profitNow: b.profit_now, bets, playing: scored.some((x) => x.status === "winning" || x.status === "losing") };
}

export const statusWord = (status) => WORD[status] ?? WORD.open;
