export const STALE_AFTER_HOURS = 48;

export function pct(p) {
  return `${Math.round(p * 100)}%`;
}

export function slug(text) {
  return String(text)
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

export function isStale(updatedAt, now = new Date(), hours = STALE_AFTER_HOURS) {
  const updated = new Date(updatedAt).getTime();
  if (Number.isNaN(updated)) return true;
  return new Date(now).getTime() - updated > hours * 3600 * 1000;
}

export function topDrivers(drivers, max = 3) {
  return [...(drivers ?? [])].sort((a, b) => b.strength - a.strength).slice(0, max);
}

export function formatUpdated(updatedAt) {
  const d = new Date(updatedAt);
  if (Number.isNaN(d.getTime())) return "unknown";
  return d.toISOString().replace("T", " ").slice(0, 16) + " UTC";
}

const LEAGUES = {
  E0: "Premier League",
  SP1: "La Liga",
  I1: "Serie A",
  D1: "Bundesliga",
  F1: "Ligue 1",
  CL: "Champions League",
};

export function leagueName(code) {
  return LEAGUES[code] ?? code;
}

export const EXPERIMENTAL_UNTIL = 30;

/** Champions League predictions are labelled experimental until enough of them have been graded. */
export function isExperimental(trackRecord, league) {
  if (league !== "CL") return false;
  return (trackRecord?.by_league?.CL?.n ?? 0) < EXPERIMENTAL_UNTIL;
}

const WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

export function formatDay(isoDate) {
  const d = new Date(`${isoDate}T00:00:00Z`);
  if (Number.isNaN(d.getTime())) return String(isoDate);
  return `${WEEKDAYS[d.getUTCDay()]} ${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]}`;
}

/** Latest graded picks for the scoreboard strip: hits among the newest ``limit``. */
export function scoreboard(recent, limit = 10) {
  const items = (recent ?? []).slice(0, limit);
  return { items, hits: items.filter((r) => r.correct).length, total: items.length };
}

/** What the bookmakers say, shown beside (never inside) the model's chances. */
export function bookieView(fixture) {
  const b = fixture.bookies;
  if (!b) return null;
  if (b.average) return { label: "Average of bookmakers", ...b.average };
  if (b.bet365) return { label: "Bet365", ...b.bet365 };
  return null;
}

/** One honest sentence about how our right-result rate compares with the bookmakers. ``gap`` is bookmakers minus us, in points. */
export function verdictText(gap) {
  if (Math.abs(gap) < 0.05) return "We are level with the bookmakers on picking the right result.";
  if (gap > 0) {
    return `The bookmakers are about ${gap.toFixed(1)} points ahead on picking the right result. That is expected: they know about injuries and line-ups, and we use match results only. They set the bar we are chasing.`;
  }
  return `We are about ${Math.abs(gap).toFixed(1)} points ahead of the bookmakers on picking the right result, on these matches.`;
}

const toDate = (iso) => {
  if (typeof iso !== "string") return null;
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? null : d;
};

/** A kickoff instant as the visitor sees it: local calendar day, clock time and zone name. */
export function localParts(iso, timeZone) {
  const d = toDate(iso);
  if (!d) return null;
  const opts = timeZone ? { timeZone } : {};
  const day = new Intl.DateTimeFormat("en-CA", { ...opts, year: "numeric", month: "2-digit", day: "2-digit" }).format(d);
  const time = new Intl.DateTimeFormat(undefined, { ...opts, hour: "numeric", minute: "2-digit" }).format(d);
  const zone = new Intl.DateTimeFormat("en-US", { ...opts, timeZoneName: "short" }).formatToParts(d).find((p) => p.type === "timeZoneName")?.value ?? "";
  return { date: day, time, zone };
}

/** Static fallback shown before the page localizes it (and without JavaScript). */
export function utcTime(iso) {
  const d = toDate(iso);
  return d ? `${String(d.getUTCHours()).padStart(2, "0")}:${String(d.getUTCMinutes()).padStart(2, "0")} UTC` : "";
}
