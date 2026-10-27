// Pure filter / sort / URL-state logic for the fixtures list. Runs in tests (node) and in the browser.

export const RANGES = { today: 0, "3d": 2, week: 6 };
export const SORTS = ["date", "confidence", "close"];
const CONFIDENT_AT = 0.55;
const LEAN_AT = 0.45;

export function defaultState() {
  return { leagues: [], range: "all", q: "", confidentOnly: false, sort: "date" };
}

export function pick(f) {
  const options = [
    { side: "home", team: f.home, p: f.p_home },
    { side: "draw", team: "Draw", p: f.p_draw },
    { side: "away", team: f.away, p: f.p_away },
  ];
  return options.reduce((best, o) => (o.p > best.p ? o : best));
}

export function confidenceLabel(f) {
  const { p } = pick(f);
  if (p >= CONFIDENT_AT) return "Confident";
  if (p >= LEAN_AT) return "Lean";
  return "Toss-up";
}

export const fold = (s) => String(s).normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase().trim();

function addDays(isoDate, days) {
  const d = new Date(`${isoDate}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0, 10);
}

export function applyFilters(fixtures, state, today) {
  const s = { ...defaultState(), ...state };
  const q = fold(s.q);
  const last = s.range in RANGES ? addDays(today, RANGES[s.range]) : null;
  const out = fixtures.filter((f) => {
    if (s.leagues.length && !s.leagues.includes(f.league)) return false;
    if (last && (f.date < today || f.date > last)) return false;
    if (q && !fold(f.home).includes(q) && !fold(f.away).includes(q)) return false;
    if (s.confidentOnly && confidenceLabel(f) !== "Confident") return false;
    return true;
  });
  const byDate = (a, b) => a.date.localeCompare(b.date) || a.match_id.localeCompare(b.match_id);
  const top = (f) => pick(f).p;
  const sorters = {
    date: byDate,
    confidence: (a, b) => top(b) - top(a) || byDate(a, b),
    close: (a, b) => top(a) - top(b) || byDate(a, b),
  };
  return [...out].sort(sorters[s.sort] ?? byDate);
}

export function leagueCounts(fixtures) {
  const counts = {};
  for (const f of fixtures) counts[f.league] = (counts[f.league] ?? 0) + 1;
  return counts;
}

export function serializeState(state) {
  const s = { ...defaultState(), ...state };
  const p = new URLSearchParams();
  if (s.leagues.length) p.set("league", [...s.leagues].sort().join(","));
  if (s.range !== "all") p.set("range", s.range);
  if (s.q.trim()) p.set("q", s.q.trim());
  if (s.confidentOnly) p.set("confident", "1");
  if (s.sort !== "date") p.set("sort", s.sort);
  const text = p.toString();
  return text ? `?${text}` : "";
}

export function parseState(search) {
  const p = new URLSearchParams(search);
  const d = defaultState();
  const range = p.get("range");
  const sort = p.get("sort");
  return {
    leagues: (p.get("league") ?? "").split(",").map((x) => x.trim()).filter(Boolean).sort(),
    range: range === "all" || range in RANGES ? range : d.range,
    q: p.get("q") ?? "",
    confidentOnly: p.get("confident") === "1",
    sort: SORTS.includes(sort) ? sort : d.sort,
  };
}
