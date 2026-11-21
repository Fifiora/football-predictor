import { readFileSync } from "node:fs";
import { join } from "node:path";

const dir = process.env.SITE_DATA_DIR ?? join(process.cwd(), "data");
const load = (name) => JSON.parse(readFileSync(join(dir, name), "utf8"));

export const fixtures = load("fixtures.json");
export const trackRecord = load("track_record.json");
// Optional: the paper bankroll file only exists once the refresh has written it.
let sim = null;
try { sim = load("simulation.json"); } catch (e) { sim = null; }
export const simulation = sim;
export const teams = load("teams.json");
export const meta = load("meta.json");
export const now = process.env.SITE_NOW ? new Date(process.env.SITE_NOW) : new Date();

// Where the browser reads live scores from (the branch the live workflow publishes to). Empty turns the feature off.
export const liveUrl = process.env.LIVE_URL ?? "";
