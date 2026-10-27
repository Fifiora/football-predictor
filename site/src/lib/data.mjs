import { readFileSync } from "node:fs";
import { join } from "node:path";

const dir = process.env.SITE_DATA_DIR ?? join(process.cwd(), "data");
const load = (name) => JSON.parse(readFileSync(join(dir, name), "utf8"));

export const fixtures = load("fixtures.json");
export const trackRecord = load("track_record.json");
export const teams = load("teams.json");
export const meta = load("meta.json");
export const now = process.env.SITE_NOW ? new Date(process.env.SITE_NOW) : new Date();
