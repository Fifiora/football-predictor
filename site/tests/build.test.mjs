// Needs `npm install` (astro). Runs a real build against the sample JSON in tests/fixtures.
import test from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const data = join(root, "tests", "fixtures");

function build(outDir, now) {
  rmSync(join(root, outDir), { recursive: true, force: true });
  execFileSync("npx", ["astro", "build", "--outDir", outDir], {
    cwd: root,
    env: { ...process.env, SITE_DATA_DIR: data, SITE_NOW: now },
    stdio: "pipe",
  });
  return join(root, outDir);
}
const read = (dist, p) => readFileSync(join(dist, p), "utf8");

test("build succeeds and emits all page types", () => {
  const dist = build("dist-test-fresh", "2025-10-27T12:00:00Z");
  for (const p of ["index.html", "teams/index.html", "track-record/index.html", "about/index.html"]) {
    assert.ok(existsSync(join(dist, p)), p);
  }
  assert.ok(readdirSync(join(dist, "match")).length >= 1);
  assert.ok(readdirSync(join(dist, "team")).length >= 1);
  assert.match(read(dist, "index.html"), /Last updated/);
});

test("match card shows three probabilities and at most three drivers", () => {
  const dist = build("dist-test-fresh", "2025-10-27T12:00:00Z");
  const html = read(dist, "index.html");
  assert.match(html, /50%/);
  assert.match(html, /30%/);
  assert.match(html, /20%/);
  const card = html.split('data-match="E0-20251030-Arsenal-Chelsea"')[1].split("</article>")[0];
  assert.ok((card.match(/data-driver/g) ?? []).length <= 3);
});

test("staleness banner appears after 48 hours and is absent when fresh", () => {
  const fresh = read(build("dist-test-fresh", "2025-10-27T12:00:00Z"), "index.html");
  assert.doesNotMatch(fresh, /data-stale-banner="shown"/);
  const stale = read(build("dist-test-stale", "2025-10-30T12:00:00Z"), "index.html");
  assert.match(stale, /data-stale-banner="shown"/);
});

test("about page says probabilities, not certainties, are the product", () => {
  const about = read(build("dist-test-fresh", "2025-10-27T12:00:00Z"), "about/index.html");
  assert.match(about, /Football is noisy, so probabilities, not certainties, are the product/);
});

test("fixtures page has filters, league chips with counts, and date headings", () => {
  const html = read(build("dist-test-fresh", "2025-10-27T12:00:00Z"), "index.html");
  assert.match(html, /id="filters"/);
  assert.match(html, /data-league="E0"/);
  assert.match(html, /data-range="3d"/);
  assert.match(html, /id="confident"/);
  assert.match(html, /Thu 30 Oct/);
  assert.match(html, /Why this pick/);
  assert.match(html, /Most likely: <b[^>]*>Arsenal win<\/b>/);
  assert.match(html, /aria-current="page"/);
});

test("Champions League fixtures get their own chip, a 90-minute note and an experimental tag", () => {
  const html = read(build("dist-test-fresh", "2025-10-27T12:00:00Z"), "index.html");
  assert.match(html, /data-league="CL"/);
  assert.match(html, /Champions League/);
  assert.match(html, /90-minute result/);
  assert.match(html, /Experimental/);
  const league = html.split('data-match="E0-20251030-Arsenal-Chelsea"')[1].split("</article>")[0];
  assert.doesNotMatch(league, /90-minute result|Experimental/);  // league cards stay as they were
});

test("teams page lists rated teams and links to team pages", () => {
  const html = read(build("dist-test-fresh", "2025-10-27T12:00:00Z"), "teams/index.html");
  assert.match(html, /id="tq"/);
  assert.match(html, /data-name="Arsenal"/);
});

test("match page shows goal meters and reason bars", () => {
  const dist = build("dist-test-fresh", "2025-10-27T12:00:00Z");
  const id = readdirSync(join(dist, "match")).find((n) => n.startsWith("e0-") && n.includes("arsenal"));
  const html = read(dist, `match/${id}/index.html`);
  assert.match(html, /Both teams score/);
  assert.match(html, /Most likely scores/);
  assert.match(html, /What the model weighed/);
});

test("track record shows tiles and a calibration chart once matches are graded", () => {
  const html = read(build("dist-test-fresh", "2025-10-27T12:00:00Z"), "track-record/index.html");
  assert.match(html, /Right result/);
  assert.match(html, /<svg[^>]*role="img"/);
  assert.match(html, /Back-test on past seasons/);
});

test("cards show what the bookies say, labelled as reference only, and the scoreboard marks right and wrong", () => {
  const html = read(build("dist-test-fresh", "2025-10-27T12:00:00Z"), "index.html");
  const card = html.split('data-match="E0-20251030-Arsenal-Chelsea"')[1].split("</article>")[0];
  assert.match(card, /What the bookies say/);
  assert.match(card, /For reference only/);
  assert.match(card, /47%/);
  const other = html.split('data-match="E0-20251031-Fulham-Everton"')[1]?.split("</article>")[0] ?? "";
  assert.doesNotMatch(other, /What the bookies say/);
  assert.match(html, /data-scoreboard/);
  assert.match(html, /1<\/b> of 2 recent picks right/);
  assert.match(html, /Fulham v Everton: right/);
  assert.match(html, /Getafe v Betis: wrong/);
});

test("track record compares the model with the bookmakers honestly", () => {
  const html = read(build("dist-test-fresh", "2025-10-27T12:00:00Z"), "track-record/index.html");
  assert.match(html, /data-benchmark/);
  assert.match(html, /How do we compare with the bookmakers/);
  assert.match(html, /about 1\.4 points ahead/);
  assert.match(html, /We never use odds in our predictions/);
});

test("the bookmaker line keeps its own styling", () => {
  const dist = build("dist-test-fresh", "2025-10-27T12:00:00Z");
  const css = readdirSync(join(dist, "_astro")).filter((n) => n.endsWith(".css")).map((n) => read(dist, `_astro/${n}`)).join("");
  assert.match(css, /\.bookies/);
});

test("entrance animation is switched off after the first render so filtering does not replay it", () => {
  const dist = build("dist-test-fresh", "2025-10-27T12:00:00Z");
  const assets = readdirSync(join(dist, "_astro")).map((n) => read(dist, `_astro/${n}`)).join("");
  assert.match(read(dist, "index.html") + assets, /data-ready/);
});

test("cards carry the kickoff instant with a UTC fallback that the page localizes", () => {
  const html = read(build("dist-test-fresh", "2025-10-27T12:00:00Z"), "index.html");
  const card = html.split('data-match="E0-20251030-Arsenal-Chelsea"')[1].split("</article>")[0];
  assert.match(card, /data-kickoff="2025-10-30T19:45:00Z"/);
  assert.match(card, />19:45 UTC</);
});

test("the localizing script only targets time elements, never the card wrappers", () => {
  const dist = build("dist-test-fresh", "2025-10-27T12:00:00Z");
  const assets = readdirSync(join(dist, "_astro")).filter((n) => n.endsWith(".js")).map((n) => read(dist, `_astro/${n}`)).join("") + read(dist, "index.html");
  assert.match(assets, /time\[data-kickoff\]/);
  assert.doesNotMatch(assets, /querySelectorAll\(["']\[data-kickoff\]["']\)/);
});
