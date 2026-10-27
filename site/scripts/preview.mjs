// CI-only: drives the built site in a real browser, checks the filters work, and saves screenshots.
import { createRequire } from "node:module";
import { mkdirSync } from "node:fs";
const { chromium } = createRequire("/tmp/pw/")("playwright");

const url = process.env.BASE_URL ?? "http://localhost:4321";
const out = process.env.OUT_DIR ?? "shots";
mkdirSync(out, { recursive: true });

const problems = [];
const check = (ok, msg) => { if (!ok) { problems.push(msg); console.log("FAIL", msg); } else console.log("ok  ", msg); };

const browser = await chromium.launch();
async function open(path, { width, height, dark = false }) {
  const ctx = await browser.newContext({ viewport: { width, height }, colorScheme: dark ? "dark" : "light" });
  const page = await ctx.newPage();
  page.on("pageerror", (e) => problems.push(`pageerror ${path}: ${e.message}`));
  page.on("console", (m) => { if (m.type() === "error") problems.push(`console error ${path}: ${m.text()}`); });
  await page.goto(url + path, { waitUntil: "networkidle" });
  return page;
}
const shot = (page, name) => page.screenshot({ path: `${out}/${name}.png`, fullPage: true });
const visible = (page) => page.locator("li.item:not([hidden])").count();

// Fixtures: desktop and mobile
let p = await open("/", { width: 1280, height: 900 });
await shot(p, "fixtures-desktop");
check((await visible(p)) === 10, "all 10 matches shown by default");
check((await p.textContent("#count")).trim() === "10 matches", "count reads 10 matches");

await p.click('button[data-league="I1"]');
check((await visible(p)) === 4, "Serie A chip leaves 4 matches");
check((await p.textContent("#count")).trim() === "4 of 10 matches", "count reads 4 of 10");
check(p.url().includes("league=I1"), "filter is saved in the URL");
await p.fill("#q", "inter");
check((await visible(p)) === 1, "team search 'inter' leaves 1 match");
await p.fill("#q", "zzzz");
check(await p.locator("#empty").isVisible(), "empty state shows when nothing matches");
await p.click("#empty-clear");
check((await visible(p)) === 10, "clearing restores all matches");
await p.check("#confident");
const n = await visible(p);
check(n > 0 && n < 10, `confident-only narrows the list (${n})`);
await p.selectOption("#sort", "close");
check((await p.locator("li.day").count()) === 0, "no date headings when sorted by closeness");
await p.close();

// Deep link restores filters
p = await open("/?league=D1&sort=confidence", { width: 390, height: 844 });
check((await visible(p)) === 2, "deep link restores the Bundesliga filter");
await shot(p, "fixtures-filtered-mobile");
await p.close();

p = await open("/", { width: 390, height: 844 });
await shot(p, "fixtures-mobile");
const overflow = await p.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1);
check(!overflow, "no horizontal scroll on a phone");
await p.close();

p = await open("/", { width: 1280, height: 900, dark: true });
await shot(p, "fixtures-dark-desktop");
await p.close();

// Other pages
p = await open("/match/i1-20261028-venezia-inter/", { width: 1000, height: 900 });
await shot(p, "match-desktop");
check(await p.getByText("Both teams score").isVisible(), "match page shows goal meters");
await p.close();

p = await open("/teams/", { width: 390, height: 844 });
await shot(p, "teams-mobile");
const before = await p.locator(".team-list li:not([hidden])").count();
await p.fill("#tq", "bayern");
check((await p.locator(".team-list li:not([hidden])").count()) === 1, `team search narrows ${before} teams to 1`);
await p.close();

p = await open("/track-record/", { width: 1000, height: 900 });
await shot(p, "track-desktop");
await p.close();
p = await open("/track-record/", { width: 390, height: 844, dark: true });
await shot(p, "track-dark-mobile");
await p.close();

await browser.close();
if (problems.length) { console.log(problems.join("\n")); process.exit(1); }
