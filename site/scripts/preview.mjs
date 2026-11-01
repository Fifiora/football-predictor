// CI-only: drives the built site in a real browser, checks the filters work, and saves screenshots.
import { createRequire } from "node:module";
import { mkdirSync, readFileSync } from "node:fs";
const { chromium } = createRequire("/tmp/pw/")("playwright");

const url = process.env.BASE_URL ?? "http://localhost:4321";
const out = process.env.OUT_DIR ?? "shots";
mkdirSync(out, { recursive: true });

const problems = [];
const check = (ok, msg) => { if (!ok) { problems.push(msg); console.log("FAIL", msg); } else console.log("ok  ", msg); };

const LIVE_FEED = readFileSync(new URL("../tests/preview-data/live.json", import.meta.url), "utf8");
const browser = await chromium.launch();
// The page clock is fixed so the sample data (built for 27 October 2026) behaves the same whenever this runs.
async function open(path, { width, height, dark = false, at = "2026-10-27T12:00:00Z", live = "feed" }) {
  const ctx = await browser.newContext({ viewport: { width, height }, colorScheme: dark ? "dark" : "light", timezoneId: "America/Chicago" });
  await ctx.clock.setFixedTime(new Date(at));
  await ctx.route("https://live.test/**", (route) =>
    live === "feed" ? route.fulfill({ status: 200, contentType: "application/json", headers: { "access-control-allow-origin": "*" }, body: LIVE_FEED }) : route.abort());
  const page = await ctx.newPage();
  page.on("pageerror", (e) => problems.push(`pageerror ${path}: ${e.message}`));
  page.on("console", (m) => { if (m.type() === "error" && !(live !== "feed" && /live\.test|ERR_FAILED/.test(m.text()))) problems.push(`console error ${path}: ${m.text()}`); });
  await page.goto(url + path, { waitUntil: "networkidle" });
  return page;
}
const shot = (page, name) => page.screenshot({ path: `${out}/${name}.png`, fullPage: true });
const visible = (page) => page.locator("li.item:not([hidden])").count();

// Fixtures: desktop and mobile
let p = await open("/", { width: 1280, height: 900 });
await shot(p, "fixtures-desktop");
check((await visible(p)) === 11, "all 11 matches shown by default");
check((await p.textContent("#count")).trim() === "11 matches", "count reads 11 matches");

// Live scores and fixtures that are already over
check(await p.locator("#live").isVisible(), "the live section shows when the feed has live or finished matches");
check((await p.locator("#live-list .lcard").count()) === 1, "one match is live");
const liveText = await p.locator("#live-list .lcard").first().textContent();
check(liveText.includes("Everton") && liveText.includes("1 – 0") && liveText.includes("Pick is ahead"), `the live card shows teams, score and whether our pick is ahead (${liveText.replace(/\s+/g, " ").trim()})`);
check((await p.locator("#results-list .lcard").count()) === 2, "today's two finished matches are listed");
const doneText = await p.locator("#results-list").textContent();
check(doneText.includes("✗ Missed") && doneText.includes("Lens"), "a finished match says whether our pick missed, and a match without a pick still shows its score");
check((await p.locator('li.item[data-id="E0-20261027-Everton-Fulham"]').isHidden()), "a match in the live section is taken off the fixtures list");
check((await p.locator('li.item[data-id="SP1-20261027-Getafe-Betis"]').isHidden()), "a match that is over is taken off the fixtures list");
check((await p.textContent("#live-meta")).includes("2 minutes ago"), "the live section says how fresh it is");
await shot(p, "fixtures-live-desktop");

check(await p.locator("[data-scoreboard]").isVisible(), "the live scoreboard is visible");
check((await p.locator("[data-bookies]").count()) >= 1, "bookmaker lines are shown on cards that have odds");
const glass = await p.locator("article.card").first().evaluate((el) => getComputedStyle(el).backdropFilter || getComputedStyle(el).webkitBackdropFilter);
check(glass && glass !== "none", `cards use a frosted-glass blur (${glass})`);
const kick = await p.locator("li.item:not([hidden]) time.kick").first().textContent();
check(/\d{1,2}:\d{2}/.test(kick) && !kick.includes("UTC"), `kickoff time is shown in the visitor's timezone (${kick.trim()})`);
await p.waitForTimeout(1700);
const anim = await p.locator("article.card").first().evaluate((el) => getComputedStyle(el).animationName);
check(anim === "none", `entrance animation is off after load, so filtering will not replay it (${anim})`);
await p.click('button[data-league="I1"]');
check((await visible(p)) === 4, "Serie A chip leaves 4 matches");
check((await p.textContent("#count")).trim() === "4 of 11 matches", "count reads 4 of 11");
check(p.url().includes("league=I1"), "filter is saved in the URL");
await p.click('button[data-league="I1"]');  // toggle Serie A off again
await p.click('button[data-league="CL"]');
check((await visible(p)) === 1, "Champions League chip narrows the list to its one match");
const cl = p.locator('li.item:not([hidden]) article');
check((await cl.textContent()).includes("90-minute result"), "the Champions League card says it is a 90-minute result");
check((await cl.textContent()).includes("Experimental"), "the Champions League card carries the Experimental tag");
await shot(p, "fixtures-champions-league");
await p.click('button[data-league="CL"]');
await p.click('button[data-league="I1"]');
await p.fill("#q", "inter");
check((await visible(p)) === 1, "team search 'inter' leaves 1 match");
await p.fill("#q", "zzzz");
check(await p.locator("#empty").isVisible(), "empty state shows when nothing matches");
await p.click("#empty-clear");
check((await visible(p)) === 11, "clearing restores all matches");
await p.check("#confident");
const n = await visible(p);
check(n > 0 && n < 11, `confident-only narrows the list (${n})`);
await p.selectOption("#sort", "close");
check((await p.locator("li.day").count()) === 0, "no date headings when sorted by closeness");
await p.close();

// A new day: yesterday's fixtures are gone even though the page was built earlier
p = await open("/", { width: 1280, height: 900, at: "2026-10-30T12:00:00Z", live: "down" });
check((await visible(p)) === 3, "three days later only the matches still to come are listed");
check(!(await p.locator("#live").isVisible()), "the live section stays hidden when the feed cannot be read");
check((await p.textContent("#count")).trim() === "3 matches", "the count follows the visible matches");
await p.close();
p = await open("/", { width: 1280, height: 900, at: "2026-11-10T12:00:00Z", live: "down" });
check((await visible(p)) === 0 && (await p.locator("#none").isVisible()), "with every match over, the list says so instead of showing old fixtures");
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
const all = await p.textContent("#conf-pct");
await p.click('button[data-tier="confident"]');
const sure = await p.textContent("#conf-pct");
check(/^\d+%$/.test(all.trim()) && /^\d+%$/.test(sure.trim()), `accuracy shows for all picks (${all.trim()}) and for confident picks (${sure.trim()})`);
check((await p.getAttribute('button[data-tier="confident"]', "aria-pressed")) === "true", "the chosen confidence level is marked");
const nAll = Number((await p.textContent("#conf-of")).match(/of (\d+) picks/)[1]);
await p.click('button[data-lg="E0"]');
const nE0 = Number((await p.textContent("#conf-of")).match(/of (\d+) picks/)[1]);
check(nE0 < nAll && nE0 > 0, `limiting to one league narrows the picks (${nAll} to ${nE0})`);
await p.click('button[data-tier="all"]');
await p.click('button[data-lg=""]');
check(Number((await p.textContent("#conf-of")).match(/of (\d+) picks/)[1]) === 120, "all picks and all leagues counts every graded match");
await shot(p, "track-confidence");
await p.close();
p = await open("/track-record/", { width: 390, height: 844, dark: true });
await shot(p, "track-dark-mobile");
await p.close();

await browser.close();
if (problems.length) { console.log(problems.join("\n")); process.exit(1); }
