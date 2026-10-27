# Fixtures API: free-tier findings and decision

Date checked: 2026-10-27. Training never depends on this API (it uses the historical CSVs), so the
tier only limits the upcoming-fixtures and latest-results feed.

## What each candidate's free tier provides

**football-data.org** (summary from a third-party article; its own pricing page was not reachable
from the build sandbox, so verify with a real key before relying on it)
- Rate limit: 10 requests per minute for registered free users, not a monthly pool.
- Competitions: 12, including the Premier League, La Liga, Serie A, Bundesliga and Ligue 1, plus the
  Champions League, Eredivisie, Primeira Liga and others.
- Scores and schedules are delayed (no live scores); history is limited to the current season.
- No player data, match statistics or odds on the free tier.

**API-Football** (from the provider's own getting-started guide)
- 100 requests per day, with a separate per-minute cap whose value the guide does not give.
- Free plan exposes every endpoint; paid plans unlock deeper history. The guide does not say how far
  back the free plan goes, and does not confirm fixtures/results coverage explicitly.

## Decision

Use **football-data.org** (v4 API, `X-Auth-Token` header) for the daily feed:
- Its free tier explicitly covers all five target leagues and the Champions League (needed for the
  planned European-cups extension).
- 10 requests/minute is far above what one daily job needs (5 leagues x 2 calls).
- Delayed scores are acceptable: the job runs daily and after matchday nights, and predictions are
  logged before kickoff regardless.
- Current-season-only history is acceptable: history comes from the CSVs.

API-Football stays the fallback: swap in by writing another small adapter that implements
`FixturesClient` (`src/footpred/ingest/fixtures.py`).

## Competition code mapping used by the adapter

`E0 -> PL`, `SP1 -> PD`, `I1 -> SA`, `D1 -> BL1`, `F1 -> FL1`. These are football-data.org's
documented competition codes as recalled when writing the adapter; confirm them against a live
response on first run.

## Sources

- [API-Football getting-started guide](https://www.api-football.com/news/post/how-to-get-started-with-api-football-the-complete-beginners-guide)
- [football-data.org free tier limits 2026 (third-party summary)](https://www.thestatsapi.com/blog/football-data-org-free-tier-limits-2026)
