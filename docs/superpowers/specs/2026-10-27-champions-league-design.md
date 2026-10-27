# Champions League predictions: design

Date: 2026-10-27. Status: draft for review. Extends `2026-10-27-football-predictor-design.md`.

## Outcome

Upcoming UEFA Champions League matches appear on the site next to the league fixtures, with
home/draw/away chances, a reason list, and a "Champions League" chip in the existing filter bar.
Predictions are for the **90-minute result**, not who advances (two-legged ties, extra time and
penalties are a different question and are out of scope).

## Scope

In: Champions League (league phase, play-offs, knockouts, final).
Out, and why:
- Europa League, Conference League, domestic cups: the free football-data.org key is refused for
  them (verified with a probe on 2026-10-27), and the paid alternative was declined. Their *results*
  are in the open history files, so they are used to improve ratings, but they are not predicted.
- Qualifying rounds: not offered by the feed.

## Data

| Need | Source | Notes |
|---|---|---|
| Upcoming Champions League fixtures and recent results | football-data.org, code `CL` | Already works on our key: 18 matches in the next two weeks. Same client, one more league code. |
| History for ratings and for testing | `openfootball/champions-league` | CC0 licence. Seasons 2011-12 to 2025-26. Text format: stage headers, dates, `Home (CTY) v Away (CTY)  2-1 (1-0)`. Extra time and penalties are written `4-3 pen. 1-1 a.e.t. (1-1, 0-1)`; the **first** bracketed score is the 90-minute score. Files per season: `cl`, `clq`, `el`, `elq`, `conf`, `confq` (some absent). |
| Big-five league results | existing football-data.co.uk history | Unchanged. |

History is fetched in CI by the existing download step (a shallow git clone), never at build time of the site.

## Design

### 1. Parser (`ingest/european.py`)
Turns an openfootball file into rows: `date, competition, stage, home, away, home_country,
away_country, home_goals_90, away_goals_90`. Matches without a score are dropped; extra time is
discarded in favour of the 90-minute score. Tested on real excerpts (regular match, extra time,
penalties, missing halftime score).

### 2. Team identity
Three name spellings must agree: football-data.org's, openfootball's, and our league canonical names.
- A normalisation key (lower-case, accents and club suffixes such as FC/CF/AC/SC removed) catches most
  matches automatically.
- `data/european_aliases.csv` holds the exceptions (for example "Sport Lisboa e Benfica" to "Benfica") and maps
  big-five clubs to the canonical names the league model already uses.
- A club that still does not match is **kept** at the default rating and listed in the run's warnings; it
  does not stop the daily run (unlike leagues), because new clubs appear every season. Its predictions
  carry a low-confidence flag. A manual `teams` workflow lists unmatched names, as it does for leagues.

### 3. Pooled ratings (`ratings/pooled.py`, new; the league Elo is not touched)
One Elo system over the union of: the big-five league matches, and all European matches above, in date
order. Cross-league games are what make ratings comparable across leagues.
- Each club starts at an offset for its country group (E0, SP1, I1, D1, F1, and "rest of Europe"). The six
  offsets are fitted by minimising the squared error between expected and actual score (a draw counts 0.5) on
  cross-league matches, using development seasons only; Elo has no draw probability, so log loss is not defined for it.
- Home advantage applies except in the final (neutral venue).
- Same point-in-time guarantee as today: a match's ratings only reflect strictly earlier dates, checked by a
  leakage test.

### 4. Champions League model (`models/`)
Multinomial logistic regression on: pooled Elo gap, home advantage (off for the final), knockout flag, and
the two clubs' recent league-phase goal averages if they help on development seasons. Platt calibration as
elsewhere. Trained on all European matches before the prediction date; walk-forward for testing.
- Registry namespace `cl` with the existing champion/challenger gate and weekly retrain.
- **Outcome of tuning (2026-10-27):** on development seasons the extra inputs and a separate calibration block never beat the plain rating gap, so the shipped model is a logistic regression on the pooled-Elo gap with tuned rating settings (docs/european-tuning.md). The gate below therefore asks for a win over the base rate and a tie (within 0.002 log loss) with the plain Elo formula; the weekly retrain applies it on its evaluation block, and the benchmark applies it walk-forward on Champions League matches.
- **Publish gate:** Champions League predictions go live only if, on development seasons, the model beats the
  base rate and the pooled-Elo-only formula on log loss. If it does not, the fixtures are not shown and the
  report says why.

### 5. Pipeline and export
- `daily`: fetch `CL` fixtures and results with the existing client; `match_id` is `CL-YYYYMMDD-Home-Away`;
  predictions are logged with `league = "CL"`; results from the feed update the pooled ratings.
- Export adds the competition name; the site's `leagueName("CL")` is "Champions League". The Teams page and
  the league model's ratings are unchanged; clubs outside the big five have no team page.
- Track record gets a Champions League row automatically (by competition).

### 6. Site
The chip appears automatically from the data. CL cards carry the note "90-minute result" and an
"Experimental" tag until 30 Champions League predictions have been graded.

## Testing

- Parser: real excerpts, including extra time and penalties.
- Aliases: every club in the 2025-26 history and in the live feed resolves, or is reported.
- Pooled Elo: point-in-time leakage test; cross-league results move the right clubs.
- Backtest report (`docs/baseline-results.md`): base rate, pooled-Elo formula and model, with correct picks
  and log loss. Development seasons for every choice. **Hold-out 2025-26 evaluated once**, only after the
  model is chosen; the earlier hold-out was already seen twice, so this is a fresh one.
- End to end: daily run with a fake client returning a CL fixture; site build test shows the chip.

## Risks

- **Name matching** is the most likely source of silent errors; mitigated by warnings and the low-confidence flag.
- **Small data:** roughly 2,000 Champions League matches (more with Europa and Conference). Expect modest
  gains over a plain Elo formula; the gate exists so we do not ship noise.
- **Rating scale for clubs outside the big five** rests on few cross-league games; they start low-confidence.
- Two-leg and final effects are ignored beyond the neutral-venue flag.

## Success criteria

Champions League fixtures appear on the site after the daily run; the benchmark document shows the
Champions League backtest with correct-pick percentage; no run fails because of a new club name.
