# Football Match Predictor: Design Spec

Date: 2026-10-27
Status: Draft for review

## 1. Purpose and agreed understanding

**What we are building:** a football match predictor for European teams, delivered as a static dashboard for fans. It publishes win/draw/loss probabilities for upcoming fixtures, with plain-language explanations, and a public track record of how past predictions performed.

**Decisions made in brainstorming (stated by the project owner):**

| Topic | Decision |
|---|---|
| Audience / purpose | Fans (a user-facing product, not a betting tool) |
| Delivery | Static dashboard, rebuilt on a schedule, no always-on backend |
| Initial coverage | Big five leagues: Premier League, La Liga, Serie A, Bundesliga, Ligue 1 |
| Later coverage | Improve from there; European cups and more leagues are planned extensions |
| Data | Free historical CSVs for training/backtesting + a free-tier API for upcoming fixtures and latest results |
| Model | Hybrid: Elo + Dixon-Coles feeding a LightGBM meta-model, then calibration |
| Priorities | Accuracy, explainability, and a model that improves over time |

**Success criteria:**
1. Probabilities are well calibrated (a "60%" outcome happens about 60% of the time).
2. Out-of-sample log loss and RPS beat the naive, Elo-only and Dixon-Coles-only baselines, and approach bookmaker closing odds on backtests.
3. Every prediction shows its top drivers in plain language.
4. Predictions are logged before kickoff and graded afterward, with the record shown publicly.
5. New model versions are promoted only when they measurably beat the current one.
6. The system refreshes automatically at near-zero hosting cost.

**Assumptions (not explicitly confirmed, adjust if wrong):**
- Primary output is home/draw/away probabilities; scorelines and both-teams-to-score are secondary outputs from the goals model.
- Hosting on a free static host (GitHub Pages or Cloudflare Pages) and scheduling with GitHub Actions is acceptable.
- Python for the pipeline; Astro as the static site generator for the frontend.

## 2. Non-goals (for the first version)

- No betting advice, stake sizing, or odds-based product features.
- Bookmaker odds are a backtest benchmark only, never a model input.
- No user accounts, personalization, or real-time in-play predictions.
- No mobile app or chat bot.
- No European cups or leagues outside the big five (the design must allow adding them later without a rewrite).

## 3. Architecture

Five units, each with one purpose and a defined interface.

1. **Ingest** (`src/ingest/`): loads historical CSVs once, then new results and upcoming fixtures from the API each run. Normalizes team names to canonical IDs. Stores raw data untouched; derives cleaned data from it. Validates schema on every run.
2. **Ratings** (`src/ratings/`): runs Elo and Dixon-Coles forward through time, recording each team's rating *before* each match.
3. **Features** (`src/features/`): builds point-in-time features: rating gaps, rolling form, shots and xG where available, rest days, home advantage, context. Uses only data dated before kickoff.
4. **Models** (`src/models/`): LightGBM meta-model that takes ratings and features and outputs outcome probabilities, followed by calibration. Includes a model registry.
5. **Publish** (`src/publish/`): exports predictions, explanations and accuracy stats as JSON; the static site reads them.

Supporting packages: `src/explain/` (SHAP to plain-language reasons) and `src/evaluate/` (backtests, metrics, leakage tests, champion gate).

**Run flow:** daily scheduled job: ingest, update ratings, re-score upcoming fixtures, grade finished predictions, update accuracy stats, publish. A separate weekly retrain workflow handles training and promotion.

## 4. Model layer

1. **Elo:** one strength rating per team with a home-advantage offset and a margin-of-victory adjustment. Between seasons, ratings regress toward the league mean; promoted teams start from a league-appropriate prior.
2. **Dixon-Coles:** per-team attack and defense strengths with time-decay weighting. Produces expected goals for each side, a scoreline grid, and win/draw/loss probabilities.
3. **LightGBM meta-model:** inputs are the Elo gap, Dixon-Coles probabilities and expected goals, plus form, shots/xG, rest days and context features. It learns where the statistical models are systematically off.
4. **Calibration:** isotonic regression by default, falling back to Platt scaling if the held-out set is too small for isotonic to be stable. Fitted on time-ordered held-out data only.

Build order within the hybrid: Elo and Dixon-Coles first (they are also the baselines), then the meta-model, then calibration. Each stage is evaluated against the previous one so the hybrid's added value is measured, not assumed.

## 5. Explainability

- **Per match (fans):** SHAP values from the meta-model are converted into the top three plain-language drivers with direction, for example "stronger recent attack", "extra rest", "home advantage". Elo and Dixon-Coles outputs appear as readable drivers (for example, "rating gap: +85 Elo"). A driver is shown only if it is meaningful.
- **Scoreline view:** the Dixon-Coles grid gives likely scores, both-teams-to-score and over/under 2.5 goals.
- **Per model (developers):** a model card per version with global feature importance, calibration plots, and performance by league and team-strength tier.
- **Wording rule:** explanations describe what the model weighted, not real-world causes ("the model weighted X", never "X caused this").

## 6. Evaluation and the improvement loop

**Backtesting rules:**
- Walk-forward only: train on data before a date, predict the next block, roll forward. No random splits.
- All ratings and features are point-in-time. An automated leakage test fails the build if any feature changes when future rows are removed.
- The most recent full season is a final hold-out, used only for release-candidate models.

**Metrics:**
- Primary: log loss and ranked probability score (RPS).
- Trust: calibration curves and expected calibration error, per outcome and per league.
- Fan-facing: plain accuracy of the top pick, shown with a note that it ignores confidence.
- Benchmarks: league base rates, Elo alone, Dixon-Coles alone, bookmaker closing odds.
- Slices: by league, team-strength tier, matchday of season, early versus late season.

**Loop:**
1. Log every prediction before kickoff with timestamp, model version and probabilities.
2. Grade after full time; metrics update from the log.
3. On the weekly retrain, train a challenger (new features, retuned parameters) walk-forward next to the current champion.
4. Promote the challenger only if it wins on recent log loss and RPS over a minimum number of matches and calibration does not worsen.
5. Monitor drift: alert when rolling log loss or calibration worsens beyond a threshold.
6. Version everything: data snapshot, code version, hyperparameters, metrics; any past prediction must be reproducible.

**Honest expectations:** football is noisy. Top-pick accuracy across the big five typically lands around 50 to 55 percent for strong models. The real quality signals are calibration and log loss, and the site presents them that way. Matching bookmaker closing odds would be a strong result; the project does not claim to beat the market.

**Planned experiments the loop makes safe:** lineups and injuries from the API, fixture congestion and travel, manager changes, European cups.

## 7. The site

- **Fixtures:** upcoming matches by league and matchday with probabilities, likely scoreline and top three drivers.
- **Match detail:** scoreline grid, both-teams-to-score and over/under chances, rating comparison, recent form.
- **Team pages:** ratings, form, upcoming and past predictions.
- **Track record:** log loss, calibration chart, accuracy by league, every graded past prediction.
- **About / method:** how it works and what it cannot do.
- Every page shows "last updated" and a staleness warning if data is older than expected.

## 8. Operations

- Daily GitHub Actions workflow, plus an extra run late on matchday nights; exits cheaply when there are no new results.
- Weekly retrain workflow, separate from the daily job. The cadence can be tuned later.
- The site deploys only when the pipeline completes successfully; otherwise the last good version stays live.
- Hosting: GitHub Pages or Cloudflare Pages (free).

## 9. Repository layout

```
football-predictor/
  data/           raw/ (untouched), processed/, predictions_log/
  src/
    ingest/       sources, name normalization, schema checks
    ratings/      elo.py, dixon_coles.py
    features/     point-in-time feature builder
    models/       meta_model.py, calibration.py, registry
    explain/      SHAP-to-text reasons
    evaluate/     backtest, metrics, leakage tests, champion gate
    publish/      JSON export for the site
  site/           static frontend
  .github/workflows/  daily.yml, retrain.yml
  docs/superpowers/specs/
```

## 10. Error handling

- Schema checks on ingest; a source format change or bad data fails the run loudly and the last good site stays live.
- API calls are cached with backoff; training never depends on the API because it uses the historical CSVs.
- A team that cannot be mapped to a canonical ID blocks only that fixture, and is flagged for a fix.
- Staleness flag on every page.

## 11. Testing

- Unit tests for Elo and Dixon-Coles against hand-computed cases.
- The leakage test (section 6) on every change.
- Name-normalization tests across data sources.
- End-to-end test on a fixed mini dataset: the pipeline produces valid JSON and the site builds.
- Reproducibility test: re-running a past date with the same data and code gives identical predictions.

## 12. Risks and first steps

- **Free API tier limits.** It may restrict leagues, fixtures or history. The first implementation step is to confirm exactly what the chosen free tier provides, and pick the API based on that. The historical CSVs mean training is unaffected.
- **Advanced stats (xG).** Free CSVs of the main results data do not include xG. Adding scraped xG later requires checking each source's terms of use first and is treated as an experiment under the champion gate.
- **Name mapping across sources** is a common failure point; it is tested explicitly (section 11).
- **Season boundaries** (promotions, transfers) reset team strength; handled by rating regression and prior rules (section 4) and evaluated by the early-season slice.
