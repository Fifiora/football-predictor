# Football Match Predictor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a static fan-facing dashboard that publishes calibrated, explained win/draw/loss predictions for big-five-league fixtures, with a public graded track record and a retraining loop that only promotes better models.

**Architecture:** A Python pipeline (ingest, Elo and Dixon-Coles ratings, point-in-time features, LightGBM meta-model with calibration, SHAP-based reasons) writes JSON that an Astro static site renders. GitHub Actions runs a daily refresh and a weekly retrain with a champion/challenger gate.

**Tech Stack:** Python 3.11+, pandas, numpy, scipy, scikit-learn, LightGBM, SHAP, pytest, Astro, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-27-football-predictor-design.md`

## Global Constraints

- Leagues: Premier League (`E0`), La Liga (`SP1`), Serie A (`I1`), Bundesliga (`D1`), Ligue 1 (`F1`) only; league is a data column so more can be added later without rewrites.
- Outcome probability order everywhere is `[home, draw, away]`; outcome labels are `H`, `D`, `A`.
- Evaluation is walk-forward and time-ordered only; no random splits anywhere.
- Every rating and feature for a match uses only data dated strictly before that match's date.
- Bookmaker odds are a backtest benchmark only and are never a model input.
- Calibration is isotonic by default, Platt scaling if the held-out set is too small for isotonic to be stable.
- Retraining is weekly; the daily job never retrains.
- Explanation wording is "the model weighted X", never causal claims; the product gives no betting advice.
- Hosting is a free static host (GitHub Pages or Cloudflare Pages); the site deploys only after a fully successful pipeline run, otherwise the last good site stays live.
- Every page shows "last updated" and a staleness warning if data is older than expected.

## Review Focus

1. A promoted or brand-new team with no history at season start gets a league-appropriate prior rating, not a crash or a rating of zero. (Task 2, Task 3)
2. Postponed or abandoned matches with missing scores are excluded from ratings and training, and never counted as results. (Task 1)
3. An unmappable team name blocks only that fixture and is reported, not the whole run. (Task 1, Task 9)
4. Calibrated probabilities always sum to 1 and stay in [0, 1], even for extreme favorites. (Task 6)
5. An empty or failed API response exits cheaply and leaves the last good site data in place. (Task 9, Task 12)

---

### Task 1: Project scaffold, schema, team names, historical loader

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/footpred/__init__.py`
- Create: `src/footpred/ingest/schema.py`, `src/footpred/ingest/teams.py`, `src/footpred/ingest/historical.py`
- Create: `data/team_aliases.csv` (columns `league,alias,canonical`)
- Test: `tests/ingest/test_schema.py`, `tests/ingest/test_teams.py`, `tests/ingest/test_historical.py`, `tests/fixtures/E0_sample.csv`

**Interfaces:**
- Produces: `MATCH_COLUMNS: list[str]` = `match_id, league, season, date, home, away, home_goals, away_goals, result, home_shots, away_shots, home_sot, away_sot, close_h, close_d, close_a` (shots and closing odds may be NaN)
- Produces: `validate_matches(df: pd.DataFrame) -> pd.DataFrame` (raises `SchemaError`)
- Produces: `canonical_team(name: str, league: str) -> str` (raises `UnknownTeamError`)
- Produces: `load_historical(raw_dir: Path) -> pd.DataFrame` (columns `MATCH_COLUMNS`, sorted by `date`, then `match_id`)

- [ ] **Step 1: `git init`, create the package skeleton, and write `pyproject.toml` with dependencies and pytest config. Commit.**
- [ ] **Step 2: Write failing tests**
  - `test_validate_rejects_missing_column`: dropping `home_goals` raises `SchemaError`.
  - `test_validate_drops_unplayed_rows`: a row with NaN `home_goals` is removed from the output (Review Focus 2).
  - `test_canonical_team_maps_variants`: `canonical_team("Man United", "E0") == canonical_team("Manchester United", "E0")`.
  - `test_unknown_team_raises`: `canonical_team("Nonexistent FC", "E0")` raises `UnknownTeamError`.
  - `test_load_historical_sample`: loading `tests/fixtures/E0_sample.csv` yields rows with `result` in `{H,D,A}`, `league == "E0"`, unique `match_id`.
- [ ] **Step 3: Run `pytest tests/ingest -v`. Expected: FAIL (modules not defined).**
- [ ] **Step 4: Implement the three modules.** football-data.co.uk columns map as `HomeTeam, AwayTeam, FTHG, FTAG, FTR, HS, AS, HST, AST`; closing odds from the Pinnacle closing columns where present. `match_id` is `f"{league}-{date:%Y%m%d}-{home}-{away}"`. Seed `team_aliases.csv` from the team names in the sample file.
- [ ] **Step 5: Run `pytest tests/ingest -v`. Expected: PASS. Commit.**

### Task 2: Elo ratings (point-in-time)

**Files:**
- Create: `src/footpred/ratings/elo.py`
- Test: `tests/ratings/test_elo.py`

**Interfaces:**
- Consumes: `matches` DataFrame from Task 1
- Produces: `@dataclass EloParams(k: float = 20.0, home_adv: float = 60.0, season_regress: float = 0.25, initial: float = 1500.0)`
- Produces: `run_elo(matches: pd.DataFrame, params: EloParams = EloParams()) -> tuple[pd.DataFrame, dict[str, float]]`; the DataFrame is indexed by `match_id` with `elo_home_pre`, `elo_away_pre`; the dict is final ratings by team.

- [ ] **Step 1: Write failing tests**
  - `test_pre_ratings_exclude_own_result`: in a one-match dataset both pre-ratings equal `initial`.
  - `test_winner_gains_loser_loses`: after one home win, final home rating > `initial` > final away rating, and the changes are equal and opposite.
  - `test_season_regression`: with `season_regress=0.25`, a team at 1700 at season end starts the next season at `1700 - 0.25 * (1700 - league_mean)`.
  - `test_new_team_gets_league_prior`: a team first appearing in season 2 starts at the league mean of the previous season's final ratings, not `initial` (Review Focus 1).
  - `test_no_future_leak`: changing the last match's score does not change any earlier row's pre-ratings.
- [ ] **Step 2: Run `pytest tests/ratings/test_elo.py -v`. Expected: FAIL.**
- [ ] **Step 3: Implement `run_elo`.** Process matches in date order; standard Elo expected score with `home_adv` added to the home rating; scale `k` by a log margin-of-victory multiplier; apply regression at each season boundary per league.
- [ ] **Step 4: Run the tests. Expected: PASS. Commit.**

### Task 3: Dixon-Coles goals model

**Files:**
- Create: `src/footpred/ratings/dixon_coles.py`
- Test: `tests/ratings/test_dixon_coles.py`

**Interfaces:**
- Consumes: `matches` from Task 1
- Produces: `@dataclass DCPrediction(exp_home_goals: float, exp_away_goals: float, p_home: float, p_draw: float, p_away: float, grid: np.ndarray)` with `grid` a 9x9 scoreline probability matrix (goals 0 to 8, renormalized to sum to 1)
- Produces: `fit_dixon_coles(matches: pd.DataFrame, as_of: pd.Timestamp, xi: float = 0.0018) -> DCModel` using only matches with `date < as_of`, weights `exp(-xi * days_ago)`
- Produces: `DCModel.predict(home: str, away: str) -> DCPrediction` (unseen teams use the league-average attack/defense, Review Focus 1)
- Produces: `dc_pre_match(matches: pd.DataFrame, refit_days: int = 7) -> pd.DataFrame` indexed by `match_id` with `dc_exp_home, dc_exp_away, dc_p_home, dc_p_draw, dc_p_away`; refits only every `refit_days` and reuses the model between refits.

- [ ] **Step 1: Write failing tests**
  - `test_probabilities_sum_to_one`: for a fitted model on synthetic data, `p_home + p_draw + p_away` is within 1e-9 of 1 and `grid.sum()` within 1e-9 of 1.
  - `test_strong_attack_team_favored`: on synthetic data where team A scores 3/game and concedes 0.5, `predict("A","B").p_home > 0.6` against an average team at home.
  - `test_time_decay_weights_recent`: a team that was weak early and strong recently is rated above its unweighted average.
  - `test_unseen_team_does_not_crash`: `predict("A", "NewTeam")` returns valid probabilities.
  - `test_pre_match_excludes_future`: `dc_pre_match` rows for a given date are unchanged if matches on or after that date are removed.
- [ ] **Step 2: Run `pytest tests/ratings/test_dixon_coles.py -v`. Expected: FAIL.**
- [ ] **Step 3: Implement.** Poisson attack/defense/home-advantage parameters with the Dixon-Coles low-score `rho` correction, fitted by weighted maximum likelihood with `scipy.optimize.minimize` (sum-to-zero constraint on attack).
- [ ] **Step 4: Run the tests. Expected: PASS. Commit.**

### Task 4: Point-in-time feature builder and leakage test

**Files:**
- Create: `src/footpred/features/builder.py`, `src/footpred/evaluate/leakage.py`
- Test: `tests/features/test_builder.py`, `tests/evaluate/test_leakage.py`

**Interfaces:**
- Consumes: `matches`, `run_elo(...)[0]`, `dc_pre_match(...)`
- Produces: `FEATURE_COLUMNS: list[str]` = `elo_diff, dc_exp_home, dc_exp_away, dc_p_home, dc_p_draw, dc_p_away, form_goals_for_home, form_goals_against_home, form_goals_for_away, form_goals_against_away, form_shots_diff_home, form_shots_diff_away, rest_days_home, rest_days_away` (form = mean over each team's previous 5 matches; NaN allowed, handled by LightGBM)
- Produces: `build_features(matches: pd.DataFrame, elo_pre: pd.DataFrame, dc_pre: pd.DataFrame) -> pd.DataFrame` indexed by `match_id`, columns `FEATURE_COLUMNS`
- Produces: `assert_no_leakage(build_fn: Callable[[pd.DataFrame], pd.DataFrame], matches: pd.DataFrame, n_probe: int = 20, seed: int = 0) -> None`; for random probe matches it deletes that match and everything after it, rebuilds, and raises `LeakageError` if the probe row's features changed.

- [ ] **Step 1: Write failing tests**
  - `test_form_uses_only_prior_matches`: first-ever match for a team has NaN form features.
  - `test_rest_days`: a team playing 3 days after its previous match has `rest_days == 3`.
  - `test_elo_diff_matches_pre_ratings`: `elo_diff == elo_home_pre - elo_away_pre`.
  - `test_leakage_check_passes_on_builder`: `assert_no_leakage` on the real builder with the Task 1 sample data does not raise.
  - `test_leakage_check_catches_leak`: a deliberately leaky builder (feature uses the match's own goals) makes `assert_no_leakage` raise `LeakageError`.
- [ ] **Step 2: Run `pytest tests/features tests/evaluate/test_leakage.py -v`. Expected: FAIL.**
- [ ] **Step 3: Implement both modules.** Rolling form via `groupby(team).shift(1).rolling(5)` over a long-format team-match table; rest days from the previous match date per team.
- [ ] **Step 4: Run the tests. Expected: PASS. Commit.**

### Task 5: Metrics, baselines and walk-forward backtest

**Files:**
- Create: `src/footpred/evaluate/metrics.py`, `src/footpred/evaluate/backtest.py`, `src/footpred/evaluate/baselines.py`
- Test: `tests/evaluate/test_metrics.py`, `tests/evaluate/test_backtest.py`

**Interfaces:**
- Produces: `log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float` (`probs` shape `(n,3)`, `outcomes` ints 0/1/2 for H/D/A)
- Produces: `rps(probs: np.ndarray, outcomes: np.ndarray) -> float`
- Produces: `expected_calibration_error(probs: np.ndarray, outcomes: np.ndarray, bins: int = 10) -> float`
- Produces: `walk_forward_splits(dates: pd.Series, first_test_date: pd.Timestamp, step_days: int = 30) -> Iterator[tuple[np.ndarray, np.ndarray]]` yielding `(train_idx, test_idx)` where every train date is strictly before every test date.
- Produces: `backtest(fit_predict: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray], features: pd.DataFrame, matches: pd.DataFrame, first_test_date: pd.Timestamp, step_days: int = 30) -> pd.DataFrame` returning one row per test match with `match_id, p_home, p_draw, p_away, outcome`.
- Produces: `baseline_base_rate`, `baseline_elo`, `baseline_dixon_coles`, `benchmark_closing_odds` each returning a `(n,3)` probability array for given matches; closing odds are de-vigged by proportional normalization.

- [ ] **Step 1: Write failing tests**
  - `test_log_loss_perfect_and_uniform`: perfect confident predictions give ~0; uniform `[1/3]*3` gives `ln(3)`.
  - `test_rps_known_value`: probs `[0.5,0.3,0.2]`, outcome home gives `0.5 * ((0.5-1)**2 + (0.8-1)**2) = 0.145`.
  - `test_ece_zero_for_perfectly_calibrated`: constructed data with matching frequencies gives ECE < 0.02.
  - `test_splits_are_strictly_time_ordered`: for every split, `max(train dates) < min(test dates)`.
  - `test_devig_odds_sum_to_one`: de-vigged closing odds rows sum to 1.
- [ ] **Step 2: Run `pytest tests/evaluate -v`. Expected: FAIL.**
- [ ] **Step 3: Implement.** `baseline_elo` maps `elo_diff` to probabilities via a multinomial logistic regression fitted walk-forward; `baseline_dixon_coles` reads the `dc_p_*` columns.
- [ ] **Step 4: Run the tests. Expected: PASS. Commit.**

### Task 6: LightGBM meta-model and calibration

**Files:**
- Create: `src/footpred/models/meta_model.py`, `src/footpred/models/calibration.py`
- Test: `tests/models/test_meta_model.py`, `tests/models/test_calibration.py`

**Interfaces:**
- Consumes: `FEATURE_COLUMNS`, outcomes as ints 0/1/2
- Produces: `MetaModel(params: dict | None = None)` with `.fit(X: pd.DataFrame, y: np.ndarray) -> MetaModel` and `.predict_proba(X: pd.DataFrame) -> np.ndarray` of shape `(n,3)`
- Produces: `Calibrator(min_samples_isotonic: int = 1000)` with `.fit(probs: np.ndarray, y: np.ndarray) -> Calibrator` and `.transform(probs: np.ndarray) -> np.ndarray`; one-vs-rest isotonic per outcome (Platt if fewer than `min_samples_isotonic` rows), then renormalize rows to sum to 1.
- Produces: `fit_hybrid(X_train, y_train, X_cal, y_cal) -> tuple[MetaModel, Calibrator]` with `X_cal` strictly later in time than `X_train` (caller guarantees; the function asserts `X_cal.index` dates are not earlier when a date column is supplied).

- [ ] **Step 1: Write failing tests**
  - `test_predict_proba_shape_and_sum`: output shape `(n,3)` and rows sum to 1 within 1e-6.
  - `test_calibrated_rows_sum_to_one_and_bounded`: after `transform`, every row sums to 1 within 1e-9 and values are in [0,1], including inputs like `[0.99,0.005,0.005]` (Review Focus 4).
  - `test_uses_platt_below_threshold`: with 200 calibration rows the calibrator reports `method == "platt"`; with 2000 it reports `"isotonic"`.
  - `test_calibration_improves_ece_on_miscalibrated_input`: on synthetic overconfident probabilities, ECE after `transform` is lower than before.
  - `test_meta_model_beats_base_rate_on_synthetic_signal`: on synthetic data where `elo_diff` determines the outcome, log loss is below the base-rate log loss.
- [ ] **Step 2: Run `pytest tests/models -v`. Expected: FAIL.**
- [ ] **Step 3: Implement.** `lightgbm.LGBMClassifier(objective="multiclass", num_class=3)` with conservative defaults (shallow trees, small learning rate, early stopping off in `fit` because the calibration split is separate).
- [ ] **Step 4: Run the tests. Expected: PASS. Commit.**

### Task 7: Explainability (SHAP to plain language)

**Files:**
- Create: `src/footpred/explain/reasons.py`, `src/footpred/explain/templates.py`
- Test: `tests/explain/test_reasons.py`

**Interfaces:**
- Consumes: a fitted `MetaModel`, one feature row
- Produces: `@dataclass Driver(feature: str, direction: str, strength: float, text: str)` with `direction` in `{"home", "away"}` meaning which side the driver favors
- Produces: `top_drivers(model: MetaModel, x_row: pd.Series, k: int = 3, min_strength: float = 0.02) -> list[Driver]` ranked by absolute SHAP value for the home-win minus away-win outcomes; drivers below `min_strength` are omitted.
- Produces: `driver_text(feature: str, direction: str, value: float, home: str, away: str) -> str` using templates in `templates.py`, always phrased as what "the model weighted" (for example "rating gap: +85 Elo").

- [ ] **Step 1: Write failing tests**
  - `test_returns_at_most_k_and_ranked`: length ≤ 3 and `strength` is non-increasing.
  - `test_direction_follows_sign`: on a synthetic model where large positive `elo_diff` drives home wins, a high-`elo_diff` row's top driver is `elo_diff` with `direction == "home"`.
  - `test_min_strength_filters_noise`: a row whose SHAP values are all below `min_strength` returns an empty list.
  - `test_text_has_no_causal_wording`: no generated text contains "because", "caused" or "due to"; every text contains "model".
- [ ] **Step 2: Run `pytest tests/explain -v`. Expected: FAIL.**
- [ ] **Step 3: Implement using `shap.TreeExplainer`.** One template per entry in `FEATURE_COLUMNS` that can appear as a driver.
- [ ] **Step 4: Run the tests. Expected: PASS. Commit.**

### Task 8: Registry, champion gate, prediction log and grading

**Files:**
- Create: `src/footpred/models/registry.py`, `src/footpred/evaluate/gate.py`, `src/footpred/evaluate/drift.py`, `src/footpred/models/model_card.py`, `src/footpred/predictions/log.py`
- Test: `tests/models/test_registry.py`, `tests/models/test_model_card.py`, `tests/evaluate/test_gate.py`, `tests/evaluate/test_drift.py`, `tests/predictions/test_log.py`

**Interfaces:**
- Produces: `save_model(registry_dir: Path, version: str, model: MetaModel, calibrator: Calibrator, metadata: dict) -> Path` (metadata must include `data_snapshot`, `code_version`, `params`, `metrics`)
- Produces: `load_champion(registry_dir: Path) -> tuple[str, MetaModel, Calibrator]`; `set_champion(registry_dir: Path, version: str) -> None`
- Produces: `@dataclass GateResult(promote: bool, reason: str, champion_log_loss: float, challenger_log_loss: float, champion_rps: float, challenger_rps: float)`
- Produces: `should_promote(champion_probs, challenger_probs, outcomes, min_matches: int = 200) -> GateResult`; promotes only if matches ≥ `min_matches`, challenger log loss and RPS are both lower, and challenger ECE is not worse than the champion's by more than 0.005.
- Produces: `log_predictions(log_path: Path, preds: pd.DataFrame, model_version: str, logged_at: pd.Timestamp) -> None` (append-only; refuses rows whose `kickoff_date` is on or before `logged_at`); `grade_predictions(log_path: Path, results: pd.DataFrame) -> pd.DataFrame` filling in actual outcome for finished matches.

- Produces: `check_drift(graded_log: pd.DataFrame, window: int = 200, log_loss_ceiling: float | None = None, ece_ceiling: float = 0.05) -> list[str]` returning alert messages when rolling log loss exceeds the ceiling (default: 1.05 x the champion's recorded backtest log loss) or rolling ECE exceeds `ece_ceiling`; empty list means healthy.
- Produces: `write_model_card(out_path: Path, model: MetaModel, backtest_preds: pd.DataFrame, matches: pd.DataFrame) -> None` writing global feature importance, a calibration table, and log loss/RPS by league and by team-strength tier (Elo terciles).

- [ ] **Step 1: Write failing tests**
  - `test_drift_alerts_when_rolling_loss_rises`: a log whose last 200 rows are much worse than the champion's recorded loss returns a non-empty alert list; a healthy log returns `[]`.
  - `test_model_card_contains_required_sections`: the written file contains feature importance, calibration, per-league and per-tier sections.
  - `test_gate_blocks_below_min_matches`: 100 matches with a clearly better challenger returns `promote False` and a reason mentioning the minimum.
  - `test_gate_blocks_worse_calibration`: better log loss and RPS but ECE worse by 0.02 returns `promote False`.
  - `test_gate_promotes_clear_winner`: 300 matches, better on all three returns `promote True`.
  - `test_registry_roundtrip`: save then `set_champion` then `load_champion` returns the same version and identical predictions on a fixed input.
  - `test_log_is_append_only_and_rejects_late_rows`: appending twice keeps both batches; a row with `kickoff_date <= logged_at` raises `ValueError`.
  - `test_grade_fills_outcomes_without_mutating_probabilities`: graded rows keep their original probabilities.
- [ ] **Step 2: Run `pytest tests/models/test_registry.py tests/models/test_model_card.py tests/evaluate/test_gate.py tests/evaluate/test_drift.py tests/predictions -v`. Expected: FAIL.**
- [ ] **Step 3: Implement.** Registry stores each version as a directory with `model.joblib`, `calibrator.joblib`, `metadata.json`, and a `champion.txt` pointer; the log is a Parquet or CSV file under `data/predictions_log/`.
- [ ] **Step 4: Run the tests. Expected: PASS. Commit.**

### Task 9: Upcoming fixtures ingest (API)

**Files:**
- Create: `docs/api-findings.md`, `src/footpred/ingest/fixtures.py`
- Test: `tests/ingest/test_fixtures.py`, `tests/fixtures/api_fixtures_sample.json`

**Interfaces:**
- Produces: `fetch_upcoming(client: FixturesClient, leagues: list[str], cache_dir: Path) -> tuple[pd.DataFrame, list[str]]` returning upcoming fixtures (`league, date, home, away`, canonical team names) and a list of unmapped-team warnings.
- Produces: `fetch_latest_results(client: FixturesClient, leagues: list[str], since: pd.Timestamp, cache_dir: Path) -> pd.DataFrame` in `MATCH_COLUMNS` shape.
- Produces: `class FixturesClient(Protocol)` with `get_fixtures(league: str, date_from: date, date_to: date) -> list[dict]` so tests inject a fake and the real provider is one small adapter.

- [ ] **Step 1: First, confirm what the free tier actually provides.** Check each candidate (for example API-Football and football-data.org) for big-five coverage, fixtures-ahead window, results freshness and daily request limits; record the choice and the limits in `docs/api-findings.md`. Training never depends on the API, so a restrictive tier only limits the upcoming-fixtures feed.
- [ ] **Step 2: Write failing tests (using the fake client and the sample JSON)**
  - `test_unmapped_team_blocks_only_that_fixture`: one fixture with an unknown team is excluded and listed in warnings while the others are returned (Review Focus 3).
  - `test_empty_response_returns_empty_frame_without_error`: an empty API response returns an empty DataFrame and no exception (Review Focus 5).
  - `test_cache_avoids_repeat_calls`: a second call with the same arguments makes zero client calls.
  - `test_retries_with_backoff_on_failure`: a client that fails twice then succeeds returns data after three attempts; a client that always fails raises `FixturesUnavailable`.
- [ ] **Step 3: Run `pytest tests/ingest/test_fixtures.py -v`. Expected: FAIL.**
- [ ] **Step 4: Implement `fixtures.py` and the real-provider adapter chosen in Step 1.**
- [ ] **Step 5: Run the tests. Expected: PASS. Commit.**

### Task 10: Publisher (JSON export)

**Files:**
- Create: `src/footpred/publish/export.py`
- Test: `tests/publish/test_export.py`

**Interfaces:**
- Consumes: scored fixtures with `p_home, p_draw, p_away`, `DCPrediction.grid`, `Driver` lists, the graded log
- Produces: `export_site_data(out_dir: Path, fixtures: pd.DataFrame, drivers: dict[str, list[Driver]], dc_grids: dict[str, np.ndarray], graded_log: pd.DataFrame, ratings: dict[str, float], updated_at: pd.Timestamp) -> None` writing `fixtures.json`, `track_record.json`, `teams.json`, `meta.json` (`updated_at`, `model_version`).
- Produces: `track_record.json` includes log loss, RPS, accuracy, a calibration table, and per-league metrics, computed only from graded rows.

- [ ] **Step 1: Write failing tests**
  - `test_exports_expected_files_and_valid_json`: all four files exist and parse.
  - `test_fixture_probabilities_sum_to_one`: each exported fixture's three probabilities sum to 1 within 1e-6.
  - `test_scoreline_derived_fields`: for a grid with known values, exported both-teams-to-score and over-2.5 chances match hand-computed sums.
  - `test_track_record_ignores_ungraded_rows`: ungraded log rows do not affect exported metrics.
  - `test_export_is_atomic`: if writing fails midway (simulated), the previous `out_dir` contents are unchanged.
- [ ] **Step 2: Run `pytest tests/publish -v`. Expected: FAIL.**
- [ ] **Step 3: Implement.** Write to a temporary directory and swap into place so a failed run never leaves partial data.
- [ ] **Step 4: Run the tests. Expected: PASS. Commit.**

### Task 11: Astro static site

**Files:**
- Create: `site/package.json`, `site/astro.config.mjs`, `site/src/pages/index.astro`, `site/src/pages/match/[id].astro`, `site/src/pages/team/[name].astro`, `site/src/pages/track-record.astro`, `site/src/pages/about.astro`, `site/src/components/MatchCard.astro`, `site/src/components/StalenessBanner.astro`
- Test: `site/tests/build.test.mjs`, `site/tests/fixtures/` (sample JSON in the Task 10 shape)

**Interfaces:**
- Consumes: `fixtures.json`, `track_record.json`, `teams.json`, `meta.json` from Task 10, read at build time from `site/data/`.
- Produces: a static build in `site/dist/` with the five page types from the spec: fixtures, match detail, team pages, track record, about/method.

- [ ] **Step 1: Write failing build tests (run against the sample JSON)**
  - `build succeeds and emits all page types`: `dist/index.html`, `dist/track-record/index.html`, `dist/about/index.html`, at least one `dist/match/*/index.html` and one `dist/team/*/index.html`.
  - `match card shows three probabilities and at most three drivers`.
  - `staleness banner appears when meta.updated_at is older than 48 hours` and is absent when fresh.
  - `about page contains the sentence that football is noisy and probabilities, not certainties, are the product`.
- [ ] **Step 2: Run `npm test` in `site/`. Expected: FAIL.**
- [ ] **Step 3: Implement the pages and components.** Invoke the `frontend-design` skill for visual direction; keep every page working at phone width; show "last updated" on every page.
- [ ] **Step 4: Run `npm test` and `npm run build`. Expected: PASS. Commit.**

### Task 12: Pipelines, workflows, end-to-end and reproducibility tests

**Files:**
- Create: `src/footpred/pipeline/daily.py`, `src/footpred/pipeline/retrain.py`, `.github/workflows/daily.yml`, `.github/workflows/retrain.yml`
- Test: `tests/pipeline/test_e2e.py`, `tests/pipeline/test_reproducibility.py`, `tests/fixtures/mini_dataset/`

**Interfaces:**
- Consumes: everything above
- Produces: `run_daily(data_dir: Path, site_data_dir: Path, client: FixturesClient, today: pd.Timestamp) -> DailyResult` (`DailyResult(new_results: int, scored_fixtures: int, published: bool, warnings: list[str])`): ingest, update ratings, score upcoming fixtures with the champion, log predictions, grade finished matches, export.
- Produces: `run_retrain(data_dir: Path, registry_dir: Path, today: pd.Timestamp, challenger_params: dict | None = None) -> GateResult`: trains a challenger walk-forward, runs the gate, promotes through `set_champion` only if `promote`, and writes a model card for the new version either way.
- Produces: `DailyResult.warnings` also carries `check_drift` alerts, which the workflow surfaces in the run summary.

- [ ] **Step 1: Write failing tests (mini dataset: two seasons, 4 teams per league)**
  - `test_daily_end_to_end_produces_valid_site_data`: with a fake client, `run_daily` returns `published True` and the output JSON passes the Task 10 checks.
  - `test_daily_no_new_results_exits_cheaply_and_keeps_site_data`: with an empty client response, `published False`, no files in `site_data_dir` change, and no scoring runs (Review Focus 5).
  - `test_daily_schema_failure_keeps_last_good_site`: corrupted historical data raises, and `site_data_dir` is byte-identical afterward.
  - `test_retrain_keeps_champion_when_challenger_not_better`: a deliberately degraded challenger leaves `load_champion` unchanged.
  - `test_reproducibility`: running `run_daily` twice for the same `today` with the same data and champion yields identical exported probabilities.
- [ ] **Step 2: Run `pytest tests/pipeline -v`. Expected: FAIL.**
- [ ] **Step 3: Implement both pipelines and the workflows.** `daily.yml` runs once daily and again late on matchday nights, then builds and deploys the site only when `run_daily` succeeds; `retrain.yml` runs weekly. Seed ratings and the first champion by running an initial backtest and `fit_hybrid` once, and record the baseline metrics from Task 5 in the champion's metadata.
- [ ] **Step 4: Run `pytest -v` for the whole suite. Expected: PASS.**
- [ ] **Step 5: Run the full backtest on the development seasons (everything except the most recent full season) and record baseline-versus-hybrid metrics (log loss, RPS, ECE, per league) in `docs/baseline-results.md`. Only if the hybrid beats Elo-only and Dixon-Coles-only there, evaluate it once on the held-out most recent full season, record that result separately, and treat the hybrid as the first release candidate. The hold-out season is not used for tuning. Commit.**
