# Champions League Predictions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show Champions League matches (90-minute home/draw/away chances with reasons) on the site beside the league fixtures.

**Architecture:** A separate European track beside the league pipeline: parse open history, resolve club names, run one pooled Elo across leagues and European matches, train a small calibrated logistic model on the European matches, and merge the scored fixtures into the same daily export. The league models, Elo and Dixon-Coles are not touched.

**Tech Stack:** Python 3.12 (pandas, scikit-learn, existing `footpred` package), pytest, Astro site, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-27-champions-league-design.md`

## Global Constraints

- Predictions are for the **90-minute result**; extra time and penalties never enter a score. In openfootball lines the **first** bracketed score is the 90-minute score (`4-3 pen. 1-1 a.e.t. (1-1, 0-1)`).
- Outcome encoding is 0/1/2 = home/draw/away, as everywhere else.
- No home advantage in the final (neutral venue).
- A club name that cannot be matched must **never stop the daily run**; it is kept at the default rating, flagged, and listed in the run's warnings.
- Point-in-time: a match's ratings reflect only strictly earlier dates.
- All choices are made on development seasons; **2025-26 is evaluated once**, after the model is chosen.
- History comes from `openfootball/champions-league` (CC0) fetched in CI; the site build never touches the network.
- The league pipeline's behaviour and outputs are unchanged when no European data is present.

## Review Focus

- An unmatched club in a live fixture: the daily run still publishes, the club is flagged, the name is in warnings (Task 7).
- A match present in both the history files and the API feed on dates one day apart (UTC vs local): counted once (Task 5).
- The final: no home advantage, `neutral` feature is 1 (Tasks 3 and 6).
- The model fails the publish gate: no European fixtures are published and a warning says why, league fixtures unaffected (Task 7).
- European data present: league-model predictions are identical to a run without it (Task 7).

---

### Task 1: Parse openfootball files

**Files:**
- Create: `src/footpred/european/__init__.py` (empty), `src/footpred/european/parse.py`
- Test: `tests/european/test_parse.py` (create `tests/european/` with no `__init__.py`, like other test dirs)

**Interfaces:**
- Produces: `STAGES` (tuple of `"LEAGUE_STAGE","QUALIFYING","PLAYOFFS","LAST_16","QUARTER_FINALS","SEMI_FINALS","FINAL"`); `COMPETITIONS = {"cl":"CL","clq":"CLQ","el":"EL","elq":"ELQ","conf":"CONF","confq":"CONFQ"}`; `parse_openfootball(text: str, competition: str, season: str) -> pd.DataFrame` with columns `competition, season, stage, date, home, away, home_country, away_country, home_goals, away_goals, result` (raw club names, country as the 3-letter tag, `result` in H/D/A, goals are the 90-minute score). Rows without a score are dropped.

- [ ] **Step 1: Write the failing tests** in `tests/european/test_parse.py`, using small excerpts copied from the real files (in `openfootball/champions-league/2024-25/cl.txt`, if cloned at `/home/claude/openfootball/champions-league`; otherwise from the spec's examples). Tests: `test_regular_match_takes_full_time_score` (`Juventus FC (ITA) v PSV (NED) 3-1 (2-0)` gives 3-1, `H`, countries ITA/NED, stage `LEAGUE_STAGE`); `test_score_without_halftime` (`Bologna FC 1909 (ITA) v FK Shakhtar Donetsk (UKR) 0-0` gives 0-0, `D`); `test_penalties_use_the_ninety_minute_score` (`Liverpool FC (ENG) v Paris Saint-Germain FC (FRA) 1-4 pen. 0-1 a.e.t. (0-1, 0-1)` gives 0-1, `A`); `test_extra_time_draw_is_a_draw_after_ninety` (`Paris Saint-Germain FC (FRA) v Arsenal FC (ENG) 4-3 pen. 1-1 a.e.t. (1-1, 0-1)` gives 1-1, `D`, stage `FINAL`); `test_dates_carry_forward_and_roll_the_year` (a day line `Wed Sep 18` inherits 2024; a January line after December becomes 2025); `test_stage_headers_map` (headers `League, Matchday 1`, `Playoffs, Matchday 1`, `Finals, Round of 16`, `Finals, Quarterfinals`, `Finals, Semifinals`, `Finals, Final` map to the stages in `STAGES`; an old-style `Group A, Matchday 1` header maps to `LEAGUE_STAGE`); `test_unplayed_rows_are_dropped` (a line with no score is absent); `test_qualifier_files_use_qualifying_stage` (any header in a `clq` file gives `QUALIFYING`).
- [ ] **Step 2: Run** `pytest tests/european/test_parse.py -q`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `parse_openfootball` in `src/footpred/european/parse.py`. Line grammar: `▪ <header>` starts a stage; a line starting with a weekday abbreviation (optionally followed by `Mon DD [YYYY]`) sets the date; a match line is `[HH:MM] Home (CTY) v Away (CTY)  <score>` where `<score>` is the first `N-N` after the second team's country tag. Year-less dates inherit the previous year and add one when the month number drops.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Check against every real file.** If `openfootball/champions-league` is cloned, run a throwaway script that parses every `*/*.txt`, asserts no exception, every `stage` is in `STAGES`, and prints the number of rows per file. Expected: about 125 rows for `cl` in 2011-12 to 2023-24 and 189 for 2024-25 and 2025-26. Add any unmapped header to the mapping and the header test.
- [ ] **Step 6: Commit** `git add src/footpred/european tests/european && git commit -m "Parse openfootball European results (90-minute scores)"`

### Task 2: Club-name resolver

**Files:**
- Create: `src/footpred/european/names.py`, `data/european_aliases.csv` (columns `alias,canonical`, header only plus known exceptions)
- Test: `tests/european/test_names.py`

**Interfaces:**
- Consumes: `footpred.ingest.teams.canonical_team`, `UnknownTeamError`, `covered_leagues` (existing; league alias table `data/team_aliases.csv`).
- Produces: `COUNTRY_TO_LEAGUE = {"ENG":"E0","ESP":"SP1","ITA":"I1","GER":"D1","FRA":"F1"}`; `GROUPS = ["E0","SP1","I1","D1","F1","OTHER"]`; `group_for_country(country: str | None) -> str`; `normalize_key(name: str) -> str`; `class EuropeanNames` with `__init__(self, euro_aliases_path: Path | None = None, league_aliases_path: Path | None = None)`, `resolve(self, name: str, country: str | None = None) -> str` (never raises), `canonical(self, name: str, league: str) -> str` (same as `resolve`, signature used as the fetcher hook), and the attribute `unmatched: set[str]` (raw names that fell through).

- [ ] **Step 1: Write the failing tests:** `test_big_five_clubs_resolve_to_league_canonical_names` (`"FC Bayern München"` with `"GER"` gives the same name `canonical_team("FC Bayern München", "D1")` gives; use names present in `data/team_aliases.csv`, after adding any missing openfootball spelling to `data/european_aliases.csv`); `test_country_restricts_the_league_tables` (an ambiguous short name resolves within the tagged league only); `test_normalized_key_matches_suffix_variants` (`normalize_key("Sport Lisboa e Benfica")`, `"SL Benfica"` and `"Benfica"` agree once the alias `Sport Lisboa e Benfica -> Benfica` is in the CSV, and `normalize_key("FC Barcelona") == normalize_key("Barcelona")`); `test_unknown_club_is_kept_flagged_and_never_raises` (`resolve("Qarabağ FK", "AZE")` returns a clean name such as `"Qarabağ FK"`, adds it to `unmatched`); `test_same_unknown_name_resolves_the_same_each_time`.
- [ ] **Step 2: Run** `pytest tests/european/test_names.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `EuropeanNames.resolve` with this order: exact alias in the European CSV, then the league alias tables (only the tagged country's league if a country is given, otherwise all of `covered_leagues`), then a `normalize_key` match against all known canonical names, then fallback to the whitespace-cleaned input recorded in `unmatched`. `normalize_key`: casefold, strip accents, drop the tokens `fc cf ac sc afc fk sk ss as ssc us rc rcd sv vfb vfl tsg 1909 1899` and punctuation, collapse spaces.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Real-name check.** In a throwaway script, parse the 2025-26 and 2024-25 `cl.txt` files and resolve every raw name with its country. Expected: every club tagged ENG/ESP/ITA/GER/FRA resolves to a canonical name that exists in `data/team_aliases.csv` (`unmatched` contains no big-five club); add exceptions to `data/european_aliases.csv` until that holds and add a regression test listing the exceptions you needed.
- [ ] **Step 6: Commit** `git add src/footpred/european data/european_aliases.csv tests/european && git commit -m "Resolve European club names to canonical names"`

### Task 3: Pooled Elo across leagues and European matches

**Files:**
- Create: `src/footpred/european/pooled.py`
- Test: `tests/european/test_pooled.py`

**Interfaces:**
- Consumes: `footpred.ratings.elo._margin_multiplier`, `expected_home_score` (existing); `GROUPS` from Task 2.
- Produces: `@dataclass(frozen=True) PooledParams(k: float = 20.0, home_adv: float = 60.0, season_regress: float = 0.25, initial: float = 1500.0, offsets: Mapping[str, float] = zeros for GROUPS)`; `run_pooled_elo(matches: pd.DataFrame, team_group: dict[str, str], params: PooledParams = PooledParams()) -> tuple[pd.DataFrame, dict[str, float]]` where `matches` has `match_id, date, season, home, away, home_goals, away_goals, neutral` and the result is a frame indexed by `match_id` with `elo_home_pre, elo_away_pre`, plus final ratings; `fit_offsets(matches, team_group, train_end: pd.Timestamp, base: PooledParams = PooledParams()) -> dict[str, float]`.

- [ ] **Step 1: Write the failing tests:** `test_pre_ratings_ignore_same_day_and_future_results` (changing a result on day D leaves every pre-rating on days ≤ D unchanged); `test_cross_group_result_moves_both_clubs_the_right_way` (a OTHER-group club that beats an E0 club gains, the E0 club loses, by the same amount when both start equal); `test_new_club_starts_at_its_group_offset` (offsets `{"E0": 0, "OTHER": -100}` gives an unseen OTHER club `initial - 100`); `test_neutral_match_has_no_home_advantage` (equal clubs, `neutral=True`, expected rating change is symmetric); `test_unplayed_rows_get_pre_ratings_but_never_update`; `test_fit_offsets_recovers_a_weaker_group` (synthetic data where OTHER clubs lose to E0 clubs more than equal ratings predict: fitted `offsets["OTHER"] < 0`; E0 offset is fixed at 0).
- [ ] **Step 2: Run** `pytest tests/european/test_pooled.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `run_pooled_elo` as a day-batched forward pass like `footpred.ratings.elo.run_elo` (pre-ratings for a whole day recorded before applying that day's results; season-boundary regression of 25% toward the group mean for clubs seen last season; home advantage zero when `neutral`). Implement `fit_offsets` as a coordinate search over the offsets of the five non-anchor groups on a grid from -300 to 300 in steps of 25, two passes, minimising the mean squared error between expected home score and actual score (1, 0.5, 0) on played matches before `train_end` whose two clubs are in different groups. E0 is the anchor at 0.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `git add src/footpred/european/pooled.py tests/european/test_pooled.py && git commit -m "Pooled Elo across leagues and European matches"`

### Task 4: Fixtures client learns the Champions League

**Files:**
- Modify: `src/footpred/ingest/fixtures.py` (the `_COMPETITION_CODES` table, the normalised row dict in `FootballDataOrgClient.get_fixtures`, `_canonical_pair`, `fetch_upcoming`, `fetch_latest_results`)
- Test: `tests/ingest/test_fixtures.py` (existing; extend)

**Interfaces:**
- Consumes: `EuropeanNames.canonical(name, league)` (Task 2) as the optional hook.
- Produces: `_COMPETITION_CODES["CL"] = "CL"`; normalised rows gain `stage` (football-data.org's `stage` string mapped to Task 1's `STAGES`: `LEAGUE_STAGE`, `PLAYOFFS`, `LAST_16`, `QUARTER_FINALS`, `SEMI_FINALS`, `FINAL`; anything else `None`); `fetch_upcoming(..., canonical: Callable[[str, str], str] | None = None)` and `fetch_latest_results(..., canonical: ...)` use `canonical` instead of `canonical_team` when given; the upcoming frame gains a `stage` column (NaN for leagues) and the results frame keeps its columns.

- [ ] **Step 1: Write the failing tests** with the existing fake HTTP client style: `test_champions_league_fixtures_are_fetched_and_staged` (a fake CL payload with `stage: "FINAL"` yields `league == "CL"`, `stage == "FINAL"`); `test_canonical_hook_is_used_and_never_raises_for_unknown_clubs` (an unknown club passes through `EuropeanNames.canonical` and is not a `UnknownTeamError`; with no hook the old error is raised); `test_league_fixtures_have_no_stage`. Update any existing assertion on the upcoming frame's exact columns.
- [ ] **Step 2: Run** `pytest tests/ingest/test_fixtures.py -q`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement** the changes listed under Produces.
- [ ] **Step 4: Run** `pytest tests/ingest -q`. Expected: PASS (all).
- [ ] **Step 5: Commit** `git add src/footpred/ingest/fixtures.py tests/ingest && git commit -m "Fixtures client: Champions League code, stage, club-name hook"`

### Task 5: European data store and download

**Files:**
- Create: `src/footpred/european/history.py`
- Modify: `src/footpred/pipeline/cli.py` (the `download` command)
- Test: `tests/european/test_history.py`, `tests/pipeline/test_cli.py` (extend)

**Interfaces:**
- Consumes: `parse_openfootball`, `COMPETITIONS` (Task 1); `EuropeanNames` (Task 2).
- Produces: `EURO_COLUMNS = ["match_id","competition","season","stage","date","home","away","home_country","away_country","home_goals","away_goals","result"]` where `match_id` is `f"{competition}-{date:%Y%m%d}-{home}-{away}"` using canonical names; `load_european_history(openfootball_dir: Path, names: EuropeanNames) -> pd.DataFrame`; `append_european_results(data_dir: Path, new: pd.DataFrame) -> None` (atomic write to `data_dir/"european_results.csv"`, dropping duplicate `match_id`); `load_european_matches(data_dir: Path, names: EuropeanNames) -> pd.DataFrame` (history from `data_dir/"openfootball"` plus the results file; a feed row is dropped when the history has the same `competition`, `home`, `away` within 3 days); `download_openfootball(dest: Path, run=subprocess.run) -> bool` (shallow `git clone` of `https://github.com/openfootball/champions-league`, or `git pull` if present; `False` on failure).

- [ ] **Step 1: Write the failing tests:** `test_history_loads_every_competition_file_and_canonicalises` (a temp dir with a `2024-25/cl.txt` and `2024-25/el.txt` excerpt gives rows with `competition` `CL` and `EL`, canonical names, 90-minute goals); `test_feed_row_one_day_off_is_not_double_counted` (history has Roma v Inter on D, the results file has it on D+1: one row remains, the history one); `test_feed_row_far_apart_is_kept` (same pair 20 days apart, e.g. a second leg: both kept); `test_append_is_idempotent_and_atomic` (appending twice leaves one row; no `.tmp` file remains); `test_download_reports_failure_without_raising` (a fake `run` raising `OSError` returns `False`); in `tests/pipeline/test_cli.py`, `test_download_command_survives_a_failed_openfootball_clone`.
- [ ] **Step 2: Run** `pytest tests/european/test_history.py tests/pipeline/test_cli.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the functions; derive `season` from `footpred.ingest.historical.season_label`. In the CLI, after the CSV download call `download_openfootball(data_dir / "openfootball")` and print a `::warning title=openfootball::` line when it returns `False`, without changing the exit code.
- [ ] **Step 4: Run** the same command plus `pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** `git add src/footpred tests && git commit -m "European history store, feed results file and download"`

### Task 6: European model, benchmark and publish gate

**Files:**
- Create: `src/footpred/european/model.py`, `src/footpred/european/benchmark.py`
- Modify: `src/footpred/pipeline/cli.py` (new command `benchmark-european`), `.github/workflows/benchmark.yml` (run it and commit `docs/european-results.md`)
- Test: `tests/european/test_model.py`, `tests/european/test_benchmark.py`; extend `tests/pipeline/conftest.py` only if a fixture is needed for European data

**Interfaces:**
- Consumes: `run_pooled_elo`, `fit_offsets`, `PooledParams` (Task 3); `load_european_matches` (Task 5); `footpred.pipeline.common.load_all_matches`; `footpred.evaluate.backtest.backtest`; `footpred.evaluate.baselines.baseline_base_rate`, `baseline_elo`; `footpred.models.meta_model.MetaModel`; `footpred.models.calibration.Calibrator`; metrics from `footpred.evaluate.metrics` (including `accuracy`).
- Produces: `EURO_FEATURES = ["elo_diff", "neutral", "knockout"]`; `european_features(matches: pd.DataFrame, pre: pd.DataFrame, home_adv: float = 60.0) -> pd.DataFrame` indexed by `match_id` (`elo_diff` = home pre-rating − away pre-rating + `home_adv` unless `neutral`; `neutral` = 1.0 when `stage == "FINAL"`; `knockout` = 1.0 for `PLAYOFFS` and later stages); `fit_european(X_train, y_train, X_cal, y_cal) -> tuple[MetaModel, Calibrator]` (logistic backend `{"backend":"logistic","C":0.1}` and Platt calibration); `predict_european(model, calibrator, X) -> np.ndarray` (n×3); `european_publish_gate(development: dict) -> tuple[bool, str]` (true only when the model's log loss is below both the base rate's and the pooled-Elo baseline's; the string states the comparison); `run_european_benchmark(data_dir: Path, out_path: Path, aliases_path: Path | None = None, today: pd.Timestamp | None = None, test_seasons: int = 3, step_days: int = 60) -> dict` returning `{"development", "holdout", "holdout_season", "publish_ok", "reason"}` and writing `out_path`.

- [ ] **Step 1: Write the failing tests:** `test_features_final_is_neutral_and_has_no_home_advantage` (two equal clubs: `elo_diff == 0` in a `FINAL`, `== 60` in a league-stage match); `test_knockout_flag_by_stage`; `test_fit_and_predict_return_probabilities_that_sum_to_one`; `test_gate_requires_beating_both_baselines` (three constructed metric dicts: beats both → true; loses to base rate → false; loses to Elo baseline → false); `test_benchmark_writes_report_and_respects_holdout_rule` (synthetic European history in a temp `data_dir/openfootball` plus synthetic league CSVs from the existing pipeline test world; the report lists base rate, pooled-Elo baseline and model with correct picks and log loss; the hold-out section appears only when `publish_ok`; `holdout_season == "2025-26"`).
- [ ] **Step 2: Run** `pytest tests/european/test_model.py tests/european/test_benchmark.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.** Benchmark data flow: league matches (`competition` = league code) plus European matches, `team_group` from the club country tags and league membership, `fit_offsets` with `train_end` at the start of the first test season, `run_pooled_elo`, features for the **European competitions only**, walk-forward with `backtest(...)` over the last `test_seasons` development seasons (everything before 2025-26), baselines through the same `backtest`. Write the table in the format of `docs/baseline-results.md` with a "Correct picks" column. Evaluate the hold-out season once, only when `european_publish_gate` passes on development.
- [ ] **Step 4: Run** the tests, then `pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** `git add src .github tests && git commit -m "European model, walk-forward benchmark and publish gate"`

### Task 7: Retrain and daily integration

**Files:**
- Create: `src/footpred/pipeline/european.py`
- Modify: `src/footpred/pipeline/daily.py` (`run_daily` gains `european: bool = False`), `src/footpred/pipeline/retrain.py` (or a new `run_european_retrain` in `pipeline/european.py`), `src/footpred/pipeline/cli.py` (`daily` and `retrain` commands), `src/footpred/publish/export.py` only if the new `stage` or a `competition` field must be added to `fixtures.json`
- Test: `tests/pipeline/test_european_e2e.py`

**Interfaces:**
- Consumes: Tasks 2 to 6; `footpred.models.registry` (`save_model`, `set_champion`, `load_champion`), used with the registry directory `registry_dir / "european"` and the same metadata keys as `run_retrain`; `footpred.pipeline.common.append_api_results` is **not** used for European rows.
- Produces: `run_european_retrain(data_dir: Path, registry_dir: Path, today: pd.Timestamp, aliases_path: Path | None = None) -> GateResult` (trains on all played European matches before `today` with the 70/15/15 time blocks that `run_retrain` uses, promotes through `should_promote`, first model unconditional only when `european_publish_gate` passes); `score_european(data_dir: Path, client: FixturesClient, today: pd.Timestamp, registry_dir: Path, aliases_path: Path | None = None, sleep=time.sleep) -> EuropeanScores` where `EuropeanScores` is a dataclass `(fixtures: pd.DataFrame, drivers: dict, grids: dict, results: pd.DataFrame, warnings: list[str], published: bool)`; `run_daily(..., european=True)` merges `fixtures`, `drivers` and `grids` into the league `scored` frame and dicts before logging and export, passes `results` to `append_european_results` and to the graded-log inputs, and adds `warnings`.

- [ ] **Step 1: Write the failing tests** in `tests/pipeline/test_european_e2e.py`, built on the pipeline `world` fixture plus synthetic European history and a fake client that also serves a `CL` fixture: `test_daily_publishes_a_champions_league_fixture_with_reasons` (the exported `fixtures.json` contains it with `league == "CL"`, three probabilities summing to 1 and at most three drivers); `test_unmatched_club_does_not_stop_the_run_and_is_warned` (fixture with an unknown club still publishes the others, the name appears in `DailyResult.warnings`); `test_failed_publish_gate_hides_european_fixtures_only` (champion absent or gate false: no `CL` rows in `fixtures.json`, league rows present, a warning mentioning the gate); `test_league_predictions_are_unchanged_by_european_data` (the league rows' probabilities with `european=True` and European data equal those with `european=False`); `test_european_results_are_graded_in_the_track_record` (a logged CL prediction and a later feed result yield a graded row with `league == "CL"`); `test_retrain_promotes_first_model_only_when_gate_passes`.
- [ ] **Step 2: Run** `pytest tests/pipeline/test_european_e2e.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `run_european_retrain` and `score_european` (fetch `CL` fixtures and results with `canonical=names.canonical`; build pooled ratings over league plus European matches including the upcoming rows with NaN goals; features through `european_features`; calibrated probabilities; reasons through `footpred.explain.reasons.top_drivers` with the logistic model; no Dixon-Coles grid for European rows, so the goals section is simply absent). Wire `european=True` in the CLI `daily`; the weekly `retrain` command also runs `run_european_retrain` and prints its result.
- [ ] **Step 4: Run** the new tests, then `pytest -q`. Expected: PASS, with the earlier 163 tests unchanged.
- [ ] **Step 5: Commit** `git add src tests && git commit -m "Daily and retrain run the Champions League track"`

### Task 8: Site shows the Champions League

**Files:**
- Modify: `site/src/lib/format.mjs` (`LEAGUES` gains `CL: "Champions League"`), `site/src/components/MatchCard.astro` (note and tag), `site/tests/format.test.mjs`, `site/tests/build.test.mjs`, `site/tests/preview-data/fixtures.json`, `site/scripts/preview.mjs`
- Test: the files above

**Interfaces:**
- Consumes: `fixtures.json` rows with `league == "CL"`.
- Produces: `leagueName("CL") === "Champions League"`; `isExperimental(trackRecord, league) -> boolean` in `site/src/lib/format.mjs` (true when `league === "CL"` and `trackRecord.by_league.CL?.n ?? 0` is below 30); European cards show the text "90-minute result" and, while `isExperimental` is true, an "Experimental" tag (`MatchCard.astro` gets an `experimental` prop that the pages compute from `trackRecord`).

- [ ] **Step 1: Write the failing tests:** in `site/tests/format.test.mjs` `leagueName("CL") === "Champions League"` and `isExperimental` (CL with 29 graded: true; with 30: false; a league code: false; missing `by_league`: true for CL); in `site/tests/build.test.mjs` a fixture with `league: "CL"` in `site/tests/fixtures/fixtures.json` produces a `data-league="CL"` chip, the text "Champions League", "90-minute result" and "Experimental" in `index.html`; add a CL match to `site/tests/preview-data/fixtures.json` and, in `site/scripts/preview.mjs`, a check that the Champions League chip narrows the list to that match.
- [ ] **Step 2: Run** `cd site && node --test tests/format.test.mjs`. Expected: the new format test FAILS (build tests run in CI only).
- [ ] **Step 3: Implement** the label, the card note and the tag (reuse the existing badge styles).
- [ ] **Step 4: Run** `node --test tests/format.test.mjs tests/filters.test.mjs`. Expected: PASS.
- [ ] **Step 5: Commit** `git add site && git commit -m "Site: Champions League label, note and chip"`

### Task 9: Roll out and verify on real data

**Files:**
- Modify: `.github/workflows/daily.yml`, `.github/workflows/retrain.yml` (state save lists gain `data/european_results.csv` and `data/openfootball` is **not** saved, since CI re-downloads it), `.github/workflows/probe.yml` (delete it), `docs/european-results.md` (written by the workflow)

**Interfaces:**
- Consumes: Tasks 1 to 8.

- [ ] **Step 1:** Push, then run the `benchmark` workflow. Read `docs/european-results.md` from the commit it makes. Expected: base rate, pooled-Elo baseline and model rows with correct picks and log loss; the gate line states whether the model beats both baselines. If the gate fails, stop and report; do not publish.
- [ ] **Step 2:** Run `retrain`, then `daily`, then `preview`. Expected: all succeed; the daily notice lists a Champions League fixture; the daily warnings list any unmatched clubs (fix aliases with a regression test if a big-five club is listed).
- [ ] **Step 3:** Read the `preview` screenshots and log. Expected: the Champions League chip and card render, filters still pass.
- [ ] **Step 4:** Tell the user the backtest accuracy against the base rate, what the gate decided, which clubs were unmatched, and that European cards are labelled experimental.
- [ ] **Step 5: Commit** any alias fixes and `git rm .github/workflows/probe.yml`.
