# sportspred architecture and plugin contract

`sportspred` is one shared platform with a separate plugin per sport. Shared
code never contains sport logic; plugins never re-implement validation, the
ledger, statuses or combos.

```
src/sportspred/
  core/        UTC storage + Pacific/Honolulu display, stable IDs, enums
  storage/     Store: SQLite (events, results, provenance, ledger, prices) + parquet tables
  adapters/    Fetcher (cached, retried, provenance), CSV import, synthetic demo data
  features/    point-in-time primitives: Elo, opponent-adjusted offence/defence ratings, rest/travel
  models/      candidates (base rate, logistic, gradient boosting), production models with bootstrap replicas
  validation/  chronological folds, metrics, calibration, walk-forward backtest, leakage tests
  sports/      one sub-package per sport, each exporting PLUGIN (auto-discovered)
  predict/     pregame engine: checks, FINAL/PROVISIONAL/CONDITIONAL/PASS, confidence labels
  props/       player props (playing time x opportunity x outcome distribution)
  combos/      multi-leg combinations with explicit dependency handling
  ev/          optional, separate price comparison (wraps evmodel); paper tracking only
  ledger/      grading, settlement, reviews, lessons, drift monitoring
  dashboard/   Streamlit app + static HTML export
  cli/         `sportspred` command; cmd_*.py modules are auto-discovered
  coverage.py  coverage table computed from the registry + recorded validation runs
  config.py    Settings (TOML + environment)
  pipeline.py  ingest -> validate -> train -> predict
src/evmodel/   pre-existing price/EV toolkit (odds conversion, de-vig, EV, Kelly, bet ledger)
```

## Non-negotiable rules

1. **No invented data.** If a source is unreachable raise
   `sportspred.sports.DataUnavailable`; never substitute numbers. Synthetic
   data lives only in leagues whose key starts with `demo_` and is flagged
   `is_synthetic`; the store refuses to mix it with real leagues.
2. **Point-in-time features.** A feature of an event may use only information
   from earlier *slates* (Pacific/Honolulu calendar days) plus the public
   schedule. Results of a slate are applied when the slate is over. Columns
   that use later information (for example who actually played) go in
   `FeatureFrame.retrospective_cols` and can only feed candidates flagged
   `uses_retrospective`, which can never be selected or put in production.
3. **As-of semantics.** `feature_frame(store, league, market, now=T)` treats
   every event starting after `T` as upcoming and ignores whatever the store
   knows about it. `validation.leakage.assert_asof_invariance` must pass.
4. **Side A cannot depend on the result.** Home side where one exists;
   otherwise a deterministic rule on IDs (never "winner listed first").
5. **Selection on development folds only.** `run_validation` reads the final
   test period once, after selection. Update rates and other hyper-parameters
   are tuned on burn-in blocks only.
6. **Odds never enter the sports model.** Only `sportspred.ev` reads prices.
   Market benchmarks appear in validation reports as comparisons only.
7. **UTC in storage** (`YYYY-MM-DDTHH:MM:SSZ`), **HST in display**
   (`core.timeutil.fmt_hst`). Naive datetimes are rejected.
8. **Coverage is earned.** `coverage.py` derives every status from ingested
   data and recorded validation runs. Do not hard-code a status.

## Writing a sport plugin

Create `src/sportspred/sports/<sport>/` with `__init__.py` exporting
`PLUGIN = <YourPlugin>()`. Use `sports/basketball/` as the reference
implementation. A plugin implements `sports/base.py::SportPlugin`:

| Method | Contract |
|---|---|
| `leagues` | tuple of `LeagueSpec`; one key per competition level / gender / discipline; never mix them |
| `ingest(store, league, seasons, refresh, fetcher)` | download via `adapters.http.Fetcher`, register every file with `Fetched.register(store)`, resolve participants with `store.resolve_participants`, write `store.upsert_events` / `store.upsert_results` and bulk tables with `store.write_table`; return `IngestReport` |
| `feature_frame(store, league, market, now)` | return a `FeatureFrame` (see below) containing completed and upcoming events as of `now` |
| `candidates(league, market, frame)` | baselines first (`PriorCandidate` with complexity 0, a simple rating model flagged `is_baseline`), then models in increasing `complexity` |
| `split_plan(...)` | `SplitPlan`: blocks (seasons), burn-in, final test blocks |
| `required_checks(...)` | `CheckSpec`s that must be complete for a FINAL status (injury report, starting goalie, probable pitcher, lineup, weather ...) with freshness windows |
| `market_benchmark(...)` | optional market-implied probabilities for comparison (label the timing honestly: opening / closing) |
| `explain(...)` | supporting factors, opponent strength, upset risks, availability assumptions, missing data |
| `apply_availability(...)` | optional: fold published availability into a row, list uncertain players, build worst-case rows |

### FeatureFrame

`df` has one row per event with `event_id`, `start_time_utc` (tz-aware),
`block` (season; sortable; blocks must not overlap in time), `y` (integer
index into `classes`, NaN if no result), `side_a_id`, `side_b_id`, optional
`info_cutoff_utc` (must precede the start) and the feature columns.
Binary markets use `classes=("away","home")` or `("b","a")` so column 1 of a
probability matrix is side A. Three-way markets use `("home","draw","away")`.
Rows for events without a result as of `now` are the upcoming rows the engine
predicts. Add `n_pending_prior` (unplayed earlier games of either side) when
the plugin declares the `prior_games_complete` check.

### Shared primitives

* `features.ratings.elo_pregame / tune_elo` - Elo with optional margin of
  victory, season carry-over, batch semantics.
* `features.ratings.offdef_pregame` - opponent-adjusted offence/defence
  ratings for any per-event quantity (points per 100 possessions, goals, runs,
  serve points won ...); call it with a fast and a slow update rate for
  "recent 5-10 events versus long-term".
* `features.schedule.team_schedule_features` - rest, back-to-backs,
  congestion, travel.
* `models.candidates` - `PriorCandidate`, `logit_candidate`, `gbm_candidate`,
  `SkCandidate` (any scikit-learn classifier; estimators must be picklable).
  Custom models subclass `sports.base.Candidate` (fit on a DataFrame, return
  an `(n, k)` probability matrix).

### Validation, training, prediction

`pipeline.validate_league` runs `validation.backtest.run_validation`
(walk-forward folds, one-standard-error selection on development folds, a
single read of the final test period, baselines, optional market benchmark,
PASS-rule table, confidence table, same-slate residual correlation, gates)
and stores the report plus out-of-sample predictions (`oos_<market>` table).
`pipeline.train_league` fits the selected candidate with bootstrap replicas;
`predict.predict_league` writes immutable ledger records.

## Statuses

Recommendation status (per prediction): FINAL, PROVISIONAL, CONDITIONAL, PASS
- defined in `core/enums.py::REC_STATUS_DEFINITIONS`, decided in
`predict/engine.py`.

Coverage status (per sport/league/market): validated, experimental,
demo-only, adapter-only, unavailable - defined in
`core/enums.py::COVERAGE_DEFINITIONS`, derived in `coverage.py`.
