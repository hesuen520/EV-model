# Status: what works, what is a placeholder, what comes next

As of 2026-11-13. "Works" means it was run here on real data or is covered by
tests that were run; anything else is listed as not run or not built.

## Working (run here on real data)

| Area | State |
|---|---|
| Shared foundations | UTC storage with Pacific/Honolulu display; stable IDs; entity matching by provider ID and normalised name; cross-provider event de-duplication; provenance on every row (URL, collection time, publication time, content hash); event change log; result corrections kept; append-only, hash-chained prediction ledger; synthetic data isolated under `demo_*` keys |
| Data adapters (public datasets, no key) | NBA, WNBA, NCAA men's and women's basketball; NFL; college football; NHL; MLB (through 2025); men's and women's international soccer; Premier League, La Liga, Bundesliga, Serie A, Ligue 1; tennis singles at eight levels (ATP and WTA main draw and qualifying, Challenger, WTA 125, ITF men and women; historical results to mid-2026, non-commercial licence); the official NBA injury report archive |
| Validation | 26 league-markets validated on real data with the walk-forward protocol and a reserved final test: 17 pass every gate, 9 are experimental (NFL spread and total, WNBA, ITF men, men's international soccer, Premier League, La Liga, Bundesliga, Serie A). Most were run more than once after fixes; every run is listed in the validation report's run history. As-of leakage test passes with zero difference for every plugin |
| Prediction engine | FINAL / PROVISIONAL / CONDITIONAL / PASS with pregame checks, freshness measured from publication time, availability applied (basketball rotation, hockey goalie), bootstrap model-uncertainty intervals, validated confidence labels, explanations; 470 real pregame predictions recorded over two days of slates (369 PROVISIONAL, 101 PASS, none FINAL); the ledger's hash chain verifies |
| Combos | pairwise dependency classification, three-leg priority, alternatives, PASS message, provisional preview, historical-slate backtest |
| Ledger | grading against results, regrade chain after corrections, reviews (variance / model error / data issue / availability change), lessons with a validation-run requirement, calibration and input drift monitor |
| CSV import | events, results, availability, check records; whole-file validation with line numbers |
| EV module | CSV price import, raw implied / no-vig / fair odds / edge / EV, fee schedules, settlement rules, eligibility presets, paper positions and resettlement |
| Dashboard and reports | static self-contained HTML dashboard; generated coverage table and validation results with charts |
| CLI | `coverage, ingest, import, templates, validate, train, predict, slate, combos, grade, review, lessons, monitor, live, ev, dashboard, report, ledger, demo` |

## Works in tests only (not exercised on real events yet)

- **FINAL and CONDITIONAL statuses on real slates.** The logic is tested end
  to end on synthetic data and was checked on a scratch copy of the real
  store, but no real prediction has reached FINAL: the newest official NBA
  report in the public archive is too old to satisfy the three-hour re-check
  rule, and no other availability feed is reachable here.
- **Grading, reviews, lessons and monitoring on real results.** Tested, but
  no real prediction has been graded yet. The first were recorded on
  2026-11-12. When this was written (2026-11-13, 05:30 UTC) the public result
  files had not yet published the previous evening's NBA, NHL and college
  games; the one predicted game with a result on file (an NFL game) had only
  PASS predictions, which are deliberately not graded. Run `sportspred ingest
  ... && sportspred grade && sportspred review` once results are published.
- **EV evaluation with real prices.** Maths and rules are tested against hand
  calculations; no real quotes were available.

## Written but not run against the live service

- ESPN adapter (`sportspred live injuries`, `live scoreboard`): host blocked
  here; fixture-tested only.
- The Odds API client and parser (`sportspred ev import-prices --odds-api`):
  host blocked here; written from the documentation; fixture-tested only.

## Not built

- **Player props.** Not implemented in this platform (the coverage table says
  unavailable). The earlier `evmodel` package still has its own prop
  projection and scanner.
- **NBA / WNBA / college spreads and totals, NHL and MLB totals.** No
  historical lines exist in the reachable sources; not implemented.
- **Golf and motorsport.** Declared so the coverage table is complete; they
  need a finish-distribution model for fields of competitors.
- **Tennis doubles, and any live tennis feed.** Singles are validated on
  historical results; there is no schedule, order of play or withdrawal feed,
  so tennis produces predictions only for events imported by CSV, and those
  stay PROVISIONAL until newer results are imported too.
- **Sport-specific combat, cricket, rugby, volleyball, racket-sport and
  esports models.** These leagues run on a results-only rating model via CSV
  import and are adapter-only.
- **Same-game combos.** Rejected by design until a joint model exists.
- **Kalshi and Polymarket API clients, closing-line value, dead-heat
  settlement.**
- **An interactive (server-based) dashboard.** The dashboard is a static file
  regenerated on demand.
- Full per-plugin unit-test suites for football, hockey, baseball and soccer.
  They have regression tests for the defects the audit found, the as-of
  leakage test and one real-data validation run per market, but no broad
  dedicated test files.

## How this was built, and what was independently reviewed

The plan was to have each unit built and then audited by an independent
reviewer. A usage limit ended that work twice, and two worker restarts lost
reviews in progress. What actually happened:

- **Audited independently (line by line, by a separate reviewer):** the
  hockey, football, baseball and soccer plugins. Outcome: **no leakage and no
  critical defect in any of them.** The reviewers found robustness and
  live-path defects, which were fixed and given regression tests: a rating
  update that could diverge when one team had many events in a single batch;
  a partial table write that dropped other events' rows; goalie name matching
  and stale goalie confirmations in hockey; the listed NFL quarterback not
  always being the one who played; duplicate college football records; club
  kick-off times lost by date-format inference; a renamed club being stored
  twice; stale fixture placeholders; draw selections explained with the home
  side's reasons; and an event starting exactly at the prediction time
  belonging to neither the completed nor the upcoming set.
- **Found and not fixed** (listed under known limitations in the model card):
  MLB's live path is unexercised and its time-zone travel feature ignores the
  date line; tied MLB games are dropped; NFL international games get ordinary
  road-game travel; the NHL adapter refreshes only the current season by
  default; a renamed soccer club stops the ingest instead of being matched.
- **Not independently audited:** basketball, tennis, the results-only model,
  and the shared core (store, validation protocol, prediction engine, combos,
  ledger, EV module, dashboard). These were written and tested by the main
  author only. An audit of the documentation's claims against the code was
  started and lost in a restart; it was **not** re-run. The author's own pass
  for this status found and fixed one overstated claim - the "results
  current" check did not recognise a source that had simply stopped (tennis,
  MLB) - and added a run history because most final-test periods have been
  read more than once.

## Tests run

`pytest` on 2026-11-13: **4,362 passed, 13 expected failures** (the expected
failures are in the earlier `evmodel` suite), of which 123 are the
`tests/sportspred` suite: time zones and HST display, stable IDs and
de-duplication, ledger immutability and tamper detection, synthetic isolation,
point-in-time availability, as-of leakage, chronological folds, retrospective
columns, stale and missing checks (including a source that has stopped
publishing), CSV validation, the results-only builder, tennis normalisation
and features, the official injury report parser, combo dependencies and the
three-leg rule, grading and resettlement, EV maths and settlement rules, the
wall between predictions and prices, dashboard rendering, regression tests
for every audit fix, and an end-to-end smoke test.

Also run on 2026-11-13, on real data: ingest, validation and training for all
26 league-markets (logs under `data/logs/`, which is not committed), the NBA
combo backtest, predictions and combos for the current slates, and
`sportspred ledger verify`.

## Next steps, in order of value

1. **Backtest NBA availability honestly.** The archive of as-published NBA
   injury reports (2018 on) allows the roster features to be validated with
   what was actually known before each game. This is the largest missing
   piece of realism.
2. **Collect what history cannot give.** Snapshot injury reports, starting
   goalies, probable pitchers, line-ups and prices with collection timestamps
   from now on, so future validation can use them.
3. **A timestamped market benchmark.** Odds snapshots at prediction time, to
   test whether any edge exists against an executable price.
4. **Tennis for live use**: a schedule / order-of-play and withdrawal feed
   with real match dates, so in-tournament form and fatigue can be used and
   predictions can be made; a commercially licensed results source if the
   platform is ever used commercially; doubles.
5. **Player props and NBA spreads/totals**, with playing-time, opportunity and
   outcome-distribution models, validated against real lines once a line
   history exists.
6. **Finish-distribution models** for golf and motorsport.
7. **Independent review** of what has not had one: basketball, tennis, the
   shared core, and the documentation's claims; fix the audit items listed as
   known and not fixed; broader per-plugin test suites.
8. Grade and review the first real slates; turn review patterns into lessons
   and test them before promoting any change.
