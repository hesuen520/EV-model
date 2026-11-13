# Status: what works, what is a placeholder, what comes next

As of 2026-11-13. "Works" means it was run here on real data or is covered by
tests that were run; anything else is listed as not run or not built.

## Working

The last column says what the claim rests on: **real data** means it was run
here on real events and the result is in the store; **tests** means it is
covered by the test suite but has not yet happened with real events.

| Area | State | Rests on |
|---|---|---|
| Shared foundations | UTC storage with Pacific/Honolulu display; stable IDs; entity matching by provider ID and normalised name; cross-provider event de-duplication; a source record behind every event and result (URL, collection time, content hash, and the publication time where the source states one); event change log; result corrections kept; append-only, hash-chained prediction ledger; synthetic data isolated under `demo_*` keys | real data |
| Data adapters (public datasets, no key) | NBA, WNBA, NCAA men's and women's basketball; NFL; college football; NHL; MLB (through 2025); men's and women's international soccer; Premier League, La Liga, Bundesliga, Serie A, Ligue 1; tennis singles at eight levels (historical results to mid-2026, non-commercial licence); the official NBA injury report archive | real data |
| Validation | @@VALIDATION_SUMMARY@@ | real data |
| Point-in-time features | the as-of leakage test passes with zero difference for all 26 league-markets on the real store (`sportspred check-leakage`, and the opt-in test `tests/sportspred/test_real_store_leakage.py`) | real data |
| Prediction engine | PROVISIONAL and PASS statuses with pregame checks, availability applied, bootstrap model-uncertainty intervals, validated confidence labels, explanations: @@PRED_SUMMARY@@ | real data |
| FINAL and CONDITIONAL statuses | every check complete and fresh; a selection that depends on a doubtful player | tests |
| Combos | the PASS message and the provisional preview on real slates; the per-league backtest of the selection rule | real data |
| Combos that are recommended | a FINAL three-leg combo, alternatives, larger sizes, dependency handling | tests |
| Ledger: grading, reviews, lessons, drift monitor | grading against results, regrade chain after corrections, reviews (variance / model error / data issue / availability change), lessons with a validation-run requirement | tests |
| CSV import | events, results, availability, check records; whole-file validation with line numbers | tests |
| EV module | CSV price import, raw implied / no-vig / fair odds / edge / EV, fee schedules, settlement rules, eligibility presets, paper positions and resettlement | tests |
| Dashboard and reports | static self-contained HTML dashboard; generated coverage table and validation results with charts | real data |
| CLI | `coverage, ingest, import, templates, validate, train, check-leakage, predict, slate, combos, grade, review, lessons, monitor, live, ev, dashboard, report, ledger, demo` | real data for the commands above; tests for the rest |

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

Each unit was to be built and then audited by a separate reviewer. A usage
limit ended that work twice and two restarts lost reviews in progress, so the
audits were run later, one area at a time. Every reviewer was read-only and
had to prove a defect by experiment. What they covered and found:

**1. Hockey, football, baseball and soccer plugins.** No leakage and no
critical defect. Robustness and live-path defects were fixed: a rating update
that could diverge when one team had many events in a single batch; a partial
table write that dropped other events' rows; goalie name matching and stale
goalie confirmations; the listed NFL quarterback not always being the one who
played; duplicate college football records; club kick-off times lost by
date-format inference; a renamed club being stored twice; stale fixture
placeholders; draw selections explained with the home side's reasons; an
event starting exactly at the prediction time belonging to neither the
completed nor the upcoming set.

**2. The shared core** (validation protocol, prediction engine, store and
ledger, grading and reviews, combos, time handling). *Confirmed correct:* all
recorded validation runs have training blocks before the calibration block
before the test block, no final block appears in a development fold, the
gates recompute to the stored values, no retrospective candidate was
selected, and the selected models' metrics recomputed from the stored
out-of-sample tables match the reports; every production model was calibrated
on blocks later than its training blocks; no recorded prediction was created
at or after its event's start. *Found and fixed* (each with a regression
test):

- **critical** - three-way results were graded as draws whenever the graded
  batch mixed leagues (a missing overtime flag was read as "true"). Nothing
  had been graded yet, so no wrong grade was recorded;
- a player dropped from a newer injury report stayed "out" indefinitely;
- combos could use legs on cancelled or moved events, and superseded
  predictions (a prediction at an old line stayed current);
- events imported without a provider ID could not be updated, and a stale
  schedule row could reopen a finished game;
- over/under and draw picks were never CONDITIONAL;
- one final event without a result aborted the whole grading run;
- publishing a kick-off time voided earlier predictions as "rescheduled";
- the ledger accepted any stated creation time. For a real event it must now
  match the machine's clock, which is stored and hashed with the row;
- the validation label could be attached to a trained model that the newest
  validation run had not selected;
- smaller: reviews not refreshed after a regrade; a changed line not
  recorded; independence accepted on a wide interval; a candidate able to
  read an undeclared column; date-only events on the wrong slate, and
  predictable after their date had begun east of Greenwich; real and
  synthetic legs combinable; "Last check" showing the prediction time.

*Found and not fixed:* a multi-game series or doubleheader arriving from a
second provider with dates only can be stored twice (logged as an ambiguous
match); home and away swapped at a neutral site are two events; combos are
append-only but not hash-chained; removing the newest rows of the ledger
cannot be detected from inside the database (`sportspred ledger verify`
prints the head hash to keep elsewhere); availability statements have no
maximum age beyond the freshness of their check.

**3. The documentation's claims against the code and the store.** The numbers
in the README, model card and this file matched the recorded runs. Sixteen
claims about behaviour were wrong or overstated and were corrected - in the
code where the code fell short (report publication time for CSV availability,
sport-specific settlement rules, a hard stop on price columns as model
inputs, the missing regression tests) and in the text otherwise (freshness,
"untouched" final tests, what the combo backtest covers, the NFL live
quarterback input, packages that did not exist).

@@LEAK_AUDIT@@

**Not independently audited:** the results-only CSV model beyond its feature
builder, the EV module's maths, the dashboard, and the adapters written for
hosts that could not be reached (ESPN, The Odds API). Whether any rule,
candidate list or gate was ever changed because of a final-test number cannot
be proved from the record; the run history shows what every run produced.

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
