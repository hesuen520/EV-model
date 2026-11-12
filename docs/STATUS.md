# Status: what works, what is a placeholder, what comes next

As of 2026-11-12. "Works" means it was run here on real data or is covered by
tests that were run; anything else is listed as not run or not built.

## Working (run here on real data)

| Area | State |
|---|---|
| Shared foundations | UTC storage with Pacific/Honolulu display; stable IDs; entity matching by provider ID and normalised name; cross-provider event de-duplication; provenance on every row (URL, collection time, publication time, content hash); event change log; result corrections kept; append-only, hash-chained prediction ledger; synthetic data isolated under `demo_*` keys |
| Data adapters (public datasets, no key) | NBA, WNBA, NCAA men's and women's basketball; NFL; college football; NHL; MLB (through 2025); men's and women's international soccer; Premier League, La Liga, Bundesliga, Serie A, Ligue 1; the official NBA injury report archive |
| Validation | 18 league-markets validated on real data with the walk-forward protocol and a reserved final test: 11 pass every gate, 7 are experimental. As-of leakage test passes with zero difference for every plugin |
| Prediction engine | FINAL / PROVISIONAL / CONDITIONAL / PASS with pregame checks, freshness measured from publication time, availability applied (basketball rotation, hockey goalie), bootstrap model-uncertainty intervals, validated confidence labels, explanations; 326 real pregame predictions recorded for the current slates |
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
- **Grading, reviews, lessons and monitoring on real results.** Tested; the
  first real predictions were recorded on 2026-11-12 and had not been played
  when this was written. Run `sportspred ingest ... && sportspred grade &&
  sportspred review` after the games.
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
- **Sport-specific tennis, combat, cricket, rugby, volleyball, racket-sport
  and esports models.** These leagues run on a results-only rating model via
  CSV import and are adapter-only. A reachable tennis archive exists (see
  DATA_SOURCES.md) but is not wired in.
- **Same-game combos.** Rejected by design until a joint model exists.
- **Kalshi and Polymarket API clients, closing-line value, dead-heat
  settlement.**
- **An interactive (server-based) dashboard.** The dashboard is a static file
  regenerated on demand.
- Per-plugin unit tests for football, hockey, baseball and soccer. Those
  plugins were verified by running the as-of leakage test and validation on
  real data, not by dedicated test files.

## How this was built, and what was not independently reviewed

The plan was to have each unit built and then audited by an independent
reviewer. A usage limit ended that work twice before any audit ran. The
football, hockey, baseball and soccer plugins were started by build agents,
left partial, and then completed and verified by the main author: as-of
leakage test, a check that no price is a model input, one validation run per
market, and a sanity check of the numbers against what is plausible for each
sport. They have **not** had a line-by-line independent review.

## Tests run

`pytest` on 2026-11-12: **4,345 passed, 13 expected failures** (the expected
failures are in the earlier `evmodel` suite), of which 106 are the new
`tests/sportspred` suite: time zones and HST display, stable IDs and
de-duplication, ledger immutability and tamper detection, synthetic isolation,
point-in-time availability, as-of leakage, chronological folds, retrospective
columns, stale and missing checks, CSV validation, the results-only builder,
combo dependencies and the three-leg rule, grading and resettlement, EV maths
and settlement rules, the wall between predictions and prices, dashboard
rendering, and an end-to-end smoke test.

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
4. **Tennis with real data**: surface-specific ratings and serve/return
   strength from the reachable archive, with care over tournament-week dates
   and its non-commercial licence.
5. **Player props and NBA spreads/totals**, with playing-time, opportunity and
   outcome-distribution models, validated against real lines once a line
   history exists.
6. **Finish-distribution models** for golf and motorsport.
7. **Independent review** of the four completed sport plugins, and dedicated
   tests for them.
8. Grade and review the first real slates; turn review patterns into lessons
   and test them before promoting any change.
