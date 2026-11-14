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
| Validation | 39 league-markets validated on real data with the walk-forward protocol and a reserved final test: 17 pass every gate; 9 event markets and the three Formula 1 markets are experimental (WNBA, NFL spread and total, ITF men, men's international soccer, Premier League, La Liga, Bundesliga, Serie A); the 10 player-statistic markets are experimental by construction. Runs were repeated after fixes; every run is listed in the validation report's run history | real data |
| Player statistics (NBA, WNBA) | points, rebounds, assists, three-pointers made and the combined statistic, over or under a threshold: point-in-time player-game inputs (minutes, per-minute production, opponent adjustment, absent team-mates), baselines and two-stage count models, validation against reference thresholds with a check at shifted and whole-number thresholds, trained models, a projection table | real data |
| Point-in-time features | the as-of leakage test passes with zero difference for all 26 league-markets on the real store (`sportspred check-leakage`, and the opt-in test `tests/sportspred/test_real_store_leakage.py`) | real data |
| NBA availability, point-in-time | the official injury report as published before each slate (archive since 2018-12-17) feeds the roster inputs; the report-aware model was selected on development folds and improves the final test (log loss 0.5959 against 0.5999) | real data |
| Formula 1 | winner, podium and points finish for every starter: ingest from a public database (523 races since 2000), point-in-time inputs from the grid, driver and car form, track fit and reliability; six candidates including a model of the whole race; judged against the grid alone as well as the base rate, with calibration judged on the offered side - all three experimental; trained models; starting-grid import | real data |
| Prediction engine | PROVISIONAL and PASS statuses with pregame checks, availability applied, bootstrap model-uncertainty intervals, validated confidence labels, explanations: 826 real pregame predictions in the ledger (470 current event-markets over seven slates: 354 PROVISIONAL, 116 PASS, none FINAL); the hash chain and the clock audit verify | real data |
| FINAL and CONDITIONAL statuses | every check complete and fresh; a selection that depends on a doubtful player | tests |
| Combos | the PASS message and the provisional preview on real slates; the per-league backtest of the selection rule | real data |
| Combos that are recommended | a FINAL three-leg combo, alternatives, larger sizes, dependency handling | tests |
| Ledger: grading and reviews | grading against results, linked correction grades, reviews (variance / model error / data issue / availability change): 63 real selections graded and reviewed (13 and 14 November 2026) | real data (two runs) |
| Ledger: lessons and drift monitor | lessons with a validation-run requirement; calibration and input drift | tests |
| CSV import | events, results, availability, check records, thresholds of player-statistic markets; whole-file validation with line numbers | tests |
| EV module | CSV price import, raw implied / no-vig / fair odds / edge / EV, fee schedules, settlement rules, eligibility presets, paper positions and resettlement | tests |
| Dashboard and reports | static self-contained HTML dashboard; generated coverage table and validation results with charts | real data |
| CLI | `coverage, ingest, import, templates, validate, train, check-leakage, predict, props, slate, combos, grade, review, lessons, monitor, live, ev, dashboard, report, ledger, demo` | real data for the commands above; tests for the rest |

## Works in tests only (not exercised on real events yet)

- **FINAL and CONDITIONAL statuses on real slates.** The logic is tested end
  to end on synthetic data and was checked on a scratch copy of the real
  store, but no real prediction has reached FINAL: the newest official NBA
  report in the public archive is too old to satisfy the three-hour re-check
  rule, and no other availability feed is reachable here.
- **Lessons and drift monitoring on real results.** Tested, but they need
  far more graded results than exist. Grading and reviews did run on real
  results for the first time on 2026-11-13: 59 selections (all PROVISIONAL)
  were graded, 48 correct and 11 incorrect - 81% against an average stated
  probability of 85%, on one day of mostly college basketball. That is far
  too little to say anything about the models. It was enough to expose a
  defect: six college football predictions made when only the date was known
  were voided as "rescheduled" once kick-off times were published. The rule
  was fixed and the six grades withdrawn by linked correction grades. 52 of
  the 59 reviews are labelled "data issue" because those predictions were
  made with travel inputs missing - inputs since removed from the college
  models. A second run on 2026-11-14 added the four NBA games of 12 November
  (the public sources were a day behind): 63 selections in all, 52 correct
  and 11 incorrect - 82.5% against 83.8% stated.
- **Formula 1 predictions on a real grid.** The next race is on 22 November
  and its grid does not exist yet, so the real ledger holds no Formula 1
  prediction. The path - grid import, pricing, the grid check, withdrawal,
  grading from the classification - is covered by tests on a synthetic
  series, and was dry-run on a scratch copy of the real store with the
  previous race's grid standing in (nothing was written to the real store).
- **EV evaluation with real prices.** Maths and rules are tested against hand
  calculations; no real quotes were available.
- **Player-statistic predictions at real thresholds.** No source of market
  thresholds is reachable, so none has been imported and the real ledger
  holds no such prediction. The path - import, pricing over / under / push,
  availability, the ledger record, withdrawal, grading from the box score - is
  covered by tests on a synthetic league, and was dry-run on a scratch copy of
  the real store with each player's reference threshold standing in for a
  line (16 players in 8 games on 13 and 14 November, repeated on the final
  code; nothing was written to the real store).

## Written but not run against the live service

- ESPN adapter (`sportspred live injuries`, `live scoreboard`): host blocked
  here; fixture-tested only.
- The Odds API client and parser (`sportspred ev import-prices --odds-api`):
  host blocked here; written from the documentation; fixture-tested only.

## Not built

- **A history or feed of market thresholds for player statistics.** Thresholds
  come in by CSV only. Until a history of real ones has been collected and
  scored, the player-statistic markets stay experimental and unrated, and
  nothing is known about how they fare against a bookmaker's line. Statistics
  beyond the five modelled ones (steals, blocks, turnovers and other
  combinations) and props for other sports are not built.
- **NBA / WNBA / college spreads and totals, NHL and MLB totals.** No
  historical lines exist in the reachable sources; not implemented.
- **Golf, NASCAR, IndyCar, MotoGP.** Declared so the coverage table is
  complete; no data for them is reachable here. Formula 1 is built (above):
  its plugin and field model are the pattern to follow. Not built for
  Formula 1: a model for use before qualifying, sprint races, fastest lap,
  head-to-head and constructor markets.
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

**4. Tennis, basketball, the results-only builder and the shared rating
code,** for point-in-time leakage. *No leakage found:* flipping every result
of one slate (or tournament week) changes none of that slate's features;
features rebuilt as of past instants equal the full-history values on real
data; update rates are tuned on burn-in seasons only; side A does not depend
on the result; no price is read anywhere. *Found and fixed* (each with a
regression test):

- every tennis main-draw quarter-final was dropped at ingestion - about one
  match in ten per league. Tennis was re-ingested and re-validated;
- a tournament in progress was treated as finished (its rows carry the start
  date), and a CSV fixture could not be linked to the archive's players;
- a blank winner in a CSV file could be stored as an away win, and a bad file
  could leave events behind with no results;
- an out-of-date "out" statement was applied to a player who had played
  since; the rotation snapshot belonged to the wrong game when two seasons'
  schedules were on file; college travel inputs were mostly missing and
  mislocated (removed).

*Found and not fixed:* in a league that mixes timed and date-only events, a
timed game can see the same-date result of a date-only game between *other*
teams (a team's own ratings are unaffected; the CSV import refuses the case
that would matter, one competitor with both kinds on one date); the rating
code seeds its league mean from the first 200 values (burn-in only, and it
cancels in every feature); a two-week tennis event counts as finished after
its first week; eight pairs of college games that may be duplicates were not
investigated.

**5. The fixes themselves.** Four further reviewers were each given two
areas and told to try to break the fixes above, not to confirm them. *Held
under attack:* three-way grading in mixed leagues, the reschedule rule at
every boundary tried, one current prediction per event and market, combos
and slates judged on the event's current state, the exact PASS message, the
ledger's clock rule for predictions (including through the engine and the
command line), the migration of an older store, conditional status for
totals and draws, the model-mismatch rule, the push-probability arithmetic,
supersession of injury reports, the stale-statement and roster-snapshot
rules, and tennis name matching for CSV fixtures. *Broken or found beside
them, and fixed* (each with a regression test):

- results and check records imported by CSV could attach to the wrong event
  when an earlier event had no ID;
- combos could still be backdated, and two writers at once could break the
  ledger's hash chain permanently;
- paper settlement booked a game with no score as a loss; the EV evaluation
  and the dashboard ignored a cancellation or a move; the dashboard kept
  showing an earlier combo run;
- "NaN" was accepted as a score; the same-day timed/untimed check could be
  bypassed with a different spelling or an event already on file;
- which injury report is "newest" went by when it was read, not when it was
  published; a list that dropped a team with no game soon never cleared it;
- an event marked final without a score counted as a result; short histories
  gave a wrong "typical break".

The reviewer of the validation protocol and tennis confirmed, by experiment:
randomising every final-test outcome changes neither the selected model nor
any development metric; flipping every result of one tournament week changes
that week's features by exactly zero; a candidate that names an undeclared
column is refused, and one that peeks at a numeric outcome cannot score. It
also caught a **regression introduced by one of the fixes**: blinding hid the
home/neutral flag from the base-rate baseline, so the "better than naive"
gate was briefly measured against a weaker baseline. That was fixed, a
real-data test now checks every candidate predicts the same on blinded rows,
and every league was validated again afterwards. Also fixed from that review:
an outcome kept in a text or date column was not blinded; the price guard
did not know this project's own price columns; the as-of check compared the
wrong slate for tennis; Olympic singles and some WTA qualifying belonged to
no league.

*Found and not fixed:* an event imported without an ID and then moved across
midnight UTC becomes a second event (give events an `event_id`); the
reschedule rule's window for a date-only event (14 hours before to 36 hours
after its stored instant) does not know the league's time zone, so a game
moved to the next local day can fall inside it; a source that stopped less
than 1.5 typical off-seasons ago, with no unplayed fixtures listed, still
counts as current; two
candidates written to share state with each other could distort selection
on development folds (not the final test) - candidates come from plugin
code, not from users; a tennis match's features are complete only from the
Monday of its week, so a prediction made earlier uses less than the backtest
did.

**6. The player-statistic markets**, reviewed after they were built and
again after each round of fixes; every reviewer was read-only and had to prove
a defect by experiment.

*Confirmed correct:* the point-in-time inputs. One reviewer recomputed twelve
of them by brute force on about 300 rows; a second recomputed every input,
with no code shared with the builder, on synthetic leagues (including a day
on which every team plays twice, a trade between the two games and a missing
box score) and on 2,292 rows of the real NBA tables, among them the real
double day of 2020-03-01 and the games after each finished game without a
box score - exact agreement. The distribution maths matched an independent
library to 1e-14 on 20,000 thresholds.

*Found and fixed* (each with a regression test):

- a finished game with no box score on file counted as a game the whole
  rotation had missed, and every statistic market on it was graded void; such
  a game is now neither, a prediction waits while a recent box score is
  missing, and nothing is graded until both teams' box scores are in;
- availability edited the report-based inputs whatever the trained model
  read, treated one stray statement - or a check from another week or another
  game - as the team's whole report, matched team-mates by name only, and
  described a player listed out as someone the selection "holds only if he
  plays". Inputs are now replaced only when the model reads them and the
  team's report is on file and fresh for that game; everything else is a
  condition recorded in words with its direction; matching goes by id, then
  name, then joined initials, and a mere surname match is a condition;
- at a whole-number threshold a calibrated over/under was mixed with an
  uncalibrated tie, so "over 23" and "over 23.5" differed; both sides are now
  read from the calibrated half points and the tie is the remainder, and the
  selection rules use the probability given a decision;
- a threshold read after tip-off was attached to the team's next game; two
  thresholds for one market at one instant were both kept; zero was accepted;
- inputs of a second game on one day saw the first; the frame cache could
  serve a build from before a report version or a schedule change; the frame's
  columns depended on when it was built; a played row without statistics
  counted as an "under"; opening night had no rows; a selection about a
  player who had left the frame stayed current;
- a threshold never went out of date and the record did not say when or where
  it was read (it now does, and one read more than 24 hours earlier is not
  priced); a second file could bring a different threshold for the same
  instant; "2_5" was read as 25; a base rate, had validation ever selected
  it, would have "priced" any threshold with one number (now PASS);
- grading read only the first of a player's rows and called statistics
  without minutes "did not play"; its box-score cache trusted the file time
  alone; the projection table did not say when its inputs were a game old.

One older defect surfaced on the way and was fixed: football spread and total
models compute the chance of landing exactly on a whole-number line, but the
engine never asked for it, so such a prediction recorded two outcomes and no
push. No such model is in use (the base rate is selected for both NFL
markets), so no recorded prediction is affected.

The ten validations were run again after the first round of fixes (the first
runs stay in the run history). The second round changed no validated row:
the completed rows of all ten frames hash identically before and after it.

Every area was attacked again after the first round of fixes: the feature
builder and the availability handling (the second round came from there), then
the prediction engine and the import and grading code (the third). *Held
under attack in the last pass:* forty-two thresholds on one slate - whole and
half numbers six either side of the reference - gave probabilities that sum
to one, a push exactly at whole numbers, and PASS exactly where the rules say;
unchanged predictions were not rewritten on later runs; a blocker always
gives PASS, an incomplete check PROVISIONAL before any condition is looked at;
withdrawal touched only the model market being predicted and wrote nothing in
a dry run; the ledger's chain verified after every step; monitoring grouped
by model market and compared hits with the probability given a decision; the
import guards for tip-off, zero, future times and name collisions; half a
box score graded nothing; a corrected box score re-graded by a linked grade.
The fixes of the third round have tests but have not themselves been
attacked by a further reviewer.

*Found and not fixed:* the report-based history matches names exactly, and
thirteen player-seasons were found in which the official report and the box
score name a rotation player differently (for example a nickname), so his
absences are missing from team-mates' report-based inputs there - the live
path matches more loosely; a player's first game for a new team cannot be
predicted although such games are in the history; the correlation between
team-mates' outcomes in one game is not measured (same-game combinations are
not offered); the winner model has no "box score missing" check, so its
roster inputs can be a game behind without saying so.

**7. Formula 1**, reviewed after it was built and first reported.

*Confirmed correct:* all three final-test log losses and the grid-only
figures recompute exactly from the stored out-of-sample tables; folds are
chronological and no final season is in a development fold; the rank
likelihood sums to one over all finishing orders; the simulation puts one
car in each place; blinded and unblinded rows predict the same; ratings, the
season fade, field means and ranks use earlier races only; keys, the
"yes"-only rule and the grid check's windows behave as specified; grading
is right for every kind of entry; a full re-ingest changes nothing.

*Found and fixed* (each with a regression test):

- **the labels.** The win market passed the gate against the grid alone on
  one random seed and failed on others; the gate now uses 20,000 resamples and
  the win market fails it. The tables behind confidence labels, the PASS rule
  and the calibration gate were dominated by thousands of confident, correct
  "no" answers, while only "yes" is ever offered; they are now built from the
  offered side, on which no market passes. **Three markets first reported as
  validated are experimental;**
- an earlier race with no result on file did not hold the next race back from
  FINAL;
- `ingest --seasons` rebuilt the tables from the named seasons only, dropping
  the rest of the history;
- a three-letter code could be matched to a driver who last raced in 2007; a
  grid read during a race was stored for the next one; a grid could be
  imported before qualifying and used at any age; slot 99 and an all-pit-lane
  file were accepted;
- drivers who were on the grid but did not start were missing from the
  simulated field, which flattered the others; a race stored with only a date
  "knew" its grid two hours before that placeholder instant; the count of
  independent races could differ from the final test actually cut; the
  simulation's result depended on the order of the rows; the source
  renumbering its races would have overwritten other Grands Prix (now
  refused).

*Found and not fixed:* a correction of a grid dated earlier than the wrong
reading never takes effect (the newest reading wins); the intervals in the
PASS-rule and confidence tables treat a race's drivers as independent (the
bootstrap intervals elsewhere are by race); the points model is somewhat too
confident at the top and no better calibration method was tried, since
choosing one on these numbers would be tuning on the final test. The fixes of
this round have tests but have not themselves been independently reviewed.

**Not independently audited:** the results-only CSV model beyond its feature
builder, the EV module's maths, the dashboard, and the adapters written for
hosts that could not be reached (ESPN, The Odds API). Whether any rule,
candidate list or gate was ever changed because of a final-test number cannot
be proved from the record; the run history shows what every run produced.

## Tests run

`pytest` on 2026-11-14: **4,500 passed, 13 expected failures, 2 skipped** (the
expected failures are in the earlier `evmodel` suite; the two skipped are the
opt-in real-data tests below). 261 of those are the `tests/sportspred` suite:
time zones and HST display, stable IDs and de-duplication, ledger
immutability, tamper detection, the clock rule and concurrent writers,
synthetic isolation, point-in-time availability and report supersession,
as-of leakage, chronological folds, blinded scoring rows, retrospective and
price columns, stale and missing checks (including a source that has stopped
publishing), CSV validation, the results-only builder, tennis, the official
injury report parser and which archived version counts before a slate,
combo dependencies and the three-leg rule, grading,
reschedules and correction grades, EV maths and settlement rules, dashboard
rendering, the player-statistic markets (the count distribution and pushes,
point-in-time inputs including two games on one day and a missing box score,
threshold import, pricing, availability, withdrawal and grading), Formula 1
(the field model, the grid rule, the grid import, the adapter on a file in
the source's format, the gates), a regression test for every defect the audits and verifiers found, and an
end-to-end smoke test.

Opt-in tests on the real store (`SPORTSPRED_REAL_STORE=data/store python -m
pytest tests/sportspred/test_real_store_leakage.py`), run on 2026-11-13 after
the last change to any of their feature builders: **52 passed** - for each of
the 26 event league-markets, features rebuilt as of past instants equal the
full-history values, and every candidate predicts the same on blinded rows.
The same two tests for the ten player-statistic markets were run on
2026-11-14, after the last change to their builder: **20 passed**; and for
the three Formula 1 markets after the last change to theirs: **6 passed**.

Also run on 2026-11-13, on real data: ingest for every league over 12-13
November, validation and training for all 26 league-markets (logs under
`data/logs/`, which is not committed), the NBA combo backtest, grading and
reviews of the first decided predictions, predictions and combos for the
current slates, and `sportspred ledger verify`. On 13-14 November:
validation and training of the ten player-statistic markets (twice for
most - before and after the first round of review fixes), and
`sportspred ledger verify` again (784 predictions, chain intact; none of them
a player-statistic prediction).

## Next steps, in order of value

1. **Collect market thresholds for player statistics.** The prop models are
   built and calibrated against reference thresholds; whether they hold up
   against a bookmaker's line is unknown and cannot be learned from history
   that does not exist here. Import real thresholds from now on
   (`sportspred import prop-lines`), let the ledger and grading accumulate,
   then score the recorded predictions against results - that is the only
   route to a rated, validated prop market.
2. **Use the later versions of the NBA report.** The availability backtest
   takes the report as it stood an hour before the day's first game. The
   versions published later in the day, before each game's own tip-off, would
   capture late scratches; using them needs a per-game information time in
   the leakage test.
3. **Collect what history cannot give.** Snapshot injury reports, starting
   goalies, probable pitchers, line-ups and prices with collection timestamps
   from now on, so future validation can use them.
4. **A timestamped market benchmark.** Odds snapshots at prediction time, to
   test whether any edge exists against an executable price.
5. **Tennis for live use**: a schedule / order-of-play and withdrawal feed
   with real match dates, so in-tournament form and fatigue can be used and
   predictions can be made; a commercially licensed results source if the
   platform is ever used commercially; doubles.
6. **NBA spreads and totals**, and props for other sports, validated against
   real lines once a line history exists; a mapping of report names to
   box-score names for the handful of players the exact match misses.
7. **Finish-distribution models** for golf and motorsport.
8. **Close the items listed as found and not fixed**, and have the parts no
   reviewer covered looked at: the EV module's arithmetic, the dashboard, and
   the adapters for hosts that could not be reached here (ESPN, The Odds API)
   against the live services.
9. Keep grading and reviewing. One day of results is on file; calibration
   monitoring and lessons need weeks of them before they mean anything. Turn
   review patterns into lessons and test them before promoting any change.
