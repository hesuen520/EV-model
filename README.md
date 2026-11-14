# sportspred

Pregame outcome probabilities for many sports, estimated from verified sports
evidence - with strength of opposition and whole-team evaluation at the centre,
chronological validation behind every number, and an honest label on
everything that is not proven.

> Probabilities are estimates. A well calibrated 70% selection still loses
> three times in ten. **Nothing here guarantees a winning pick or a profit.**
> Prediction accuracy, probability calibration and betting value are three
> different questions; this platform reports them separately. In its own
> validation the models do **not** beat the market benchmarks that were
> available (NFL, club soccer).

The repository holds two packages:

- **`sportspred`** - the prediction platform described here. Its models never
  see betting odds.
- **`evmodel`** - the earlier price toolkit (odds conversion, vig removal, EV,
  Kelly, a prop scanner). Kept intact; see [docs/EVMODEL.md](docs/EVMODEL.md).
  `sportspred` reaches it only through its separate, optional EV module.

## What is operational

Statuses are **earned**, not typed in: the coverage table is computed from
ingested data and recorded validation runs
([docs/COVERAGE.md](docs/COVERAGE.md), `sportspred coverage`).

Results below are from walk-forward validation on real data. In each run the
final test period is read once, after the model has been selected on earlier
seasons; leagues that were re-run after a data or defect fix are listed with
every earlier run in the report's run history. Lower log loss is better. Full
detail, intervals, calibration charts and limitations:
[docs/validation/VALIDATION.md](docs/validation/VALIDATION.md).

<!-- results:start -->
| League | Market | Status | Final-test events | Accuracy | Log loss | Base rate | Rating baseline | Market benchmark |
|---|---|---|---|---|---|---|---|---|
| Major League Baseball | winner | validated | 4,949 | 56.5% | 0.680 | 0.691 | 0.680 |  |
| NBA | winner | validated | 2,813 | 67.8% | 0.596 | 0.692 | 0.604 |  |
| NCAA Division I men | winner | validated | 13,251 | 75.0% | 0.488 | 0.627 | 0.511 |  |
| NCAA Division I women | winner | validated | 12,313 | 78.3% | 0.439 | 0.656 | 0.473 |  |
| WNBA | winner | experimental | 664 | 65.4% | 0.626 | 0.688 | 0.644 |  |
| NCAA Division I FBS | winner | validated | 2,168 | 70.8% | 0.546 | 0.675 | 0.578 |  |
| NFL | spread (at the historical line) | experimental | 696 | 50.3% | 0.693 | (selected) | 0.695 | **0.694** |
| NFL | total (at the historical line) | experimental | 704 | 47.3% | 0.694 | (selected) | 0.704 | **0.693** |
| NFL | winner | validated | 705 | 65.4% | 0.627 | 0.693 | 0.631 | **0.603** |
| NHL | winner (incl. overtime and shootout) | validated | 3,090 | 57.0% | 0.672 | 0.689 | 0.679 |  |
| NHL | three-way after regulation | validated | 3,090 | 45.0% | 1.050 | 1.069 | 1.057 |  |
| German Bundesliga | three-way (home / draw / away) | experimental | 648 | 51.7% | 1.000 | 1.080 | 0.994 | **0.969** |
| English Premier League | three-way (home / draw / away) | experimental | 810 | 49.6% | 1.015 | 1.086 | 1.020 | **0.994** |
| Spanish La Liga | three-way (home / draw / away) | experimental | 829 | 52.6% | 0.979 | 1.060 | (selected) | **0.956** |
| French Ligue 1 | three-way (home / draw / away) | validated | 656 | 52.4% | 0.992 | 1.065 | 0.997 | **0.969** |
| Italian Serie A | three-way (home / draw / away) | experimental | 810 | 52.6% | 0.980 | 1.093 | 0.991 | **0.962** |
| Men's international football | three-way (home / draw / away) | experimental | 1,803 | 61.6% | 0.843 | 1.044 | 0.854 |  |
| Women's international football | three-way (home / draw / away) | validated | 1,350 | 67.3% | 0.772 | 1.006 | 0.797 |  |
| ATP Tour main draw singles | winner | validated | 3,898 | 66.0% | 0.610 | 0.694 | 0.621 |  |
| ATP Challenger main draw singles | winner | validated | 9,950 | 63.5% | 0.638 | 0.693 | 0.640 |  |
| ATP Tour qualifying singles | winner | validated | 1,799 | 65.0% | 0.618 | 0.693 | 0.636 |  |
| ITF men's singles | winner | experimental | 43,092 | 71.1% | 0.561 | 0.693 | 0.583 |  |
| ITF women's singles | winner | validated | 25,094 | 70.0% | 0.570 | 0.693 | 0.585 |  |
| WTA Tour main draw singles | winner | validated | 3,673 | 66.7% | 0.604 | 0.693 | 0.612 |  |
| WTA 125 main draw singles | winner | validated | 2,341 | 65.6% | 0.611 | 0.694 | 0.618 |  |
| WTA Tour qualifying singles | winner | validated | 3,278 | 67.5% | 0.590 | 0.693 | 0.599 |  |
<!-- results:end -->

*(This table is generated: `sportspred report summary --into README.md`.)*

How to read this honestly:

- **"validated" means the probabilities held up out of sample** (enough
  events, better than the naive baseline with the interval excluding zero,
  calibration error under 3 points, point-in-time features). It does **not**
  mean the model beats a market.
- Where a market benchmark exists for a winner market (NFL and the five club
  soccer leagues), **the market is better** than the model every time. The
  benchmark prices are undated in their source (possibly closing numbers,
  which contain later news), so the comparison is indicative only.
- Against the **NFL spread and total**, no sports-only model beat a coin flip
  (nor, at these lines, does the price itself), so the base rate was selected
  and every such prediction is PASS.
- **A model that failed a gate still makes selections by default.** Each
  prediction carries its validation status, and "experimental" ones are
  labelled as such. Set `allowed_coverage = ["validated"]` in the settings to
  turn every experimental selection into PASS. Combos already use validated
  legs only.
- **MLB** gains almost nothing over a plain Elo rating; MLB and the NHL are
  close to coin flips for most games, and the PASS rule passes on most of them.
- **Tennis** is validated on a snapshot archive that ends in mid-2026 and
  lists no upcoming matches, so it produces no live predictions until a
  schedule is imported by CSV (fixtures are matched to the archive's
  players). Matches are dated by tournament week, so a match never sees
  results from its own tournament or week. The archive's licence is
  **non-commercial** (CC BY-NC-SA 4.0). ITF men is experimental only because
  its history allows two walk-forward folds where the gate asks for three.
  An audit found that quarter-finals had been dropped at ingestion; the
  numbers shown are from runs after that was fixed.
- **College basketball** headline figures pool two kinds of game: about one
  in fifteen is against an opponent with almost no history (mostly
  lower-division teams) and is close to certain. The validation report shows
  each kind separately; see the model card.
- **Availability news is in one backtest only.** The NBA model uses the
  league's official injury report exactly as it was published before each
  day's games (archived since December 2018), and that improves it out of
  sample. No other league has such a history here: their backtests contain no
  injury reports, confirmed line-ups, starting goalies or pitchers as
  announced. Features that need the actual starter are measured separately as
  retrospective upper bounds and can never be selected.

### Player statistics (NBA and WNBA): built, experimental by construction

Points, rebounds, assists, three-pointers made and points + rebounds +
assists, each over or under a threshold. Three things are modelled: playing
time (expected minutes), production per minute (the player's own rates, what
the opponent allows, what absent team-mates leave behind) and the outcome
distribution, so the answer is a probability of finishing over, under or
exactly on a threshold - not a projected average.

<!-- results:start props -->
| League | Statistic | Status | Selected model | Final-test player-games | Accuracy | Log loss | Base rate | Season average + Poisson | Calibration error | Worst calibration error at a shifted threshold |
|---|---|---|---|---|---|---|---|---|---|---|
| NBA | assists | experimental | report_gbm | 46,004 | 64.9% | 0.627 | 0.668 | 0.652 | 0.009 | 0.013 |
| NBA | points | experimental | report_gbm | 46,004 | 60.9% | 0.654 | 0.690 | 0.678 | 0.006 | 0.016 |
| NBA | pra | experimental | report_gbm | 46,004 | 61.9% | 0.647 | 0.690 | 0.675 | 0.008 | 0.022 |
| NBA | rebounds | experimental | report_gbm | 46,004 | 63.0% | 0.641 | 0.677 | 0.662 | 0.007 | 0.013 |
| NBA | threes | experimental | report_gbm | 46,004 | 68.8% | 0.573 | 0.637 | 0.596 | 0.006 | 0.014 |
| WNBA | assists | experimental | two_stage_glm | 8,800 | 64.4% | 0.633 | 0.666 | 0.651 | 0.007 | 0.022 |
| WNBA | points | experimental | two_stage_glm | 8,800 | 59.9% | 0.664 | 0.690 | 0.679 | 0.009 | 0.026 |
| WNBA | pra | experimental | two_stage_gbm | 8,800 | 60.2% | 0.658 | 0.692 | 0.674 | 0.010 | 0.022 |
| WNBA | rebounds | experimental | two_stage_glm | 8,800 | 63.1% | 0.640 | 0.674 | 0.660 | 0.005 | 0.014 |
| WNBA | threes | experimental | two_stage_gbm | 8,800 | 71.0% | 0.560 | 0.625 | 0.584 | 0.020 | 0.018 |
<!-- results:end -->

*(Generated: `sportspred report summary --into README.md`.)*

- **These numbers are against a reference threshold, not a bookmaker's
  line.** No history of market thresholds is reachable, so each completed
  player-game is scored against the half point just above the median of that
  player's previous 15 games. Beating a naive threshold shows the
  distribution is informative and calibrated; it says nothing about beating a
  market line, which already contains most of what the model knows and news
  it does not. A gate records this and keeps every such market
  **experimental** until real thresholds have been collected and scored.
- The last column is the worst calibration error when the same model is read
  at thresholds moved away from the reference (for points, up to 6 either
  way), including whole-number thresholds where a tie is a push.
- A prediction needs a threshold you import
  (`sportspred import prop-lines`); without one there is no selection. A
  player listed out gets PASS; one listed doubtful or questionable makes the
  selection conditional on him playing.

### Formula 1: winner, podium and points finish

One row per driver and race, three yes-or-no questions about every starter.
The inputs are the official starting grid, driver and car form (the car's
with most weight dropped at each season change), each relative to the field
of that race, pace against team-mates, track fit and reliability. Data: the
F1DB database (CC BY 4.0), every race since 2000 for the model.

<!-- results:start field -->
| Series | Market | Status | Selected model | Final-test races | Driver starts | Log loss | Base rate | Grid position alone | Calibration error | Model's top picks that did, per race | First on the grid that did, per race |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Formula 1 | podium finish (top three) | validated | full_logit | 112 | 2,258 | 0.215 | 0.421 | 0.235 | 0.012 | 1.98 of 3 | 1.99 of 3 |
| Formula 1 | points finish (top ten) | validated | field_rank | 112 | 2,258 | 0.462 | 0.694 | 0.506 | 0.013 | 7.88 of 10 | 7.70 of 10 |
| Formula 1 | race winner (each starter: yes or no) | validated | full_logit | 112 | 2,258 | 0.101 | 0.197 | 0.110 | 0.006 | 0.54 of 1 | 0.58 of 1 |
<!-- results:end -->

*(Generated: `sportspred report summary --into README.md`.)*

- **The baseline to beat is the grid, not the base rate.** All three models
  give better probabilities than starting position alone on the final test
  (2022-2026, 112 races), with the interval excluding zero - for the win
  market only just.
- **Better probabilities, not better picks of the winner.** The driver the
  model rates most likely to win won 54% of those races; the pole-sitter won
  58%. For the podium the model's three and the first three on the grid do
  equally well; for points the model's ten are slightly better. What the
  model adds is a calibrated number for every starter.
- Rows are drivers, but the events are races: the gate that counts
  independent events counts 112 races, not 2,258 driver starts.
- **A prediction needs the starting grid**, which you import after
  qualifying (`sportspred import grid`). Before that there is none. Only the
  "yes" side is ever offered as a selection. Weather, tyres, strategy,
  practice and sprint results are not in the data.

Everything else:

- **adapter-only** (a model and a documented CSV import exist; no real data
  ingested and validated here): tennis doubles, UFC and boxing, cricket (T20,
  ODI, Test), rugby union and league, volleyball, badminton, table tennis,
  esports (CS2, Valorant, LoL, Dota 2) and international basketball run on a
  results-only rating model that knows nothing sport-specific; MLS and NWSL
  are declared under the soccer model with no data ingested.
- **unavailable** (declared, no model): golf, NASCAR, IndyCar and MotoGP (no
  reachable data), NBA / WNBA / college spreads and totals, NHL and MLB
  totals.
- **demo-only**: two synthetic leagues used for the offline demo and tests.

## Install

```bash
git clone https://github.com/hesuen520/EV-model && cd EV-model
pip install -e ".[dev]"          # Python 3.11+
pytest                           # whole suite; no network needed
sportspred demo                  # offline end-to-end run on SYNTHETIC data, clearly labelled
```

## Daily use

```bash
sportspred ingest --league nba            # public dataset: schedule, results, box scores (no key)
sportspred validate --league nba          # walk-forward validation; records the run
sportspred train --league nba             # fits the model the validation run selected
sportspred live nba-injury-report         # official injury report (public archive of every version)
sportspred predict --league nba --detail  # records pregame predictions in the immutable ledger
sportspred import prop-lines --league nba --csv lines.csv        # player thresholds you read (not prices)
sportspred predict --league nba --market player_prop:points      # over / under / push at those thresholds
sportspred import grid --league f1 --csv grid.csv                # the starting grid, after qualifying
sportspred predict --league f1 --market finish:podium            # also finish:win, finish:points
sportspred slate                          # today's slate (Pacific/Honolulu), with statuses
sportspred combos                         # strongest supported three-leg combo, or the PASS message
sportspred dashboard                      # writes data/reports/dashboard.html (open it; no server)
# after the games:
sportspred ingest --league nba && sportspred grade && sportspred review
sportspred monitor --league nba
```

Leagues with a working public-data adapter: `nba`, `wnba`, `ncaam`, `ncaaw`,
`nfl`, `ncaaf`, `nhl`, `mlb`, `soccer_intl_men`, `soccer_intl_women`, `epl`,
`laliga`, `bundesliga`, `seriea`, `ligue1`, and tennis singles `atp`,
`atp_qual`, `atp_challenger`, `itf_men`, `wta`, `wta_qual`, `wta_125`,
`itf_women` (historical results only). For any other league - or when a
feed is unreachable - use the CSV import ([docs/CSV_IMPORT.md](docs/CSV_IMPORT.md)):

```bash
sportspred templates my_csv/
sportspred import events --league ufc --csv my_csv/events.csv --create-participants
sportspred import availability --league nhl --csv goalies.csv --check-key starting_goalie
```

No data is ever invented or substituted from elsewhere. If a source cannot be
reached the command says so and records nothing new; if an earlier download of
the same file is in the local cache, that copy is reused with a warning that
gives its collection time.

## What a prediction contains

Competition, participants, start time in HST, venue, market and exact line;
the model probability; validation status; three separate kinds of uncertainty
(event: the chance it loses even if the probability is right; model: a
bootstrap interval; data quality); supporting factors, opponent strength and
upset risks; availability assumptions; missing data; source links with
collection and publication times; the last check time; and a status:

| Status | Meaning |
|---|---|
| **FINAL** | every configured pregame check is complete and fresh. The result is still uncertain. |
| **PROVISIONAL** | a probability exists, but a required check is missing, stale or due a re-check. The reasons name the check; no confirmation time is invented. |
| **CONDITIONAL** | holds only if a stated condition is true, e.g. a named questionable player plays. |
| **PASS** | evidence inadequate or a selection rule failed. |

How old is a report? Where the source states when the report was
**published** - the official NBA injury report, `import checks` rows, and
`import availability --published-at` - its age is measured from that time,
not from the download. A live list with no publication time (the ESPN
adapter, or a CSV imported without one) is as old as the moment it was read.
Confidence labels come from the observed out-of-sample hit rate of the
probability band a selection falls in, never from the raw probability.

Tonight's real NBA slate, for example, is PROVISIONAL: the newest official
injury report in the public archive was published the previous evening, and
the rule needs one from within three hours of tip-off.

## The ledger

Every prediction is appended to a ledger that cannot be edited: updates and
deletes are refused by the database, a later prediction for the same event
and market is a new row linked to the one it replaces, and each row carries a
hash of its content and of the row before it (`sportspred ledger verify`).
Three things make "pregame" mean something. The event must be scheduled and
not yet started. For a real event the stated creation time must match the
machine's clock, which is stored and hashed with the row, so a prediction
cannot be written afterwards with an earlier time. And an event that has only
a date is closed once that date has begun anywhere in the world.

Combos are recorded under the same rules (after their legs, before any leg's
event, at the machine's time) and `ledger verify` checks them as well. Writers
are serialised, so two processes recording at once cannot break the chain.

What it cannot prove: that the machine's clock was right, or that the newest
rows were never removed - `ledger verify` prints the head hash and the count
so you can keep them somewhere else.

## Combos

`sportspred combos` prioritises the strongest supported **three-leg** combo by
the estimated probability that every leg wins, and also shows strong singles, a
two-leg alternative, and larger combos only while they stay supported. Every
pair of legs is classified: same-game legs, the same team or player in two
events, and mutually exclusive outcomes are **rejected** (no joint model
exists); legs from different games of one league use the validated same-slate
residual correlation (independence when it is indistinguishable from zero, a
Gaussian copula otherwise). With fewer than three qualifying legs the output is
fewer picks and `PASS — insufficient support for a three-leg combo.`
Only FINAL legs can form a recommendation; PROVISIONAL legs appear, if at all,
in a preview labelled as not a recommendation. A leg is judged against its
event as it stands now: it is dropped if the event has started, been
cancelled or postponed, or moved since the prediction, and only the newest
prediction of each event and market counts.

The core of the rule - take the most probable qualifying legs of one league's
slate and multiply - is back-tested over historical slates
(`sportspred combos validate --league nba`). That backtest does **not** cover
the FINAL-status filter (no historical check states exist), combos across
leagues, or the dependency handling. For NBA three-leg combos on the
final-test slates the estimated all-win probability averaged 50.7% and 57.2%
actually won (271 slates). It does not always hold up: on NHL development
slates the three-leg estimate averaged 38.9% and 31.5% won (317 slates).
Adding a leg can never raise the probability that every selection wins.

## Optional EV module

Separate from the prediction engine, which cannot import it; paper tracking
only. Given timestamped executable prices it reports raw implied probability,
no-vig probability, fair odds, edge and expected value from actual payouts,
fees and settlement rules; a missing price means "EV unavailable". Includes the
editable 40%-70% raw-implied band for NBA / NHL / NFL team-win selections only.
See [docs/EV_MODULE.md](docs/EV_MODULE.md).

## Documentation

| Document | Contents |
|---|---|
| [docs/MODEL_CARD.md](docs/MODEL_CARD.md) | what each model uses, how it is validated, what it cannot know, known limitations |
| [docs/validation/VALIDATION.md](docs/validation/VALIDATION.md) | generated results: every league and market, intervals, calibration charts, PASS-rule tables |
| [docs/COVERAGE.md](docs/COVERAGE.md) | generated coverage table and status definitions |
| [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) | every source: access, cost, licence, coverage, what was reachable here |
| [docs/CSV_IMPORT.md](docs/CSV_IMPORT.md) | the import path for any league |
| [docs/EV_MODULE.md](docs/EV_MODULE.md) | prices, fees, settlement rules, paper tracking |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | design, the plugin contract, how to add a sport |
| [docs/STATUS.md](docs/STATUS.md) | working features versus placeholders, tests run, next steps |

## Configuration

Every selection and freshness rule is in one TOML file:
[examples/sportspred.toml](examples/sportspred.toml)
(`sportspred --config FILE ...` or `SPORTSPRED_CONFIG`). These are explicit
rules, not model weights - model weights are learned in validation, and the
validation report shows accuracy and coverage at each threshold. Credentials
are read from environment variables only ([.env.example](.env.example)); the
core platform needs none.

## Layout

```
src/sportspred/
  core/ storage/        UTC storage + HST display, stable IDs, provenance, append-only ledger
  adapters/             cached downloads, CSV import, official NBA injury report, ESPN (adapter-only), synthetic demo
  features/ models/     opponent-adjusted ratings, schedule context, candidates, production models
  validation/           chronological folds, metrics, calibration, backtests, leakage tests, charts
  sports/               one plugin per sport (basketball, football, hockey, baseball, soccer, tennis, generic, demo)
  predict/              pregame engine: checks, statuses, confidence labels
  combos/ ledger/       dependency-aware combos; grading, reviews, lessons, drift monitoring
  ev/                   optional price comparison (paper only)
  dashboard/ cli/       static dashboard; the `sportspred` command
src/evmodel/            the earlier price/EV toolkit
tests/sportspred/       leakage, time zones, stale data, settlement, probability validity, combo dependencies, smoke test
```
