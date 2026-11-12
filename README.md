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

Results below are from walk-forward validation on real data; the final test
period was read once, after the model had been selected on earlier seasons.
Lower log loss is better. Full detail, intervals, calibration charts and
limitations: [docs/validation/VALIDATION.md](docs/validation/VALIDATION.md).

| League | Market | Status | Final-test games | Accuracy | Log loss | Base rate | Rating baseline | Market benchmark |
|---|---|---|---|---|---|---|---|---|
| NBA | winner | validated | 2,813 | 67.7% | 0.600 | 0.692 | 0.604 | none available |
| WNBA | winner | experimental | 664 | 65.4% | 0.626 | 0.688 | 0.644 | |
| NCAA Division I men | winner | validated | 13,251 | 74.9% | 0.488 | 0.627 | 0.511 | |
| NCAA Division I women | winner | validated | 12,313 | 78.3% | 0.439 | 0.656 | 0.473 | |
| NFL | winner | validated | 704 | 65.2% | 0.627 | 0.693 | 0.630 | **0.603** |
| NFL | spread (at the historical line) | experimental | 695 | 50.2% | 0.694 | | 0.695 | 0.694 |
| NFL | total (at the historical line) | experimental | 703 | 47.4% | 0.694 | | 0.704 | 0.693 |
| College football (FBS) | winner | validated | 2,168 | 71.2% | 0.552 | 0.675 | 0.578 | |
| NHL | winner (incl. overtime) | validated | 3,090 | 57.0% | 0.672 | 0.689 | 0.679 | |
| NHL | regulation three-way | validated | 3,090 | 45.0% | 1.050 | 1.069 | 1.057 | |
| MLB | winner | validated | 4,949 | 56.5% | 0.680 | 0.691 | 0.680 | |
| Men's international soccer | three-way | experimental | 1,803 | 61.5% | 0.843 | 1.044 | 0.854 | |
| Women's international soccer | three-way | validated | 1,350 | 67.3% | 0.772 | 1.006 | 0.797 | |
| Premier League | three-way | experimental | 810 | 49.6% | 1.015 | 1.086 | 1.020 | **0.994** |
| La Liga | three-way | experimental | 829 | 52.6% | 0.979 | 1.060 | (selected) | **0.956** |
| Bundesliga | three-way | experimental | 648 | 51.7% | 1.000 | 1.080 | 0.994 | **0.969** |
| Serie A | three-way | validated | 810 | 53.0% | 0.980 | 1.093 | 0.991 | **0.962** |
| Ligue 1 | three-way | validated | 656 | 52.4% | 0.992 | 1.065 | 0.997 | **0.969** |

How to read this honestly:

- **"validated" means the probabilities held up out of sample** (enough
  events, better than the naive baseline with the interval excluding zero,
  calibration error under 3 points, point-in-time features). It does **not**
  mean the model beats a market.
- Where a market benchmark exists, **the market is better** than the model in
  every case. The NFL and soccer benchmark prices are undated in their source
  (possibly closing numbers, which contain later news), so the comparison is
  indicative only.
- Against the **NFL spread and total**, no sports-only model beat a coin flip,
  so the base rate was selected and every such prediction is PASS.
- **MLB** gains almost nothing over a plain Elo rating; MLB and the NHL are
  close to coin flips for most games, and the PASS rule passes on most of them.
- None of the backtests include **late availability news** (injury reports,
  confirmed line-ups, starting goalies or pitchers as announced): no
  as-published history of those was used. Features that need the actual
  starter are measured separately as retrospective upper bounds and can never
  be selected.

Everything else:

- **adapter-only** (a model and a documented CSV import exist; no real data
  ingested and validated here): tennis (ATP, WTA, Challenger, WTA 125, ITF,
  doubles), UFC and boxing, cricket (T20, ODI, Test), rugby union and league,
  volleyball, badminton, table tennis, esports (CS2, Valorant, LoL, Dota 2),
  international basketball, MLS, NWSL. These run on a results-only rating
  model that knows nothing sport-specific.
- **unavailable** (declared, no model): golf and motorsport (field events need
  a finish-distribution model), player props, NBA / WNBA / college spreads and
  totals, NHL and MLB totals.
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
sportspred slate                          # today's slate (Pacific/Honolulu), with statuses
sportspred combos                         # strongest supported three-leg combo, or the PASS message
sportspred dashboard                      # writes data/reports/dashboard.html (open it; no server)
# after the games:
sportspred ingest --league nba && sportspred grade && sportspred review
sportspred monitor --league nba
```

Leagues with a working public-data adapter: `nba`, `wnba`, `ncaam`, `ncaaw`,
`nfl`, `ncaaf`, `nhl`, `mlb`, `soccer_intl_men`, `soccer_intl_women`, `epl`,
`laliga`, `bundesliga`, `seriea`, `ligue1`. For any other league - or when a
feed is unreachable - use the CSV import ([docs/CSV_IMPORT.md](docs/CSV_IMPORT.md)):

```bash
sportspred templates my_csv/
sportspred import events --league ufc --csv my_csv/events.csv --create-participants
sportspred import availability --league nhl --csv goalies.csv
```

No data is ever substituted: if a source cannot be reached the command says so
and records nothing.

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

A report's age is measured from when it was **published**, not when it was
downloaded. Confidence labels come from the observed out-of-sample hit rate of
the probability band a selection falls in, never from the raw probability.

Tonight's real NBA slate, for example, is PROVISIONAL: the newest official
injury report in the public archive was published the previous evening, and
the rule needs one from within three hours of tip-off.

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
in a preview labelled as not a recommendation.

The whole selection rule is back-tested over historical slates
(`sportspred combos validate --league nba`). For NBA three-leg combos on the
final-test slates the estimated all-win probability averaged 50.8% and 52.2%
actually won (268 slates). Adding a leg can never raise the probability that
every selection wins.

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
  sports/               one plugin per sport (basketball, football, hockey, baseball, soccer, generic, demo)
  predict/              pregame engine: checks, statuses, confidence labels
  combos/ ledger/       dependency-aware combos; grading, reviews, lessons, drift monitoring
  ev/                   optional price comparison (paper only)
  dashboard/ cli/       static dashboard; the `sportspred` command
src/evmodel/            the earlier price/EV toolkit
tests/sportspred/       leakage, time zones, stale data, settlement, probability validity, combo dependencies, smoke test
```
