# evmodel

Find +EV (positive expected value) player-prop bets.

A sportsbook price is +EV when it pays more than the bet's true probability
justifies. `evmodel` estimates that true probability two independent ways and
flags the prices that beat it:

1. **Market consensus.** Some books price props much better than others.
   The vig is removed from the sharp books' prices, their views are pooled,
   and every other book's price is compared with the result.
2. **Projection model.** A statistical model turns each player's game logs
   into a full probability distribution for the stat, fitted and tested on
   real NBA, NHL and NFL results.

It then sizes each bet with capped fractional Kelly and gives you a ledger
that measures closing-line value, so you can find out whether the edges are
real before the bankroll does.

> **Read this first.** Nothing here guarantees profit. Expected value is an
> estimate that is only as good as its inputs; a real 3% edge still loses
> over hundreds of bets a meaningful share of the time; sportsbooks limit
> accounts that win. Sports betting is illegal in some places and age
> restricted everywhere. Bet only money you can afford to lose. If gambling
> stops being fun, the US helpline is 1-800-GAMBLER.

## Install

```bash
git clone https://github.com/hesuen520/EV-model && cd EV-model
pip install -e .            # Python 3.11+
```

## Five-minute tour

```bash
evmodel demo                          # full run on a simulated slate (no key, no network)
evmodel devig -- -125 +105            # remove the vig from a two-way price
evmodel ev --prob 0.56 --price -105   # EV and Kelly stake of one bet
evmodel project --sport nba --market player_points --line 24.5
                                      # real projections from current game logs
evmodel backtest --sport nba          # how well calibrated the model is on held-out seasons
```

`evmodel project` and `evmodel backtest` download free public game logs the
first time they run (about 1 MB per season) and cache them under `data/raw/`.

## Finding bets

You need prices. There are two ways to supply them.

**Type them in.** No account needed:

```bash
evmodel odds-template odds.csv        # writes an example file
# edit odds.csv: one row per book and prop (player, market, line, book, over, under)
evmodel scan --odds-csv odds.csv --sport nba --html board.html
```

**Pull them live** from [The Odds API](https://the-odds-api.com) (free tier
available; you need your own key):

```bash
export ODDS_API_KEY=your_key
evmodel fetch-odds --sport nba        # shows the credit cost and asks before spending
evmodel scan --odds data/odds/latest.json --out picks.csv --html board.html
```

`scan` prints the bets that clear the thresholds, best first:

```
        player    market       bet        book price fair  win%    EV units refs    src flags
  Jalen Example    Points over 24.5  draftkings  -105 -115 53.4% +4.2%  1.10    2  blend
```

* `price` is what the book offers, `fair` is the vig-free price of that side.
* `EV` is expected profit per unit staked. `units` is the suggested stake
  (one unit = 1% of bankroll by default).
* `refs` is how many reference books support the fair price; `src` says
  whether it came from the market, the model, or a blend.
* `flags` are cautions: the fair price was translated from a different line,
  only one reference book, references disagree, and so on.

The HTML board (`--html`) shows the same bets with an *edge bar*: where the
price breaks even, where the fair probability sits, and the gap between them.

**Then check before you bet.** Prices move within minutes. Confirm the line
and price are still there, and skip anything that moved against you. A
snapshot cannot see injury news that broke after it was taken.

## How it works

### Market consensus (`evmodel.market`)

For every prop, each reference book's two-way price is de-vigged
(`shin` by default; `multiplicative`, `additive`, `power`, `odds_ratio` and
`probit` are available) and the books are pooled in log-odds space using the
weights in the config (Pinnacle and Circa heaviest, recreational books light
or zero).

* **Leave-one-out.** A price is only ever called +EV on the strength of
  *other* books. The book being judged can shrink its own edge (its quote is
  evidence too, at its own weight) but can never create one.
* **Different lines.** When a reference book hangs 24.5 and the target book
  hangs 23.5, the probability is moved across lines through the market's
  fitted distribution, and the bet is flagged. Same-line references are
  always preferred.
* **Worst-case de-vig gate.** How the vig is removed is an assumption. A bet
  must stay non-negative EV under whichever method is least favourable to
  it, so no edge exists only because of that choice.
* **Pick'em apps** (PrizePicks, Underdog) have no per-leg price, so a leg is
  valued against the break-even probability of the entry type you play
  (`evmodel dfs` lists them).

### Projection model (`evmodel.projection`)

For each player and stat: recent per-minute production (exponentially
weighted, shrunk towards positional norms, carrying less weight across
seasons) times expected minutes gives a mean. Around that mean sits a
distribution whose shape was fitted per market: a negative binomial for
counts, and a *scaled* negative binomial for stats that arrive in chunks
(basketball points, football yardage), which reproduces the skew those stats
really have.

The model is deliberately a **second opinion**, blended at 20% weight by
default and never used alone unless you ask (`--model-only`). It does not
know about injuries, lineup changes or matchups announced since a player's
last game, and sportsbooks do. A model that ignores that news will "find"
large edges that are simply the market knowing something it does not.

### Staking (`evmodel.scanner`)

Quarter Kelly by default, halved for low-confidence bets, capped per bet,
per player and per game (props on one player or game are correlated, so
independent Kelly stakes would over-bet them).

## Does it work? What was tested

**The math.** `tests/test_golden.py` checks odds conversion, all six de-vig
methods, EV, Kelly with pushes and line translation against reference values
computed independently with 50-digit arithmetic: 216 checks, agreement to
1e-9.

**The projection model, on real results.** Every game was projected using
only earlier games; distribution shapes were fitted on early seasons, chosen
on middle seasons, and scored on later seasons the fitting never saw. The
score is calibration error: among props the model prices at 60%, how far
from 60% do they actually land? Lines are placed where books place them (at
the model's median and a little either side), because free historical prop
lines do not exist.

| Sport | Held-out seasons | Player-games | Calibration error (percentage points) |
|-------|------------------|--------------|----------------------------------------|
| NBA   | 2024-25 to Nov 2026 | 14,500 to 38,600 per market | 0.4 to 0.9 across 12 markets |
| NHL   | 2025-26 to Nov 2026 | 3,500 to 31,200 per market | 0.4 to 1.5 across 7 markets |
| NFL   | 2024 to week 9 of 2026 | 1,400 to 8,600 per market | 1.0 to 3.6 across 12 markets |

Reproduce with `evmodel backtest --sport nba|nhl|nfl` (add `--detail` for
the reliability tables). NFL numbers are noisier because there are far fewer
games; passing touchdowns is the weakest market (3.6).

What this does and does not show: the model's *probabilities are honest* on
average. It does **not** show the model beats sportsbook lines, because
there were no historical lines to test against.

**The scanner, in a simulated market.** `evmodel simulate` builds fictional
books with known true probabilities, runs the scanner, and compares claims
with truth. On the default settings the predicted EV of recommended bets is
within about half a point of their true EV and roughly nine in ten are truly
+EV. That validates the arithmetic and selection logic. It says nothing
about whether real books are as soft as the simulated ones.

**Not tested here:** the live odds client was written against The Odds API's
documentation and official samples but has not been run against the live
service in this repository's development environment. Its parser is covered
by fixture tests.

## Keeping score

Results are too noisy to judge an edge quickly: at typical prop prices you
need thousands of bets to see a 3% edge in profit. Closing-line value shows
up far sooner, so track both.

```bash
evmodel bets add-picks picks.csv --unit 10        # record the bets you placed
evmodel bets close <bet_id> --over -130 --under +105   # prices just before the game
evmodel bets settle <bet_id> --actual 27          # the stat's final value
evmodel bets summary --by market
```

The summary puts results next to what the model expected, with a confidence
interval, how lucky or unlucky the run was, and how many more bets you would
need before the edge is statistically distinguishable from zero.

## Configuration

Every threshold, book weight and staking rule lives in one TOML file; see
[`examples/config.toml`](examples/config.toml).

```bash
evmodel scan --odds data/odds/latest.json --config examples/config.toml
```

The book weights and pick'em break-evens are judgment calls and change over
time. Tune them to the books and markets you actually bet.

## Data

| What | Source | Key needed |
|------|--------|------------|
| Live prop odds | [The Odds API](https://the-odds-api.com) v4 | yes (yours) |
| Odds you typed | CSV (`evmodel odds-template`) | no |
| NBA and NHL game logs | [sportsdataverse](https://github.com/sportsdataverse/sportsdataverse-data) releases | no |
| NFL weekly stats, schedules, snaps | [nflverse](https://github.com/nflverse/nflverse-data) releases | no |

NHL logs are available from the 2023-24 season. MLB markets are defined for
the market engine but have no projection model yet.

## Using it from Python

```python
from evmodel import scan, Config
from evmodel.providers.csv_odds import read_odds_csv

picks = scan(read_odds_csv("odds.csv", sport="nba"), Config())
```

[`examples/quickstart.py`](examples/quickstart.py) walks through the pieces.

## Layout

```
src/evmodel/
  odds.py devig.py ev.py        prices, vig removal, expected value, Kelly
  distributions.py              integer-valued predictive distributions
  markets.py data/              market registry and fitted shapes
  market.py                     sharp-book consensus and line translation
  projection/                   walk-forward features, fitting, calibration
  scanner.py                    valuation, filtering, staking
  dfs.py                        pick'em entry maths
  tracking.py                   bet ledger, closing-line value, statistics
  simulation.py                 simulated-market check
  providers/                    odds (API, CSV, simulated) and game logs
  report.py cli.py              HTML board and command line
tests/                          pytest suite
```

## Development

```bash
pip install -e ".[dev]"
pytest
evmodel calibrate --sport nba          # re-fit distribution shapes (add --write to keep them)
```
