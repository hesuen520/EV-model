# EV module (optional, separate, paper tracking only)

The prediction engine produces sports-based probabilities and records them in
the ledger. It cannot import this module, and no price is ever a model input
(a test scans the source to enforce that). Only afterwards, and only if you
supply prices, are those probabilities compared with timestamped executable
quotes.

**Nothing here places a bet, and nothing here guarantees a profit.** Expected
value is an estimate that is only as good as the probability and the price
behind it. In this repository's own validation the models do not beat the
market benchmarks that were available (NFL, club soccer), so treat any
positive EV figure with scepticism.

```bash
sportspred ev template prices.csv                 # example file
sportspred ev import-prices --csv prices.csv      # or --odds-api nba (needs ODDS_API_KEY; spends credits)
sportspred ev evaluate --date 2026-11-12          # implied, no-vig, fair odds, edge, EV per selection
sportspred ev evaluate --open-paper               # record eligible ones as PAPER positions
sportspred ev paper settle                        # after results are ingested
sportspred ev paper report                        # price-dependent returns, separate from accuracy
sportspred ev rules ; sportspred ev fees          # the assumptions in use
```

## What is computed

| Term | Meaning |
|---|---|
| raw implied probability | the quote's own break-even, margin included: `1 / decimal odds` for a sportsbook; ask + fee for a contract |
| no-vig probability | the margin removed (method set by `ev.devig_method`); needs every outcome of the same market from the same venue at the same quote time, otherwise it is reported as unavailable |
| fair odds | `1 / model probability` |
| edge | model probability minus the break-even probability (pushes accounted for) |
| EV per unit | sportsbook: `p * (d - 1) - P(lose)`, a push refunds the stake; contract paying 1: `(p - cost) / cost` with `cost = ask + fee`; exchange commission is taken from net winnings |

Raw implied and no-vig probabilities are always shown separately. A selection
with no quote at its **exact** line is reported as `EV unavailable`; a price is
never assumed and a nearby line is never substituted.

## Eligibility

A selection is eligible only if all of these hold: the prediction is `FINAL`
and pregame; the quote is executable, was taken before the start and is at
most `ev.max_quote_age_minutes` old; the quote's settlement rules match the
prediction's market; the edge is at least `ev.min_edge`; and, **for NBA, NHL
and NFL team-win selections only**, the raw market-implied probability lies in
the editable band of the `us_major_team_win` preset (default 0.40-0.70). The
preset is not applied to any other league, market or prop. Edit it in the
settings file (`[[ev.presets]]`).

## Settlement rules (`rules_key` on every quote)

A model probability for "wins, overtime included" says nothing about a
regulation-time three-way price. Each quote therefore names the rule set it
settles under, and a quote whose rules do not match the prediction's market is
rejected as *not comparable*.

| Situation | Handling |
|---|---|
| overtime | `moneyline_incl_ot`, `spread_incl_ot`, `total_incl_ot` include it; `three_way_regulation` settles on the regulation result (a game that needed overtime was a draw) |
| drawn event in a winner market | push (default), loss (`moneyline_incl_ot_tie_loses`), or 50-50 on prediction markets |
| landing exactly on a spread or total | push |
| postponed / cancelled | void at sportsbooks; 50-50 under the prediction-market rule set (venues publish per-market windows - read them) |
| tennis retirement | three selectable policies: void, action after one completed set (needs `sets_completed` on file, otherwise left unsettled for manual handling), or action |
| walkover, no contest | void |
| player did not play (props) | void |
| result corrected later | a new settlement is appended and linked to the one it replaces; nothing is overwritten |

Books differ. Pick the rule set that matches the venue you are comparing
against, or add one in `src/sportspred/ev/rules.py`.

## Fees (`fee_key` on every quote)

Read on 2026-11-12 from the venues' published pages. Venues change these and
publish per-series or per-market parameters that override the defaults -
verify before relying on them.

| Key | Formula | Source |
|---|---|---|
| `kalshi_taker` | round up to the cent(0.07 x contracts x price x (1 - price)), times a per-series multiplier (default 1); some series also charge maker fees | https://kalshi.com/fee-schedule |
| `polymarket_us_taker` | 0.0695 x contracts x price x (1 - price) | https://docs.polymarket.us/fees |
| `polymarket_intl_sports_taker` | 0.05 x shares x price x (1 - price); US persons may not trade on the international platform | https://docs.polymarket.com/trading/fees |
| `exchange_2pct` | illustrative 2% commission on net winnings | set your exchange's rate |
| `sportsbook` | none (the margin is in the odds) | - |

Evaluation prices a single contract, so Kalshi's round-up-to-the-cent is
applied at its worst case.

## What is and is not implemented

- **Working and tested:** CSV price import (sportsbook and prediction-market
  quotes), all the maths above (checked against hand calculations), the
  eligibility rules, the settlement rules, paper positions, resettlement, the
  paper report.
- **Written from documentation, not run here:** `--odds-api` (The Odds API v4
  `/odds` parser and client). The host is blocked in the environment this was
  built in; the parser is covered by a fixture test in the documented response
  shape. Check https://the-odds-api.com/liveapi/guides/v4/ before use. Each
  call costs credits (featured markets: markets x regions).
- **Not implemented:** API clients for Kalshi and Polymarket (use the CSV
  import with `venue_kind=prediction_market`), dead-heat settlement for field
  events, closing-line value (needs a later quote per position), any form of
  order placement (by design).
