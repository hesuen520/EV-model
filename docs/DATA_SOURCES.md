# Data sources: access, cost, licence, coverage

Researched on 2026-11-12. Facts marked *(verified)* were read that day from
the source's own repository, documentation or pricing page; facts marked
*(unverified)* come from secondary sources or memory and must be checked before
you rely on them. Prices and terms change - re-check before any commercial use.

**This is not legal advice.** Several sources below restrict commercial use or
automated access. If you use this platform to make or sell betting decisions,
read each licence yourself.

## What the development environment could reach

The sandbox this was built in allowed only GitHub-hosted files
(`raw.githubusercontent.com` and release downloads) and PyPI. Every official
API host that was tried was blocked by network policy: ESPN, NBA, NHL, MLB
StatsAPI, The Odds API, football-data.co.uk, TheSportsDB. Consequently:

- everything marked **in use** below was actually downloaded and validated;
- adapters for blocked hosts (ESPN injuries and scoreboard, The Odds API) were
  written from documented or commonly observed response shapes and tested on
  hand-written fixtures. They have **not** been run against the live services.

## In use (downloaded, ingested, validated)

| Source | Used for | Coverage on 2026-11-12 | Access and cost | Licence / terms |
|---|---|---|---|---|
| [sportsdataverse-data](https://github.com/sportsdataverse/sportsdataverse-data) release files (hoopR, wehoop, fastRhockey, cfbfastR exports) | NBA, WNBA, NCAA basketball (schedules, team and player box scores); NHL (schedules, goalie box scores, play-by-play shots); college football schedules | NBA 2002-03 to yesterday plus the rest of 2026-27; WNBA 2003-2026; NCAA basketball 2005-06 on; NHL 2009-10 on; refreshed daily upstream | free, no key | repository **MIT** *(verified)*. The data originates from ESPN and the NHL; the MIT licence covers the repository's code and cannot grant rights the original publishers hold. Treat redistribution and commercial use as unsettled. |
| sportsdataverse `nba_stats_injury_report` release | official NBA injury report, **every published version**, 2018-12-17 to present | newest version in the archive lags the league's live report by hours | free, no key | as above *(verified description)* |
| [nflverse-data](https://github.com/nflverse/nflverse-data) | NFL schedules and results incl. rest, roof, surface, quarterbacks, one spread / total / moneyline per game; team and quarterback statistics | 1999 to this week; future 2026 games included | free, no key | **CC BY 4.0** *(verified)*: commercial use allowed with attribution. The betting numbers are described by nflfastR as the *closing* line sourced from Pro-Football-Reference *(verified wording)*; the schedules dictionary itself names neither provider nor timing, and the moneylines are undocumented - so they are used only as an undated benchmark. |
| [Retrosheet](https://www.retrosheet.org) game logs via the [Chadwick Bureau mirror](https://github.com/chadwickbureau/retrosheet), and `retrosplits` day-by-day files | MLB results, parks, actual starting pitchers | 2000-2025; no 2026 file or future schedule yet | free, no key | Required notice: *"The information used here was obtained free of charge from and is copyrighted by Retrosheet. Interested parties may contact Retrosheet at www.retrosheet.org."* retrosplits: ODbL 1.0 *(verified)*. |
| [martj42/international_results](https://github.com/martj42/international_results) and the women's equivalent | men's and women's international soccer results (date-only) | 1872 to October 2026; no future fixtures | free, no key | **CC0** *(verified for the men's repository)* |
| [xgabora/Club-Football-Match-Data-2000-2025](https://github.com/xgabora/Club-Football-Match-Data-2000-2025) | results, shots and a Bet365 1X2 price for five club leagues | 2000 to 2026-09-20 | free, no key | repository MIT; the underlying data comes from Football-Data.co.uk, whose own terms apply *(unverified)*. The price's quote time is not recorded. |
| [openfootball/football.json](https://github.com/openfootball/football.json) | 2026-27 club fixtures | current season | free, no key | CC0 public domain |

## Reachable but not yet used

| Source | What it offers | Why it matters |
|---|---|---|
| A mirror of Jeff Sackmann's tennis archive (`Aneeshers/tennis-sackmann-archive`) | ATP / WTA tour, qualifying, Challenger and ITF match files with serve and return statistics, to about June 2026 | The original `tennis_atp` and `tennis_wta` repositories returned 404 from 2026-09-20 *(verified)*. Licence **CC BY-NC-SA 4.0**: attribution, share-alike and **non-commercial only**, which the author enforces. Tennis is therefore CSV-only here for now. Tournament dates are week starts, not match dates. |
| sportsdataverse NBA injury-report archive (history) | as-published injury reports since 2018 | allows an honest point-in-time backtest of availability features - the most valuable next step for the NBA model |
| nflverse `injuries` | weekly NFL injury reports since 2009 | only a last-modified date, not each day's report as published *(verified)* |

## Official feeds worth an adapter (blocked here)

| Feed | Notes |
|---|---|
| ESPN site API (`site.api.espn.com`: scoreboard, injuries) | Unofficial and undocumented; free, no key. Disney's terms of use bar automated access without written permission *(seen only as a third-party copy - unverified)*. Returns a current snapshot only: an as-published history exists only if you snapshot it yourself. An adapter is included (`sportspred live injuries`, `live scoreboard`), untested against the live service. |
| NBA official injury report | Game-day reports are due 11 a.m.-1 p.m. local time and refreshed through the day (every 15 minutes since December 2025) *(secondary sources)*. Available through the archive above. |
| NFL injury report | Practice reports Wednesday-Friday and a game-status report (Out / Doubtful / Questionable) *(secondary sources)*. No API. |
| NHL (`api-web.nhle.com`) | Unofficial reference only; no injury or line-up endpoint; NHL.com terms prohibit unauthorised scraping *(secondary)*. No official injury list exists. Starting goalies need a separate source. |
| MLB StatsAPI | Probable pitchers and line-ups; MLB's notice permits only "individual, non-commercial, non-bulk use" *(quoted from a secondary copy)*. Returns the current probable pitcher, not the history of announcements. |

## Odds and commercial providers (for the optional EV module)

| Provider | What it offers | Cost and access (2026-11-12) | Timestamps |
|---|---|---|---|
| [The Odds API](https://the-odds-api.com) v4 | sportsbook odds for US sports, soccer, tennis majors, MMA; no esports found | free 500 credits/month; $30 / 20K, $59 / 100K, $119 / 5M, $249 / 15M credits per month *(verified, pricing page extract may be dated)*. Featured markets cost markets x regions per call; historical snapshots cost 10x and need a paid plan. | Snapshots every 5-10 minutes back to 2020: the practical way to get an honestly timestamped market benchmark. |
| Kalshi | prediction-market contracts incl. sports | public market data without a key; taker fee = round up(0.07 x contracts x price x (1 - price)) *(verified)*; per-series multipliers and maker fees vary | timestamped trades and 1-minute candles |
| Polymarket (US and international) | contracts incl. sports | US: taker 0.0695 x contracts x price x (1 - price); international sports: 0.05 x ...; US persons may not trade on the international platform *(verified)* | timestamped history; fine-grained history ages out |
| Sportradar, SportsDataIO, Stats Perform / Opta, Genius Sports | official and licensed feeds incl. injuries and line-ups | enterprise licences, price on request *(unverified)* | whether as-published injury history is retained must be asked |
| Betfair historical data | exchange prices to the second | paid product; commission on net winnings | the best timestamped benchmark where liquidity exists |

Fee formulas and settlement rules are stored as dated data in
`src/sportspred/ev/rules.py`; see [EV_MODULE.md](EV_MODULE.md).

## Sports with no reachable feed

Cricket, rugby, volleyball, badminton, table tennis, boxing, UFC, esports,
golf and motorsport: nothing was ingested. Sources that exist (terms
unverified unless stated): Cricsheet (ball-by-ball, open licence for its
registers), f1db (**CC BY 4.0**, *verified*) and Jolpica-F1 (non-commercial
API terms), UFCStats-derived scrapes (site terms unverified), OpenDota,
Oracle's Elixir, HLTV and vlr.gg (scraping restricted). Use the CSV import
([CSV_IMPORT.md](CSV_IMPORT.md)); those leagues stay adapter-only until a
validation run on imported data is recorded.

## What an honest backtest still lacks

| Need | As-published history available? |
|---|---|
| NBA injury reports | **Yes** - the archive above (not yet used for backtesting) |
| NFL injury reports | No verified per-day archive |
| NHL starting goalies and scratches | No |
| MLB probable pitchers and line-ups as announced | No |
| Odds at prediction time | Not in any free source used here; needs a snapshot provider or your own collection |

The practical consequence: snapshot live injury, line-up and price feeds
yourself, with your own collection timestamps, from the day you start. The
store keeps every statement with its publication and collection time for
exactly that purpose.
