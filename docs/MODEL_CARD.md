# Model card

Last updated 2026-11-13. Numbers quoted here are from the newest validation run
recorded in the store for each league and market; the generated source of truth is
[validation/VALIDATION.md](validation/VALIDATION.md).

## Purpose and limits of use

The models estimate **pregame outcome probabilities** from sports evidence:
who is likely to win (or draw) before the event starts. They are meant for
analysis and paper tracking.

They are **not** a betting system. No model sees betting odds. In the
validation runs where a market benchmark existed (NFL, five club soccer
leagues) the market's implied probabilities were more accurate than the
model's in every winner market; on the NFL spread and total neither the
models nor the price beat a coin flip. Nothing here guarantees a winning pick or a profit.

## What every model has in common

**Design priorities.** Strength of opposition and whole-team evaluation.
A result counts for more against a strong opponent than a weak one, because
every rating is updated by the *surprise* relative to what that opponent
normally allows. Reputation, published rankings, raw win-loss records, star
names and odds are not inputs.

**Feature families** (the weight of each is learned in validation, never set
by hand):

- *Opponent-adjusted results*: Elo, with margin of victory where it helps;
  update rates tuned on the earliest ("burn-in") seasons only.
- *Opponent-adjusted production*: offence and defence ratings on the sport's
  scoring unit (points per 100 possessions, goals, runs, expected points added
  per play), each with a slow memory (long-term level) and a fast one (roughly
  the last 5-10 events), plus their difference (recent form relative to the
  long-term level).
- *Matchup*: an offence's rating set against the opposing defence (four
  factors in basketball, pass/rush in football, shots and expected goals in
  hockey, attack/defence in soccer).
- *Roster and availability*, where player data exists: quality of the players
  expected to take the floor, inferred from who has actually been playing;
  the primary goalie; the incumbent quarterback.
- *Surface and serve*, in tennis: a rating per surface beside the all-surface
  one, and opponent-adjusted service points won set against the opponent's
  return.
- *Schedule*: rest, back-to-backs, congestion, travel distance, time zones,
  home advantage, neutral sites, playoffs.

**Point-in-time rule.** A feature of an event uses only events on earlier
slates (Pacific/Honolulu calendar days; earlier dates for date-only sources)
plus the public schedule. Results of a slate are applied when the slate is
over. This is enforced by a generic test, *as-of invariance*: features are
rebuilt as of a past instant and must equal the full-history values exactly.
On its first run that test found a real leak (a traded player's rating update
reaching his new opponent's features within one game, about 1e-6); it was
fixed, and the test passes with zero difference for every plugin.

**Retrospective inputs.** Some information is only known at or after the
start: who actually played, the actual starting goalie, pitcher or
quarterback, observed weather. Such columns are flagged retrospective, can
only feed candidates labelled as retrospective experiments, and can never be
selected or put in production. They are reported as upper bounds, not as
realistic backtests.

**Outcome structures.** Two-way winner (binary), three-way with a draw
(soccer, NHL regulation time, Test cricket), and over/under against an exact
line (NFL spread and total). Multi-competitor fields (golf, motorsport) are
declared but not modelled.

**Candidates.** For each market: a base-rate baseline, a rating-only baseline
(Elo logistic), then regularised logistic regressions with progressively more
features and a shallow gradient-boosted model; soccer adds a Poisson /
Dixon-Coles goals model, NFL spread and total add margin and total
regressions. Complexity is only accepted when it improves chronological
out-of-sample log loss by more than one standard error over a simpler
candidate.

## Validation protocol

1. Completed events are cut into chronological blocks (seasons).
2. The most recent blocks are the **final test period**. No selection,
   tuning or calibration code reads them. Their scores are, however, printed
   at the end of every run - including runs made while the code was being
   written - so they were visible to the author throughout, and "untouched"
   would overstate it.
3. **Walk-forward development folds**: fit on earlier blocks, fit the
   calibrator (Platt scaling on log-probabilities) on the next block, score on
   the block after that. Imputation and scaling live inside the fitted
   pipeline, so they are fitted per fold. A candidate may read only the
   columns the feature frame declares as inputs, and the rows it is scored on
   are blinded - the outcome and every other undeclared column, whatever its
   type, are blanked first - so a model that peeked could not score. A
   real-data test checks that every candidate predicts the same with and
   without the blinding.
4. **Selection** on pooled development-fold log loss with a one-standard-error
   preference for the simpler candidate.
5. The selected candidate and the baselines are scored **once per run** on the
   final test period. Every league has four or five recorded runs, because
   data handling and defects found in review were fixed and the run repeated.
   No candidate list, selection rule or gate was changed because of a
   final-test number - a statement of the author's intent that the record
   cannot prove. Fixes did change outcomes: after duplicate college football
   records were removed the selected candidate changed (ratings-only to full
   logistic), and Serie A changed status. Every recorded run is kept and
   listed under "Run history" in the generated report. Runs made in scratch
   stores while a plugin was being written were not recorded.
6. Reported: sample sizes and dates; accuracy, Brier score, log loss and
   expected calibration error; reliability tables and charts; intervals from a
   bootstrap that resamples whole slates; season-by-season stability against
   each baseline; performance and coverage after the PASS rule at several
   probability thresholds; results per segment where a league mixes kinds of
   game (college basketball: games against opponents with almost no history);
   a market benchmark where one exists; the same-slate residual correlation
   used by the combo builder.
7. **Gates for "validated"**: real data; point-in-time features; at least 300
   final-test events; at least 3 development folds; final-test log-loss
   improvement over the base rate with a 95% interval above zero; expected
   calibration error at most 0.03. Anything else is "experimental".

Production models are refitted on all history with the most recent block held
out for calibration, plus 30 bootstrap refits that give the model-uncertainty
interval shown with every prediction.

## Models by sport

### Basketball (NBA, WNBA, NCAA Division I men and women)

Data: schedules, team box scores and player box scores (ESPN-derived public
files). Features: home court, Elo, opponent-adjusted efficiency per 100
possessions (slow, fast, form), four-factor matchups (shooting, turnovers,
offensive rebounding, free-throw rate, each offence against the opposing
defence), expected-rotation quality and recently missing quality from player
box scores, rest, back-to-backs, congestion, travel, time zones, playoffs.

<!-- results:start sport=basketball -->
| League | Market | Status | Final-test events | Accuracy | Log loss | Base rate | Rating baseline | Market benchmark |
|---|---|---|---|---|---|---|---|---|
| NBA | winner | validated | 2,813 | 67.8% | 0.596 | 0.692 | 0.604 |  |
| NCAA Division I men | winner | validated | 13,251 | 75.0% | 0.488 | 0.627 | 0.511 |  |
| NCAA Division I women | winner | validated | 12,313 | 78.3% | 0.439 | 0.656 | 0.473 |  |
| WNBA | winner | experimental | 664 | 65.4% | 0.626 | 0.688 | 0.644 |  |
<!-- results:end -->

On development folds the NBA model beat the Elo baseline in 15 of 15 seasons.

**The NBA model uses the official injury report, validated point-in-time.**
The league's report is archived in every published version since 2018-12-17.
For each game the backtest takes the report as it stood one hour before the
first tip-off of that day - one information time per slate, like every other
feature - and applies it exactly as the live path does: players listed out or
doubtful are removed from the expected rotation and their minutes go to the
rest in proportion; questionable and probable players are assumed to play.
That gives report-adjusted versions of the two roster inputs, and a candidate
that uses them in place of the unadjusted ones.

- In the nine development seasons before the archive begins the two
  candidates are identical, as they must be.
- In every development season with reports (2019-2024) the report-aware
  candidate is better; over 2020-2024 by 0.0034 log loss on average.
- It was selected on development folds, and on the final test it scores
  0.5959 against 0.5999 for the same model without the report (accuracy
  67.8% against 67.7%). For scale: adding the rotation that *actually* played
  - which nobody knows beforehand - scores 0.5933.

What this does not include: versions of the report published later in the day
(late scratches before a night game), confirmed starting line-ups, and
anything before December 2018. The live path may see a later version than the
backtest did, never an earlier one.

*College headline numbers need the segment split.* College schedules include
games against lower-division opponents, which are close to certain.
In the men's final test 9.6% of games (1,271) involve a side with fewer
than ten games on file in the previous year; the model picks those at 98.0%
(log loss 0.095). On the other 90.4% (11,980 games) accuracy is 72.5% and log
loss 0.529. For women: 7.8% of games at 94.0%, and 76.9% (log loss 0.459) on
the other 11,349. The pooled headline (75.0% and 78.3%) sits between the two;
the second figure is the one to compare with Division I games as usually
understood. The label is known before the game and is never a model input. Travel inputs are not used for the college leagues: the
built-in venue table locates under a tenth of their games.

*Availability.* A report says who is doubtful now: once a newer report of the
same kind has been read for a team, statements it carried from an older
report no longer apply, and an "out" statement older than the player's last
game is ignored (and listed as out of date). The roster feature assumes each
team's rotation matches who has played in its recent games. When an availability report is on file, players
listed out or doubtful are removed and their minutes go to the rest of the
rotation in proportion; questionable players make a selection CONDITIONAL if
their absence would break it. That redistribution rule was checked against
history: over 26,798 NBA team-games with an absent rotation player it predicts
the quality of the rotation that actually played with an error (RMSE) of 0.0104
on development seasons and 0.0130 on the final period, against 0.0161 and
0.0200 with no adjustment; a replacement-level alternative fitted on
development seasons was essentially the same (0.0103 and 0.0130). The
analysis is `scripts/analysis/nba_availability_redistribution.py`. *Caveat:* player quality is a box-score
composite (Game Score per minute, a published formula used only to define the
input). It undervalues low-usage defenders and shooters, so ruling such a
player out can nudge the estimate *up*. History shows the measure does move
that way when such players sit, but small availability adjustments should be
read with caution.

The WNBA and college leagues have no archived report history, so for them the
availability rule remains what it was: checked against who actually played,
not against results.

### Player statistics (NBA, WNBA)

**What is predicted.** For a player and a threshold: the probability that he
finishes over it, under it, or exactly on it (a push, possible only at a whole
number), for points, rebounds, assists, three-pointers made, and points +
rebounds + assists. The model is conditional on the player appearing - a
statistic market is void when he does not play - and covers rotation players
only (at least 8 earlier appearances and 15 recent minutes per appearance).

**Three parts, as a threshold market needs.**

1. *Playing time.* Expected minutes from his recent minutes (exponentially
   weighted, last three games, this season, longer run), starting role, rest,
   back-to-backs, games he has just missed, the gap between the two teams'
   recent scoring margins (lopsided games shorten minutes) and the minutes of
   team-mates expected to be absent.
2. *Production per minute.* The statistic per minute, recent and longer-run;
   how much of it the opponent has allowed relative to the league (the
   opponent adjustment); shot volume per minute as opportunity; and the share
   of a typical team total produced by team-mates expected to be absent - all
   of them, and separately those whose absence is new.
3. *The distribution.* A negative binomial around minutes x rate, with a
   spread learned from history as a function of the mean. Reading it at the
   threshold gives over, under and push.

Every input of a player-game comes from earlier Pacific/Honolulu days. The
as-of test rebuilds the frame at past instants - including the two days in the
data on which a team played twice - and finds zero difference.

Candidates, simplest first: the base rate; the player's season average with a
Poisson spread ("a projected average and nothing else"); recent minutes x
recent rate with a negative binomial; the two-stage model, log-linear; the
two-stage model with gradient-boosted trees; and, for the NBA, the last two
with team-mates' absences taken from the official injury report as published
before the slate instead of from who missed the previous game.

**How it is validated - and what that cannot show.** No history of
bookmakers' thresholds is reachable. Each completed player-game is therefore
scored against a *reference threshold* built only from that player's earlier
games: the half point just above the median of his previous 15 appearances.
The protocol is the shared one (walk-forward folds, calibration on a later
block, selection on development folds, the final two seasons read once).
Beating a naive threshold shows the distribution is informative and honest
about its spread. It does not show an edge against a market threshold, which
already contains most of what this model knows and news it does not have.
A gate (`market_thresholds`) records that, so these markets are
**experimental** whatever the other gates say, and a prop selection carries
no confidence label.

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

**What the runs show (NBA).** On development folds (eight seasons, 2016-17
to 2023-24, about 169,000 player-games) every step down the candidate list
helped, for every statistic. For points, pooled log loss: base rate 0.690;
season average with a Poisson spread 0.678; recent minutes x recent rate
0.671; two-stage log-linear 0.664; two-stage trees 0.662; and the same with
absences from the official report 0.661 - the model selected for all five
statistics. It beat the season-average baseline in 8 of 8 development
seasons. On the final test (2024-25, 2025-26 and the start of 2026-27; 46,004
player-games) points score 0.654 against 0.690 for the base rate and 0.678 for
the season average, with a calibration error of 0.006.

Accuracy differs between statistics mostly because the reference threshold
splits outcomes differently: "over" happens 46% of the time for points but
about 32% for three-pointers, where saying "under" every time is already
right 67.5% of the time. Log loss against the base rate is the comparison that
means something. Starters are harder than the rest of the rotation: for
points, 59.1% for players expected to play 28 minutes or more against 62.2%
for those expected to play 15 to 28 (log loss 0.665 and 0.646).

**WNBA.** The history is seven times smaller (59,660 player-games) and has no
report archive. Development folds select the log-linear two-stage model for
points, rebounds and assists and the tree version for three-pointers and the
combined statistic; calibration is looser (0.005 to 0.020 at the reference
threshold).

Across all pairs of players on one day the residual correlation is zero
(points: -0.0003, interval -0.0008 to +0.0002). Within one game it has not
been measured.

**Thresholds away from the reference.** A real threshold need not sit at the
player's recent median, and the calibration step is fitted at the reference
threshold only. So each run also scores the selected model at thresholds
moved up and down (for points by 0.5, 3 and 6), on the last development fold
and on the final test. A half-point shift gives a whole number, where a tie is
a push: over and under are then read from the calibrated model at the half
points either side, and the tie is what remains.

On the NBA final test, points: at the reference threshold 45.8% went over
against 45.4% predicted; three points lower 64.8% against 64.4%; three points
higher 29.1% against 29.6%; six lower 79.6% against 80.4%; six higher 16.7%
against 18.2%. The probability stays within about a point and a half out to
six points either side, and the error grows with distance - the fitted
distribution's upper tail is a little heavy. Over the five statistics the
worst calibration error at any shifted threshold is 0.013 to 0.022 for the
NBA and 0.014 to 0.026 for the WNBA (last column of the table). Predicted and
observed tie rates at whole-number thresholds agree to within about a point
and a half (NBA: points 6.2% predicted against 6.5% observed just below the
reference, rebounds 15.7% against 16.6%, assists 22.5% against 21.9%,
three-pointers 28.2% against 26.8%).

**Live use.** A prediction needs a threshold you import
(`sportspred import prop-lines`); without one there is no selection, and
`sportspred props` only lists projections. The player himself listed out:
PASS. Listed doubtful, questionable or probable: conditional on him playing.
Team-mates' statements change the model's inputs only when the trained model
reads the report-based inputs (the NBA models do; the WNBA models take
absences from who missed the previous game) *and* the team's report is on file
and fresh for that game. Otherwise nothing is changed, and each difference
between what the inputs assume and what is listed is a condition recorded in
words with its direction - "computed as if he plays, but he is listed out",
"computed as if he sits, but he is not on the team's report". A statement is
matched to a player by id, then name, then name with initials joined; one that
only shares a surname with a rotation player is a condition, never an
application. A prediction waits (PROVISIONAL) while the box score of either
team's last two finished games is missing, and a selection about a player who
has since left the modelled rotation is withdrawn. A team's first game of a
season is served from last season's rotation; a player's first game for a new
team has no row until he has played it. Each prediction records its threshold,
when it was read and where; one read more than 24 hours earlier is not priced
as a selection. If validation ever selected the base rate for one of these
markets, every prediction would be PASS: a base rate is about the thresholds
it was measured at, not the one imported.

*Not in the model:* confirmed starting line-ups, announced minutes limits,
foul trouble, garbage time beyond the margin-gap input, the opponent's
absences, position-specific defence, and any price. Same-game combinations of
these markets are not offered: team-mates' outcomes are not independent.

### American football (NFL, college FBS)

Data: nflverse schedules, team and quarterback statistics; college schedules
and results. Features: Elo with margin, opponent-adjusted points and expected
points added per play (slow, fast), pass/rush matchups, turnovers, the
incumbent quarterback's shrunk efficiency (whoever started the team's previous
game), rest, byes, short weeks, travel, division games, roof and surface.
Seasons are short, so ratings carry over with regression to the mean. The
schedule file's listed quarterback is not always the one who played: for a
completed game the starter is the listed quarterback only if he has a stat
line for that team in that game, otherwise the team's quarterback with the
most plays.

- **NFL winner** (validated): 705 games, Sep 2024 - Nov 2026, accuracy 65.4%
  (62.0-68.6), log loss 0.627; Elo 0.631; the de-vigged price in the schedule
  file 0.603. Knowing the actual starting quarterback would give 0.619.
  *Live predictions use one input the backtest did not.* The schedule file
  lists a projected starter for upcoming games. It is a third-party
  projection, not a team announcement; it is stored as a "probable"
  statement and, when present, replaces the incumbent in the quarterback
  feature. That substitution has not been validated. Expected points added
  come from nflfastR's play-value model, which was itself estimated on
  seasons inside the backtest period (a mild look-ahead in the scale, not in
  the results).
- **NFL spread and total** (experimental): scored against the historical line
  in the schedule file. No candidate beat the 50% base rate on development
  folds, so the base rate is the selected "model" and every such prediction is
  PASS. This is the honest result for an efficient market.
- **College football winner** (validated): 2,168 games, accuracy 70.8%
  (68.7-72.7), log loss 0.546; Elo 0.578. Results-and-ratings only: no
  quarterback or injury information.

Observed temperature and wind are retrospective and excluded from selectable
models. Sixteen tied NFL games have no binary outcome and are excluded from
scoring. Travel features treat an international-series game as a normal road
game for the designated visitor; the neutral venue's distance from both
teams is not modelled.

### Hockey (NHL)

Data: schedules and results, goalie box scores, located shot attempts from
play-by-play. Features: Elo, opponent-adjusted goals (slow, fast; shoot-out
goals not counted), five-on-five expected goals from a shot-location model
fitted on the burn-in seasons only, shots on goal, special teams, the shrunk
save percentage of each team's recent primary goalie, rest, back-to-backs,
travel.

- **Winner incl. overtime and shoot-out** (validated): 3,090 games, Oct 2024 -
  Nov 2026, accuracy 57.0%, log loss 0.672; Elo 0.679; base rate 0.689.
- **Regulation three-way** (validated): log loss 1.050; base rate 1.069.

The starting goalie is assumed, not known; the actual starter is a
retrospective-only feature and a required pregame check. Skater injuries are
not modelled. The expected-goals model is deliberately simple (distance,
angle, shot type).

### Baseball (MLB)

Data: Retrosheet game logs 2000-2025. Features: Elo, opponent-adjusted runs
(slow, fast), park factor from prior seasons, day/night, rest, travel,
doubleheaders, bullpen workload in the preceding days.

- **Winner** (validated): 4,949 games, 2024-2025, accuracy 56.5%, log loss
  0.6798; Elo 0.6800; base rate 0.6912. **The gain over Elo is negligible.**
  With the actual starting pitcher (retrospective) the log loss is 0.6774.

The selectable models do not know who starts: probable starters as announced
are not in the source. No 2026 game log or future schedule is in the mirror,
so there are no upcoming MLB events and ratings end on 2025-11-01.

### Soccer (internationals and five club leagues)

Data: international results (date-only), club results with shots, and a
bookmaker 1X2 price per club match used **only** as a benchmark. Three-way
market. Features: Elo, opponent-adjusted attack and defence on goals (slow,
fast, form), expected goal difference and total, home advantage and neutral
venues, rest, friendlies versus competitive matches, promoted clubs, shots on
target. Expected-goals (xG) data was not available from any verified source.

Final-test three-way log loss (lower is better): women's internationals 0.772
and Ligue 1 0.992 (validated); men's internationals 0.843, La Liga 0.979,
Serie A 0.980, Bundesliga 1.000 and Premier League 1.015 (experimental:
calibration error above 0.03). The bookmaker benchmark is better in all five
club leagues (0.956-0.994). The benchmark column in the source is Bet365 in
most seasons and appears to be a multi-bookmaker average from 2019/20 to
2024/25; the file does not say when a quote was taken.

**Serie A changed status between runs.** It passed the calibration gate at
0.028 and, after a fix to kick-off-time parsing that left its log loss
unchanged (0.980), failed it at 0.033. With about 800 three-way matches in a
final test the calibration estimate is too noisy for the gate to be stable
near its threshold. The newest run is the one reported; treat "validated" on
a club league as marginal.

No line-ups, injuries or suspensions are in the data. International scores
include extra time, so a match won in extra time counts as a win although it
was a draw after 90 minutes. The men's international source also lists
non-FIFA representative sides; they are rated on the same scale. Club results
in the source end on 2026-09-20; the built-in "results current" check keeps
every later prediction PROVISIONAL. A team renamed upstream stops the ingest
with an instruction to add an alias, rather than being merged automatically.

### Tennis (ATP, WTA, qualifying, Challenger, WTA 125, ITF; singles)

Data: a mirror of Jeff Sackmann's match archive (results, surface, round,
best-of, serve and return counts), 2000 to mid-2026. Each level is its own
league with its own model, but **ratings are pooled within a tour** (all men's
levels together, all women's together), so a player arriving at tour level
from Challengers carries a rating earned there. Qualifying and main draw are
separate leagues; doubles is not modelled (CSV import only).

Features: all-surface Elo, surface-specific Elo, opponent-adjusted service
points won against the opponent's return (long-term and recent), days since
the previous tournament, matches in the previous 14 days, experience, and a
best-of-five interaction. Rankings, seedings and head-to-head records are
deliberately not inputs. Side A is assigned by player ID, never by strength,
so the base rate is a coin flip by construction.

<!-- results:start sport=tennis -->
| League | Market | Status | Final-test events | Accuracy | Log loss | Base rate | Rating baseline | Market benchmark |
|---|---|---|---|---|---|---|---|---|
| ATP Tour main draw singles | winner | validated | 3,898 | 66.0% | 0.610 | 0.694 | 0.621 |  |
| ATP Challenger main draw singles | winner | validated | 9,950 | 63.5% | 0.638 | 0.693 | 0.640 |  |
| ATP Tour qualifying singles | winner | validated | 1,799 | 65.0% | 0.618 | 0.693 | 0.636 |  |
| ITF men's singles | winner | experimental | 43,092 | 71.1% | 0.561 | 0.693 | 0.583 |  |
| ITF women's singles | winner | validated | 25,094 | 70.0% | 0.570 | 0.693 | 0.585 |  |
| WTA Tour main draw singles | winner | validated | 3,673 | 66.7% | 0.604 | 0.693 | 0.612 |  |
| WTA 125 main draw singles | winner | validated | 2,341 | 65.6% | 0.611 | 0.694 | 0.618 |  |
| WTA Tour qualifying singles | winner | validated | 3,278 | 67.5% | 0.590 | 0.693 | 0.599 |  |
<!-- results:end -->

Selected: a full logistic model for ATP and WTA main draw and WTA
qualifying, boosted trees for ATP qualifying, Challenger and both ITF
leagues, and the two-rating (all-surface plus surface) logistic model for
WTA 125. On development folds the selected model beat the Elo baseline in
every season of every tennis league. ITF men is experimental only because
its history allows two walk-forward folds where the gate asks for three.

What this does **not** establish:

- **Dates are tournament weeks, not match days.** A match uses nothing from
  its own tournament or from any tournament starting the same week, including
  earlier rounds, so the features are point-in-time but coarser than a real
  pregame view: in-tournament form and fatigue are invisible. A two-week event
  counts as finished after its first week; that reaches other players'
  ratings only.
- **Live use would lag the backtest slightly.** A match's features are
  complete from the Monday of its week. A prediction made before that (on
  the Sunday, say) does not yet know the week just ending, so it is built on
  less than the backtest gave the same match.
- **An ingestion bug was found by audit and fixed.** Until 2026-11-13 every
  main-draw quarter-final was dropped (the round code "QF" was read as a
  qualifying round): about one match in ten per league, missing from the
  pooled ratings too. The numbers here are from runs made after the fix; the
  earlier runs remain in the run history and were slightly optimistic,
  because quarter-finals are harder to call than earlier rounds.
- **No live use yet.** The archive is a snapshot with no upcoming matches, no
  order of play and no withdrawals. Tennis produces no predictions until a
  schedule is imported by CSV (names are matched to the archive's players;
  give the surface and best-of). The "results current" check will hold those
  PROVISIONAL, because the newest result on file is from mid-2026.
- No injury, fitness or withdrawal information; no court speed, altitude,
  indoor/outdoor or weather. Retirements count as wins for the player who
  advanced; walkovers are excluded.
- No market benchmark was available. Tennis markets are known to be sharp;
  beating an Elo baseline says nothing about beating a price.
- The archive's licence is CC BY-NC-SA 4.0: **non-commercial use only**.

### Motorsport (Formula 1)

**What is predicted.** For every starter of a Grand Prix, three yes-or-no
questions, each its own market with its own validation: does he win, finish
on the podium (top three), finish in the points (top ten). A driver who does
not take the start has no row; his markets are void.

**Data.** The F1DB database (CC BY 4.0): every race with its classification,
qualifying and starting grid. The model uses 2000 onwards - 523 races, 10,905
driver starts to 8 November 2026.

**Inputs, all from races on earlier days.**

- *The starting grid* - the strongest pregame fact; a pit-lane start counts
  as last.
- *Driver form*: recent and longer-run results on a 0-1 scale relative to the
  field of each race, qualifying form, and how he qualifies against his
  team-mates, which separates the driver from the car.
- *Car form*: the constructor's recent and longer-run results and
  qualifying. Most of that weight is dropped at a season change, when cars
  are new.
- *Field strength*: driver and car form relative to this race's field, and
  the car's rank in it.
- *Track fit*: how the driver has gone at this circuit before relative to
  his form then, and how the car goes at this type of circuit.
- *Reliability*: how often the driver, the car and the field as a whole have
  recently failed to be classified.

The official starting grid of a past race counts as known from two hours
before its start; the as-of test rebuilds the frame at past instants under
that rule and finds zero difference. For a race to come, the grid has to be
imported, and a prediction uses the newest grid read by then.

**Candidates, simplest first:** the base rate; the grid position alone; form
without the grid (what is known before qualifying); a logistic model on
everything; a model of the whole race - Plackett-Luce strengths fitted to
finishing orders, a retirement model, and a simulated finish distribution
whose probabilities for one race add up as they must; and boosted trees.

<!-- results:start field -->
| Series | Market | Status | Selected model | Final-test races | Driver starts | Log loss | Base rate | Grid position alone | Improvement over the grid (95% interval) | Possible 'yes' selections: won / stated (n) | Their calibration error | Model's top picks that did, per race | First on the grid that did, per race |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Formula 1 | podium finish (top three) | experimental | full_logit | 112 | 2,258 | 0.215 | 0.421 | 0.235 | 0.020 (0.009 to 0.031) | 73.2% / 70.4% (265) | 0.044 | 1.98 of 3 | 1.99 of 3 |
| Formula 1 | points finish (top ten) | experimental | field_rank | 112 | 2,258 | 0.463 | 0.694 | 0.506 | 0.043 (0.026 to 0.061) | 82.2% / 81.9% (1000) | 0.031 | 7.86 of 10 | 7.70 of 10 |
| Formula 1 | race winner (each starter: yes or no) | experimental | full_logit | 112 | 2,258 | 0.101 | 0.197 | 0.110 | 0.009 (-0.000 to 0.018) | 75.0% / 62.7% (44) | 0.169 | 0.54 of 1 | 0.58 of 1 |
<!-- results:end -->

**What the runs show.** Development folds are the seasons 2012 to 2021; the
final test is 2022 to 2026 (112 races, 2,258 driver starts). **All three
markets are experimental.**

- The grid alone is a strong baseline: for the win market it takes log loss
  from 0.193 (base rate) to 0.107 on development folds. Form without the
  grid reaches 0.120 - qualifying tells more than the season so far.
- Adding form, track fit and reliability to the grid helps every market on
  development folds, in 9 or 10 of 10 seasons. On the final test the
  improvement over the grid alone is 0.020 for the podium (95% interval 0.009
  to 0.031) and 0.043 for points (0.026 to 0.061). For the win market it is
  0.009 with an interval of -0.000 to 0.018: **not shown to be better than
  reading the grid**, and worse than the grid in 2025 and 2026.
- For win and podium the plain logistic model was selected: the whole-race
  model was fractionally better on development folds, within one standard
  error, and the simpler one wins a tie. For points the whole-race model was
  selected outright.
- **Better probabilities, not better picks.** On the final test the driver
  the model rates most likely to win won 54% of races; the pole-sitter won
  58%. The model's three for the podium included 1.98 podium finishers per
  race, the first three on the grid 1.99; its ten for points 7.86 against
  7.70.
- **Calibration is judged on what would be offered.** Only "yes" is ever a
  selection, so the selection tables, the confidence bands and the
  calibration gate are built from the rows where "yes" is the likelier
  answer. In the final test that is 44 rows for the win market and 265 for
  the podium - too few to call the offered side calibrated (the minimum is
  300) - and 1,000 for points, whose calibration error is 0.031 against a
  limit of 0.03. The points model is too confident at the top: over all
  out-of-sample rows, picks stated at 90-95% won 88%, and those at 95% and
  above won 92%. Over *all* rows the calibration error is under 0.013 for
  every market, because most rows are confident, correct "no" answers - which
  is exactly why that figure is not the one used.
- The headline "accuracy" of these markets is not a useful number for the
  same reason: saying "nobody wins" is right 96% of the time.

**What the labels rest on.** Rows are drivers, but the independent events are
races, and a gate counts them: 112 in the final test, against a minimum of
100. Intervals are resampled by race. Four runs are on record for each market:

1. the first, which passed every gate then defined;
2. with a gate added that the model must beat the grid alone;
3. with the qualifying gap to pole removed from the inputs - a dry run showed
   that a grid read before a race carries slots, not lap times (on development
   folds that input had been worth 0.002 log loss or less);
4. after an independent review: the gate against the grid recomputed with
   20,000 resamples instead of 500 (the win market's pass had depended on the
   random seed), calibration judged on the offered side, and drivers who were
   on the grid but did not start put back into the simulated field.

Runs 1 to 3 called all three markets validated. Run 4 is the one reported.
The final test was read four times, by models that differed slightly; the
labels did not get easier as a result, they got stricter.

**Live use.** `sportspred import grid` after qualifying, then
`sportspred predict --league f1 --market finish:win` (or `finish:podium`,
`finish:points`). Only the "yes" side is ever a selection. The grid check is
complete only when the grid was read within four hours of the start; a
reading more than 36 hours old builds no prediction; a later grid replaces an
earlier one, and a driver no longer on it has his selection withdrawn. A
selection cannot be FINAL while an earlier race of the season has no result on
file. The two logistic models answer each driver separately, so their
probabilities for one race need not add up exactly (final-test average: 0.92
for the win, 3.11 for the podium); the whole-race model's do.

*Not in the model:* weather and track conditions, tyres and strategy,
upgrades, practice and sprint results, penalties announced after the grid
was read, the start itself. Rule eras differ - points for the top six, then
eight, then ten; far fewer retirements now than in 2000 - and inputs are
relative to each race's field, but older seasons are not the same sport in
every respect. The 2026 rule change falls inside the final test.

### Results-only leagues (CSV import)

Tennis doubles, UFC and boxing, cricket, rugby, volleyball, badminton, table
tennis, esports and international basketball share one model: Elo,
opponent-adjusted scoring where results carry scores, recent form, rest and
experience, with side A chosen by a rule that ignores the result. It knows
nothing sport-specific - no style, reach, weight class, maps, vetoes,
pairings or line-ups - and has not been validated on real data here.
Head-to-head records are not used at all, which avoids overweighting tiny
samples. These leagues are adapter-only until results are imported and a
validation run is recorded.

## Selection rules and confidence

A probability becomes a selection only if the league-market's coverage status
is allowed, enough inputs are present, the probability clears the configured
minimum (0.60 two-way, 0.45 three-way, 0.55 over/under by default) and, for
two-way and over/under markets, its model-uncertainty interval stays above a
coin flip (a three-way selection has no such test). Where a market can end
in a push (a whole-number threshold or line), the recorded probability is the
chance of winning and the push is recorded as an outcome of its own; the
minimum, the coin-flip test and the confidence band are applied to the
probability given a decision, because a push is graded as neither hit nor
miss. By default both
"validated" and "experimental" league-markets may produce selections; the
status is shown on every prediction. These are explicit,
editable rules; the validation report shows accuracy and coverage at each
threshold on both development folds and the final test.

Confidence labels are assigned from the **observed** out-of-sample hit rate of
the probability band the selection falls in (High: lower 95% bound at least
70%; Medium: at least 60%; otherwise Low; Unrated below 150 validated events
in the band). The label's text always carries the observed rate, its interval
and the sample size.

## Combos

Legs from different games of one league are combined using the validated
same-slate residual correlation (NBA: +0.006, 95% interval -0.000 to +0.013
over 80,517 event pairs, so independence is a justified approximation there).
Same-game legs and repeated participants are rejected. The top-three rule was
back-tested over historical slates: for the NBA, development slates 50.6%
estimated versus 49.4% observed (1,755 slates), final-test slates 50.7% versus
57.2% (271 slates, interval 51.2-63.0%: more won than estimated, which one
favourable stretch can produce). Selecting the most confident legs also
selects the most overstated ones: at a leg minimum of 0.80 the development
slates show 64.6% estimated versus 58.7% observed. The NHL shows the rule failing: on
development slates three-leg combos were estimated at 38.9% and won 31.5%
(317 slates, interval 26.7-36.9%); its 67 final-test slates went the other
way (35.0% estimated, 49.3% observed), and the NFL has 40. Those final-test
samples are too small to conclude anything. The backtest covers "most
probable qualifying legs of one league, multiplied"; it does not cover the
FINAL-status filter, combos across leagues, or the dependency handling.

## Known limitations

- Only the NBA backtest includes availability news, and only the official
  report as it stood an hour before the day's first game. No other league's
  backtest has any; late scratches and confirmed line-ups are in none. Real
  pregame performance depends on checks the history cannot reproduce.
- Market benchmarks are undated in their sources; no timestamped odds history
  was available, so "does it beat an executable price" is untested.
- Several final-test periods include the current, partial season.
- MLB, club-soccer and tennis results in the reachable sources lag
  (2025-11-01, 2026-09-20 and mid-2026). A prediction built on results that
  old is held PROVISIONAL by the "results current" check, which recognises
  both unplayed fixtures on file and a source that has simply stopped (newest
  result older than 1.5 typical off-seasons of that league).
- Tennis is validated only on tournament-week dates and has no live schedule
  feed. Tennis doubles, combat sports, cricket, rugby, volleyball, racket
  sports and esports have no validated model; golf, NASCAR, IndyCar and MotoGP
  have no model; NBA spreads and totals are not implemented in this platform.
- Formula 1 predictions need a starting grid imported by hand. All three
  markets are experimental: the win model is not shown to beat the grid
  alone, there are too few possible win and podium selections to judge their
  calibration, and the points selections are slightly too confident. None
  picks winners better than the grid does. The final test is 112 races.
- Player-statistic markets have never been scored against a market
  threshold, only against reference thresholds built from each player's own
  earlier games. They are experimental by construction and carry no
  confidence label. Thirteen player-seasons were found in which the official
  report and the box score spell a rotation player differently (a nickname or
  a different first name); for those the report-based history does not see
  his absences.
- The calibration gate (0.03) is unstable for final tests of under about
  1,000 events: Serie A and the WNBA sit within a few thousandths of it.
- Known and not fixed, from the code audit: MLB's live path has never run on
  a real upcoming game (no 2026 schedule is reachable) and its travel
  time-zone feature does not handle a date-line crossing; tied MLB games are
  dropped; NFL international games use ordinary road-game travel; the NHL
  adapter re-downloads only the current season unless told otherwise.
- Mixed precision: where a league has both timed and date-only events, a
  timed game can see the same-date result of a date-only game between other
  teams. A team's own ratings cannot be affected (it plays once a day), and
  the CSV import refuses one competitor having both kinds on one date.
- The rating code seeds its league mean from the first 200 values of a
  history. This reaches only the earliest events, inside every league's
  burn-in, and cancels in every feature.
- Availability statements with no report behind them, and statements for a
  team that has had no newer report, stay in force until replaced; only an
  "out" older than the player's last game is discarded automatically.
- Selection thresholds are rules chosen for readability, not optimised. Their
  measured effect is in the validation report.
- Each model was validated on one cut of history, and most final-test
  periods were read more than once because runs were repeated after fixes
  (see the run history in the validation report). The repeated reads did not
  drive any model choice, but the final-test numbers are no longer strictly
  first-look numbers. Calibration can drift; `sportspred monitor` compares
  live graded selections with what validation says to expect.
