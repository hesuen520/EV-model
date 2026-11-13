# CSV import

The CSV import is the supported way to load any league whose feed cannot be
reached, and the only way to supply availability reports where no feed exists.
It works for every league key in `sportspred coverage --all`, including the
ones that also have a public-dataset adapter.

```bash
sportspred templates my_csv/                       # writes four example files
sportspred import events       --league cricket_t20_men --csv my_csv/events.csv --create-participants
sportspred import results      --league cricket_t20_men --csv my_csv/results.csv
sportspred import availability --league nba --csv injuries.csv --league-wide
sportspred import checks       --league nhl --csv checks.csv
```

After importing results: `sportspred validate --league <key>`, then `train`,
then `predict`. A league stays **adapter-only** in the coverage table until a
validation run on imported real data is on file.

## Rules that apply to every file

- **A wrong row is an error, with its line number.** Nothing is skipped or
  repaired, and nothing is written unless the whole file is valid.
- **Timestamps need a zone**: `2026-11-13T00:30:00Z`, `2026-11-12 19:30-05:00`,
  or pass `--tz America/New_York` to state the zone the file is written in.
  A zone is never guessed. Everything is stored in UTC and shown in
  Pacific/Honolulu.
- **Blank cells stay missing.** Nothing is filled in or estimated.
- **Unknown names are errors** unless `--create-participants` is given (use it
  for the first import of a league), so a typo cannot create a phantom team.
- **Every import is recorded** as a source (file path, SHA-256, collection
  time). Re-importing the same file changes nothing.
- Lines starting with `#` are comments.

## events

| Column | Required | Meaning |
|---|---|---|
| `season` | yes | season label used for validation blocks (a year works best) |
| `start_time` | yes | start instant; with `time_precision=date` a calendar date is enough |
| `status` | yes | `scheduled`, `live`, `final`, `postponed`, `canceled`, `unknown` |
| `home`, `away` | yes | the two sides. No home side (tennis, combat, esports)? put them in any order and set `neutral_site=1`; the model orders sides by a rule that ignores the result |
| `event_id` | no | your stable ID; keeps the event's identity through a reschedule and lets `results` refer to it |
| `time_precision` | no | `datetime` (default), `date` (start time unknown), `period` (only a tournament window known) |
| `neutral_site` | no | `1` neutral, `0` true home game, blank unknown |
| `venue`, `venue_city`, `stage`, `discipline` | no | stored as given; `stage` values `preseason`, `exhibition`, `allstar` are excluded from modelling |
| `best_of`, `surface`, `game_number`, `overs` | no | stored in the event's format |
| `home_score`, `away_score` | no | both or neither; requires `status=final` |
| `winner` | no | name of the winning side, for results without scores |
| `draw` | no | `1` for a drawn event |
| `method` | no | e.g. `KO`, `SUB`, `DEC`, `RET`, `W/O`, `D/L` |

A `final` row must carry a score, a winner or a draw. A row with a result must
be `final`.

## results

`event_id` (required: the ID from the events file, or the store's own event
ID) plus `home_score` + `away_score`, or `winner`, or `draw=1`. A changed
result replaces the old one, which is kept in a history table; graded
predictions are re-graded with a linked new grade.

## availability

| Column | Required | Meaning |
|---|---|---|
| `team` | yes | must already exist in the league |
| `player` | yes | name as published |
| `status` | yes | `out` (confirmed absence), `doubtful`, `questionable`, `probable`, `uncertain` (all four: uncertain availability), `available` (confirmed) |
| `role` | no | e.g. `starting_goalie`, `probable_pitcher`, `starting_qb` |
| `detail` | no | free text |
| `published_at` | no | when the source published this one statement. It dates the statement, not the report: a long-term injury is old news on a current list, so this does not age the pregame check |
| `collected_at` | no | when you read the source (default: the time of the import - state it, or pass `--collected-at`, when importing a file you saved earlier). A prediction only sees statements collected before it |

Three situations are kept apart: a **confirmed absence** (`out`), **uncertain
availability** (the middle four statuses) and **missing information** (no row
at all). Importing a file logs the pregame check (`--check-key`, default
`injury_report`) as *reported* for each team in it. With `--league-wide`, teams
that have an event in the next 48 hours and are not in the file are logged as
*checked, nothing listed* - which completes their check - instead of being
left as *never checked*.

**How old the check is.** Pass `--published-at` with the time the report you
transcribed was published, and the check's age is measured from then. Without
it, the check is as old as its collection time: the file is taken to be the
source's current list at the moment you read it.

## checks

`check_key`, `outcome` (`reported`, `nothing_listed`, `not_published`),
`checked_at`, and a `team` or an `event_id`; optional `published_at`,
`detail`. `not_published` records that you looked and the source had nothing
yet; it does not complete the check.

Each league's required checks and their freshness windows are listed by
`sportspred predict --detail`; they can be changed - tightened or loosened -
in the settings file (`[selection.max_age_hours]`,
`[selection.recheck_within_hours]`). The gates for "validated" are settings
too (`[validation]`). Loosening either weakens what FINAL and "validated"
mean; the defaults are the ones every recorded result here was produced with.

## Prices

Prices are not part of this import. They belong to the optional EV module:
`sportspred ev import-prices` (see `docs/EV_MODULE.md`).
