-- sportspred durable store.  All timestamps are ISO-8601 UTC strings with a
-- trailing Z at second precision, so string comparison is time comparison.

CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- Provenance of everything that enters the store.
CREATE TABLE IF NOT EXISTS sources (
  source_id TEXT PRIMARY KEY,
  provider TEXT NOT NULL,
  url TEXT,
  external_id TEXT,
  collected_at_utc TEXT NOT NULL,      -- when this installation fetched/imported it
  published_at_utc TEXT,               -- when the provider published it, if known
  sha256 TEXT,
  n_bytes INTEGER,
  license TEXT,
  notes TEXT,
  is_synthetic INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS participants (
  participant_id TEXT PRIMARY KEY,
  sport TEXT NOT NULL,
  scope TEXT NOT NULL,                 -- matching pool: a league key for teams, a tour pool for athletes
  kind TEXT NOT NULL,                  -- team | player | pair | fighter | driver | golfer ...
  name TEXT NOT NULL,
  short_name TEXT,
  country TEXT,
  meta_json TEXT,
  is_synthetic INTEGER NOT NULL DEFAULT 0,
  created_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS participant_xref (
  provider TEXT NOT NULL,
  scope TEXT NOT NULL,
  provider_id TEXT NOT NULL,
  participant_id TEXT NOT NULL REFERENCES participants(participant_id),
  PRIMARY KEY (provider, scope, provider_id)
);

CREATE TABLE IF NOT EXISTS participant_alias (
  scope TEXT NOT NULL,
  alias_norm TEXT NOT NULL,
  participant_id TEXT NOT NULL REFERENCES participants(participant_id),
  source TEXT,
  PRIMARY KEY (scope, alias_norm)
);

CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY,
  sport TEXT NOT NULL,
  league TEXT NOT NULL,
  season TEXT NOT NULL,
  stage TEXT NOT NULL DEFAULT 'regular',   -- regular | playoffs | preseason | qualifying | main_draw | ...
  discipline TEXT,                         -- singles | doubles | ... where relevant
  start_time_utc TEXT NOT NULL,
  time_precision TEXT NOT NULL DEFAULT 'datetime' CHECK (time_precision IN ('datetime','date','period')),
  status TEXT NOT NULL CHECK (status IN ('scheduled','live','final','postponed','canceled','unknown')),
  home_id TEXT,
  away_id TEXT,
  neutral_site INTEGER,                    -- 1 neutral, 0 true home game, NULL unknown
  venue_name TEXT,
  venue_city TEXT,
  venue_id TEXT,
  participants_json TEXT,                  -- multi-competitor fields
  format_json TEXT,                        -- best_of, surface, game_number, map pool, ...
  provider TEXT NOT NULL,
  provider_event_id TEXT,
  source_id TEXT REFERENCES sources(source_id),
  first_seen_utc TEXT NOT NULL,
  updated_at_utc TEXT NOT NULL,
  is_synthetic INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_events_league_start ON events (league, start_time_utc);
CREATE INDEX IF NOT EXISTS ix_events_league_status ON events (league, status);

CREATE TABLE IF NOT EXISTS event_xref (
  provider TEXT NOT NULL,
  league TEXT NOT NULL,
  provider_event_id TEXT NOT NULL,
  event_id TEXT NOT NULL REFERENCES events(event_id),
  PRIMARY KEY (provider, league, provider_event_id)
);

-- Audit trail of changes to an event's start time, status, venue or sides.
CREATE TABLE IF NOT EXISTS event_changes (
  change_id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id TEXT NOT NULL,
  changed_at_utc TEXT NOT NULL,
  field TEXT NOT NULL,
  old_value TEXT,
  new_value TEXT,
  source_id TEXT
);

CREATE TABLE IF NOT EXISTS results (
  event_id TEXT PRIMARY KEY REFERENCES events(event_id),
  home_score REAL,
  away_score REAL,
  winner_id TEXT,
  is_draw INTEGER NOT NULL DEFAULT 0,
  reg_home_score REAL,                     -- score at the end of regulation where overtime exists
  reg_away_score REAL,
  went_overtime INTEGER,
  went_shootout INTEGER,
  method TEXT,                             -- KO/TKO, SUB, DEC, RET, W/O, D/L, forfeit ...
  ranking_json TEXT,                       -- multi-competitor finishing order
  detail_json TEXT,
  completed_at_utc TEXT,
  source_id TEXT REFERENCES sources(source_id),
  verified_at_utc TEXT NOT NULL,
  is_synthetic INTEGER NOT NULL DEFAULT 0
);

-- Result corrections are kept: the previous row is copied here before an overwrite.
CREATE TABLE IF NOT EXISTS result_history (
  history_id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id TEXT NOT NULL,
  replaced_at_utc TEXT NOT NULL,
  previous_json TEXT NOT NULL
);

-- Availability statements exactly as published; rows are never updated, so the
-- state "as of" any past instant can be reconstructed.
CREATE TABLE IF NOT EXISTS availability (
  report_id TEXT PRIMARY KEY,
  league TEXT NOT NULL,
  event_id TEXT,
  team_id TEXT,
  player_id TEXT,
  player_name TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('out','doubtful','questionable','probable','uncertain','available')),
  role TEXT,                               -- starter, QB1, starting_goalie, probable_pitcher ...
  detail TEXT,
  published_at_utc TEXT,
  collected_at_utc TEXT NOT NULL,
  source_id TEXT REFERENCES sources(source_id),
  is_synthetic INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_avail_team ON availability (league, team_id, collected_at_utc);
CREATE INDEX IF NOT EXISTS ix_avail_event ON availability (event_id);

-- A record that a pregame check was performed, including when it found nothing
-- to report.  "Checked, nobody listed" is different from "never checked".
CREATE TABLE IF NOT EXISTS check_log (
  check_id TEXT PRIMARY KEY,
  league TEXT NOT NULL,
  check_key TEXT NOT NULL,                 -- injury_report | lineup | starting_goalie | probable_pitcher | weather | ...
  event_id TEXT,
  team_id TEXT,
  outcome TEXT NOT NULL CHECK (outcome IN ('reported','nothing_listed','not_published')),
  published_at_utc TEXT,
  checked_at_utc TEXT NOT NULL,
  source_id TEXT REFERENCES sources(source_id),
  detail TEXT,
  is_synthetic INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_check_event ON check_log (league, check_key, event_id);
CREATE INDEX IF NOT EXISTS ix_check_team ON check_log (league, check_key, team_id, checked_at_utc);

CREATE TABLE IF NOT EXISTS data_issues (
  issue_id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at_utc TEXT NOT NULL,
  league TEXT,
  event_id TEXT,
  kind TEXT NOT NULL,
  detail TEXT
);

CREATE TABLE IF NOT EXISTS tables_manifest (
  league TEXT NOT NULL,
  name TEXT NOT NULL,
  n_rows INTEGER NOT NULL,
  updated_at_utc TEXT NOT NULL,
  source_ids_json TEXT,
  columns_json TEXT,
  is_synthetic INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (league, name)
);

-- ---------------------------------------------------------------- validation
CREATE TABLE IF NOT EXISTS validation_runs (
  run_id TEXT PRIMARY KEY,
  created_at_utc TEXT NOT NULL,
  sport TEXT NOT NULL,
  league TEXT NOT NULL,
  market TEXT NOT NULL,
  selected_model TEXT,
  gates_passed INTEGER NOT NULL DEFAULT 0,
  is_synthetic INTEGER NOT NULL DEFAULT 0,
  point_in_time INTEGER NOT NULL DEFAULT 0,
  report_json TEXT NOT NULL                -- the full ValidationReport
);
CREATE INDEX IF NOT EXISTS ix_valruns ON validation_runs (league, market, created_at_utc);

CREATE TABLE IF NOT EXISTS models (
  model_version TEXT PRIMARY KEY,
  sport TEXT NOT NULL,
  league TEXT NOT NULL,
  market TEXT NOT NULL,
  model_name TEXT NOT NULL,
  trained_at_utc TEXT NOT NULL,
  train_start_utc TEXT,
  train_end_utc TEXT,
  n_train INTEGER,
  artifact_path TEXT NOT NULL,
  validation_run_id TEXT REFERENCES validation_runs(run_id),
  promoted INTEGER NOT NULL DEFAULT 0,
  is_synthetic INTEGER NOT NULL DEFAULT 0,
  info_json TEXT
);

-- ------------------------------------------------------------------- ledger
-- Append-only.  Triggers reject UPDATE and DELETE; a SHA-256 hash chain makes
-- any out-of-band edit detectable (see Store.verify_ledger).
CREATE TABLE IF NOT EXISTS predictions (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  prediction_id TEXT NOT NULL UNIQUE,
  created_at_utc TEXT NOT NULL,            -- prediction time
  event_id TEXT NOT NULL,
  event_start_utc TEXT NOT NULL,           -- pregame cutoff snapshot used for the check below
  sport TEXT NOT NULL,
  league TEXT NOT NULL,
  market TEXT NOT NULL,
  line REAL,
  selection TEXT NOT NULL,                 -- participant_id | draw | over | under | none
  selection_label TEXT NOT NULL,
  prob REAL,
  prob_low REAL,
  prob_high REAL,
  probs_json TEXT NOT NULL,                -- full outcome distribution (incl. push where relevant)
  model_name TEXT NOT NULL,
  model_version TEXT NOT NULL,
  validation_status TEXT NOT NULL,
  rec_status TEXT NOT NULL CHECK (rec_status IN ('FINAL','PROVISIONAL','CONDITIONAL','PASS')),
  reasons_json TEXT NOT NULL,
  checks_json TEXT NOT NULL,
  confidence_json TEXT NOT NULL,
  factors_json TEXT NOT NULL,
  features_json TEXT NOT NULL,
  data_quality_json TEXT NOT NULL,
  sources_json TEXT NOT NULL,
  display_json TEXT NOT NULL,
  supersedes_id TEXT,
  is_synthetic INTEGER NOT NULL DEFAULT 0,
  prev_hash TEXT,
  content_hash TEXT NOT NULL,
  recorded_at_utc TEXT,                    -- this machine's clock when the row was written (hashed when present)
  CHECK (created_at_utc < event_start_utc),
  CHECK (prob IS NULL OR (prob >= 0.0 AND prob <= 1.0)),
  CHECK (prob_low IS NULL OR prob_high IS NULL OR prob_low <= prob_high)
);
CREATE INDEX IF NOT EXISTS ix_pred_event ON predictions (event_id, market);
CREATE INDEX IF NOT EXISTS ix_pred_league ON predictions (league, event_start_utc);

CREATE TRIGGER IF NOT EXISTS predictions_no_update BEFORE UPDATE ON predictions
BEGIN SELECT RAISE(ABORT, 'predictions are immutable: insert a superseding prediction instead'); END;
CREATE TRIGGER IF NOT EXISTS predictions_no_delete BEFORE DELETE ON predictions
BEGIN SELECT RAISE(ABORT, 'predictions are immutable: they cannot be deleted'); END;

CREATE TABLE IF NOT EXISTS combos (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  combo_id TEXT NOT NULL UNIQUE,
  created_at_utc TEXT NOT NULL,
  slate_date_hst TEXT NOT NULL,
  n_legs INTEGER NOT NULL,
  role TEXT NOT NULL,                      -- primary_three_leg | two_leg_alternative | single | larger | none
  leg_ids_json TEXT NOT NULL,              -- prediction_ids
  joint_prob REAL,
  joint_prob_low REAL,
  joint_method TEXT NOT NULL,
  dependency_json TEXT NOT NULL,
  rec_status TEXT NOT NULL CHECK (rec_status IN ('FINAL','PROVISIONAL','CONDITIONAL','PASS')),
  message TEXT,
  rules_json TEXT NOT NULL,
  is_synthetic INTEGER NOT NULL DEFAULT 0,
  content_hash TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS combos_no_update BEFORE UPDATE ON combos
BEGIN SELECT RAISE(ABORT, 'combos are immutable'); END;
CREATE TRIGGER IF NOT EXISTS combos_no_delete BEFORE DELETE ON combos
BEGIN SELECT RAISE(ABORT, 'combos are immutable'); END;

-- Grading a prediction against the verified sporting result (no prices involved).
CREATE TABLE IF NOT EXISTS grades (
  grade_id INTEGER PRIMARY KEY AUTOINCREMENT,
  prediction_id TEXT NOT NULL,
  graded_at_utc TEXT NOT NULL,
  outcome TEXT NOT NULL CHECK (outcome IN ('correct','incorrect','push','void','no_result')),
  result_json TEXT NOT NULL,
  result_source_id TEXT,
  supersedes_grade_id INTEGER              -- regrade after a result correction
);
CREATE INDEX IF NOT EXISTS ix_grades_pred ON grades (prediction_id);

CREATE TABLE IF NOT EXISTS reviews (
  review_id INTEGER PRIMARY KEY AUTOINCREMENT,
  prediction_id TEXT NOT NULL,
  created_at_utc TEXT NOT NULL,
  category TEXT NOT NULL,                  -- variance | model_error | data_issue | availability_change | unreviewable
  evidence_json TEXT NOT NULL,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS lessons (
  lesson_id TEXT PRIMARY KEY,
  created_at_utc TEXT NOT NULL,
  league TEXT,
  market TEXT,
  title TEXT NOT NULL,
  hypothesis TEXT NOT NULL,
  proposed_change TEXT NOT NULL,
  test_plan TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('proposed','testing','supported','rejected','promoted')),
  evidence_json TEXT,
  validation_run_id TEXT,
  updated_at_utc TEXT NOT NULL
);

-- ------------------------------------------------------- prices (EV module)
-- Written and read only by sportspred.ev.  The prediction engine never joins it.
CREATE TABLE IF NOT EXISTS prices (
  price_id TEXT PRIMARY KEY,
  quoted_at_utc TEXT NOT NULL,             -- timestamp of the quote itself
  collected_at_utc TEXT NOT NULL,
  event_id TEXT NOT NULL,
  league TEXT NOT NULL,
  market TEXT NOT NULL,
  line REAL,
  selection TEXT NOT NULL,
  venue_kind TEXT NOT NULL CHECK (venue_kind IN ('sportsbook','prediction_market','exchange')),
  venue TEXT NOT NULL,
  price_decimal REAL,                      -- sportsbook decimal odds
  contract_price REAL,                     -- prediction market executable price in [0,1]
  is_executable INTEGER NOT NULL DEFAULT 1,
  rules_key TEXT,                          -- settlement rule set this quote is governed by
  detail_json TEXT,
  source_id TEXT REFERENCES sources(source_id),
  is_synthetic INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_prices_event ON prices (event_id, market);

CREATE TABLE IF NOT EXISTS paper_positions (
  position_id TEXT PRIMARY KEY,
  created_at_utc TEXT NOT NULL,
  prediction_id TEXT NOT NULL,
  price_id TEXT NOT NULL,
  stake REAL NOT NULL,
  detail_json TEXT NOT NULL,               -- implied, no-vig, fair odds, edge, EV, fees at entry
  mode TEXT NOT NULL DEFAULT 'paper' CHECK (mode = 'paper')
);

CREATE TABLE IF NOT EXISTS settlements (
  settlement_id INTEGER PRIMARY KEY AUTOINCREMENT,
  position_id TEXT NOT NULL,
  settled_at_utc TEXT NOT NULL,
  outcome TEXT NOT NULL CHECK (outcome IN ('win','loss','push','void','half_win','half_loss','dead_heat')),
  payout REAL NOT NULL,                    -- total returned incl. stake, net of fees
  pnl REAL NOT NULL,
  rules_key TEXT NOT NULL,
  rule_notes TEXT,
  result_source_id TEXT,
  supersedes_settlement_id INTEGER         -- resettlement chain
);
CREATE INDEX IF NOT EXISTS ix_settle_pos ON settlements (position_id);
