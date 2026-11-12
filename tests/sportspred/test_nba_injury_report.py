"""Official NBA injury report adapter: versions, statuses, not-yet-submitted
teams, and freshness judged from the publication time."""

from __future__ import annotations

import pandas as pd

from sportspred.adapters.nba_injury_report import import_latest_report, normalize_injury_report
from sportspred.config import Settings
from sportspred.predict.checks import evaluate_checks
from sportspred.sports.base import CheckSpec

# report_datetime is US Eastern clock time; 2030-01-01 13:30 ET = 18:30 UTC.
RAW = pd.DataFrame([
    {"report_datetime": "2030-01-01 09:30:00", "team": "Boston Celtics", "report_player_name": "Old, News",
     "current_status": "Questionable", "reason": "Injury/Illness - Knee"},
    {"report_datetime": "2030-01-01 13:30:00", "team": "Boston Celtics", "report_player_name": "Starter, A",
     "current_status": "Out", "reason": "Injury/Illness - Ankle"},
    {"report_datetime": "2030-01-01 13:30:00", "team": "Boston Celtics", "report_player_name": "Guard Jr., B",
     "current_status": "Questionable", "reason": "Injury/Illness - Back"},
    {"report_datetime": "2030-01-01 13:30:00", "team": "Los Angeles Lakers", "report_player_name": None,
     "current_status": None, "reason": "NOT YET SUBMITTED"},
    {"report_datetime": "2030-01-01 13:30:00", "team": "Seattle Expansion", "report_player_name": "X, Y",
     "current_status": "Out", "reason": "Rest"},
    {"report_datetime": "2030-01-02 23:00:00", "team": "Boston Celtics", "report_player_name": "Future, News",
     "current_status": "Out", "reason": "published after 'now'"},
]).assign(game_date="2030-01-01", report_matchup="LAL@BOS",
          report_datetime=lambda d: pd.to_datetime(d["report_datetime"]))

SPEC = [CheckSpec("injury_report", "injury report", max_age_hours=24.0, recheck_within_hours=3.0)]


def test_versions_are_converted_from_eastern_time_and_names_reordered():
    rep = normalize_injury_report(RAW)
    assert rep["published_at"].iloc[1] == pd.Timestamp("2030-01-01T18:30:00Z")
    assert list(rep["player_name"].iloc[1:3]) == ["A Starter", "B Guard Jr."]
    assert rep["status"].iloc[1] == "out" and pd.isna(rep["status"].iloc[3]) and bool(rep["not_submitted"].iloc[3])


def test_only_the_newest_version_known_at_now_is_stored(store, two_teams, scheduled_event):
    out = import_latest_report(store, RAW, now="2030-01-01T20:00:00Z")
    assert out["version_utc"] == "2030-01-01T18:30:00Z"
    assert (out["statements"], out["teams_reported"], out["teams_not_submitted"], out["unknown_teams"]) == (2, 1, 1, 1)
    avail = store.availability_as_of("nba", "2030-01-01T20:00:00Z")
    assert set(avail["player_name"]) == {"A Starter", "B Guard Jr."}          # neither the old nor the future version
    assert set(avail["published_at_utc"]) == {"2030-01-01T18:30:00Z"}
    log = store.checks_as_of("nba", "2030-01-01T20:00:00Z").set_index("team_id")["outcome"].to_dict()
    assert log == {two_teams[0]: "reported", two_teams[1]: "not_published"}


def test_freshness_is_judged_from_publication_not_from_download(store, two_teams, scheduled_event):
    """The archive is downloaded 30 minutes before the start, but its newest
    version was published nine hours before it: a re-check is still needed."""
    import_latest_report(store, RAW, now="2030-01-02T03:00:00Z")              # start is 03:30Z
    row = {"event_id": scheduled_event, "start_time_utc": pd.Timestamp("2030-01-02T03:30:00Z"),
           "side_a_id": two_teams[0], "side_b_id": two_teams[1]}
    now = pd.Timestamp("2030-01-02T03:00:00Z")
    checks = {c["subject"]: c for c in evaluate_checks(SPEC, row, store.checks_as_of("nba", now), now=now,
                                                       settings=Settings(), names={two_teams[0]: "BOS", two_teams[1]: "LAL"})}
    assert checks["BOS"]["state"] == "recheck_needed" and "published" in checks["BOS"]["detail"]
    assert checks["LAL"]["state"] == "missing"                                 # not yet submitted completes nothing
    # A version published inside the window does complete the check.
    fresh = RAW.copy()
    fresh.loc[1:2, "report_datetime"] = pd.Timestamp("2030-01-01 21:00:00")    # 02:00 UTC, 90 minutes before the start
    import_latest_report(store, fresh.iloc[[1, 2]], now="2030-01-02T03:00:00Z")
    again = {c["subject"]: c for c in evaluate_checks(SPEC, row, store.checks_as_of("nba", now), now=now,
                                                      settings=Settings(), names={two_teams[0]: "BOS", two_teams[1]: "LAL"})}
    assert again["BOS"]["state"] == "complete"
