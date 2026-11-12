"""ESPN adapter parsers, on hand-written fixtures in the observed response
shape.  The live service is not reachable from the development environment,
so these tests cover parsing and storage semantics only."""

from __future__ import annotations

import pandas as pd
import pytest

from sportspred.adapters.espn import import_injuries, import_scoreboard, parse_injuries, parse_scoreboard
from sportspred.predict.checks import evaluate_checks
from sportspred.config import Settings
from sportspred.sports.base import CheckSpec

INJURIES = {"injuries": [
    {"id": "2", "displayName": "Boston Celtics", "injuries": [
        {"status": "Out", "date": "2029-12-31T22:13Z", "athlete": {"id": "111", "displayName": "A Starter"},
         "details": {"type": "Knee", "fantasyStatus": {"description": "OUT"}}, "shortComment": "knee"},
        {"status": "Day-To-Day", "date": "2029-12-31T23:00Z", "athlete": {"id": "112", "displayName": "B Guard"},
         "details": {"type": "Ankle", "fantasyStatus": {"description": "GTD"}}},
        {"status": "Something New", "athlete": {"displayName": "C Wing"}},
        {"status": "Out", "athlete": {}},                                   # no name: dropped
    ]},
    {"id": "9999", "displayName": "Unknown Expansion Team", "injuries": [
        {"status": "Out", "athlete": {"id": "5", "displayName": "X"}}]},
]}

SCOREBOARD = {"events": [
    {"id": "401", "date": "2030-01-02T03:30Z", "season": {"year": 2030, "type": 2},
     "status": {"type": {"name": "STATUS_FINAL", "completed": True}},
     "competitions": [{"neutralSite": False, "venue": {"id": "1", "fullName": "TD Garden",
                                                        "address": {"city": "Boston"}},
                       "competitors": [
                           {"homeAway": "home", "score": "112", "team": {"id": "2", "displayName": "Boston Celtics"}},
                           {"homeAway": "away", "score": "104", "team": {"id": "13", "displayName": "Los Angeles Lakers"}}]}]},
    {"id": "777", "date": "2030-01-02T03:30Z", "season": {"year": 2030},
     "status": {"type": {"name": "STATUS_SCHEDULED"}},
     "competitions": [{"competitors": [
         {"homeAway": "home", "team": {"id": "9999", "displayName": "Unknown Expansion Team"}},
         {"homeAway": "away", "team": {"id": "13", "displayName": "Los Angeles Lakers"}}]}]},
]}


def test_injury_statuses_map_conservatively():
    df = parse_injuries(INJURIES)
    status = dict(zip(df["player_name"], df["status"]))
    assert status == {"A Starter": "out", "B Guard": "questionable", "C Wing": "uncertain", "X": "out"}
    assert df.loc[df["player_name"] == "A Starter", "published_at_utc"].iloc[0] == "2029-12-31T22:13:00Z"
    assert df.loc[df["player_name"] == "C Wing", "published_at_utc"].isna().all()     # unknown stays unknown


def test_injury_import_logs_reported_and_nothing_listed_checks(store, two_teams, scheduled_event):
    out = import_injuries(store, "nba", INJURIES, collected_at="2030-01-02T01:00:00Z")
    assert out == {"reports": 3, "teams_reported": 1, "teams_nothing_listed": 1, "unknown_teams": 1}
    avail = store.availability_as_of("nba", "2030-01-02T02:00:00Z")
    assert set(avail["player_name"]) == {"A Starter", "B Guard", "C Wing"}
    assert store.availability_as_of("nba", "2030-01-02T00:00:00Z").empty          # not visible before collection
    log = store.checks_as_of("nba", "2030-01-02T02:00:00Z")
    by_team = dict(zip(log["team_id"], log["outcome"]))
    assert by_team == {two_teams[0]: "reported", two_teams[1]: "nothing_listed"}

    spec = [CheckSpec("injury_report", "injury report", max_age_hours=24.0, recheck_within_hours=3.0)]
    row = {"event_id": scheduled_event, "start_time_utc": pd.Timestamp("2030-01-02T03:30:00Z"),
           "side_a_id": two_teams[0], "side_b_id": two_teams[1]}
    fresh = evaluate_checks(spec, row, log, now=pd.Timestamp("2030-01-02T02:00:00Z"), settings=Settings())
    assert [c["state"] for c in fresh] == ["complete", "complete"]


def test_checks_are_missing_stale_or_due_a_recheck(store, two_teams, scheduled_event):
    spec = [CheckSpec("injury_report", "injury report", max_age_hours=24.0, recheck_within_hours=3.0)]
    row = {"event_id": scheduled_event, "start_time_utc": pd.Timestamp("2030-01-02T03:30:00Z"),
           "side_a_id": two_teams[0], "side_b_id": two_teams[1]}
    empty = store.checks_as_of("nba", "2030-01-01T00:00:00Z")
    assert {c["state"] for c in evaluate_checks(spec, row, empty, now=pd.Timestamp("2030-01-01T00:00:00Z"),
                                                settings=Settings())} == {"missing"}
    # Checked 30 hours before the start: fresh at first, later stale; never within the re-check window.
    for team in two_teams:
        store.log_check("nba", "injury_report", "nothing_listed", checked_at="2029-12-31T21:30:00Z", team_id=team)
    early = evaluate_checks(spec, row, store.checks_as_of("nba", "2030-01-01T00:00:00Z"),
                            now=pd.Timestamp("2030-01-01T00:00:00Z"), settings=Settings())
    assert {c["state"] for c in early} == {"recheck_needed"}
    late = evaluate_checks(spec, row, store.checks_as_of("nba", "2030-01-02T03:00:00Z"),
                           now=pd.Timestamp("2030-01-02T03:00:00Z"), settings=Settings())
    assert {c["state"] for c in late} == {"stale"}
    # "The report is not out yet" does not complete a check.
    store.log_check("nba", "injury_report", "not_published", checked_at="2030-01-02T03:00:00Z", team_id=two_teams[0])
    again = evaluate_checks(spec, row, store.checks_as_of("nba", "2030-01-02T03:10:00Z"),
                            now=pd.Timestamp("2030-01-02T03:10:00Z"), settings=Settings())
    assert {c["state"] for c in again} == {"stale"}


def test_scoreboard_updates_known_events_and_skips_unknown_teams(store, two_teams, scheduled_event):
    games = parse_scoreboard(SCOREBOARD, "nba")
    assert list(games["status"]) == ["final", "scheduled"] and str(games["start_time_utc"].iloc[0].tzinfo) == "UTC"
    out = import_scoreboard(store, "nba", SCOREBOARD)
    assert out == {"events": 1, "results": 1, "skipped": 1}
    ev = store.events("nba").iloc[0]
    assert ev["event_id"] == scheduled_event and ev["status"] == "final" and ev["winner_id"] == two_teams[0]
    assert len(store.events("nba")) == 1                       # the unknown-team game was not invented into the store
