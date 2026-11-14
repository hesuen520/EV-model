"""Store integrity: time zones, stable IDs, de-duplication, ledger immutability,
synthetic isolation."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime

import pandas as pd
import pytest

from sportspred.core import NaiveDatetimeError, fmt_hst, hst_date, hst_day_bounds_utc, iso_z, parse_ts, stable_id
from sportspred.storage import LedgerError, StoreError, SyntheticIsolationError

from sp_helpers import prediction


def test_naive_timestamps_are_rejected_not_guessed():
    with pytest.raises(NaiveDatetimeError):
        parse_ts("2026-11-12 19:30:00")
    with pytest.raises(NaiveDatetimeError):
        parse_ts(datetime(2026, 11, 12, 19, 30))
    assert iso_z("2026-11-12 19:30:00", assume_tz="America/New_York") == "2026-11-13T00:30:00Z"


def test_hst_display_and_slate_date_cross_midnight_utc():
    # 00:30 UTC on the 13th is 2:30 PM on the 12th in Honolulu (UTC-10, no DST).
    assert fmt_hst("2026-11-13T00:30:00Z") == "Thu Nov 12, 2026 2:30 PM HST"
    assert hst_date("2026-11-13T00:30:00Z") == date(2026, 11, 12)
    # July: still UTC-10 because Hawaii has no daylight saving.
    assert fmt_hst("2026-07-04T05:00:00Z") == "Fri Jul 3, 2026 7:00 PM HST"
    lo, hi = hst_day_bounds_utc(date(2026, 11, 12))
    assert (iso_z(lo), iso_z(hi)) == ("2026-11-12T10:00:00Z", "2026-11-13T10:00:00Z")


def test_stable_ids_are_deterministic_and_accent_insensitive():
    assert stable_id("p", "basketball", "Luka Dončić") == stable_id("p", "basketball", "LUKA  DONCIC")
    assert stable_id("p", "basketball", "a") != stable_id("p", "basketball", "b")


def test_reingest_is_idempotent_and_cross_provider_events_merge(store, two_teams, scheduled_event):
    row = {"league": "nba", "season": "2030", "start_time_utc": "2030-01-02T03:30:00Z", "status": "scheduled",
           "provider": "espn", "provider_event_id": "401", "home_id": two_teams[0], "away_id": two_teams[1]}
    _, again = store.upsert_events(pd.DataFrame([row]), sport="basketball")
    assert (again.inserted, again.unchanged) == (0, 1)
    other = dict(row, provider="nba_api", provider_event_id="0022", start_time_utc="2030-01-02T03:40:00Z")
    out, rep = store.upsert_events(pd.DataFrame([other]), sport="basketball")
    assert rep.merged == 1 and out.iloc[0]["event_id"] == scheduled_event
    assert len(store.events("nba")) == 1


def test_same_provider_distinct_native_ids_are_never_folded(store, two_teams, scheduled_event):
    twin = {"league": "nba", "season": "2030", "start_time_utc": "2030-01-02T04:30:00Z", "status": "scheduled",
            "provider": "espn", "provider_event_id": "402", "home_id": two_teams[0], "away_id": two_teams[1]}
    out, rep = store.upsert_events(pd.DataFrame([twin]), sport="basketball")
    assert rep.inserted == 1 and out.iloc[0]["event_id"] != scheduled_event


def test_event_time_must_be_timezone_aware(store, two_teams):
    bad = {"league": "nba", "season": "2030", "start_time_utc": "2030-01-02 19:30:00", "status": "scheduled",
           "provider": "espn", "provider_event_id": "9", "home_id": two_teams[0], "away_id": two_teams[1]}
    with pytest.raises(NaiveDatetimeError):
        store.upsert_events(pd.DataFrame([bad]), sport="basketball")


def test_reschedule_is_recorded_not_silently_overwritten(store, two_teams, scheduled_event):
    moved = {"league": "nba", "season": "2030", "start_time_utc": "2030-01-05T03:30:00Z", "status": "scheduled",
             "provider": "espn", "provider_event_id": "401", "home_id": two_teams[0], "away_id": two_teams[1]}
    store.upsert_events(pd.DataFrame([moved]), sport="basketball")
    changes = store.event_changes(scheduled_event)
    assert changes[changes["field"] == "start_time_utc"].iloc[0]["old_value"] == "2030-01-02T03:30:00Z"


def test_upcoming_excludes_started_postponed_and_final(store, two_teams, scheduled_event):
    assert len(store.upcoming("nba", now="2030-01-01T00:00:00Z")) == 1
    assert len(store.upcoming("nba", now="2030-01-02T03:30:00Z")) == 0     # at the start: not upcoming
    store.upsert_results(pd.DataFrame([{"event_id": scheduled_event, "home_score": 110, "away_score": 100}]))
    assert len(store.upcoming("nba", now="2030-01-01T00:00:00Z")) == 0     # final is never upcoming


def test_prediction_must_be_pregame(store, two_teams, scheduled_event):
    store.insert_prediction(prediction(scheduled_event, two_teams))
    with pytest.raises(LedgerError, match="pregame"):
        store.insert_prediction(prediction(scheduled_event, two_teams, created_at_utc="2030-01-02T03:30:00Z"))
    with pytest.raises(LedgerError, match="pregame"):
        store.insert_prediction(prediction(scheduled_event, two_teams, created_at_utc="2030-01-03T00:00:00Z"))


def test_no_prediction_for_completed_event(store, two_teams, scheduled_event):
    store.upsert_results(pd.DataFrame([{"event_id": scheduled_event, "home_score": 110, "away_score": 100}]))
    with pytest.raises(LedgerError, match="scheduled"):
        store.insert_prediction(prediction(scheduled_event, two_teams))


@pytest.mark.parametrize("probs", [{"a": 0.7, "b": 0.4}, {"a": 1.2, "b": -0.2}, {"a": float("nan"), "b": 0.5}])
def test_invalid_probabilities_are_rejected(store, two_teams, scheduled_event, probs):
    with pytest.raises(LedgerError):
        store.insert_prediction(prediction(scheduled_event, two_teams, probs=probs))


def test_ledger_is_append_only_and_tamper_evident(store, two_teams, scheduled_event):
    first = store.insert_prediction(prediction(scheduled_event, two_teams))
    second = store.insert_prediction(prediction(
        scheduled_event, two_teams, created_at_utc="2030-01-01T06:00:00Z", prob=0.65,
        probs={two_teams[0]: 0.65, two_teams[1]: 0.35}, supersedes_id=first))
    assert second != first
    assert len(store.predictions()) == 2 and len(store.predictions(latest_only=True)) == 1
    for sql in ("UPDATE predictions SET prob = 0.99", "DELETE FROM predictions"):
        with pytest.raises(sqlite3.DatabaseError, match="immutable"):
            with store.conn() as con:
                con.execute(sql)
    assert store.verify_ledger() == (True, [])
    # An edit that bypasses the triggers is caught by the hash chain.
    with store.conn() as con:
        con.execute("DROP TRIGGER predictions_no_update")
        con.execute("UPDATE predictions SET prob = 0.99 WHERE prediction_id = ?", (first,))
    ok, problems = store.verify_ledger()
    assert not ok and any("edited" in p for p in problems) and any("trigger" in p for p in problems)


def test_synthetic_rows_cannot_enter_a_real_league_and_vice_versa(store, two_teams):
    row = {"league": "nba", "season": "2030", "start_time_utc": "2030-02-02T03:30:00Z", "status": "scheduled",
           "provider": "synthetic", "provider_event_id": "s1", "home_id": two_teams[0], "away_id": two_teams[1]}
    with pytest.raises(SyntheticIsolationError):
        store.upsert_events(pd.DataFrame([dict(row, is_synthetic=True)]), sport="basketball")
    with pytest.raises(SyntheticIsolationError):
        store.upsert_events(pd.DataFrame([dict(row, league="demo_nba")]), sport="basketball")
    out, _ = store.upsert_events(pd.DataFrame([dict(row, league="demo_nba", is_synthetic=True)]), sport="basketball")
    assert len(store.events("demo_nba")) == 0                       # hidden unless asked for
    assert len(store.events("demo_nba", include_synthetic=True)) == 1


def test_result_must_be_consistent_and_corrections_are_kept(store, two_teams, scheduled_event):
    with pytest.raises(StoreError, match="contradicts"):
        store.upsert_results(pd.DataFrame([{"event_id": scheduled_event, "home_score": 90, "away_score": 100,
                                            "winner_id": two_teams[0]}]))
    store.upsert_results(pd.DataFrame([{"event_id": scheduled_event, "home_score": 110, "away_score": 100}]))
    store.upsert_results(pd.DataFrame([{"event_id": scheduled_event, "home_score": 111, "away_score": 100}]))
    assert "result_corrected" in set(store.issues("nba")["kind"])
    assert len(store.query("SELECT * FROM result_history")) == 1


def test_availability_is_point_in_time_and_distinguishes_absent_uncertain_missing(store, two_teams):
    rows = pd.DataFrame([
        {"league": "nba", "team_id": two_teams[0], "player_name": "A Star", "status": "questionable",
         "collected_at_utc": "2030-01-01T18:00:00Z", "published_at_utc": "2030-01-01T17:30:00Z"},
        {"league": "nba", "team_id": two_teams[0], "player_name": "A Star", "status": "out",
         "collected_at_utc": "2030-01-02T01:00:00Z"},
    ])
    assert store.add_availability(rows) == 2
    early = store.availability_as_of("nba", "2030-01-01T20:00:00Z")
    late = store.availability_as_of("nba", "2030-01-02T02:00:00Z")
    assert early.iloc[0]["status"] == "questionable" and late.iloc[0]["status"] == "out"
    # Before anything was collected there is *no information*, which is not "available".
    assert store.availability_as_of("nba", "2030-01-01T00:00:00Z").empty
    with pytest.raises(StoreError):
        store.add_availability(pd.DataFrame([{"league": "nba", "player_name": "x", "status": "maybe",
                                              "collected_at_utc": "2030-01-01T00:00:00Z"}]))
