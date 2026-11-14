"""Regression tests for the soccer ingest defects found by the audit."""

from __future__ import annotations

import pandas as pd
import pytest

from sportspred.sports import DataUnavailable
from sportspred.sports.base import Explanation, IngestReport
from sportspred.sports.soccer import PLUGIN as SOCCER
from sportspred.sports.soccer.adapter import refuse_mass_disappearance, resolve_teams, write_matches
from sportspred.sports.soccer.clubs import (FIXTURES_PROVIDER, RESULTS_PROVIDER, _retire_played_fixtures,
                                            kickoff_utc)

LEAGUE = "epl"      # a temporary store; rows are not flagged synthetic so the default reads see them


def _report() -> IngestReport:
    return IngestReport(league=LEAGUE)


def _matches(store, n=20, provider=RESULTS_PROVIDER):
    names = pd.Series(["Alpha FC", "Beta FC", "Gamma FC", "Delta FC"])
    ids = list(resolve_teams(store, LEAGUE, names, provider))
    rows = []
    for i in range(n):
        h, a = ids[i % 4], ids[(i + 1 + i // 4) % 4]
        if h == a:
            a = ids[(i + 2) % 4]
        rows.append({
            "league": LEAGUE, "season": "2029", "start_time_utc": pd.Timestamp("2029-01-01", tz="UTC")
            + pd.Timedelta(days=3 * i), "time_precision": "date", "status": "final", "provider": provider,
            "provider_event_id": f"m{i}", "home_id": h, "away_id": a, "home_score": 1, "away_score": 0,
        })
    return write_matches(store, pd.DataFrame(rows), _report()), ids


def test_kickoff_time_is_kept_when_the_first_row_has_no_time():
    dates = pd.Series(pd.to_datetime(["2019-08-09", "2019-08-09", "2020-01-11"]))
    times = pd.Series([None, "20:00", "12:30"])
    start, precision = kickoff_utc(dates, times, "Europe/London")
    assert list(precision) == ["date", "datetime", "datetime"]
    assert start.iloc[0] == pd.Timestamp("2019-08-09T00:00:00Z")
    assert start.iloc[1] == pd.Timestamp("2019-08-09T19:00:00Z")      # British Summer Time
    assert start.iloc[2] == pd.Timestamp("2020-01-11T12:30:00Z")


def test_ingest_stops_when_many_stored_matches_are_missing_from_the_new_file(store):
    events, _ = _matches(store)
    refuse_mass_disappearance(store, LEAGUE, RESULTS_PROVIDER, events["provider_event_id"])
    with pytest.raises(DataUnavailable, match="renamed upstream"):
        refuse_mass_disappearance(store, LEAGUE, RESULTS_PROVIDER, events["provider_event_id"].iloc[:10])
    # Nothing was changed by the refused ingest.
    assert (store.events(LEAGUE, with_results=False)["status"] == "final").all()


def test_fixture_placeholder_is_retired_once_the_result_is_on_file(store):
    events, ids = _matches(store, n=4)
    played = events.iloc[0]
    other_pair = (ids[2], ids[0])
    assert not ((events["home_id"] == other_pair[0]) & (events["away_id"] == other_pair[1])).any()
    placeholders = pd.DataFrame([
        # Same season and pairing as a played match, but the feed has it on a different day.
        {"league": LEAGUE, "season": "2029", "start_time_utc": pd.Timestamp("2029-06-01", tz="UTC"),
         "time_precision": "date", "status": "scheduled", "provider": FIXTURES_PROVIDER,
         "provider_event_id": "fx-played", "home_id": played["home_id"], "away_id": played["away_id"]},
        # A pairing with no result yet must stay scheduled.
        {"league": LEAGUE, "season": "2029", "start_time_utc": pd.Timestamp("2029-06-08", tz="UTC"),
         "time_precision": "date", "status": "scheduled", "provider": FIXTURES_PROVIDER,
         "provider_event_id": "fx-open", "home_id": other_pair[0], "away_id": other_pair[1]},
    ])
    write_matches(store, placeholders, _report())
    report = _report()
    _retire_played_fixtures(store, LEAGUE, report)
    after = store.events(LEAGUE, with_results=False).set_index("provider_event_id")["status"]
    assert after["fx-played"] == "canceled"
    assert after["fx-open"] == "scheduled"
    assert any("retired" in w for w in report.warnings)


def test_draw_selection_is_explained_as_a_draw_not_as_the_home_side():
    row = pd.Series({"elo_diff": 20.0, "elo_a": 1510.0, "elo_b": 1490.0, "goal_sum_exp": 2.1, "home_edge": 1.0,
                     "_pick_class": "draw", "_pick_is_a": False})
    exp = SOCCER._explain_draw(Explanation(), row, "Alpha FC", "Beta FC")
    text = " ".join(exp.supporting)
    assert "closely matched" in text and "low-scoring" in text
    assert any("home advantage favours Alpha FC" in r for r in exp.upset_risks)
