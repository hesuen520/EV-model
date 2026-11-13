"""Regression tests for the defects found by the independent audit of the
shared core (grading, availability, combos, event updates, statuses, ledger)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sp_helpers import prediction
from sportspred.core.results import clean_row, flag, regulation_result
from sportspred.ledger import grade_predictions
from sportspred.ledger.grading import grade_one, rescheduled


def _teams(store, league, sport, names):
    frame = pd.DataFrame({"name": names})
    return list(store.resolve_participants(frame, sport=sport, scope=league))


def _event(store, league, sport, ids, eid, start="2030-01-02T19:00:00Z", status="scheduled", **more):
    row = {"league": league, "season": "2030", "start_time_utc": start, "status": status, "provider": "test",
           "provider_event_id": eid, "home_id": ids[0], "away_id": ids[1], **more}
    return store.upsert_events(pd.DataFrame([row]), sport=sport)[0].iloc[0]["event_id"]


# ------------------------------------------------------------------ grading
def test_missing_values_are_never_read_as_true():
    assert flag(float("nan")) is None and flag(None) is None and flag(pd.NA) is None
    assert flag(1.0) is True and flag(0) is False
    assert clean_row({"a": float("nan"), "b": 2, "c": pd.NaT}) == {"a": None, "b": 2, "c": None}


def test_three_way_grading_is_not_a_draw_just_because_another_league_has_overtime_flags(store):
    """Events of several leagues are loaded into one frame; a soccer match has
    no overtime flag, which arrives as NaN beside an NHL row's 0/1."""
    epl = _teams(store, "epl", "soccer", ["Arsenal", "Chelsea"])
    nhl = _teams(store, "nhl", "hockey", ["Boston Bruins", "Montreal Canadiens"])
    soccer = _event(store, "epl", "soccer", epl, "s1")
    hockey = _event(store, "nhl", "hockey", nhl, "h1")
    store.insert_prediction(prediction(soccer, epl, market="winner_3way", rec_status="FINAL", selection=epl[0],
                                       probs={epl[0]: 0.5, "draw": 0.25, epl[1]: 0.25}, prob=0.5))
    store.insert_prediction(prediction(hockey, nhl, market="winner_3way", rec_status="FINAL", selection="draw",
                                       probs={nhl[0]: 0.4, "draw": 0.25, nhl[1]: 0.35}, prob=0.25))
    store.upsert_results(pd.DataFrame([{"event_id": soccer, "home_score": 3, "away_score": 0}]))
    store.upsert_results(pd.DataFrame([{"event_id": hockey, "home_score": 3, "away_score": 2, "went_overtime": 1,
                                        "went_shootout": 0}]))
    graded = grade_predictions(store).set_index("league")
    assert graded.loc["epl", "outcome"] == "correct"            # 3-0 at home is a home win, not a draw
    assert graded.loc["nhl", "outcome"] == "correct"            # needed overtime: level after regulation
    detail = json.loads(store.grades().set_index("prediction_id").loc[graded.loc["epl", "prediction_id"], "result_json"])
    assert detail["regulation_result"] == epl[0] and detail["went_overtime"] is None


def test_regulation_result_is_unknown_when_an_overtime_sport_does_not_say_whether_overtime_was_played():
    row = {"sport": "hockey", "home_id": "H", "away_id": "A", "home_score": 3.0, "away_score": 2.0,
           "reg_home_score": np.nan, "reg_away_score": np.nan, "went_overtime": np.nan, "went_shootout": np.nan}
    assert regulation_result(row)[0] is None
    assert regulation_result({**row, "went_overtime": 0.0})[0] == "H"
    assert regulation_result({**row, "went_shootout": 1.0})[0] == "draw"
    assert regulation_result({**row, "reg_home_score": 2.0, "reg_away_score": 2.0})[0] == "draw"
    assert regulation_result({**row, "sport": "soccer"})[0] == "H"     # no overtime inside a league score


def test_one_event_marked_final_without_a_result_does_not_stop_the_grading_run(store, two_teams, scheduled_event):
    other = _event(store, "nba", "basketball", two_teams[::-1], "402", start="2030-01-03T03:30:00Z")
    store.insert_prediction(prediction(scheduled_event, two_teams, rec_status="FINAL", prob=0.7,
                                       probs={two_teams[0]: 0.7, two_teams[1]: 0.3}))
    store.insert_prediction(prediction(other, two_teams, rec_status="FINAL", prob=0.6,
                                       probs={two_teams[0]: 0.6, two_teams[1]: 0.4}))
    # The provider marks the first game final but no score ever arrives; the second has a result.
    row = {"league": "nba", "season": "2030", "start_time_utc": "2030-01-02T03:30:00Z", "status": "final",
           "provider": "espn", "provider_event_id": "401", "home_id": two_teams[0], "away_id": two_teams[1]}
    store.upsert_events(pd.DataFrame([row]), sport="basketball")
    store.upsert_results(pd.DataFrame([{"event_id": other, "home_score": 90, "away_score": 101}]))
    graded = grade_predictions(store)
    assert len(graded) == 1 and graded.iloc[0]["outcome"] == "correct"


def _pred(start, date_only=False):
    checks = [{"key": "start_time_published", "state": "missing"}] if date_only else []
    return {"event_start_utc": start, "checks_json": json.dumps(checks), "display_json": "{}"}


def test_publishing_a_kick_off_time_is_not_a_reschedule_but_a_move_to_another_day_is():
    # Predicted when only the date was known (stored at 00:00Z); the kick-off is later published for that date.
    assert rescheduled(_pred("2030-03-09T00:00:00Z", date_only=True),
                       {"start_time_utc": "2030-03-09T19:30:00Z", "time_precision": "datetime"}) is None
    # ... also when the local evening kick-off falls on the next UTC day.
    assert rescheduled(_pred("2030-03-09T00:00:00Z", date_only=True),
                       {"start_time_utc": "2030-03-10T02:00:00Z", "time_precision": "datetime"}) is None
    # Still date-only, but a different date: that is a move.
    moved = rescheduled(_pred("2030-03-09T00:00:00Z", date_only=True),
                        {"start_time_utc": "2030-03-16T00:00:00Z", "time_precision": "date"})
    assert moved and moved["moved_hours"] == 168.0
    # A timed game delayed by ten minutes is the same game; the next day is not.
    timed = _pred("2030-03-09T19:30:00Z")
    assert rescheduled(timed, {"start_time_utc": "2030-03-09T19:40:00Z", "time_precision": "datetime"}) is None
    assert rescheduled(timed, {"start_time_utc": "2030-03-10T19:30:00Z", "time_precision": "datetime"})["moved_hours"] == 24.0


def test_a_prediction_on_a_date_only_event_is_graded_after_its_time_is_published(store, two_teams):
    eid = _event(store, "nba", "basketball", two_teams, "d1", start="2030-03-09T00:00:00Z", time_precision="date")
    store.insert_prediction(prediction(
        eid, two_teams, rec_status="PROVISIONAL", prob=0.7, probs={two_teams[0]: 0.7, two_teams[1]: 0.3},
        checks=[{"key": "start_time_published", "label": "start time published", "subject": "event",
                 "state": "missing", "detail": "date only"}]))
    _event(store, "nba", "basketball", two_teams, "d1", start="2030-03-09T23:30:00Z", status="final",
           time_precision="datetime")
    store.upsert_results(pd.DataFrame([{"event_id": eid, "home_score": 110, "away_score": 100}]))
    graded = grade_predictions(store)
    assert len(graded) == 1 and graded.iloc[0]["outcome"] == "correct"
    pred, ev = store.predictions().iloc[0], store.events(event_ids=[eid]).iloc[0].to_dict()
    assert grade_one(store, pred, ev)[0] == "correct"


# -------------------------------------------------------------- availability
def test_a_player_dropped_from_a_newer_report_is_no_longer_listed(store, two_teams):
    """A report lists who is doubtful now.  The store used to keep the latest
    statement per player for ever, so a recovered player stayed 'out'."""
    team = two_teams[0]
    r1, r2, r3 = (store.add_source("test", external_id=f"report-{i}", sha256=f"r{i}") for i in (1, 2, 3))
    first = pd.DataFrame([
        {"league": "nba", "team_id": team, "player_name": "Star One", "status": "out",
         "collected_at_utc": "2030-01-01T18:00:00Z", "source_id": r1},
        {"league": "nba", "team_id": team, "player_name": "Old Hand", "status": "questionable",
         "collected_at_utc": "2030-01-01T18:00:00Z", "source_id": r1},
    ])
    store.add_availability(first)
    store.log_check("nba", "injury_report", "reported", checked_at="2030-01-01T18:00:00Z", team_id=team,
                    source_id=r1)
    assert set(store.availability_as_of("nba", "2030-01-01T19:00:00Z")["player_name"]) == {"Star One", "Old Hand"}

    # Nine days later the report lists only a different player.
    store.add_availability(pd.DataFrame([{
        "league": "nba", "team_id": team, "player_name": "Bench Guy", "status": "doubtful",
        "collected_at_utc": "2030-01-10T18:00:00Z", "source_id": r2}]))
    store.log_check("nba", "injury_report", "reported", checked_at="2030-01-10T18:00:00Z", team_id=team,
                    source_id=r2)
    now = store.availability_as_of("nba", "2030-01-10T19:00:00Z")
    assert list(now["player_name"]) == ["Bench Guy"]
    # History is not rewritten: as of the earlier date the earlier report still stands.
    assert set(store.availability_as_of("nba", "2030-01-05T00:00:00Z")["player_name"]) == {"Star One", "Old Hand"}

    # A later league-wide report that lists nobody for the team clears it.
    store.log_check("nba", "injury_report", "nothing_listed", checked_at="2030-01-12T18:00:00Z", team_id=team,
                    source_id=r3)
    assert store.availability_as_of("nba", "2030-01-12T19:00:00Z").empty


def test_a_report_of_another_kind_does_not_clear_the_injury_list(store, two_teams):
    team = two_teams[0]
    injuries = store.add_source("test", external_id="injuries", sha256="i")
    lineups = store.add_source("test", external_id="lineups", sha256="l")
    store.add_availability(pd.DataFrame([{
        "league": "nba", "team_id": team, "player_name": "Star One", "status": "out",
        "collected_at_utc": "2030-01-01T18:00:00Z", "source_id": injuries}]))
    store.log_check("nba", "injury_report", "reported", checked_at="2030-01-01T18:00:00Z", team_id=team,
                    source_id=injuries)
    store.log_check("nba", "lineup", "nothing_listed", checked_at="2030-01-01T22:00:00Z", team_id=team,
                    source_id=lineups)
    assert list(store.availability_as_of("nba", "2030-01-01T23:00:00Z")["player_name"]) == ["Star One"]
    # A statement with no logged report behind it is kept, as before.
    store.add_availability(pd.DataFrame([{
        "league": "nba", "team_id": team, "player_name": "Loose Note", "status": "questionable",
        "collected_at_utc": "2030-01-01T10:00:00Z"}]))
    assert "Loose Note" in set(store.availability_as_of("nba", "2030-01-01T23:00:00Z")["player_name"])


def test_a_check_cannot_be_published_after_it_was_read(store, two_teams):
    from sportspred.storage.store import StoreError

    with pytest.raises(StoreError, match="later than its checked_at"):
        store.log_check("nba", "injury_report", "reported", checked_at="2030-01-01T18:00:00Z",
                        published_at="2030-01-01T20:00:00Z", team_id=two_teams[0])


# ------------------------------------------------------------------- combos
def _final_slate(store, probs=(0.85, 0.80, 0.78, 0.76, 0.66)):
    teams = pd.DataFrame({"name": [f"Team {i}" for i in range(2 * len(probs))]})
    ids = list(store.resolve_participants(teams, sport="basketball", scope="nba"))
    rows = [{"league": "nba", "season": "2030", "start_time_utc": "2030-01-02T03:30:00Z", "status": "scheduled",
             "provider": "t", "provider_event_id": str(i), "home_id": ids[2 * i], "away_id": ids[2 * i + 1]}
            for i in range(len(probs))]
    events, _ = store.upsert_events(pd.DataFrame(rows), sport="basketball")
    pids = []
    for i, (eid, p) in enumerate(zip(events["event_id"], probs)):
        pids.append(store.insert_prediction(prediction(
            eid, [ids[2 * i], ids[2 * i + 1]], prob=p, probs={ids[2 * i]: p, ids[2 * i + 1]: 1 - p},
            rec_status="FINAL", validation_status="validated", prob_low=p - 0.02, prob_high=p + 0.02,
            display={"side_a": f"Team {2 * i}", "side_b": f"Team {2 * i + 1}", "side_a_role": "home"})))
    store.save_validation_run({"sport": "basketball", "league": "nba", "market": "winner", "gates_passed": True,
                               "same_day_residual_correlation": {"rho": 0.004, "low": -0.002, "high": 0.011,
                                                                 "n_pairs": 50000}})
    return rows, list(events["event_id"]), ids, pids


def test_a_cancelled_event_is_not_a_combo_leg_even_though_its_prediction_was_final(store):
    from datetime import date
    from sportspred.combos import build_slate_combos

    rows, eids, _, pids = _final_slate(store)
    before = build_slate_combos(store, date(2030, 1, 1), now="2030-01-01T12:00:00Z", write=False)
    assert pids[0] in {l.prediction_id for l in before.three_leg[0].legs}
    store.cancel_events([eids[0]], "called off")
    after = build_slate_combos(store, date(2030, 1, 1), now="2030-01-01T12:00:00Z", write=False)
    assert pids[0] not in {l.prediction_id for l in after.three_leg[0].legs}
    assert any("event is now canceled" in x["reason"] for x in after.excluded)


def test_a_moved_event_drops_out_of_its_old_slate_and_its_old_prediction_is_not_used(store):
    from datetime import date
    from sportspred.combos import build_slate_combos

    rows, eids, ids, pids = _final_slate(store)
    # The top game moves two days later; until it is predicted again its old FINAL row must not be a leg.
    store.upsert_events(pd.DataFrame([{**rows[0], "start_time_utc": "2030-01-04T03:30:00Z"}]), sport="basketball")
    moved = build_slate_combos(store, date(2030, 1, 1), now="2030-01-01T12:00:00Z", write=False)
    assert pids[0] not in {l.prediction_id for c in moved.three_leg for l in c.legs}
    assert any("start has changed" in x["reason"] for x in moved.excluded)
    # Predicted again (PROVISIONAL, new start): the superseded FINAL row is gone from the old slate altogether.
    store.insert_prediction(prediction(
        eids[0], ids[:2], created_at_utc="2030-01-01T13:00:00Z", prob=0.85, probs={ids[0]: 0.85, ids[1]: 0.15},
        rec_status="PROVISIONAL", validation_status="validated", supersedes_id=pids[0],
        display={"side_a": "Team 0", "side_b": "Team 1", "side_a_role": "home"}))
    again = build_slate_combos(store, date(2030, 1, 1), now="2030-01-01T14:00:00Z", write=False)
    assert pids[0] not in {l.prediction_id for c in again.three_leg for l in c.legs}
    assert eids[0] not in set(store.predictions(slate=date(2030, 1, 1), latest_only=True)["event_id"])
    assert eids[0] in set(store.predictions(slate=date(2030, 1, 3), latest_only=True)["event_id"])


def test_a_new_line_supersedes_the_old_one_for_the_same_event_and_market(store, two_teams, scheduled_event):
    first = store.insert_prediction(prediction(
        scheduled_event, two_teams, market="total", line=44.5, selection="over", selection_label="Over 44.5",
        probs={"over": 0.58, "under": 0.42}, prob=0.58, rec_status="FINAL"))
    second = store.insert_prediction(prediction(
        scheduled_event, two_teams, created_at_utc="2030-01-01T06:00:00Z", market="total", line=45.5,
        selection="over", selection_label="Over 45.5", probs={"over": 0.52, "under": 0.48}, prob=0.52,
        rec_status="PASS", supersedes_id=first))
    latest = store.predictions(latest_only=True)
    assert list(latest["prediction_id"]) == [second]            # one current row per event and market
    assert len(store.predictions()) == 2                        # the first is still stored


def test_a_date_only_event_belongs_to_the_slate_of_its_own_date(store, two_teams):
    from datetime import date
    from sportspred.core.slates import slate_date_of

    eid = _event(store, "nba", "basketball", two_teams, "d9", start="2030-03-09T00:00:00Z", time_precision="date")
    store.insert_prediction(prediction(
        eid, two_teams, rec_status="PROVISIONAL", prob=0.7, probs={two_teams[0]: 0.7, two_teams[1]: 0.3},
        checks=[{"key": "start_time_published", "label": "start time published", "subject": "event",
                 "state": "missing", "detail": "date only"}]))
    row = store.predictions().iloc[0]
    assert slate_date_of(row) == date(2030, 3, 9)                # 00:00Z is the previous afternoon in Honolulu
    assert len(store.predictions(slate=date(2030, 3, 9))) == 1
    assert store.predictions(slate=date(2030, 3, 8)).empty


# ------------------------------------------------------------ event updates
def test_an_event_without_a_provider_id_can_be_updated_by_importing_it_again(store, two_teams):
    """A CSV row need not carry an event_id.  Re-importing such a row used to
    be treated as a foreign duplicate: nothing changed and an issue was logged."""
    def row(**over):
        return pd.DataFrame([{"league": "nba", "season": "2030", "start_time_utc": "2030-01-02T03:30:00Z",
                              "status": "scheduled", "provider": "csv", "home_id": two_teams[0],
                              "away_id": two_teams[1], **over}])

    first, _ = store.upsert_events(row(), sport="basketball")
    eid = first.iloc[0]["event_id"]
    again, report = store.upsert_events(row(), sport="basketball")
    assert again.iloc[0]["event_id"] == eid and report.unchanged == 1 and report.duplicates == 0
    assert store.issues("nba").empty

    _, report = store.upsert_events(row(start_time_utc="2030-01-02T04:30:00Z"), sport="basketball")
    assert report.updated == 1
    assert store.events(event_ids=[eid], with_results=False).iloc[0]["start_time_utc"] == pd.Timestamp("2030-01-02T04:30:00Z")

    _, report = store.upsert_events(row(start_time_utc="2030-01-02T04:30:00Z", status="postponed"), sport="basketball")
    assert report.updated == 1
    assert store.events(event_ids=[eid], with_results=False).iloc[0]["status"] == "postponed"
    assert len(store.events("nba", with_results=False)) == 1


def test_a_stale_schedule_row_does_not_reopen_a_game_whose_result_is_on_file(store, two_teams, scheduled_event):
    final = {"league": "nba", "season": "2030", "start_time_utc": "2030-01-02T03:30:00Z", "status": "final",
             "provider": "espn", "provider_event_id": "401", "home_id": two_teams[0], "away_id": two_teams[1]}
    store.upsert_events(pd.DataFrame([final]), sport="basketball")
    store.upsert_results(pd.DataFrame([{"event_id": scheduled_event, "home_score": 101, "away_score": 99}]))
    _, report = store.upsert_events(pd.DataFrame([{**final, "status": "scheduled"}]), sport="basketball")
    ev = store.events(event_ids=[scheduled_event]).iloc[0]
    assert ev["status"] == "final" and ev["home_score"] == 101
    assert any("finished game" in msg for msg in report.issues)
    # An explicit cancellation is still honoured.
    store.upsert_events(pd.DataFrame([{**final, "status": "canceled"}]), sport="basketball")
    assert store.events(event_ids=[scheduled_event]).iloc[0]["status"] == "canceled"


# ------------------------------------------------------------------- ledger
def test_insert_or_replace_cannot_rewrite_an_append_only_row(store, two_teams, scheduled_event):
    """REPLACE deletes the old row without firing DELETE triggers unless
    recursive triggers are on; they are, on every connection."""
    import sqlite3

    pid = store.insert_prediction(prediction(scheduled_event, two_teams))
    with pytest.raises(sqlite3.Error, match="immutable"):
        with store.conn() as con:
            cols = [r["name"] for r in con.execute("PRAGMA table_info(predictions)")]
            row = dict(con.execute("SELECT * FROM predictions WHERE prediction_id=?", (pid,)).fetchone())
            row["prob"] = 0.99
            con.execute(f"INSERT OR REPLACE INTO predictions ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                        [row[c] for c in cols])
    assert store.predictions().iloc[0]["prob"] == 0.6 and store.verify_ledger()[0]


def test_a_real_events_prediction_cannot_be_backdated(store, two_teams, monkeypatch):
    """The ledger compares the stated creation time with its own clock, so a
    'pregame' record cannot be written afterwards with an earlier time."""
    from sportspred.core.timeutil import iso_z, utcnow
    from sportspred.storage import Store
    from sportspred.storage.store import LedgerError

    monkeypatch.setattr(Store, "wall_clock_tolerance_seconds", 300.0)
    now = pd.Timestamp(utcnow())
    eid = _event(store, "nba", "basketball", two_teams, "live1", start=iso_z(now + pd.Timedelta(hours=5)))
    with pytest.raises(LedgerError, match="does not match this machine's clock"):
        store.insert_prediction(prediction(eid, two_teams, created_at_utc=iso_z(now - pd.Timedelta(days=2))))
    with pytest.raises(LedgerError, match="does not match this machine's clock"):
        store.insert_prediction(prediction(eid, two_teams, created_at_utc=iso_z(now + pd.Timedelta(hours=2))))
    pid = store.insert_prediction(prediction(eid, two_teams, created_at_utc=iso_z(now)))
    row = store.predictions().set_index("prediction_id").loc[pid]
    assert abs((pd.Timestamp(row["recorded_at_utc"]) - now).total_seconds()) < 60
    assert store.verify_ledger()[0]              # the recorded clock time is part of the hash
    # Synthetic demo leagues replay invented dates and are exempt, but labelled synthetic.
    demo = list(store.resolve_participants(pd.DataFrame({"name": ["D1", "D2"]}), sport="demo",
                                           scope="demo_league", is_synthetic=True))
    old = store.upsert_events(pd.DataFrame([{
        "league": "demo_league", "season": "2020", "start_time_utc": "2020-01-02T00:00:00Z", "status": "scheduled",
        "provider": "synthetic", "provider_event_id": "x", "home_id": demo[0], "away_id": demo[1],
        "is_synthetic": True}]), sport="demo")[0].iloc[0]["event_id"]
    store.insert_prediction(prediction(old, demo, created_at_utc="2020-01-01T00:00:00Z", is_synthetic=True))
    assert len(store.predictions(include_synthetic=True)) == 2 and store.verify_ledger()[0]


# ------------------------------------------------------------------- engine
DEMO = "demo_basketball"
DEMO_NOW = "2026-11-12T12:00:00Z"


@pytest.fixture(scope="module")
def demo_world(tmp_path_factory):
    from sportspred.config import Settings
    from sportspred.pipeline import train_league, validate_league
    from sportspred.sports import get_league
    from sportspred.storage import Store

    settings = Settings()
    settings.paths.data_dir = str(tmp_path_factory.mktemp("core_audit"))
    settings.validation.n_boot, settings.validation.n_replicas = 40, 4
    settings.selection.allowed_coverage = ["validated", "experimental", "demo-only"]
    settings.selection.horizon_hours = 72.0
    store = Store(settings.paths.store)
    plugin, _ = get_league(DEMO)
    plugin.ingest(store, DEMO, now=DEMO_NOW)
    report, _ = validate_league(store, DEMO, "winner", settings=settings, now=DEMO_NOW)
    train_league(store, DEMO, "winner", settings=settings, now=DEMO_NOW)
    upcoming = store.upcoming(DEMO, now=DEMO_NOW, horizon_hours=24, include_synthetic=True)
    start = upcoming["start_time_utc"].min()
    for team in set(upcoming["home_id"]) | set(upcoming["away_id"]):
        store.log_check(DEMO, "injury_report", "nothing_listed", checked_at=(start - pd.Timedelta(hours=2)).isoformat(),
                        team_id=team, is_synthetic=True)
    return {"settings": settings, "store": store, "plugin": plugin, "report": report,
            "near": (start - pd.Timedelta(hours=1)).isoformat()}


def test_a_pick_with_no_side_is_conditional_on_either_teams_doubtful_players(demo_world, monkeypatch):
    """Over, under and draw have no 'own side'.  They used to be FINAL with
    questionable players on both teams, because only a side pick was tested."""
    from sportspred.core.enums import OutcomeType
    from sportspred.predict import predict_league
    from sportspred.sports.base import AvailabilityEffect

    store, settings, plugin = demo_world["store"], demo_world["settings"], demo_world["plugin"]
    kind = type(plugin)
    monkeypatch.setattr(kind, "apply_availability", lambda self, league, market, row, frame, avail: AvailabilityEffect(
        row=row, uncertain_a=["Home Star (questionable)"], uncertain_b=["Away Star (questionable)"]))
    side_picks = predict_league(store, DEMO, settings=settings, now=demo_world["near"], horizon_hours=6, write=False)
    assert {p["rec_status"] for p in side_picks} <= {"CONDITIONAL", "PASS"}
    assert any(p["rec_status"] == "CONDITIONAL" for p in side_picks)

    original = kind.feature_frame

    def as_total(self, store_, league, market, **kw):          # the same events, posed as an over/under market
        frame = original(self, store_, league, market, **kw)
        frame.outcome, frame.classes = OutcomeType.OVER_UNDER, ("under", "over")
        frame.df["line"] = 200.5
        return frame

    monkeypatch.setattr(kind, "feature_frame", as_total)
    monkeypatch.setattr(settings.selection, "min_prob_over_under", 0.60)
    totals = predict_league(store, DEMO, settings=settings, now=demo_world["near"], horizon_hours=6, write=False)
    picks = [p for p in totals if p["rec_status"] != "PASS"]
    assert picks and {p["rec_status"] for p in picks} == {"CONDITIONAL"}
    reasons = " ".join(r for p in picks for r in p["reasons"])
    assert "Home Star" in reasons and "Away Star" in reasons    # both sides' doubts are conditions


def test_the_validation_label_is_not_lent_to_a_model_the_newest_run_did_not_select(demo_world):
    from sportspred.predict import predict_league

    store, settings, report = demo_world["store"], demo_world["settings"], demo_world["report"]
    ok = predict_league(store, DEMO, settings=settings, now=demo_world["near"], horizon_hours=6, write=False)
    assert any(p["rec_status"] != "PASS" for p in ok)
    other = "elo_logit" if report["selected_model"] != "elo_logit" else "ratings_logit"
    store.save_validation_run({**report, "run_id": "val_later_run", "selected_model": other,
                               "created_at_utc": "2099-01-01T00:00:00Z"})
    stale = predict_league(store, DEMO, settings=settings, now=demo_world["near"], horizon_hours=6, write=False)
    assert stale and {p["rec_status"] for p in stale} == {"PASS"}
    assert all(any("is not the candidate the newest validation run selected" in r for r in p["reasons"]) for p in stale)
    # Restore the consistent state for any later test in this module.
    store.save_validation_run({**report, "run_id": "val_restored", "created_at_utc": "2099-01-02T00:00:00Z"})


def test_a_changed_line_is_recorded_even_when_the_probability_is_the_same(demo_world, monkeypatch):
    from sportspred.core.enums import OutcomeType
    from sportspred.predict import predict_league

    store, settings, plugin = demo_world["store"], demo_world["settings"], demo_world["plugin"]
    kind = type(plugin)
    original = kind.feature_frame
    line = {"value": 200.5}

    def as_total(self, store_, league, market, **kw):
        frame = original(self, store_, league, market, **kw)
        frame.outcome, frame.classes = OutcomeType.OVER_UNDER, ("under", "over")
        frame.df["line"] = line["value"]
        return frame

    monkeypatch.setattr(kind, "feature_frame", as_total)
    first = predict_league(store, DEMO, settings=settings, now=demo_world["near"], horizon_hours=6)
    line["value"] = 207.5
    second = predict_league(store, DEMO, settings=settings, now=demo_world["near"], horizon_hours=6)
    assert first and len(second) == len(first)
    assert not any(p.get("unchanged") for p in second)
    assert {p["line"] for p in second} == {207.5} and all(p.get("supersedes_id") for p in second)
    latest = store.predictions(latest_only=True, include_synthetic=True)
    assert set(latest["line"]) == {207.5}


# ------------------------------------------------------- combos and reviews
def test_independence_needs_a_tight_interval_not_just_one_that_contains_zero():
    from sportspred.combos import classify_pair
    from sportspred.combos.dependency import Leg

    def leg(i):
        return Leg(prediction_id=f"p{i}", event_id=f"e{i}", league="nba", sport="basketball", market="winner",
                   line=None, selection=f"t{i}", label=f"t{i}", prob=0.7, prob_low=0.67, rec_status="FINAL",
                   validation_status="validated", start_utc="2030-01-02T03:30:00Z",
                   participants=frozenset({f"t{i}", f"u{i}"}), entities=frozenset({f"t{i}"}))

    tight = {("nba", "winner"): {"rho": 0.004, "low": -0.002, "high": 0.011, "n_pairs": 50000}}
    vague = {("nba", "winner"): {"rho": 0.029, "low": -0.30, "high": 0.50, "n_pairs": 40}}
    assert classify_pair(leg(1), leg(2), tight, 0.03).treatment == "independent"
    unknown = classify_pair(leg(1), leg(2), vague, 0.03)
    assert unknown.treatment == "copula" and "conservative figure uses -0.3000" in unknown.note


def test_a_review_is_written_again_when_a_corrected_result_changes_the_grade(store, two_teams, scheduled_event):
    from sportspred.ledger import review_predictions
    from sportspred.ledger.review import review_summary

    store.insert_prediction(prediction(scheduled_event, two_teams, rec_status="FINAL", prob=0.7,
                                       probs={two_teams[0]: 0.7, two_teams[1]: 0.3}))
    store.upsert_results(pd.DataFrame([{"event_id": scheduled_event, "home_score": 105, "away_score": 104}]))
    grade_predictions(store)
    first = review_predictions(store)
    assert len(first) == 1 and first.iloc[0]["outcome"] == "correct"
    assert review_predictions(store).empty                       # nothing changed: no second review
    store.upsert_results(pd.DataFrame([{"event_id": scheduled_event, "home_score": 99, "away_score": 104}]))
    grade_predictions(store)
    second = review_predictions(store)
    assert len(second) == 1 and second.iloc[0]["outcome"] == "incorrect"
    assert len(store.reviews()) == 2                              # the first review is kept
    summary = review_summary(store)
    assert summary["outcomes"] == {"incorrect": 1} and sum(summary["categories"].values()) == 1


def test_two_provider_ids_sharing_a_name_are_merged_on_the_record_not_silently(store):
    first = store.resolve_participants(pd.DataFrame({"name": ["Julian Alonso"], "pid": [103031]}), sport="tennis",
                                       scope="tennis_men", kind="player", provider="archive", provider_id_col="pid")
    second = store.resolve_participants(pd.DataFrame({"name": ["Julian Alonso"], "pid": [211655]}), sport="tennis",
                                        scope="tennis_men", kind="player", provider="archive", provider_id_col="pid")
    assert first.iloc[0] == second.iloc[0]
    issues = store.issues("tennis_men")
    assert list(issues["kind"]) == ["name_shared_by_provider_ids"] and "211655" in issues.iloc[0]["detail"]
    # The same ID seen again raises nothing new.
    store.resolve_participants(pd.DataFrame({"name": ["Julian Alonso"], "pid": [211655]}), sport="tennis",
                               scope="tennis_men", kind="player", provider="archive", provider_id_col="pid")
    assert len(store.issues("tennis_men")) == 1
