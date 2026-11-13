"""Regression tests for defects found when independent reviewers tried to
break the fixes made after the audits (settlement, combo ledger, concurrent
writers, availability reports, results-current, EV and dashboard state)."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import textwrap
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sp_helpers import prediction
from sportspred.core.timeutil import iso_z, utcnow
from sportspred.storage import Store
from sportspred.storage.store import LedgerError

SRC = str(Path(__file__).resolve().parents[2] / "src")


def _teams(store, league, sport, names):
    return list(store.resolve_participants(pd.DataFrame({"name": names}), sport=sport, scope=league))


def _event(store, league, sport, ids, eid, start, status="scheduled", **more):
    row = {"league": league, "season": "2030", "start_time_utc": start, "status": status, "provider": "test",
           "provider_event_id": eid, "home_id": ids[0], "away_id": ids[1], **more}
    return store.upsert_events(pd.DataFrame([row]), sport=sport)[0].iloc[0]["event_id"]


# --------------------------------------------------------------- settlement
def test_a_final_event_with_no_result_is_unsettled_not_a_loss(store, two_teams, scheduled_event):
    from sportspred.ev.rules import RULES, settle_outcome

    row = {"league": "nba", "season": "2030", "start_time_utc": "2030-01-02T03:30:00Z", "status": "final",
           "provider": "espn", "provider_event_id": "401", "home_id": two_teams[0], "away_id": two_teams[1]}
    store.upsert_events(pd.DataFrame([row]), sport="basketball")
    other = _event(store, "nba", "basketball", two_teams[::-1], "402", "2030-01-03T03:30:00Z")
    store.upsert_results(pd.DataFrame([{"event_id": other, "home_score": 90, "away_score": 101}]))
    events = store.events("nba").set_index("event_id")          # two rows: the missing winner arrives as NaN
    outcome, why = settle_outcome(RULES["moneyline_incl_ot"], "winner", two_teams[0], None,
                                  events.loc[scheduled_event].to_dict())
    assert outcome == "unsettled" and "no winner" in why
    assert settle_outcome(RULES["moneyline_incl_ot"], "winner", two_teams[0], None,
                          events.loc[other].to_dict())[0] == "win"


# ------------------------------------------------------------- combo ledger
def _real_slate(store, hours_ahead=5.0, n=3, status="FINAL"):
    """Real (non-synthetic) events starting soon, predicted now - so the clock rule can stay on."""
    now = pd.Timestamp(utcnow())
    ids = _teams(store, "nba", "basketball", [f"Team {i}" for i in range(2 * n)])
    pids, eids = [], []
    for i in range(n):
        eid = _event(store, "nba", "basketball", ids[2 * i:2 * i + 2], f"g{i}", iso_z(now + pd.Timedelta(hours=hours_ahead)))
        eids.append(eid)
        pids.append(store.insert_prediction(prediction(
            eid, ids[2 * i:2 * i + 2], created_at_utc=iso_z(now), rec_status=status, prob=0.8,
            probs={ids[2 * i]: 0.8, ids[2 * i + 1]: 0.2}, validation_status="validated")))
    return now, eids, pids


def test_a_combo_of_real_predictions_cannot_be_backdated_or_outrank_its_legs(store, monkeypatch):
    monkeypatch.setattr(Store, "wall_clock_tolerance_seconds", 300.0)
    now, eids, pids = _real_slate(store, status="PROVISIONAL")
    base = {"slate_date_hst": "2030-01-01", "role": "primary_three_leg", "leg_ids": pids, "joint_prob": 0.5,
            "joint_prob_low": 0.45, "joint_method": "product", "rec_status": "PROVISIONAL"}
    with pytest.raises(LedgerError, match="before one of its own legs"):
        store.insert_combo({**base, "created_at_utc": iso_z(now - pd.Timedelta(days=2))})
    with pytest.raises(LedgerError, match="does not match this machine's clock"):
        store.insert_combo({**base, "created_at_utc": iso_z(now + pd.Timedelta(hours=2))})
    with pytest.raises(LedgerError, match="once one of its events has started"):
        store.insert_combo({**base, "created_at_utc": iso_z(now + pd.Timedelta(hours=6))})
    with pytest.raises(LedgerError, match="only as final as its least final leg"):
        store.insert_combo({**base, "created_at_utc": iso_z(utcnow()), "rec_status": "FINAL"})
    with pytest.raises(LedgerError, match="mix synthetic and real"):
        store.insert_combo({**base, "created_at_utc": iso_z(utcnow()), "is_synthetic": True})
    store.insert_combo({**base, "created_at_utc": iso_z(utcnow())})
    combos = store.combos()
    assert len(combos) == 1 and combos.iloc[0]["recorded_at_utc"] and store.verify_ledger() == (True, [])


def test_a_date_only_leg_is_closed_once_its_date_has_begun_anywhere(store, two_teams):
    from sportspred.combos import build_slate_combos

    ids = _teams(store, "nba", "basketball", [f"Team {i}" for i in range(6)])
    for i in range(3):
        eid = _event(store, "nba", "basketball", ids[2 * i:2 * i + 2], f"d{i}", "2030-03-09T00:00:00Z",
                     time_precision="date")
        store.insert_prediction(prediction(eid, ids[2 * i:2 * i + 2], created_at_utc="2030-03-01T00:00:00Z",
                                           rec_status="FINAL", prob=0.8, validation_status="validated",
                                           probs={ids[2 * i]: 0.8, ids[2 * i + 1]: 0.2}))
    store.save_validation_run({"sport": "basketball", "league": "nba", "market": "winner", "gates_passed": True,
                               "same_day_residual_correlation": {"rho": 0.0, "low": -0.01, "high": 0.01,
                                                                 "n_pairs": 9999}})
    early = build_slate_combos(store, date(2030, 3, 9), now="2030-03-08T08:00:00Z", write=False)
    assert early.three_leg                                             # 16 hours before the stored instant
    late = build_slate_combos(store, date(2030, 3, 9), now="2030-03-08T12:00:00Z", write=False)
    assert not late.three_leg and any("only a date" in x["reason"] for x in late.excluded)
    # The record itself says the event had only a date, even without the engine's marker.
    assert '"date_only":true' in store.predictions().iloc[0]["display_json"]


# ------------------------------------------------------------------- ledger
def test_replace_is_refused_on_any_connection_and_combo_edits_are_detected(store, two_teams, scheduled_event):
    pid = store.insert_prediction(prediction(scheduled_event, two_teams))
    store.insert_combo({"created_at_utc": "2030-01-01T01:00:00Z", "slate_date_hst": "2030-01-01", "role": "single",
                        "leg_ids": [pid], "joint_prob": 0.6, "joint_prob_low": 0.55, "joint_method": "single",
                        "rec_status": "PROVISIONAL"})
    raw = sqlite3.connect(store.db_path)                     # a plain connection: no pragma set by the store
    cols = [r[1] for r in raw.execute("PRAGMA table_info(predictions)")]
    row = raw.execute("SELECT * FROM predictions").fetchone()
    with pytest.raises(sqlite3.Error, match="immutable"):
        raw.execute(f"INSERT OR REPLACE INTO predictions ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", row)
    # With the triggers removed a combo can be edited - and the audit says so.
    assert store.verify_ledger() == (True, [])
    for name in ("combos_no_update", "combos_no_replace"):
        raw.execute(f"DROP TRIGGER {name}")
    raw.execute("UPDATE combos SET joint_prob = 0.99")
    raw.commit()
    raw.close()
    ok, problems = store.verify_ledger()
    assert not ok and any("combo content does not match" in p for p in problems)
    assert any("combos_no_update is missing" in p for p in problems)


def test_the_audit_flags_a_real_prediction_written_after_its_event_started(store, two_teams, monkeypatch):
    now = pd.Timestamp(utcnow())
    eid = _event(store, "nba", "basketball", two_teams, "late", iso_z(now - pd.Timedelta(hours=1)))
    # The clock rule switched off (as only the test-suite does): a "pregame" row for a game already under way.
    store.insert_prediction(prediction(eid, two_teams, created_at_utc=iso_z(now - pd.Timedelta(hours=3))))
    assert store.verify_ledger()[0]                           # not audited while the rule is off
    monkeypatch.setattr(Store, "wall_clock_tolerance_seconds", 300.0)
    ok, problems = store.verify_ledger()
    assert not ok and "not before the event start" in problems[0]


def test_concurrent_writers_do_not_break_the_hash_chain(tmp_path):
    store = Store(tmp_path / "store")
    ids = _teams(store, "nba", "basketball", ["A", "B"])
    eids = [_event(store, "nba", "basketball", ids, f"c{i}", "2030-01-02T03:30:00Z", format={"game_number": i + 1})
            for i in range(24)]
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {SRC!r}); sys.path.insert(0, {str(Path(__file__).parent)!r})
        from sp_helpers import prediction
        from sportspred.storage import Store
        Store.wall_clock_tolerance_seconds = None
        st = Store({str(tmp_path / 'store')!r})
        for eid in sys.argv[1:]:
            st.insert_prediction(prediction(eid, {ids!r}))
    """)
    procs = [subprocess.Popen([sys.executable, "-c", script, *eids[k::3]]) for k in range(3)]
    assert [p.wait(timeout=120) for p in procs] == [0, 0, 0]
    assert len(store.predictions()) == 24 and store.verify_ledger() == (True, [])


def test_changes_in_the_same_second_get_their_own_rows(store, two_teams, scheduled_event):
    base = dict(created_at_utc="2030-01-01T00:00:00Z", rec_status="PROVISIONAL", prob=0.6,
                probs={two_teams[0]: 0.6, two_teams[1]: 0.4})
    first = store.insert_prediction(prediction(scheduled_event, two_teams, **base))
    moved = store.insert_prediction(prediction(scheduled_event, two_teams, **{**base, "prob": 0.65,
                                               "probs": {two_teams[0]: 0.65, two_teams[1]: 0.35}}, supersedes_id=first))
    relabelled = store.insert_prediction(prediction(scheduled_event, two_teams, **base,
                                                    validation_status="validated", supersedes_id=moved))
    assert len({first, moved, relabelled}) == 3 and store.verify_ledger()[0]


# ------------------------------------------------------------- availability
def test_the_report_in_force_is_the_one_with_the_newest_information(store, two_teams):
    """A live list read at 23:00 against an official report read later but published that
    afternoon: the engine shows the newer information, and the statements must be its too."""
    team = two_teams[0]
    live = store.add_source("espn", external_id="live", sha256="a")
    official = store.add_source("nba", external_id="official", sha256="b")
    store.add_availability(pd.DataFrame([{"league": "nba", "team_id": team, "player_name": "A Starter", "status": "out",
                                          "collected_at_utc": "2030-01-01T23:00:00Z", "source_id": live}]))
    store.log_check("nba", "injury_report", "reported", checked_at="2030-01-01T23:00:00Z", team_id=team, source_id=live)
    store.add_availability(pd.DataFrame([{"league": "nba", "team_id": team, "player_name": "B Guard",
                                          "status": "probable", "published_at_utc": "2030-01-01T14:30:00Z",
                                          "collected_at_utc": "2030-01-01T23:30:00Z", "source_id": official}]))
    store.log_check("nba", "injury_report", "reported", checked_at="2030-01-01T23:30:00Z",
                    published_at="2030-01-01T14:30:00Z", team_id=team, source_id=official)
    assert list(store.availability_as_of("nba", "2030-01-02T00:00:00Z")["player_name"]) == ["A Starter"]


def test_two_versions_from_one_source_and_reports_without_a_source_still_supersede(store, two_teams):
    team = two_teams[0]
    archive = store.add_source("nba", external_id="archive", sha256="x")
    for when, player in (("2030-01-01T18:00:00Z", "First Listed"), ("2030-01-01T22:00:00Z", "Second Listed")):
        store.add_availability(pd.DataFrame([{"league": "nba", "team_id": team, "player_name": player, "status": "out",
                                              "collected_at_utc": when, "source_id": archive}]))
        store.log_check("nba", "injury_report", "reported", checked_at=when, team_id=team, source_id=archive)
    assert list(store.availability_as_of("nba", "2030-01-01T23:00:00Z")["player_name"]) == ["Second Listed"]
    other = two_teams[1]
    for when, player in (("2030-01-01T18:00:00Z", "Old Note"), ("2030-01-01T22:00:00Z", "New Note")):
        store.add_availability(pd.DataFrame([{"league": "nba", "team_id": other, "player_name": player,
                                              "status": "questionable", "collected_at_utc": when}]))
        store.log_check("nba", "injury_report", "reported", checked_at=when, team_id=other)
    now = store.availability_as_of("nba", "2030-01-01T23:00:00Z")
    assert list(now[now["team_id"] == other]["player_name"]) == ["New Note"]


def test_a_league_wide_file_clears_a_team_it_no_longer_lists_even_with_no_game_soon(store, tmp_path):
    from sportspred.adapters.csv_import import import_availability, import_events

    league = "cricket_t20_men"
    events = tmp_path / "e.csv"
    events.write_text("event_id,season,start_time,status,home,away\n"
                      "e1,2030,2030-06-20T14:00:00Z,scheduled,Alpha,Beta\n")       # weeks away
    import_events(store, league, events, create_participants=True)
    first = tmp_path / "a1.csv"
    first.write_text("team,player,status\nAlpha,Opener,out\n")
    import_availability(store, league, first, league_wide=True, collected_at="2030-06-01T10:00:00Z")
    second = tmp_path / "a2.csv"
    second.write_text("team,player,status\nBeta,Keeper,questionable\n")
    import_availability(store, league, second, league_wide=True, collected_at="2030-06-03T10:00:00Z")
    assert list(store.availability_as_of(league, "2030-06-04T00:00:00Z")["player_name"]) == ["Keeper"]
    assert list(store.availability_as_of(league, "2030-06-02T00:00:00Z")["player_name"]) == ["Opener"]


# ---------------------------------------------------------- results current
def test_an_event_marked_final_without_a_score_is_not_a_result(store, two_teams):
    from sportspred.predict.engine import _unresolved_past_events

    now = pd.Timestamp("2030-03-10T00:00:00Z")
    done = _event(store, "nba", "basketball", two_teams, "r1", "2030-03-01T03:00:00Z")
    store.upsert_results(pd.DataFrame([{"event_id": done, "home_score": 100, "away_score": 90}]))
    _event(store, "nba", "basketball", two_teams, "r2", "2030-03-08T03:00:00Z", status="final")   # no score ever came
    behind, through, latest, _ = _unresolved_past_events(store, "nba", now)
    assert behind == {two_teams[0]: 1, two_teams[1]: 1} and through == "2030-03-01"
    # A years-old gap of the same kind does not block a league for ever.
    _event(store, "nba", "basketball", two_teams, "r0", "2027-01-05T03:00:00Z", status="final")
    behind, _, _, _ = _unresolved_past_events(store, "nba", now)
    assert behind == {two_teams[0]: 1, two_teams[1]: 1}


def test_the_typical_break_ignores_the_partial_first_and_last_years(store, two_teams):
    from sportspred.predict.engine import _unresolved_past_events

    rows, results = [], []
    for season in range(2022, 2029):                       # October to April seasons, one game a week
        for week in range(26):
            day = pd.Timestamp(f"{season}-10-05T03:00:00Z") + pd.Timedelta(days=7 * week)
            rows.append({"league": "nba", "season": str(season + 1), "start_time_utc": day, "status": "final",
                         "provider": "test", "provider_event_id": f"{season}-{week}", "home_id": two_teams[0],
                         "away_id": two_teams[1]})
    events, _ = store.upsert_events(pd.DataFrame(rows), sport="basketball")
    store.upsert_results(events[["event_id"]].assign(home_score=100, away_score=90))
    _, _, _, usual = _unresolved_past_events(store, "nba", pd.Timestamp("2029-10-01T00:00:00Z"))
    assert 180 < usual < 200                                # the off-season, not the 7 days of a partial year
    short = Store(store.root.parent / "short")
    ids = _teams(short, "nba", "basketball", ["A", "B"])
    ev, _ = short.upsert_events(pd.DataFrame([{**r, "home_id": ids[0], "away_id": ids[1]} for r in rows[:60]]),
                                sport="basketball")
    short.upsert_results(ev[["event_id"]].assign(home_score=100, away_score=90))
    assert _unresolved_past_events(short, "nba", pd.Timestamp("2025-10-01T00:00:00Z"))[3] is None   # too little history


# ----------------------------------------------------------- EV and dashboard
def test_ev_evaluation_and_the_dashboard_follow_the_event_as_it_stands_now(store, two_teams, scheduled_event):
    from sportspred.dashboard import build_dashboard
    from sportspred.ev.core import event_state_reasons

    pid = store.insert_prediction(prediction(
        scheduled_event, two_teams, rec_status="FINAL", prob=0.7, probs={two_teams[0]: 0.7, two_teams[1]: 0.3},
        display={"competition": "NBA", "side_a": "Boston Celtics", "side_b": "Los Angeles Lakers",
                 "side_a_role": "home", "start_hst": "x"}))
    pred = store.predictions().iloc[0]
    event = store.events(event_ids=[scheduled_event], with_results=False).iloc[0]
    assert event_state_reasons(pred, event, "2030-01-01T12:00:00Z") == []
    store.insert_combo({"created_at_utc": "2030-01-01T01:00:00Z", "slate_date_hst": "2030-01-01", "role": "single",
                        "leg_ids": [pid], "joint_prob": 0.7, "joint_prob_low": 0.66, "joint_method": "single",
                        "rec_status": "FINAL"})
    store.cancel_events([scheduled_event], "called off")
    event = store.events(event_ids=[scheduled_event], with_results=False).iloc[0]
    assert event_state_reasons(pred, event, "2030-01-01T12:00:00Z") == ["event is now canceled"]
    # A later combo run (nothing qualifies any more) replaces the earlier recommendation on the page.
    store.insert_combo({"created_at_utc": "2030-01-01T02:00:00Z", "slate_date_hst": "2030-01-01", "role": "none",
                        "leg_ids": [], "rec_status": "PASS", "message": "PASS — insufficient support for a three-leg combo."})
    page = build_dashboard(store, now="2030-01-01T12:00:00Z")
    assert "This event is now canceled" in page
    combos = page[page.index("<h3>Combos</h3>"):page.index("</section>", page.index("<h3>Combos</h3>"))]
    assert "insufficient support" in combos and "single" not in combos and "nan" not in combos
