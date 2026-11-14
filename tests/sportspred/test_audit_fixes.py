"""Regression tests for defects found by the independent audit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sportspred.core.enums import OutcomeType
from sportspred.features.ratings import offdef_pregame
from sportspred.sports.base import FeatureFrame
from sportspred.sports.football import PLUGIN as FOOTBALL
from sportspred.sports.hockey import PLUGIN as HOCKEY


def test_many_updates_in_one_batch_cannot_make_a_rating_diverge():
    """A pooled opponent that plays 30 games in every slate used to take the
    sum of 30 updates and oscillate without bound."""
    rng = np.random.default_rng(0)
    n_slates, per_slate = 60, 30
    a_idx = np.tile(np.arange(1, per_slate + 1), n_slates)          # 30 real teams ...
    b_idx = np.zeros(n_slates * per_slate, dtype=int)                # ... all against entity 0
    batch = np.repeat(np.arange(n_slates), per_slate)
    val_a = 35 + rng.normal(0, 10, len(a_idx))                       # teams score 35 on the pooled side
    val_b = 14 + rng.normal(0, 7, len(a_idx))
    out = offdef_pregame(a_idx, b_idx, val_a, val_b, np.zeros(len(a_idx), dtype=int), batch,
                         np.zeros(len(a_idx)), lr=0.25, n_entities=per_slate + 1)
    assert np.abs(out["def_b"]).max() < 40 and np.abs(out["off_b"]).max() < 40
    late = batch >= 40                                               # settled: no slate-to-slate oscillation
    assert np.std(out["def_b"][late]) < 3.0


def test_single_event_batches_are_unchanged_by_the_damping():
    a = np.array([0, 1, 0]); b = np.array([1, 0, 1]); batch = np.array([0, 1, 2])
    va = np.array([110.0, 100.0, 105.0]); vb = np.array([100.0, 108.0, 99.0])
    out = offdef_pregame(a, b, va, vb, np.zeros(3, dtype=int), batch, np.zeros(3), lr=0.2, n_entities=2)
    mean0 = np.concatenate([va, vb]).mean()                          # the starting league mean
    assert out["off_a"][1] == pytest.approx(0.2 * (vb[0] - mean0))   # entity 1's offence after one plain update


def test_partial_table_write_keeps_other_events(store):
    full = pd.DataFrame({"event_id": ["e1", "e2", "e3"], "x": [1, 2, 3]})
    store.write_table("nba", "stats", full)
    store.write_table("nba", "stats", pd.DataFrame({"event_id": ["e3", "e4"], "x": [30, 40]}))   # a later, partial ingest
    got = store.read_table("nba", "stats").set_index("event_id")["x"].to_dict()
    assert got == {"e1": 1, "e2": 2, "e3": 30, "e4": 40}
    store.write_table("nba", "stats", pd.DataFrame({"event_id": ["e9"], "x": [9]}), mode="replace")
    assert store.read_table("nba", "stats")["event_id"].tolist() == ["e9"]


GOALIES = {"H": [{"player_id": "1", "name": "C. Hellebuyck", "start_share": 0.8, "rating": 0.925, "shots": 900.0},
                 {"player_id": "2", "name": "E. Comrie", "start_share": 0.2, "rating": 0.900, "shots": 300.0}],
           "A": [{"player_id": "3", "name": "J. Allen", "start_share": 1.0, "rating": 0.910, "shots": 800.0}]}


def _hockey_row():
    return pd.Series({"event_id": "evt1", "start_time_utc": pd.Timestamp("2030-01-02T00:00:00Z"), "side_a_id": "H",
                      "side_b_id": "A", "goalie_sv_a": 0.925, "goalie_sv_b": 0.910, "goalie_diff": 0.015})


def _hockey_frame():
    return FeatureFrame("nhl", "winner", OutcomeType.BINARY, ("away", "home"), pd.DataFrame(), [],
                        extra={"goalies": GOALIES})


def _stmt(team, player, status, role="starting_goalie", collected="2030-01-01T18:00:00Z", event_id=None):
    return {"team_id": team, "player_name": player, "status": status, "role": role, "player_id": None,
            "collected_at_utc": collected, "event_id": event_id}


def test_goalie_published_under_his_full_name_is_matched():
    eff = HOCKEY.apply_availability("nhl", "winner", _hockey_row(), _hockey_frame(),
                                    pd.DataFrame([_stmt("H", "Eric Comrie", "available")]))
    assert eff.row["goalie_sv_a"] == pytest.approx(0.900) and "E. Comrie confirmed as starting goalie" in eff.applied
    assert not eff.unmatched


def test_unknown_confirmed_starter_leaves_the_input_missing_not_the_wrong_goalie():
    eff = HOCKEY.apply_availability("nhl", "winner", _hockey_row(), _hockey_frame(),
                                    pd.DataFrame([_stmt("H", "Some Newgoalie", "available")]))
    assert np.isnan(eff.row["goalie_sv_a"]) and any("no record of him" in x for x in eff.applied)


def test_questionable_starter_builds_a_worst_case_with_the_backup():
    eff = HOCKEY.apply_availability("nhl", "winner", _hockey_row(), _hockey_frame(),
                                    pd.DataFrame([_stmt("H", "Connor Hellebuyck", "questionable", role="")]))
    assert eff.uncertain_a and eff.row["goalie_sv_a"] == pytest.approx(0.925)
    assert eff.worst_case_rows["a"]["goalie_sv_a"] == pytest.approx(0.900)


def test_old_starter_confirmation_is_not_reused_for_a_later_game():
    old = _stmt("H", "Eric Comrie", "available", collected="2029-12-28T18:00:00Z")
    eff = HOCKEY.apply_availability("nhl", "winner", _hockey_row(), _hockey_frame(), pd.DataFrame([old]))
    assert eff.row["goalie_sv_a"] == pytest.approx(0.925) and any("not for this game" in x for x in eff.unmatched)
    named = _stmt("H", "Eric Comrie", "available", collected="2029-12-28T18:00:00Z", event_id="evt1")
    eff2 = HOCKEY.apply_availability("nhl", "winner", _hockey_row(), _hockey_frame(), pd.DataFrame([named]))
    assert eff2.row["goalie_sv_a"] == pytest.approx(0.900)


QB_TABLE = {"prior": -0.05, "qbs": {"q1": {"name": "Starter One", "rating": 0.12, "plays": 900.0}},
            "teams": {"H": {"incumbent": "q1"}, "A": {}}}


def _nfl(statements):
    row = pd.Series({"side_a_id": "H", "side_b_id": "A", "qb_inc_a": 0.12, "qb_inc_b": 0.02, "qb_base_a": 0.10,
                     "qb_base_b": 0.02, "qb_inc_diff": 0.10, "qb_inc_adj_diff": 0.02, "qb_inc_sum": 0.14})
    frame = FeatureFrame("nfl", "winner", OutcomeType.BINARY, ("away", "home"), pd.DataFrame(), [],
                         extra={"rosters": {"qb": QB_TABLE}})
    rows = [{"team_id": "H", "player_name": "Starter One", "status": st, "role": role, "player_id": None,
             "collected_at_utc": when} for st, role, when in statements]
    return FOOTBALL.apply_availability("nfl", "winner", row, frame, pd.DataFrame(rows))


def test_newest_quarterback_statement_decides():
    confirmed_later = _nfl([("doubtful", "", "2030-01-01T18:00:00Z"), ("available", "QB1", "2030-01-03T18:00:00Z")])
    assert confirmed_later.row["qb_inc_a"] == pytest.approx(0.12) and not confirmed_later.uncertain_a
    ruled_out_later = _nfl([("available", "QB1", "2030-01-01T18:00:00Z"), ("out", "", "2030-01-03T18:00:00Z")])
    assert ruled_out_later.row["qb_inc_a"] == pytest.approx(-0.05)
    questionable = _nfl([("questionable", "", "2030-01-03T18:00:00Z")])
    assert questionable.uncertain_a and questionable.row["qb_inc_a"] == pytest.approx(0.12)
    assert questionable.worst_case_rows["a"]["qb_inc_a"] == pytest.approx(-0.05)


# ------------------------------------------------- results that have stopped
def test_results_are_not_current_when_the_source_has_simply_stopped(store):
    """A source with no schedule lists no unplayed fixtures, so a stale store
    must be recognised from the age of its newest result."""
    from sp_helpers import synthetic_two_sided
    from sportspred.predict.engine import _unresolved_past_events, results_current

    synthetic_two_sided(store, "demo_league", n_teams=4, n_seasons=5, rounds_per_season=2, upcoming_rounds=0)
    now = pd.Timestamp("2021-06-01T00:00:00Z")
    behind, through, latest, usual = _unresolved_past_events(store, "demo_league", now)
    assert behind == {} and latest is not None
    assert 300 < usual < 366                      # one season a year: the yearly break is the off-season

    days_to_next_opener = usual                   # an ordinary off-season
    ok, _ = results_current(0, days_to_next_opener, usual, 1.5, through)
    assert ok
    stale, why = results_current(0, 2 * usual, usual, 1.5, through)
    assert not stale and "very likely missing" in why and through in why
    # Unplayed fixtures on file are reported as such, whatever the age.
    behind_flag, why = results_current(2, 1.0, usual, 1.5, through)
    assert not behind_flag and "2 earlier event(s)" in why
    # Too little history to know what a normal break is: no claim either way.
    assert results_current(0, 5000.0, None, 1.5, through)[0]
