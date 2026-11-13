"""Availability handling in the basketball plugin: confirmed absences are
applied, uncertain players create worst-case scenarios, unknown names are
reported rather than guessed."""

from __future__ import annotations

import pandas as pd
import pytest

from sportspred.core.enums import Availability, OutcomeType
from sportspred.sports.base import FeatureFrame
from sportspred.sports.basketball import PLUGIN
from sportspred.sports.basketball.features import roster_quality

HOME, AWAY = "team_home", "team_away"
ROSTER = [
    {"player_id": "1", "name": "Star Guard", "exp_min": 34.0, "role_min": 34.0, "rating": 0.60},
    {"player_id": "2", "name": "Second Option", "exp_min": 32.0, "role_min": 32.0, "rating": 0.45},
    {"player_id": "3", "name": "Role Player", "exp_min": 28.0, "role_min": 28.0, "rating": 0.30},
    {"player_id": "4", "name": "Returning Big", "exp_min": 2.0, "role_min": 30.0, "rating": 0.50},
    {"player_id": "5", "name": "Deep Bench", "exp_min": 4.0, "role_min": 4.0, "rating": 0.25},
] + [{"player_id": str(10 + i), "name": f"Rotation {i}", "exp_min": 20.0, "role_min": 20.0, "rating": 0.35}
     for i in range(7)]
RATE = 0.38


def _frame():
    return FeatureFrame("nba", "winner", OutcomeType.BINARY, ("away", "home"), pd.DataFrame(), [],
                        extra={"rosters": {HOME: ROSTER, AWAY: ROSTER}, "league_rate": RATE,
                               "regulation_minutes": 240.0})


def _row():
    q, ab = roster_quality(ROSTER, RATE, 240.0)
    return pd.Series({"side_a_id": HOME, "side_b_id": AWAY, "roster_q_a": q, "roster_q_b": q, "absence_a": ab,
                      "absence_b": ab, "roster_q_diff": 0.0, "absence_diff": 0.0})


def _report(*rows):
    return pd.DataFrame([{"team_id": t, "player_name": n, "status": s, "player_id": None} for t, n, s in rows])


def test_statuses_distinguish_absent_uncertain_and_available():
    assert Availability.OUT.is_confirmed_absence and not Availability.OUT.is_uncertain
    assert all(Availability(s).is_uncertain for s in ("doubtful", "questionable", "probable", "uncertain"))
    assert not Availability.AVAILABLE.is_uncertain and not Availability.AVAILABLE.is_confirmed_absence


def test_no_report_changes_nothing_and_is_not_treated_as_everyone_available():
    effect = PLUGIN.apply_availability("nba", "winner", _row(), _frame(), pd.DataFrame())
    assert effect.row["roster_q_diff"] == 0.0 and not effect.applied and not effect.uncertain_a


def test_confirmed_out_star_lowers_his_side_only():
    effect = PLUGIN.apply_availability("nba", "winner", _row(), _frame(), _report((HOME, "Star Guard", "out")))
    assert effect.row["roster_q_a"] < _row()["roster_q_a"] and effect.row["roster_q_b"] == _row()["roster_q_b"]
    assert effect.row["roster_q_diff"] < 0 and effect.row["absence_diff"] > 0
    assert any("Star Guard out" in text and "34 expected minutes" in text for text in effect.applied)


def test_name_matching_ignores_accents_and_case_but_never_guesses():
    effect = PLUGIN.apply_availability("nba", "winner", _row(), _frame(),
                                       _report((HOME, "STAR  guard", "out"), (HOME, "Somebody Else", "out")))
    assert effect.row["roster_q_diff"] < 0
    assert any("Somebody Else" in text and "no effect" in text for text in effect.unmatched)


def test_questionable_rotation_player_builds_a_worst_case_for_his_side():
    effect = PLUGIN.apply_availability("nba", "winner", _row(), _frame(),
                                       _report((HOME, "Star Guard", "questionable")))
    assert effect.uncertain_a and not effect.uncertain_b
    assert effect.row["roster_q_a"] == pytest.approx(_row()["roster_q_a"])            # assumed to play
    assert effect.worst_case_rows["a"]["roster_q_a"] < effect.row["roster_q_a"]       # ... unless he sits
    assert effect.worst_case_rows["b"]["roster_q_a"] == pytest.approx(effect.row["roster_q_a"])


def test_questionable_deep_bench_player_does_not_make_a_selection_conditional():
    effect = PLUGIN.apply_availability("nba", "winner", _row(), _frame(),
                                       _report((AWAY, "Deep Bench", "questionable")))
    assert not effect.uncertain_a and not effect.uncertain_b


def test_doubtful_is_treated_as_out_probable_as_playing_and_returns_are_restored():
    doubtful = PLUGIN.apply_availability("nba", "winner", _row(), _frame(), _report((AWAY, "Star Guard", "doubtful")))
    out = PLUGIN.apply_availability("nba", "winner", _row(), _frame(), _report((AWAY, "Star Guard", "out")))
    assert doubtful.row["roster_q_b"] == pytest.approx(out.row["roster_q_b"])
    probable = PLUGIN.apply_availability("nba", "winner", _row(), _frame(), _report((AWAY, "Star Guard", "probable")))
    assert probable.row["roster_q_b"] == pytest.approx(_row()["roster_q_b"])
    back = PLUGIN.apply_availability("nba", "winner", _row(), _frame(), _report((HOME, "Returning Big", "available")))
    assert back.row["roster_q_a"] > _row()["roster_q_a"]
    assert any("restored to a 30-minute role" in text for text in back.applied)


def test_an_out_statement_older_than_the_players_last_game_is_not_applied():
    """Listed out three weeks ago, has played nine games since: the statement
    is out of date and must not remove him from the rotation."""
    roster = [dict(p, last_played_utc="2030-01-10T03:00:00Z") for p in ROSTER]
    frame = FeatureFrame("nba", "winner", OutcomeType.BINARY, ("away", "home"), pd.DataFrame(), [],
                         extra={"rosters": {HOME: roster, AWAY: roster}, "league_rate": RATE,
                                "regulation_minutes": 240.0})
    stale = pd.DataFrame([{"team_id": HOME, "player_name": "Star Guard", "status": "out", "player_id": None,
                           "published_at_utc": None, "collected_at_utc": "2029-12-20T18:00:00Z"}])
    effect = PLUGIN.apply_availability("nba", "winner", _row(), frame, stale)
    assert effect.row["roster_q_a"] == pytest.approx(_row()["roster_q_a"]) and not effect.applied
    assert any("has played since" in text and "not applied" in text for text in effect.unmatched)
    # The same statement made after his last game is applied.
    fresh = stale.assign(collected_at_utc="2030-01-11T18:00:00Z")
    assert PLUGIN.apply_availability("nba", "winner", _row(), frame, fresh).row["roster_q_a"] < _row()["roster_q_a"]


def test_no_rotation_for_this_game_leaves_the_inputs_alone_instead_of_blanking_them():
    row = _row()
    row["block"] = 2030.0
    # The only snapshot on file belongs to another season's game.
    frame = FeatureFrame("nba", "winner", OutcomeType.BINARY, ("away", "home"), pd.DataFrame(), [],
                         extra={"rosters": {HOME: ROSTER, AWAY: ROSTER}, "roster_block": {HOME: 2031, AWAY: 2030},
                                "league_rate": RATE, "regulation_minutes": 240.0})
    effect = PLUGIN.apply_availability("nba", "winner", row, frame,
                                       _report((HOME, "Star Guard", "out"), (AWAY, "Star Guard", "out")))
    assert effect.row["roster_q_a"] == pytest.approx(row["roster_q_a"])          # home: wrong game, untouched
    assert effect.row["roster_q_b"] < row["roster_q_b"]                          # away: applied as usual
    assert any("no rotation on file for this game" in text for text in effect.unmatched)
    # An empty rotation does not turn the input into a missing value either.
    empty = FeatureFrame("nba", "winner", OutcomeType.BINARY, ("away", "home"), pd.DataFrame(), [],
                         extra={"rosters": {HOME: [], AWAY: ROSTER}, "league_rate": RATE, "regulation_minutes": 240.0})
    effect = PLUGIN.apply_availability("nba", "winner", row, empty, _report((HOME, "Anybody", "questionable")))
    assert effect.row["roster_q_a"] == pytest.approx(row["roster_q_a"])
    # What cannot be applied is not assumed away: it becomes a stated, unquantified condition
    # (no worst-case row for that side, so the engine makes a selection CONDITIONAL).
    assert effect.uncertain_a and "not quantified" in effect.uncertain_a[0] and "a" not in effect.worst_case_rows
