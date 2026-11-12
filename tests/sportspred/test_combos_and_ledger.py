"""Combo dependencies, the three-leg rule, grading and reviews."""

from __future__ import annotations

import pandas as pd
import pytest

from sp_helpers import prediction
from sportspred.combos import (
    PASS_MESSAGE,
    Leg,
    classify_pair,
    combo_backtest,
    joint_probability,
    latent_correlation,
    select_combos,
)
from sportspred.config import ComboConfig, Settings
from sportspred.ledger import grade_predictions, review_predictions, set_lesson_status, add_lesson

CORR_OK = {("nba", "winner"): {"rho": 0.004, "low": -0.002, "high": 0.011, "n_pairs": 50000}}
CORR_HIGH = {("nba", "winner"): {"rho": 0.20, "low": 0.15, "high": 0.25, "n_pairs": 50000}}


def leg(i, prob, *, event=None, league="nba", market="winner", teams=None, selection=None, status="FINAL",
        coverage="validated", line=None, entities=None):
    teams = teams or (f"t{i}a", f"t{i}b")
    selection = selection or teams[0]
    return Leg(prediction_id=f"p{i}", event_id=event or f"e{i}", league=league, sport="basketball", market=market,
               line=line, selection=selection, label=selection, prob=prob, prob_low=prob - 0.03, rec_status=status,
               validation_status=coverage, start_utc="2030-01-02T03:30:00Z", participants=frozenset(teams),
               entities=frozenset(entities or {selection}))


def test_independent_legs_multiply_and_adding_a_leg_never_helps():
    legs = [leg(1, 0.80), leg(2, 0.75), leg(3, 0.70), leg(4, 0.68)]
    previous = 1.0
    for k in range(1, 5):
        joint = joint_probability(legs[:k], CORR_OK, 0.03)
        assert joint.valid and joint.prob <= previous + 1e-12
        previous = joint.prob
    assert joint_probability(legs[:3], CORR_OK, 0.03).prob == pytest.approx(0.80 * 0.75 * 0.70)
    assert joint_probability(legs[:3], CORR_OK, 0.03).prob_low == pytest.approx(0.77 * 0.72 * 0.67)


def test_same_game_legs_are_rejected_not_multiplied():
    a = leg(1, 0.70, event="e1", teams=("x", "y"))
    b = leg(2, 0.60, event="e1", teams=("x", "y"), market="total", selection="over", line=220.5)
    joint = joint_probability([a, b], CORR_OK, 0.03)
    assert not joint.valid and joint.prob is None and joint.rejected[0].kind == "same_event"


def test_two_outcomes_of_one_market_are_mutually_exclusive():
    a = leg(1, 0.55, event="e1", teams=("x", "y"), selection="x")
    b = leg(2, 0.45, event="e1", teams=("x", "y"), selection="y")
    assert classify_pair(a, b, CORR_OK, 0.03).kind == "exclusive"
    assert not joint_probability([a, b], CORR_OK, 0.03).valid


def test_same_team_or_player_in_two_events_is_rejected():
    a = leg(1, 0.70, event="e1", teams=("x", "y"))
    b = leg(2, 0.70, event="e2", teams=("x", "z"))
    assert classify_pair(a, b, CORR_OK, 0.03).kind == "same_participant"
    prop1 = leg(3, 0.60, event="e3", market="player_prop:points", selection="over", line=24.5, entities={"player9"})
    prop2 = leg(4, 0.60, event="e4", market="player_prop:assists", selection="over", line=6.5, entities={"player9"})
    assert classify_pair(prop1, prop2, CORR_OK, 0.03).treatment == "rejected"


def test_same_slate_legs_need_a_validated_dependence_estimate():
    a, b = leg(1, 0.70), leg(2, 0.70)
    assert classify_pair(a, b, {}, 0.03).treatment == "rejected"
    assert classify_pair(a, b, CORR_OK, 0.03).treatment == "independent"
    high = classify_pair(a, b, CORR_HIGH, 0.03)
    assert high.treatment == "copula"
    joint = joint_probability([a, b], CORR_HIGH, 0.03)
    # Positive dependence makes both winning more likely than the product, never above the weaker leg.
    assert 0.70 * 0.70 < joint.prob < 0.70


def test_copula_reproduces_the_binary_correlation():
    r = latent_correlation(0.7, 0.6, 0.2)
    assert 0.2 < r < 0.6
    assert latent_correlation(0.7, 0.6, 0.0) == 0.0


def test_different_competitions_are_independent_by_stated_assumption():
    pair = classify_pair(leg(1, 0.7, league="nba"), leg(2, 0.7, league="nhl"), {}, 0.03)
    assert pair.kind == "unrelated" and pair.treatment == "independent" and "assumed" in pair.note


def test_three_leg_selection_skips_combinations_with_unmodelled_dependencies():
    cfg = ComboConfig()
    legs = [leg(1, 0.85, event="e1", teams=("a", "b")), leg(2, 0.84, event="e2", teams=("a", "c")),   # share team a
            leg(3, 0.80), leg(4, 0.75), leg(5, 0.70)]
    picked = select_combos(legs, cfg, CORR_OK)["by_size"]
    best3 = picked[3][0]
    ids = {l.prediction_id for l in best3.legs}
    assert not {"p1", "p2"} <= ids                      # the two strongest legs cannot be combined
    assert best3.prob == pytest.approx(0.85 * 0.80 * 0.75)
    assert picked[3][0].prob >= picked[3][-1].prob      # candidates are compared within a size


def _slate_store(store, n_events, probs, status):
    teams = pd.DataFrame({"name": [f"Team {i}" for i in range(2 * n_events)]})
    ids = list(store.resolve_participants(teams, sport="basketball", scope="nba"))
    rows = [{"league": "nba", "season": "2030", "start_time_utc": "2030-01-02T03:30:00Z", "status": "scheduled",
             "provider": "t", "provider_event_id": str(i), "home_id": ids[2 * i], "away_id": ids[2 * i + 1]}
            for i in range(n_events)]
    events, _ = store.upsert_events(pd.DataFrame(rows), sport="basketball")
    for i, (eid, p) in enumerate(zip(events["event_id"], probs)):
        store.insert_prediction(prediction(
            eid, [ids[2 * i], ids[2 * i + 1]], prob=p, probs={ids[2 * i]: p, ids[2 * i + 1]: 1 - p},
            rec_status=status, validation_status="validated", prob_low=p - 0.02, prob_high=p + 0.02,
            display={"side_a": f"Team {2 * i}", "side_b": f"Team {2 * i + 1}", "side_a_role": "home"}))
    store.save_validation_run({"sport": "basketball", "league": "nba", "market": "winner", "gates_passed": True,
                               "same_day_residual_correlation": CORR_OK[("nba", "winner")]})
    return events, ids


def test_fewer_than_three_qualifying_legs_returns_the_pass_message(store):
    from datetime import date
    from sportspred.combos import build_slate_combos

    _slate_store(store, 4, [0.80, 0.75, 0.55, 0.52], "FINAL")
    result = build_slate_combos(store, date(2030, 1, 1), now="2030-01-01T12:00:00Z")
    assert result.message.startswith(PASS_MESSAGE) and result.status.value == "PASS"
    assert not result.three_leg and len(result.two_leg) >= 1 and len(result.eligible) == 2
    recorded = store.combos()
    assert "none" in set(recorded["role"]) and "primary_three_leg" not in set(recorded["role"])


def test_provisional_legs_are_never_a_recommendation_only_a_labelled_preview(store):
    from datetime import date
    from sportspred.combos import build_slate_combos

    _slate_store(store, 4, [0.80, 0.75, 0.72, 0.70], "PROVISIONAL")
    result = build_slate_combos(store, date(2030, 1, 1), now="2030-01-01T12:00:00Z")
    assert result.status.value == "PASS" and not result.three_leg and not result.eligible
    assert result.preview is not None and result.preview.status.value == "PROVISIONAL"
    assert "not a recommendation" in result.preview.message and result.preview.three_leg
    roles = store.combos().set_index("role")["rec_status"].to_dict()
    assert roles["none"] == "PASS" and roles["provisional_preview"] == "PROVISIONAL"


def test_final_legs_give_a_final_three_leg_combo_and_supported_larger_sizes(store):
    from datetime import date
    from sportspred.combos import build_slate_combos

    _slate_store(store, 5, [0.85, 0.80, 0.78, 0.76, 0.66], "FINAL")
    result = build_slate_combos(store, date(2030, 1, 1), now="2030-01-01T12:00:00Z")
    assert result.status.value == "FINAL"
    assert result.three_leg[0].prob == pytest.approx(0.85 * 0.80 * 0.78)
    assert 4 in result.larger and result.larger[4][0].prob < result.three_leg[0].prob
    assert 5 not in result.larger or result.larger[5][0].prob >= 0.25      # only while supported
    # Once the events have started nothing is eligible any more.
    late = build_slate_combos(store, date(2030, 1, 1), now="2030-01-02T04:00:00Z", write=False)
    assert not late.eligible and late.status.value == "PASS"


def test_combo_backtest_reports_dev_and_final_separately():
    import numpy as np
    rng = np.random.default_rng(3)
    n = 4000
    p = rng.uniform(0.5, 0.9, n)
    y = (rng.uniform(size=n) < p).astype(int)
    start = pd.Timestamp("2020-01-01T03:00:00Z") + pd.to_timedelta(np.arange(n) // 8, unit="D")
    oos = pd.DataFrame({"event_id": [f"e{i}" for i in range(n)], "start_time_utc": start, "block": 1, "y": y,
                        "p_away": 1 - p, "p_home": p, "selected": True, "is_final_test": np.arange(n) >= 3000})
    result = combo_backtest(oos, min_leg_prob=0.65, sizes=(1, 3))
    dev3 = result["development"][1]
    assert dev3["legs"] == 3 and dev3["n_combos"] > 200
    assert abs(dev3["observed_minus_predicted"]) < 0.06           # calibrated legs -> calibrated combos
    assert result["final_test"][1]["n_combos"] < dev3["n_combos"]


def test_grading_review_and_regrade_after_a_result_correction(store, two_teams, scheduled_event):
    pid = store.insert_prediction(prediction(scheduled_event, two_teams, rec_status="FINAL", prob=0.7,
                                             probs={two_teams[0]: 0.7, two_teams[1]: 0.3}))
    assert grade_predictions(store).empty                       # nothing to grade before the result
    store.upsert_results(pd.DataFrame([{"event_id": scheduled_event, "home_score": 99, "away_score": 104}]))
    first = grade_predictions(store)
    assert first.iloc[0]["outcome"] == "incorrect" and not first.iloc[0]["regrade"]
    reviews = review_predictions(store)
    assert reviews.iloc[0]["category"] == "variance" and "expected to lose 30%" in reviews.iloc[0]["note"]
    # The score is corrected: a new grade is appended and linked, the old one stays.
    store.upsert_results(pd.DataFrame([{"event_id": scheduled_event, "home_score": 105, "away_score": 104}]))
    second = grade_predictions(store)
    assert second.iloc[0]["outcome"] == "correct" and second.iloc[0]["regrade"]
    assert len(store.grades(latest_only=False)) == 2 and len(store.grades()) == 1
    assert store.predictions().iloc[0]["prediction_id"] == pid and store.verify_ledger()[0]


def test_late_availability_news_is_reviewed_as_availability_change_not_model_error(store, two_teams, scheduled_event):
    store.insert_prediction(prediction(scheduled_event, two_teams, rec_status="FINAL", prob=0.7,
                                       probs={two_teams[0]: 0.7, two_teams[1]: 0.3}))
    store.add_availability(pd.DataFrame([{
        "league": "nba", "team_id": two_teams[0], "player_name": "Star Player", "status": "out",
        "collected_at_utc": "2030-01-02T02:00:00Z"}]))       # after the prediction, before the start
    store.upsert_results(pd.DataFrame([{"event_id": scheduled_event, "home_score": 90, "away_score": 104}]))
    grade_predictions(store)
    assert review_predictions(store).iloc[0]["category"] == "availability_change"


def test_postponed_event_is_void_and_lessons_need_a_validation_run(store, two_teams, scheduled_event):
    store.insert_prediction(prediction(scheduled_event, two_teams, rec_status="FINAL", prob=0.7,
                                       probs={two_teams[0]: 0.7, two_teams[1]: 0.3}))
    row = {"league": "nba", "season": "2030", "start_time_utc": "2030-01-02T03:30:00Z", "status": "postponed",
           "provider": "espn", "provider_event_id": "401", "home_id": two_teams[0], "away_id": two_teams[1]}
    store.upsert_events(pd.DataFrame([row]), sport="basketball")
    assert grade_predictions(store).iloc[0]["outcome"] == "void"
    lesson = add_lesson(store, title="t", hypothesis="h", proposed_change="c", test_plan="p")
    set_lesson_status(store, lesson, "testing")
    with pytest.raises(ValueError, match="validation run"):
        set_lesson_status(store, lesson, "supported")
    with pytest.raises(ValueError, match="cannot become"):
        set_lesson_status(store, lesson, "promoted")
    run = store.save_validation_run({"sport": "basketball", "league": "nba", "market": "winner"})
    set_lesson_status(store, lesson, "supported", validation_run_id=run)
    set_lesson_status(store, lesson, "promoted")
