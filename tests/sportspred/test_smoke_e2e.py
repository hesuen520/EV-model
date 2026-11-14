"""End-to-end smoke test on the SYNTHETIC demo league, through the public
pipeline: generate -> validate -> train -> predict -> status transitions ->
combos -> results -> grade -> review -> ledger verification."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from sportspred.combos import PASS_MESSAGE, build_slate_combos
from sportspred.config import Settings
from sportspred.coverage import coverage_status, coverage_table
from sportspred.ledger import grade_predictions, review_predictions
from sportspred.pipeline import train_league, validate_league
from sportspred.predict import predict_league
from sportspred.sports import get_league
from sportspred.storage import Store

LEAGUE = "demo_basketball"
NOW = "2026-11-12T12:00:00Z"            # the first synthetic slate starts 2026-11-13T03:00Z


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    settings = Settings()
    settings.paths.data_dir = str(tmp_path_factory.mktemp("demo"))
    settings.validation.n_boot = 60
    settings.validation.n_replicas = 6
    settings.selection.allowed_coverage = ["validated", "experimental", "demo-only"]
    settings.selection.horizon_hours = 72.0
    settings.combos.allowed_coverage = ["validated", "demo-only"]
    store = Store(settings.paths.store)
    plugin, _ = get_league(LEAGUE)
    plugin.ingest(store, LEAGUE, now=NOW)
    report, oos = validate_league(store, LEAGUE, "winner", settings=settings, now=NOW)
    model = train_league(store, LEAGUE, "winner", settings=settings, now=NOW)
    return {"settings": settings, "store": store, "report": report, "oos": oos, "model": model}


def test_validation_report_is_complete_and_never_validates_synthetic_data(world):
    report = world["report"]
    assert report["is_synthetic"] and not report["gates_passed"]
    assert report["protocol"]["final_test_used_for_selection"] is False
    assert report["selected_model"] in {"elo_logit", "ratings_logit"}            # real signal exists in the demo
    for key in ("candidates", "dev", "final_test", "confidence_table", "same_day_residual_correlation", "gates"):
        assert report[key]
    sel = report["final_test"]["selected"]
    assert sel["n"] > 100 and 0 < sel["log_loss"] < 0.69 and sel["accuracy_ci"][0] < sel["accuracy"] < sel["accuracy_ci"][1]
    assert coverage_status(world["store"], LEAGUE, "winner").value == "demo-only"
    table = coverage_table(world["store"])
    assert set(table[table["league"].str.startswith("demo_")]["status"]) == {"demo-only"}


def test_statuses_move_with_the_checks_and_nothing_is_rewritten(world):
    store, settings = world["store"], world["settings"]
    first = predict_league(store, LEAGUE, settings=settings, now=NOW)
    assert first and {p["rec_status"] for p in first} <= {"PROVISIONAL", "PASS"}
    assert all(p["validation_status"] == "demo-only" for p in first)
    picks = [p for p in first if p["rec_status"] == "PROVISIONAL"]
    assert picks, "the demo slate should contain at least one selection"
    for p in picks:
        assert any("nothing on file" in r for r in p["reasons"])
        assert p["prob_low"] <= p["prob"] <= p["prob_high"] and abs(sum(p["probs"].values()) - 1) < 1e-9
    # Unchanged inputs are not written twice.
    again = predict_league(store, LEAGUE, settings=settings, now=NOW)
    assert all(p.get("unchanged") for p in again) and len(store.predictions(include_synthetic=True)) == len(first)

    # A fresh check close to the start completes the checks -> FINAL (the result is still uncertain).
    upcoming = store.upcoming(LEAGUE, now=NOW, horizon_hours=24, include_synthetic=True)
    start = upcoming["start_time_utc"].min()
    near = (start - pd.Timedelta(hours=1)).isoformat()
    for team in set(upcoming["home_id"]) | set(upcoming["away_id"]):
        store.log_check(LEAGUE, "injury_report", "nothing_listed", checked_at=(start - pd.Timedelta(hours=2)).isoformat(),
                        team_id=team, is_synthetic=True)
    final = predict_league(store, LEAGUE, settings=settings, now=near, horizon_hours=6)
    by_status = {p["rec_status"] for p in final}
    assert "FINAL" in by_status and "PROVISIONAL" not in by_status
    done = next(p for p in final if p["rec_status"] == "FINAL")
    assert done["supersedes_id"] and "complete and fresh" in done["reasons"][0]
    rows = store.predictions(include_synthetic=True)
    assert len(rows) > len(first) and store.verify_ledger() == (True, [])        # old rows are still there

    # The same check, seen 30 hours later for the next slate, is stale again.
    later = store.upcoming(LEAGUE, now=near, horizon_hours=72, include_synthetic=True)
    next_start = later["start_time_utc"].max()
    stale = predict_league(store, LEAGUE, settings=settings, now=(next_start - pd.Timedelta(minutes=30)).isoformat(),
                           horizon_hours=2)
    assert stale and all(p["rec_status"] in ("PROVISIONAL", "PASS") for p in stale)
    assert any("older than" in r or "nothing on file" in r for p in stale for r in p["reasons"]
               if p["rec_status"] == "PROVISIONAL")


def test_no_prediction_is_written_once_an_event_has_started(world):
    store, settings = world["store"], world["settings"]
    before = len(store.predictions(include_synthetic=True))
    upcoming = store.upcoming(LEAGUE, now=NOW, include_synthetic=True)
    after_all = (upcoming["start_time_utc"].max() + pd.Timedelta(minutes=1)).isoformat()
    assert predict_league(store, LEAGUE, settings=settings, now=after_all) == []
    assert len(store.predictions(include_synthetic=True)) == before


def test_combos_grading_and_review_close_the_loop(world):
    store, settings = world["store"], world["settings"]
    upcoming = store.upcoming(LEAGUE, now=NOW, horizon_hours=24, include_synthetic=True)
    start = upcoming["start_time_utc"].min()
    near = (start - pd.Timedelta(minutes=30)).isoformat()
    slate = (start.tz_convert("Pacific/Honolulu")).date()
    result = build_slate_combos(store, slate, settings=settings, now=near, include_synthetic=True, write=True)
    finals = [l for l in result.eligible]
    if len(finals) >= 3:
        assert result.status.value == "FINAL" and len(result.three_leg[0].legs) == 3
        assert result.three_leg[0].prob <= min(l.prob for l in result.three_leg[0].legs)
        if result.two_leg:
            assert result.two_leg[0].prob >= result.three_leg[0].prob           # adding a leg never helps
    else:
        assert result.message.startswith(PASS_MESSAGE)
    assert len(store.combos(include_synthetic=True)) >= 1

    # Results arrive: grade (no prices involved) and review.
    played = store.events(LEAGUE, include_synthetic=True, event_ids=upcoming["event_id"].tolist())
    truth = store.read_table(LEAGUE, "truth").set_index("event_id")
    results = pd.DataFrame({
        "event_id": played["event_id"],
        "home_score": [101 if truth.loc[e, "p_home_true"] >= 0.5 else 95 for e in played["event_id"]],
        "away_score": 98, "is_synthetic": True,
    })
    store.upsert_results(results)
    grades = grade_predictions(store, include_synthetic=True)
    assert len(grades) and set(grades["outcome"]) <= {"correct", "incorrect"}
    reviews = review_predictions(store, include_synthetic=True)
    assert len(reviews) and set(reviews["category"]) <= {"variance", "model_error", "data_issue", "availability_change"}
    assert store.verify_ledger() == (True, [])
