"""The static dashboard: self-contained, honest about synthetic data, and
showing every field a prediction is required to carry."""

from __future__ import annotations

import re

import pandas as pd

from sp_helpers import prediction
from sportspred.adapters.synthetic import SYNTHETIC_BANNER
from sportspred.cli.cmd_dashboard import write_validation_docs
from sportspred.config import Settings
from sportspred.dashboard import NOTICE, build_dashboard
from sportspred.pipeline import train_league, validate_league
from sportspred.predict import predict_league
from sportspred.sports import get_league
from sportspred.storage import Store
from sportspred.validation.plots import improvement_svg, reliability_svg, standalone

NOW = "2026-11-12T12:00:00Z"


def test_empty_store_renders_and_explains_what_to_run(store):
    page = build_dashboard(store)
    assert "No predictions recorded yet" in page and "EV unavailable - no prices on file" in page
    assert NOTICE.split(".")[0] in page and "Pacific/Honolulu" in page


def test_page_is_self_contained_and_shows_required_fields(store, two_teams, scheduled_event):
    store.insert_prediction(prediction(
        scheduled_event, two_teams, rec_status="PROVISIONAL", prob=0.7, prob_low=0.66, prob_high=0.74,
        probs={two_teams[0]: 0.7, two_teams[1]: 0.3},
        reasons=["injury report: nothing on file for Boston Celtics"],
        checks=[{"key": "injury_report", "label": "injury report", "subject": "Boston Celtics", "state": "missing",
                 "detail": "nothing on file"}],
        confidence={"label": "High", "basis": "validated band", "event_uncertainty": {"prob_selection_loses": 0.3},
                    "model_uncertainty": {"method": "30 bootstrap refits"}},
        factors={"supporting": ["opponent-adjusted results"], "opponent_strength": ["Lakers Elo 1480"],
                 "upset_risks": ["back-to-back"], "availability_assumptions": ["recent rotation"], "missing_data": []},
        data_quality={"grade": "partial", "missing_features": []},
        sources=[{"provider": "espn", "url": "https://example.org/x", "collected_at_utc": "2029-12-31T00:00:00Z",
                  "published_at_utc": None}],
        display={"competition": "NBA", "side_a": "Boston Celtics", "side_b": "Los Angeles Lakers", "side_a_role": "home",
                 "start_hst": "Tue Jan 1, 2030 5:30 PM HST", "venue": "TD Garden", "market_note": "incl. overtime",
                 "last_check_hst": "Mon Dec 31, 2029 2:00 PM HST"}))
    page = build_dashboard(store)
    for needle in ("Los Angeles Lakers at Boston Celtics", "Tue Jan 1, 2030 5:30 PM HST", "TD Garden", "70.0%",
                   "66.0% - 74.0%", "PROVISIONAL", "nothing on file", "opponent-adjusted results", "Lakers Elo 1480",
                   "back-to-back", "recent rotation", "https://example.org/x", "Predicted Mon Dec 31, 2029 2:00 PM HST",
                   "newest source check", "experimental"):
        assert needle in page, needle
    # No external assets: nothing is loaded from the network.
    assert not re.search(r'<(script|link|img)[^>]+(src|href)="https?:', page)
    # The demo leagues are listed in the coverage table (as demo-only), but no synthetic content is shown.
    assert "SYNTHETIC DEMO DATA" not in page and "not a real event" not in page


def test_started_events_are_marked_and_the_page_opens_on_the_next_pending_slate(store, two_teams, scheduled_event):
    later = pd.DataFrame([{
        "league": "nba", "season": "2030", "start_time_utc": "2030-01-04T03:30:00Z", "status": "scheduled",
        "provider": "espn", "provider_event_id": "402", "home_id": two_teams[1], "away_id": two_teams[0],
    }])
    later_id = store.upsert_events(later, sport="basketball")[0]["event_id"].iloc[0]
    display = {"competition": "NBA", "side_a": "A", "side_b": "B", "side_a_role": "home", "start_hst": "x"}
    store.insert_prediction(prediction(scheduled_event, two_teams, display=display))
    store.insert_prediction(prediction(later_id, two_teams, display=display, event_start_utc="2030-01-04T03:30:00Z"))

    before = build_dashboard(store, now="2030-01-01T12:00:00Z")
    assert "This event has started" not in before
    assert '<option value="2030-01-01" selected>' in before

    # After the first game has started: its card says so, and the page opens on the slate still to come.
    after = build_dashboard(store, now="2030-01-02T04:00:00Z")
    assert after.count("This event has started") == 1
    assert '<option value="2030-01-03" selected>' in after


def test_synthetic_content_is_hidden_by_default_and_banner_labelled_on_request(tmp_path):
    settings = Settings()
    settings.paths.data_dir = str(tmp_path)
    settings.validation.n_boot, settings.validation.n_replicas = 40, 4
    settings.selection.allowed_coverage = ["validated", "experimental", "demo-only"]
    settings.selection.horizon_hours = 72.0
    store = Store(settings.paths.store)
    plugin, _ = get_league("demo_basketball")
    plugin.ingest(store, "demo_basketball", now=NOW)
    validate_league(store, "demo_basketball", settings=settings, now=NOW)
    train_league(store, "demo_basketball", settings=settings, now=NOW)
    assert predict_league(store, "demo_basketball", settings=settings, now=NOW)
    hidden = build_dashboard(store, settings)
    assert "Demo Club" not in hidden and "No predictions recorded yet" in hidden
    shown = build_dashboard(store, settings, include_synthetic=True)
    assert "Demo Club" in shown and "SYNTHETIC DEMO DATA" in shown and "SYNTHETIC DEMO - not a real event" in shown
    assert "demo-only" in shown and "data-tip=" in shown          # charts carry their hover text
    # Validation documents never include synthetic runs.
    path = write_validation_docs(store, tmp_path / "docs")
    assert "demo_basketball" not in path.read_text()


def test_charts_are_valid_svg_with_tooltips_and_theme_support():
    rel = [{"mean_predicted": 0.55, "observed": 0.54, "observed_low": 0.5, "observed_high": 0.58, "n": 400},
           {"mean_predicted": 0.75, "observed": 0.77, "observed_low": 0.72, "observed_high": 0.81, "n": 250}]
    svg = standalone(reliability_svg(rel))
    assert svg.startswith("<svg") and svg.endswith("</svg>") and "prefers-color-scheme: dark" in svg
    assert svg.count("<title>") == 2 and "perfect calibration" in svg and "400 selections" in svg
    bars = improvement_svg([{"block": "2020", "n": 1200, "improvement": 0.004},
                            {"block": "2021", "n": 1230, "improvement": -0.001}], reference="elo_logit")
    assert bars.count('class="bar mark"') == 2 and "worse" in bars and "better" in bars
    assert "No seasons to plot" in improvement_svg([], reference="x")
