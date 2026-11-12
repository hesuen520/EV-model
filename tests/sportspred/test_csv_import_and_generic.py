"""CSV import validation, the results-only builder, and synthetic isolation."""

from __future__ import annotations

import pandas as pd
import pytest

from sportspred.adapters.csv_import import (
    CsvImportError,
    import_availability,
    import_checks,
    import_events,
    import_results,
    write_templates,
)
from sportspred.adapters.synthetic import generate_demo_league
from sportspred.sports import DataUnavailable, get_league
from sportspred.sports.generic.features import GenericOptions, build_generic_frame
from sportspred.validation.leakage import assert_asof_invariance

LEAGUE = "cricket_t20_men"
HEADER = "event_id,season,start_time,status,home,away,neutral_site,home_score,away_score,winner,draw\n"


def _csv(tmp_path, text, name="events.csv"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_templates_are_importable_as_written(store, tmp_path):
    paths = {p.name: p for p in write_templates(tmp_path / "t")}
    out = import_events(store, LEAGUE, paths["events_template.csv"], create_participants=True)
    assert out["events_new"] == 2 and out["results_written"] == 1
    assert import_results(store, LEAGUE, paths["results_template.csv"])["results_new"] == 1
    with pytest.raises(CsvImportError, match="unknown participant"):      # the template's goalie row names a new team
        import_availability(store, LEAGUE, _csv(tmp_path, "team,player,status\nNobody FC,X,out\n", "a.csv"))
    assert import_checks(store, LEAGUE, paths["checks_template.csv"])["checks"] == 2


def test_naive_timestamps_are_refused_unless_a_zone_is_stated(store, tmp_path):
    path = _csv(tmp_path, HEADER + "e1,2026,2026-04-02 19:30:00,scheduled,A,B,0,,,,\n")
    with pytest.raises(CsvImportError, match="no time zone"):
        import_events(store, LEAGUE, path, create_participants=True)
    assert store.events(LEAGUE).empty                                    # nothing was written
    import_events(store, LEAGUE, path, tz="Asia/Kolkata", create_participants=True)
    assert store.events(LEAGUE).iloc[0]["start_time_utc"] == pd.Timestamp("2026-04-02T14:00:00Z")


def test_every_bad_row_is_reported_with_its_line_and_nothing_is_written(store, tmp_path):
    path = _csv(tmp_path, HEADER
                + "e1,2026,2026-04-02T14:00:00Z,finished,A,B,0,,,,\n"          # bad status
                + "e2,2026,2026-04-03T14:00:00Z,final,A,A,0,10,8,,\n"          # same side twice
                + "e3,2026,2026-04-04T14:00:00Z,scheduled,A,B,0,10,8,,\n"      # result on a scheduled event
                + "e3,2026,2026-04-05T14:00:00Z,final,A,B,0,,,C,\n"            # duplicate id, foreign winner
                + "e5,2026,2026-04-06T14:00:00Z,final,A,B,0,,,,\n")            # final without a result
    with pytest.raises(CsvImportError) as err:
        import_events(store, LEAGUE, path, create_participants=True)
    text = "\n".join(err.value.errors)
    for needle in ("line 2: status", "line 3: home and away are the same", "line 4: a result is given",
                   "line 5: duplicate event", "line 5: winner 'C' is neither", "line 6: status is 'final' but no"):
        assert needle in text
    assert store.events(LEAGUE).empty and store.sources().empty


def test_unknown_names_are_errors_and_reimport_is_idempotent(store, tmp_path):
    first = _csv(tmp_path, HEADER + "e1,2026,2026-04-02T14:00:00Z,final,Kings,Royals,0,180,170,,\n")
    assert import_events(store, LEAGUE, first, create_participants=True)["events_new"] == 1
    again = import_events(store, LEAGUE, first)
    assert again["events_new"] == 0 and again["events_unchanged"] == 1 and again["results_written"] == 0
    typo = _csv(tmp_path, HEADER + "e2,2026,2026-04-03T14:00:00Z,scheduled,Kingz,Royals,0,,,,\n", "typo.csv")
    with pytest.raises(CsvImportError, match="unknown participant.*Kingz"):
        import_events(store, LEAGUE, typo)
    assert len(store.participants(LEAGUE)) == 2                          # no phantom team was created
    with pytest.raises(KeyError, match="unknown league"):
        import_events(store, "no_such_league", first)


def test_blank_cells_stay_missing_and_winner_without_scores_is_accepted(store, tmp_path):
    path = _csv(tmp_path, HEADER + "e1,2026,2026-04-02T14:00:00Z,final,A,B,,,,b,\n")
    import_events(store, LEAGUE, path, create_participants=True)
    ev = store.events(LEAGUE).iloc[0]
    assert pd.isna(ev["home_score"]) and pd.isna(ev["neutral_site"]) and pd.isna(ev["venue_name"])
    assert ev["winner_id"] == ev["away_id"]


def test_league_wide_availability_distinguishes_nothing_listed_from_never_checked(store, tmp_path):
    events = _csv(tmp_path, HEADER
                  + "e1,2030,2030-04-02T14:00:00Z,scheduled,A,B,0,,,,\n"
                  + "e2,2030,2030-04-02T16:00:00Z,scheduled,C,D,0,,,,\n")
    import_events(store, LEAGUE, events, create_participants=True)
    report = _csv(tmp_path, "team,player,status,published_at,collected_at\n"
                            "A,Opener,out,2030-04-01T08:00:00Z,2030-04-01T20:00:00Z\n", "avail.csv")
    out = import_availability(store, LEAGUE, report, league_wide=True, collected_at="2030-04-01T20:00:00Z")
    assert out == {"statements": 1, "teams_reported": 1, "teams_nothing_listed": 3}
    log = store.checks_as_of(LEAGUE, "2030-04-01T21:00:00Z")
    assert log["outcome"].value_counts().to_dict() == {"nothing_listed": 3, "reported": 1}
    bad = _csv(tmp_path, "team,player,status,published_at,collected_at\n"
                         "A,Opener,injured,,\nA,Opener,out,2030-04-02T00:00:00Z,2030-04-01T00:00:00Z\n", "bad.csv")
    with pytest.raises(CsvImportError) as err:
        import_availability(store, LEAGUE, bad)
    assert "line 2: status" in str(err.value) and "line 3: published_at is later" in str(err.value)


def test_csv_only_leagues_raise_instead_of_inventing_data(store):
    plugin, spec = get_league(LEAGUE)
    assert spec.real_data_source == ""
    with pytest.raises(DataUnavailable, match="sportspred import events"):
        plugin.ingest(store, LEAGUE)
    assert store.events(LEAGUE).empty


def test_field_sports_are_declared_but_honestly_unimplemented():
    for key in ("f1", "pga_tour"):
        _, spec = get_league(key)
        assert all(not m.implemented for m in spec.markets)


def test_side_a_never_depends_on_the_result_when_there_is_no_home_side(store):
    """Sources often list the winner first.  With every event stored
    winner-first, the base rate of 'side A wins' must still be about 50%."""
    import numpy as np

    rng = np.random.default_rng(4)
    names = pd.DataFrame({"name": [f"Demo Player {i:02d}" for i in range(16)]})
    ids = list(store.resolve_participants(names, sport="demo", scope="demo_duel", is_synthetic=True))
    strength = dict(zip(ids, rng.normal(0, 1, 16)))
    rows = []
    day = pd.Timestamp("2020-01-01T12:00:00Z")
    for k in range(1200):
        a, b = rng.choice(16, 2, replace=False)
        a, b = ids[a], ids[b]
        win_a = rng.uniform() < 1 / (1 + np.exp(strength[b] - strength[a]))
        winner, loser = (a, b) if win_a else (b, a)
        rows.append({"league": "demo_duel", "season": str(2020 + k // 300), "start_time_utc": day + pd.Timedelta(days=k),
                     "status": "final", "provider": "synthetic", "provider_event_id": str(k), "home_id": winner,
                     "away_id": loser, "neutral_site": 1, "is_synthetic": True, "winner_id": winner})
    events, _ = store.upsert_events(pd.DataFrame(rows), sport="demo")
    store.upsert_results(events[["event_id"]].assign(winner_id=events["home_id"], is_synthetic=True))
    frame = build_generic_frame(store, "demo_duel", "winner", opts=GenericOptions(home_adv=False),
                                now="2030-01-01T00:00:00Z")
    done = frame.completed
    assert len(done) == 1200 and 0.45 < done["y"].mean() < 0.55
    assert (done["side_a_id"] < done["side_b_id"]).all() and "home_edge" not in frame.feature_cols
    # Stronger players win more: the rating carries real signal even though side A is arbitrary.
    late = done.tail(600)
    assert np.corrcoef(late["elo_diff"], late["y"])[0, 1] > 0.2


def test_generic_builder_is_point_in_time(store):
    plugin, _ = get_league("demo_basketball")
    generate_demo_league(store, "demo_basketball", now="2026-11-12T12:00:00Z", n_teams=8, n_seasons=5)
    result = assert_asof_invariance(plugin, store, "demo_basketball", "winner", n_cutoffs=3)
    assert result["max_abs_diff"].max() == 0.0


def test_synthetic_generator_refuses_real_league_keys(store):
    with pytest.raises(ValueError, match="demo_"):
        generate_demo_league(store, "nba")
    generate_demo_league(store, "demo_duel", now="2026-11-12T12:00:00Z", home_adv=False, n_teams=6, n_seasons=3)
    assert store.events("demo_duel").empty and len(store.events("demo_duel", include_synthetic=True)) > 0
    assert int(store.query("SELECT COUNT(*) AS n FROM events WHERE is_synthetic=0")["n"][0]) == 0
