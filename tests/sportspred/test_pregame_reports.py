"""The official injury report as published before each slate: which version
counts, what it does to the roster inputs, and that nothing published later
is ever used."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sportspred.core.ids import person_key
from sportspred.sports.basketball.features import BasketballConfig, _roster_features
from sportspred.sports.basketball.reports import pregame_reports

TEAMS = {"Alpha City": "tA", "Beta Town": "tB", "Gamma Bay": "tC", "Delta Falls": "tD"}


def _listing(published, game_date, team, player, status):
    return {"published_at": pd.Timestamp(published), "game_date": game_date, "matchup": "x", "team_name": team,
            "player_name": player, "status": status, "detail": None, "not_submitted": status is None}


def _events():
    # One slate: an early game (00:00Z) and a late one (03:00Z) on the same US evening.
    return pd.DataFrame({
        "event_id": ["early", "late"],
        "start_time_utc": pd.to_datetime(["2030-01-11T00:00:00Z", "2030-01-11T03:00:00Z"]),
        "home_id": ["tA", "tC"], "away_id": ["tB", "tD"],
    })


def test_the_version_in_force_is_the_newest_one_an_hour_before_the_slates_first_game():
    listings = pd.DataFrame([
        _listing("2030-01-10T18:30:00Z", "2030-01-10", "Alpha City", "Star One", "questionable"),
        _listing("2030-01-10T18:30:00Z", "2030-01-10", "Alpha City", "Cleared Later", "out"),
        _listing("2030-01-10T22:30:00Z", "2030-01-10", "Alpha City", "Star One", "out"),        # in force
        _listing("2030-01-10T22:30:00Z", "2030-01-10", "Beta Town", None, None),                # not yet submitted
        _listing("2030-01-10T23:30:00Z", "2030-01-10", "Alpha City", "Star One", "available"),  # after the cut-off
        _listing("2030-01-10T22:30:00Z", "2030-01-10", "Gamma Bay", "Late Guard", "doubtful"),
        # Published before the LATE game's own tip-off but after the slate's cut-off: not used.
        _listing("2030-01-11T01:30:00Z", "2030-01-10", "Gamma Bay", "Late Guard", "available"),
        _listing("2030-01-10T22:30:00Z", "2030-01-10", "Nowhere United", "Ghost", "out"),       # unknown team
    ])
    listed, covered = pregame_reports(listings, _events(), TEAMS)
    got = {(r.event_id, r.team_id, r.player_name): r.status for r in listed.itertuples()}
    assert got == {("early", "tA", "Star One"): "out", ("late", "tC", "Late Guard"): "doubtful"}
    assert set(listed["published_at_utc"]) == {"2030-01-10T22:30:00Z"}
    assert (listed["published_at_utc"] <= listed["cutoff_utc"]).all()
    assert set(listed["cutoff_utc"]) == {"2030-01-10T23:00:00Z"}            # one information time for the slate
    assert set(covered["event_id"]) == {"early", "late"}                     # Beta Town's game is covered, nobody listed
    assert listed.iloc[0]["player_key"] == person_key("Star One")


def test_no_version_before_the_cut_off_means_no_report():
    listings = pd.DataFrame([_listing("2030-01-10T23:30:00Z", "2030-01-10", "Alpha City", "Star One", "out")])
    listed, covered = pregame_reports(listings, _events(), TEAMS)
    assert listed.empty and covered.empty


def _league(n_games=6):
    """Team H plays team V every other day; H's star plays 36 minutes and scores far more than the others."""
    events, box = [], []
    for g in range(n_games):
        eid = f"g{g}"
        events.append({"event_id": eid, "home_id": "H", "away_id": "V",
                       "start_time_utc": pd.Timestamp("2030-01-01T03:00:00Z") + pd.Timedelta(days=2 * g)})
        for team in ("H", "V"):
            for k in range(8):
                star = team == "H" and k == 0
                box.append({"event_id": eid, "team_id": team, "player_id": f"{team}{k}",
                            "player_name": "Hank Star Jr." if star else f"{team} Player {k}",
                            "minutes": 36.0 if k < 5 else 20.0, "pts": 34.0 if star else 9.0, "fgm": 4.0, "fga": 9.0,
                            "fta": 2.0, "ftm": 1.0, "oreb": 1.0, "dreb": 3.0, "stl": 1.0, "ast": 2.0, "blk": 0.0,
                            "pf": 2.0, "tov": 1.0})
    return pd.DataFrame(events), pd.DataFrame(box)


def _features(reports, completed_upto=5):
    ev, box = _league()
    n = len(ev)
    completed = np.arange(n) < completed_upto
    return _roster_features(ev, box, batch=np.arange(n), season_idx=np.zeros(n, dtype=int), completed=completed,
                            cfg=BasketballConfig(), reports=reports)


def test_a_reported_absence_lowers_that_sides_rotation_and_only_that_sides():
    plain = _features({})
    assert np.allclose(plain["roster_q_rep_a"], plain["roster_q_a"], equal_nan=True)      # no report: unadjusted
    # The report names him without the suffix the box score uses.
    out = _features({("g5", "H"): [(person_key("Hank Star"), "out")]})
    assert out["roster_q_rep_a"][5] < out["roster_q_a"][5] - 1e-4                        # the star's minutes are gone
    assert out["roster_q_rep_b"][5] == out["roster_q_b"][5]
    assert np.allclose(out["roster_q_rep_a"][:5], out["roster_q_a"][:5], equal_nan=True)  # other games untouched
    assert out["roster_q_a"][5] == plain["roster_q_a"][5]                                 # the plain input never moves
    # Questionable and probable players are assumed to play, exactly as the live path assumes.
    maybe = _features({("g5", "H"): [(person_key("Hank Star"), "questionable")]})
    assert maybe["roster_q_rep_a"][5] == maybe["roster_q_a"][5]
    # A name that is not in the rotation changes nothing.
    stranger = _features({("g5", "H"): [("somebody else", "out")]})
    assert stranger["roster_q_rep_a"][5] == stranger["roster_q_a"][5]


def test_a_report_published_after_now_is_not_used_by_the_feature_builder(store):
    from sportspred.sports.basketball.features import _pregame_reports

    listed = pd.DataFrame([{"event_id": "e1", "team_id": "tA", "player_name": "Star One",
                            "player_key": "star one", "status": "out",
                            "published_at_utc": "2030-01-10T22:30:00Z", "cutoff_utc": "2030-01-10T23:00:00Z"}])
    store.write_table("nba", "pregame_report", listed, mode="replace")
    store.write_table("nba", "pregame_report_games",
                      pd.DataFrame([{"event_id": "e1", "published_at_utc": "2030-01-10T22:30:00Z"}]), mode="replace")
    before, covered_before = _pregame_reports(store, "nba", pd.Timestamp("2030-01-10T22:00:00Z"))
    after, covered_after = _pregame_reports(store, "nba", pd.Timestamp("2030-01-10T23:00:00Z"))
    assert before == {} and covered_before == set()
    assert after == {("e1", "tA"): [("star one", "out")]} and covered_after == {"e1"}
    assert _pregame_reports(store, "wnba", pd.Timestamp("2030-01-11T00:00:00Z")) == ({}, None)   # no table: no feature
