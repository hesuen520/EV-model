"""Regression tests for football audit fixes, and unit tests for the baseball
schedule features (rest, travel, bullpen workload)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sp_helpers import synthetic_two_sided
from sportspred.sports.baseball.features import bullpen_load, rest_and_travel, slate_days
from sportspred.sports.football.college import normalize_cfb
from sportspred.sports.football.features import FootballConfig, _qb_features
from sportspred.sports.generic.features import GenericOptions, build_generic_frame


# ------------------------------------------------------------------ football
def test_listed_quarterback_who_did_not_play_is_replaced_by_the_one_who_did():
    """The schedule file names a starter per game.  When he has no stat line
    for his team in that game, the quarterback with the most plays started -
    and it is that man who is the incumbent for the team's next game."""
    ev = pd.DataFrame({"event_id": ["g1", "g2"], "home_id": ["H", "H"], "away_id": ["A", "B"]})
    extra = pd.DataFrame({"home_qb_id": ["q_listed", None], "away_qb_id": ["q_a", None],
                          "home_qb_name": ["Listed Starter", None], "away_qb_name": ["Away QB", None]})
    qb_stats = pd.DataFrame({
        "event_id": ["g1", "g1", "g1"], "team_id": ["H", "H", "A"],
        "player_id": ["q_backup", "q_third", "q_a"], "player_name": ["Backup", "Third", "Away QB"],
        "plays": [38.0, 4.0, 40.0], "epa": [6.0, -1.0, 2.0],
    })
    out, table = _qb_features(ev, extra, qb_stats, batch=np.array([0, 1]), played=np.array([True, False]),
                              lr=0.1, cfg=FootballConfig())
    assert table["starters_corrected_from_box_score"] == 1
    assert table["teams"]["H"]["incumbent"] == "q_backup"
    assert table["teams"]["A"]["incumbent"] == "q_a"            # a listed starter who played is kept
    # The next game's pregame feature is the backup's rating, built only from the earlier game.
    assert out["qb_inc_a"][1] == table["qbs"]["q_backup"]["rating"] or np.isclose(
        out["qb_inc_a"][1], table["qbs"]["q_backup"]["rating"], atol=1e-5)
    assert np.isnan(out["qb_inc_a"][0])                         # nothing is known before the first game


def _cfb_row(game_id, start, tbd, home_points=None, away_points=None, completed=False):
    return {"game_id": game_id, "season_type": "regular", "home_division": "fbs", "away_division": "fbs",
            "start_date": start, "start_time_tbd": tbd, "home_points": home_points, "away_points": away_points,
            "completed": completed, "home_id": 10, "away_id": 20, "home_team": "Home U", "away_team": "Away U",
            "neutral_site": False, "venue": "Stadium", "week": 3, "conference_game": True}


def test_one_college_game_listed_twice_is_kept_once_preferring_the_timed_record():
    """The source sometimes carries a game under two IDs: a date-only copy and
    a timed copy.  Two records would apply the result to the ratings twice."""
    raw = pd.DataFrame([
        _cfb_row(111, "2025-09-13T04:00:00.000Z", True, 31, 17, True),       # placeholder at local midnight
        _cfb_row(222, "2025-09-13T19:30:00.000Z", False, 31, 17, True),      # the same game with its kick-off
        _cfb_row(333, "2025-09-20T19:30:00.000Z", False),                    # a different week
    ])
    out = normalize_cfb(raw, "ncaaf", 2025)
    assert list(out["provider_event_id"]) == ["222", "333"]
    assert out.iloc[0]["time_precision"] == "datetime" and out.iloc[0]["status"] == "final"
    assert out.iloc[1]["status"] == "scheduled"


def test_an_event_starting_exactly_at_the_prediction_time_is_upcoming_not_lost(store):
    """It was in neither the completed nor the upcoming set."""
    synthetic_two_sided(store, "demo_league", n_teams=4, n_seasons=4, rounds_per_season=2)
    scheduled = store.events("demo_league", status="scheduled", with_results=False, include_synthetic=True)
    first = scheduled["start_time_utc"].min()
    same_instant = set(scheduled.loc[scheduled["start_time_utc"] == first, "event_id"])
    frame = build_generic_frame(store, "demo_league", "winner", opts=GenericOptions(), now=first)
    assert same_instant <= set(frame.upcoming["event_id"])
    assert not same_instant & set(frame.completed["event_id"])


# ------------------------------------------------------------------ baseball
def test_slate_days_follow_the_honolulu_calendar():
    starts = pd.Series(pd.to_datetime(["2025-06-01T23:05:00Z", "2025-06-02T02:10:00Z", "2025-06-02T10:30:00Z"]))
    day = slate_days(starts)
    assert day[0] == day[1]                 # 1:05 PM and 4:10 PM HST on the same day
    assert day[2] == day[0] + 1             # 12:30 AM HST the next day


def test_rest_and_travel_use_only_the_previous_game_of_the_same_season():
    # Team 0 hosts team 1 twice on consecutive days in city X, then visits team 2 in city Y two days later.
    a = np.array([0, 0, 2])
    b = np.array([1, 1, 0])
    season = np.array([0, 0, 0])
    day = np.array([10, 11, 13])
    lat = np.array([40.0, 40.0, 34.0])
    lon = np.array([-74.0, -74.0, -118.0])
    out = rest_and_travel(a, b, season, day, lat, lon, n_teams=3, rest_cap=5.0)
    assert list(out["rest_a"]) == [5.0, 1.0, 5.0]               # first game of a season gets the cap
    assert list(out["rest_b"]) == [5.0, 1.0, 2.0]
    assert np.isnan(out["travel_km_a"][0]) and out["travel_km_a"][1] == 0.0
    assert out["travel_km_b"][2] > 3000                          # New York area to Los Angeles area
    assert out["tz_shift_b"][2] < 0                              # westward
    assert list(out["season_games_b"]) == [0, 1, 2]
    # A new season forgets the previous one's last game.
    out2 = rest_and_travel(np.array([0, 0]), np.array([1, 1]), np.array([0, 1]), np.array([10, 200]),
                           np.array([40.0, 40.0]), np.array([-74.0, -74.0]), n_teams=2, rest_cap=5.0)
    assert out2["rest_a"][1] == 5.0 and np.isnan(out2["travel_km_a"][1]) and out2["season_games_a"][1] == 0


def test_bullpen_load_counts_earlier_days_only_never_the_same_slate():
    # Team 0 plays on days 1, 2, 2 (doubleheader) and 3; its bullpen records 9, 6, 3 and 12 relief outs.
    a = np.array([0, 0, 0, 0])
    b = np.array([1, 1, 1, 1])
    day = np.array([1, 2, 2, 3])
    relief_a = np.array([9.0, 6.0, 3.0, 12.0])
    relief_b = np.zeros(4)
    load_a, _ = bullpen_load(a, b, day, relief_a, relief_b, n_teams=2, window=2)
    assert load_a[0] == 0.0
    assert load_a[1] == 9.0
    assert load_a[2] == 9.0                 # the first game of the doubleheader is the same slate: not known yet
    assert load_a[3] == 9.0 + 6.0 + 3.0     # days 1 and 2 are inside the two-day window
    load_short, _ = bullpen_load(a, b, day, relief_a, relief_b, n_teams=2, window=1)
    assert load_short[3] == 6.0 + 3.0       # day 1 has dropped out
