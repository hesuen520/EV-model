"""Tennis: winner-first source rows must not leak the result, tournament-week
ordering, walkovers and retirements, and point-in-time features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sportspred.sports.tennis import adapter, features
from sportspred.sports.tennis.adapter import TennisLeague, normalize_matches, store_matches
from sportspred.sports.tennis.features import TennisConfig, build_tennis_frame
from sportspred.validation.leakage import asof_invariance

LEAGUE = "demo_tennis"


@pytest.fixture
def demo_tour(monkeypatch):
    """A synthetic tour registered under a demo_ key so real data stays untouched."""
    monkeypatch.setitem(adapter.TENNIS_LEAGUES, LEAGUE,
                        TennisLeague(LEAGUE, "demo", "x", "x", 2015, lambda d: d["tourney_level"].eq("A")))
    monkeypatch.setitem(adapter.TOUR_GROUP, "demo", [LEAGUE])
    monkeypatch.setitem(features.TOUR_GROUP, "demo", [LEAGUE])


def _raw(n_players=24, n_weeks=140, seed=5):
    """Sackmann-format rows: the WINNER is always listed first."""
    rng = np.random.default_rng(seed)
    strength = rng.normal(0, 1.0, n_players)
    rows = []
    week = pd.Timestamp("2015-01-05")
    for w in range(n_weeks):
        order = rng.permutation(n_players)
        for m in range(n_players // 2):
            a, b = int(order[2 * m]), int(order[2 * m + 1])
            p = 1 / (1 + np.exp(-(strength[a] - strength[b]) * 1.2))
            win, lose = (a, b) if rng.uniform() < p else (b, a)
            score = "6-4 6-3"
            if m == 0 and w % 20 == 0:
                score = "W/O"
            elif m == 1 and w % 15 == 0:
                score = "6-4 2-1 RET"
            sv = 60 + int(rng.integers(0, 30))
            rows.append({
                "tourney_id": f"{week.year}-{w:03d}", "tourney_name": f"Demo Open {w}", "surface": ["Hard", "Clay"][w % 2],
                "tourney_level": "A", "tourney_date": int(week.strftime("%Y%m%d")), "match_num": m + 1,
                "winner_id": 1000 + win, "winner_name": f"Demo Player {win:02d}", "loser_id": 1000 + lose,
                "loser_name": f"Demo Player {lose:02d}", "score": score, "best_of": 5 if w % 10 == 0 else 3,
                "round": "R32", "minutes": 90.0, "w_svpt": sv, "w_1stWon": int(sv * 0.45), "w_2ndWon": int(sv * 0.22),
                "l_svpt": sv, "l_1stWon": int(sv * 0.40), "l_2ndWon": int(sv * 0.20),
            })
        week += pd.Timedelta(days=7)
    return pd.DataFrame(rows), strength


def _load(store, raw):
    parts = [normalize_matches(raw[raw["tourney_date"] // 10000 == y], LEAGUE, y)
             for y in sorted(set(raw["tourney_date"] // 10000))]
    return store_matches(store, LEAGUE, pd.concat(parts, ignore_index=True), is_synthetic=True)


def test_walkovers_are_skipped_and_retirements_flagged(demo_tour):
    raw, _ = _raw()
    norm = normalize_matches(raw, LEAGUE, 2015)
    assert len(norm) == len(raw) - raw["score"].eq("W/O").sum()
    assert (norm["method"] == "RET").sum() == raw["score"].str.contains("RET").sum()
    assert set(norm["time_precision"]) == {"period"} and str(norm["start_time_utc"].dt.tz) == "UTC"


def test_stored_order_and_side_a_do_not_reveal_the_winner(store, demo_tour):
    raw, _ = _load_and_raw = _raw()
    report = _load(store, raw)
    ev = store.events(LEAGUE, include_synthetic=True)
    assert report.n_events == len(ev) and (ev["home_id"] < ev["away_id"]).all()
    assert 0.42 < (ev["winner_id"] == ev["home_id"]).mean() < 0.58          # not winner-first
    frame = build_tennis_frame(store, LEAGUE, now="2030-01-01T00:00:00Z", cfg=TennisConfig(burn_in_years=1))
    done = frame.completed
    assert 0.42 < done["y"].mean() < 0.58                                   # a no-feature model scores about 50%
    # The ratings do carry signal once they have warmed up.
    late = done[done["start_time_utc"] >= "2016-06-01"]
    assert np.corrcoef(late["elo_diff"], late["y"])[0, 1] > 0.2
    assert frame.df["serve_edge"].notna().mean() > 0.8 and set(frame.df["bo5"]) == {0.0, 1.0}


def test_a_match_never_uses_its_own_tournament_week(store, demo_tour):
    raw, _ = _raw()
    _load(store, raw)
    frame = build_tennis_frame(store, LEAGUE, now="2030-01-01T00:00:00Z", cfg=TennisConfig(burn_in_years=1))
    df = frame.df
    assert (df["info_cutoff_utc"].dropna() < df.loc[df["info_cutoff_utc"].notna(), "start_time_utc"]).all()
    # Every match of a tournament week shares one start time, so it shares one information cut-off too.
    assert df.groupby("start_time_utc")["info_cutoff_utc"].nunique(dropna=False).max() == 1

    class Builder:
        def feature_frame(self, st, league, market, *, now=None):
            return build_tennis_frame(st, league, market, now=now, cfg=TennisConfig(burn_in_years=1))

    cut = [pd.Timestamp("2016-03-06T23:59:59Z"), pd.Timestamp("2017-02-05T23:59:59Z")]
    result = asof_invariance(Builder(), store, LEAGUE, "winner", cutoffs=cut)
    assert len(result) and result["ok"].all() and result["n_events"].min() >= 10
