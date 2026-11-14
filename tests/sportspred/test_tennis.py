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

    # At the start of a tournament week the previous week is complete: what the builder
    # knows then is exactly what the backtest gave that week's matches.
    cut = [pd.Timestamp("2016-03-07T00:00:00Z"), pd.Timestamp("2017-02-06T00:00:00Z")]
    result = asof_invariance(Builder(), store, LEAGUE, "winner", cutoffs=cut)
    assert len(result) and result["ok"].all() and result["n_events"].min() >= 10
    # One second earlier the week in progress is still unknown, so the same matches are
    # built from LESS information than in the backtest - a difference in the safe direction.
    early = build_tennis_frame(store, LEAGUE, now="2016-03-06T23:59:59Z", cfg=TennisConfig(burn_in_years=1)).df
    monday = pd.Timestamp("2016-03-07T00:00:00Z")
    a = early[early["start_time_utc"] == monday].set_index("event_id")["games_played_a"]
    b = df[df["start_time_utc"] == monday].set_index("event_id")["games_played_a"]
    assert (a <= b.loc[a.index]).all() and (a < b.loc[a.index]).any()


def test_quarter_finals_are_main_draw_matches_not_qualifying():
    """'QF' starts with a Q.  Every quarter-final was dropped from the main-draw
    leagues (and never reached the qualifying ones), about one match in ten."""
    raw = pd.DataFrame({"round": ["Q1", "Q2", "Q3", "QF", "SF", "F", "R32", "RR", "qf"]})
    is_q = adapter._is_q(raw)
    assert list(is_q) == [True, True, True, False, False, False, False, False, False]

    rows = pd.DataFrame([{
        "tourney_id": "2019-580", "tourney_name": "Demo Slam", "surface": "Hard", "tourney_level": "G",
        "tourney_date": 20190114, "match_num": 100 + i, "winner_id": 1 + i, "winner_name": f"W {i}",
        "loser_id": 50 + i, "loser_name": f"L {i}", "score": "6-4 6-4 6-4", "best_of": 5, "round": rnd,
    } for i, rnd in enumerate(["R16", "QF", "QF", "SF", "F", "Q1"])])
    main = normalize_matches(rows, "atp", 2019)
    assert sorted(main["round"]) == ["F", "QF", "QF", "R16", "SF"]            # the qualifying match is excluded
    qual = normalize_matches(rows, "atp_qual", 2019)
    assert list(qual["round"]) == ["Q1"]


def test_a_tournament_in_progress_is_not_known_until_its_week_is_over(store, demo_tour):
    """Archive rows carry the tournament's start date, whatever day the match
    was played.  Asked on the Wednesday, the builder used to know the final."""
    raw, _ = _raw()
    _load(store, raw)
    cfg = TennisConfig(burn_in_years=1)
    monday = pd.Timestamp("2016-03-07T00:00:00Z")                               # a tournament date in the demo tour
    midweek = build_tennis_frame(store, LEAGUE, now=monday + pd.Timedelta(days=2, hours=12), cfg=cfg)
    this_week = midweek.df[midweek.df["start_time_utc"] == monday]
    assert len(this_week) >= 10 and this_week["y"].isna().all()                 # results not known yet
    earlier = midweek.df[midweek.df["start_time_utc"] == monday - pd.Timedelta(days=7)]
    assert earlier["y"].notna().all()                                           # last week's are
    after = build_tennis_frame(store, LEAGUE, now=monday + pd.Timedelta(days=7), cfg=cfg)
    assert after.df[after.df["start_time_utc"] == monday]["y"].notna().all()
    # The week's own rows have the same features whether or not its results are on file.
    cols = ["event_id"] + list(midweek.feature_cols)
    a = this_week[cols].set_index("event_id").sort_index()
    b = after.df[after.df["start_time_utc"] == monday][cols].set_index("event_id").sort_index()
    assert np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True)


def test_tournaments_starting_in_the_same_week_do_not_see_each_other(store, demo_tour):
    from sportspred.sports.tennis.features import week_start

    raw, _ = _raw(n_weeks=60)
    # A second tournament three days after each Monday one, with the same players.
    late = raw.copy()
    late["tourney_id"] = late["tourney_id"] + "-B"
    late["tourney_date"] = (pd.to_datetime(late["tourney_date"].astype(str)) + pd.Timedelta(days=3)).dt.strftime("%Y%m%d").astype(int)
    both = pd.concat([raw, late], ignore_index=True)
    _load(store, both)
    frame = build_tennis_frame(store, LEAGUE, now="2030-01-01T00:00:00Z", cfg=TennisConfig(burn_in_years=1))
    df = frame.df
    weeks = week_start(df["start_time_utc"])
    # One information cut-off per week, and it lies before the week's first tournament.
    assert df.groupby(weeks)["info_cutoff_utc"].nunique(dropna=False).max() == 1
    known = df["info_cutoff_utc"].notna()
    assert (df.loc[known, "info_cutoff_utc"] < weeks[known]).all()


def test_a_fixture_imported_by_csv_is_rated_with_the_archive_players_history(store, demo_tour, tmp_path, monkeypatch):
    """The CSV import used to look names up under the league, the archive
    stores players under the tour: a fixture between two known players got two
    brand-new entries and a 50/50 rating.  Its surface and best-of were ignored."""
    from sportspred.adapters.csv_import import CsvImportError, import_events
    from sportspred.sports import get_league as real_get_league
    from sportspred.sports.tennis import PLUGIN as TENNIS

    raw, strength = _raw()
    _load(store, raw)
    monkeypatch.setattr(type(TENNIS), "leagues", TENNIS.leagues + (
        type(TENNIS.leagues[0])(LEAGUE, "tennis", "Demo tour", gender="men", level="tour", discipline="singles",
                                markets=TENNIS.leagues[0].markets),))
    best, worst = int(np.argmax(strength)), int(np.argmin(strength))
    csv = tmp_path / "fixtures.csv"
    csv.write_text("event_id,season,start_time,status,home,away,neutral_site,surface,best_of\n"
                   f"fx1,2018,2018-01-10T15:00:00Z,scheduled,Demo Player {best:02d},Demo Player {worst:02d},1,clay,5\n")
    with pytest.raises(CsvImportError, match="unknown participant"):
        bad = tmp_path / "bad.csv"
        bad.write_text(csv.read_text().replace(f"Demo Player {best:02d}", "Nobody Known"))
        import_events(store, LEAGUE, bad)
    out = import_events(store, LEAGUE, csv)                        # no --create-participants needed
    assert out["events_new"] == 1
    frame = build_tennis_frame(store, LEAGUE, now="2018-01-09T00:00:00Z", cfg=TennisConfig(burn_in_years=1))
    row = frame.upcoming.iloc[0]
    favourite_is_a = row["side_a_id"] == store.resolve_participant("tennis", "tennis_demo", f"Demo Player {best:02d}",
                                                                  create=False)
    gap = row["elo_diff"] if favourite_is_a else -row["elo_diff"]
    assert gap > 150                                               # the archive ratings are behind the fixture
    assert row["bo5"] == 1.0 and row["surface"] == "clay" and not np.isnan(row["surf_elo_diff"])
