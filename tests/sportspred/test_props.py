"""Player-statistic markets: the count distribution, point-in-time features,
threshold import, the prediction path into the ledger, and grading."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from sportspred.adapters.csv_import import CsvImportError, import_prop_lines
from sportspred.config import Settings
from sportspred.coverage import coverage_status
from sportspred.ledger.grading import grade_predictions
from sportspred.ledger.review import graded_frame
from sportspred.models.candidates import PriorCandidate
from sportspred.models.counts import (ColumnMean, CountCandidate, ProductMean, TwoStageMean, count_probabilities,
                                      fit_dispersion, nb_alpha, threshold_probabilities)
from sportspred.pipeline import train_league, validate_league
from sportspred.predict import predict_league
from sportspred.predict.engine import _outcome_probabilities, load_promoted
from sportspred.sports import get_league
from sportspred.sports.base import AvailabilityEffect
from sportspred.sports.basketball.props import (_ew, _hold_within_day, _rolling_prior, apply_prop_availability,
                                                build_prop_frame, prop_market, split_prop_market, threshold_sweep)
from sportspred.storage import Store
from sportspred.storage.store import StoreError
from sportspred.validation.backtest import _blind
from sportspred.validation.leakage import asof_invariance

LEAGUE = "nba"       # a temporary store; rows are not flagged synthetic so the default reads see them
N_TEAMS, N_PLAYERS = 6, 8
FIRST_SEASON, N_SEASONS = 2015, 13
NOW = "2027-11-20T00:00:00Z"            # after the last played round, before the scheduled one
FAR = "2099-01-01T00:00:00Z"            # "everything on file": never the machine's clock, the league is dated 2015-2027


# ---------------------------------------------------------------- distribution
def test_half_point_threshold_has_no_push_and_matches_the_negative_binomial():
    mu, params = np.array([18.0, 6.5, 2.2]), (-2.0, -0.4)
    under, push, over = count_probabilities(mu, np.array([17.5, 6.5, 1.5]), params)
    assert np.allclose(push, 0.0) and np.allclose(under + over, 1.0)
    alpha = nb_alpha(mu, params)
    n, p = 1.0 / alpha, (1.0 / alpha) / (1.0 / alpha + mu)
    assert np.allclose(under, stats.nbinom.cdf([17, 6, 1], n, p), atol=1e-9)


def test_whole_number_threshold_gives_the_push_its_own_probability():
    mu, params = np.array([18.0, 6.5]), (-2.0, -0.4)
    under, push, over = count_probabilities(mu, np.array([18.0, 7.0]), params)
    alpha = nb_alpha(mu, params)
    n, p = 1.0 / alpha, (1.0 / alpha) / (1.0 / alpha + mu)
    assert np.allclose(push, stats.nbinom.pmf([18, 7], n, p), atol=1e-9)
    assert np.allclose(under + push + over, 1.0) and (push > 0.03).all()
    # Without a dispersion the distribution is Poisson.
    u, q, o = count_probabilities(np.array([4.0]), np.array([4.0]), None)
    assert np.isclose(q[0], stats.poisson.pmf(4, 4.0)) and np.isclose(u[0], stats.poisson.cdf(3, 4.0))


def test_off_a_whole_number_the_push_is_exactly_zero():
    rng = np.random.default_rng(1)
    mu = rng.uniform(0.2, 40.0, 5000)
    line = np.floor(rng.uniform(0.0, 45.0, 5000)) + 0.5
    under, push, over = count_probabilities(mu, line, (-2.0, -0.4))
    assert (push == 0.0).all() and np.allclose(under + over, 1.0, atol=1e-12)


def test_one_event_has_one_probability_whichever_threshold_asks_for_it():
    """Over 23 and over 23.5 are the same event; so are under 23 and under 22.5.
    A whole-number threshold is read from the half points either side."""
    def over_at(h):                         # any decreasing function of the threshold stands in for a model
        return 1.0 / (1.0 + np.exp((np.asarray(h) - 20.0) / 4.0))

    line = np.array([23.0, 23.5, 22.5, 1.0, 0.5])
    under, push, over = threshold_probabilities(over_at, line)
    assert over[0] == over[1] and under[0] == under[2]
    assert np.allclose(under + push + over, 1.0) and push[0] > 0 and push[1] == 0.0 == push[2]
    assert under[3] == pytest.approx(1.0 - over_at(0.5)) and push[4] == 0.0


def test_a_row_without_a_threshold_cannot_be_priced():
    with pytest.raises(ValueError, match="without a line"):
        count_probabilities(np.array([10.0]), np.array([np.nan]), None)


def test_dispersion_is_recovered_from_simulated_counts():
    rng = np.random.default_rng(3)
    mu = rng.uniform(4.0, 30.0, 40000)
    alpha = 0.08
    y = rng.negative_binomial(1.0 / alpha, (1.0 / alpha) / (1.0 / alpha + mu))
    a, b = fit_dispersion(mu, y)
    fitted = nb_alpha(np.array([8.0, 20.0]), (a, b))
    assert np.allclose(fitted, alpha, rtol=0.25)


def test_count_candidate_reads_only_declared_inputs_and_sums_to_one():
    rng = np.random.default_rng(5)
    n = 3000
    minutes = rng.uniform(15, 38, n)
    rate = rng.uniform(0.3, 0.7, n)
    played = np.clip(minutes + rng.normal(0, 3, n), 1, 48)
    train = pd.DataFrame({"min_ew": minutes, "rate_ew": rate, "minutes_played": played,
                          "stat_value": rng.poisson(played * rate).astype(float), "block": 1})
    train["line"] = np.floor(train["min_ew"] * train["rate_ew"]) + 0.5
    train["y"] = (train["stat_value"] > train["line"]).astype(float)
    cand = CountCandidate("c", ProductMean("min_ew", "rate_ew")).fit(train, ("under", "over"))
    blind = _blind(train, cand, "block")
    assert blind["stat_value"].isna().all() and blind["minutes_played"].isna().all()     # outcomes are hidden
    P = cand.predict_proba(blind)
    assert np.allclose(P.sum(axis=1), 1.0) and np.all((P > 0) & (P < 1))
    assert abs(P[:, 1].mean() - train["y"].mean()) < 0.03
    two = CountCandidate("t", TwoStageMean(["min_ew"], ["rate_ew"], kind="glm")).fit(train, ("under", "over"))
    proj = two.projection(blind.head(5))
    assert {"exposure", "rate", "mean", "q10", "q50", "q90"} <= set(proj.columns)
    assert (proj["q10"] <= proj["q50"]).all() and (proj["q50"] <= proj["q90"]).all()
    with pytest.raises(ValueError):
        CountCandidate("x", ColumnMean(["min_ew"])).fit(train, ("b", "a"))


def test_market_keys():
    assert prop_market("points") == "player_prop:points"
    assert split_prop_market("player_prop:points:1629029") == ("points", "1629029")
    assert split_prop_market("player_prop:pra") == ("pra", None)
    assert split_prop_market("winner") == (None, None)


# ------------------------------------------------------------ synthetic league
def _build_league(store: Store, double_day: bool = False) -> dict:
    """Six teams of eight players over thirteen seasons: persistent player
    minutes and per-minute rates, opponents that allow more or less, random
    absences.  The last season has played rounds and then a scheduled one.
    With ``double_day`` one slate of a middle season is moved onto the day of
    the slate before it, so every team plays twice on that day."""
    rng = np.random.default_rng(11)
    teams = pd.DataFrame({"name": [f"Prop Team {chr(65 + i)}" for i in range(N_TEAMS)]})
    ids = list(store.resolve_participants(teams, sport="basketball", scope=LEAGUE))
    allow = dict(zip(ids, rng.uniform(0.88, 1.12, N_TEAMS)))
    base_min = np.array([36, 34, 32, 30, 26, 22, 18, 12], dtype=float)
    players = {}
    for t, team in enumerate(ids):
        for j in range(N_PLAYERS):
            players[(team, j)] = {"player_id": str(1000 + 10 * t + j), "name": f"Player {chr(65 + t)}{j}",
                                  "min": base_min[j], "pts": rng.uniform(0.35, 0.75), "reb": rng.uniform(0.1, 0.35),
                                  "ast": rng.uniform(0.05, 0.25), "fg3": rng.uniform(0.02, 0.1)}
    events, box = [], []
    for s in range(N_SEASONS):
        last = s == N_SEASONS - 1
        day = pd.Timestamp(f"{FIRST_SEASON + s}-11-01T03:00:00Z")
        for r in range(3 if last else 4):
            upcoming = last and r == 2
            order = rng.permutation(N_TEAMS)
            for k in range(N_TEAMS - 1):
                second_game_of_day = double_day and s == 6 and r == 1 and k == 2
                if second_game_of_day:
                    day = day - pd.Timedelta(days=2) + pd.Timedelta(hours=3)      # same Pacific/Honolulu day
                    doubled = day - pd.Timedelta(hours=3)
                rot = np.r_[order[0], np.roll(order[1:], k)]
                for j in range(N_TEAMS // 2):
                    h, a = ids[rot[j]], ids[rot[-1 - j]]
                    if (k + r) % 2:
                        h, a = a, h
                    gid = f"g-{s}-{r}-{k}-{j}"
                    events.append({"league": LEAGUE, "season": str(FIRST_SEASON + s + 1), "start_time_utc": day,
                                   "status": "scheduled" if upcoming else "final", "provider": "synthetic",
                                   "provider_event_id": gid, "home_id": h, "away_id": a, "neutral_site": 0})
                    if upcoming:
                        continue
                    for team, opp in ((h, a), (a, h)):
                        for jj in range(N_PLAYERS):
                            p = players[(team, jj)]
                            row = {"gid": gid, "team_id": team, "player_id": p["player_id"], "player_name": p["name"],
                                   "position": "G", "starter": int(jj < 5)}
                            if rng.random() < 0.08:            # did not play: listed with empty statistics
                                box.append({**row, "minutes": np.nan})
                                continue
                            minutes = float(np.clip(round(rng.normal(p["min"], 3.0)), 1, 48))
                            f = allow[opp]
                            pts, fg3 = rng.poisson(minutes * p["pts"] * f), rng.poisson(minutes * p["fg3"] * f)
                            box.append({**row, "minutes": minutes, "pts": float(pts), "fg3m": float(fg3),
                                        "reb": float(rng.poisson(minutes * p["reb"] * f)),
                                        "ast": float(rng.poisson(minutes * p["ast"] * f)),
                                        "fga": float(rng.poisson(minutes * 0.4)), "fta": float(rng.poisson(minutes * 0.1)),
                                        "fg3a": float(fg3 + rng.poisson(minutes * 0.06))})
                day += pd.Timedelta(days=2) - (pd.Timedelta(hours=3) if second_game_of_day else pd.Timedelta(0))
    stored, _ = store.upsert_events(pd.DataFrame(events), sport="basketball")
    eid = dict(zip(stored["provider_event_id"], stored["event_id"]))
    frame = pd.DataFrame(box)
    frame["event_id"] = frame["gid"].map(eid)
    frame = frame.drop(columns="gid")
    for col in ("fgm", "ftm", "oreb", "dreb", "stl", "blk", "tov", "pf"):
        frame[col] = np.where(frame["minutes"].notna(), 0.0, np.nan)
    played = frame[frame["minutes"].notna()]
    score = played.groupby(["event_id", "team_id"])["pts"].sum()
    results = []
    for ev in stored.itertuples():
        if ev.status != "final":
            continue
        hs, aw = int(score[(ev.event_id, ev.home_id)]), int(score[(ev.event_id, ev.away_id)])
        results.append({"event_id": ev.event_id, "home_score": hs + (hs == aw), "away_score": aw})
    store.upsert_results(pd.DataFrame(results))
    store.write_table(LEAGUE, "player_box", frame)
    return {"teams": ids, "players": players, "names": dict(zip(ids, teams["name"])), "eid": eid,
            "doubled": doubled if double_day else None}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """One league, validated and trained once.  Tests read it; a test that
    writes takes a private copy with :func:`_clone`."""
    settings = Settings()
    settings.paths.data_dir = str(tmp_path_factory.mktemp("props"))
    settings.validation.n_boot, settings.validation.n_replicas = 30, 3
    settings.selection.horizon_hours = 240.0
    store = Store(settings.paths.store)
    info = _build_league(store)
    plugin, _ = get_league(LEAGUE)
    report, _ = validate_league(store, LEAGUE, "player_prop:points", settings=settings, now=NOW)
    train_league(store, LEAGUE, "player_prop:points", settings=settings, now=NOW)
    return {"settings": settings, "store": store, "plugin": plugin, "report": report, **info}


def _clone(world, tmp_path):
    target = tmp_path / "data"
    shutil.copytree(world["settings"].paths.data_dir, target)
    settings = copy.deepcopy(world["settings"])
    settings.paths.data_dir = str(target)
    return Store(settings.paths.store), settings


def _upcoming(world, stat="points", store=None, now=NOW):
    frame = build_prop_frame(store or world["store"], LEAGUE, stat, now=now)
    return frame, frame.upcoming.sort_values(["start_time_utc", "event_id"])


def _lines_csv(tmp_path, rows, name="lines.csv"):
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _line_row(world, row, line, stat="points", **extra):
    return {"team": world["names"][row["side_a_id"]], "player": row["player_name"], "stat": stat, "line": line, **extra}


def _play(store, world, game_id, special=None, box=True):
    """Finish a scheduled game: a result, and (with ``box``) a box score in which every
    player of both teams appears - 20 minutes and 5 points unless ``special`` says otherwise
    (``{player_id: (minutes, points)}``; missing minutes = listed, did not play)."""
    special = special or {}
    ev = store.events(event_ids=[game_id], with_results=False).iloc[0]
    if box:
        table = store.read_table(LEAGUE, "player_box")
        template = table[table["minutes"].notna()].iloc[0].to_dict()
        rows = []
        for team in (ev["home_id"], ev["away_id"]):
            for j in range(N_PLAYERS):
                p = world["players"][(team, j)]
                minutes, pts = special.get(p["player_id"], (20.0, 5.0))
                rows.append({**template, "event_id": game_id, "team_id": team, "player_id": p["player_id"],
                             "player_name": p["name"], "minutes": minutes, "pts": pts if minutes == minutes else np.nan})
        store.write_table(LEAGUE, "player_box", pd.concat([table, pd.DataFrame(rows)], ignore_index=True))
    store.upsert_results(pd.DataFrame([{"event_id": game_id, "home_score": 101, "away_score": 99}]))


class _Builder:
    """Duck-typed plugin for the generic leakage test.  Without a time, "everything on
    file" - not the machine's clock, which falls in the middle of this league's dates."""

    def feature_frame(self, store, league, market, *, now=None):
        return build_prop_frame(store, league, "points", now=FAR if now is None else now)


# -------------------------------------------------------------------- features
def test_features_do_not_change_when_the_future_is_hidden(world):
    full = build_prop_frame(world["store"], LEAGUE, "points", now=FAR)
    assert full.upcoming.empty or len(full.completed) > 9000                 # the whole history, not a part of it
    result = asof_invariance(_Builder(), world["store"], LEAGUE, "player_prop:points", n_cutoffs=3)
    assert len(result) and result["n_events"].min() >= 10
    assert result["max_abs_diff"].max() == 0.0


def test_a_second_game_on_one_day_sees_nothing_of_the_first(tmp_path):
    """Every input of a slate comes from earlier days - also when a team plays twice on one
    day.  As of the day's first tip-off both games' rows equal their full-history values."""
    store = Store(tmp_path / "store")
    info = _build_league(store, double_day=True)
    cutoff = info["doubled"]
    full = build_prop_frame(store, LEAGUE, "points", now=FAR).df
    same_day = full[(full["start_time_utc"] >= cutoff) & (full["start_time_utc"] <= cutoff + pd.Timedelta(hours=4))]
    assert same_day.groupby("player_id").size().max() == 2                   # the scenario is really there
    result = asof_invariance(_Builder(), store, LEAGUE, "player_prop:points", cutoffs=[cutoff])
    assert result["n_events"].min() >= 40 and result["max_abs_diff"].max() == 0.0
    # And the two rows of one player on that day carry the same pregame inputs.
    twice = same_day[same_day.duplicated("player_id", keep=False)].sort_values(["player_id", "start_time_utc"])
    first, second = twice.groupby("player_id").nth(0), twice.groupby("player_id").nth(1)
    for col in ("min_ew", "min_last3", "min_season", "gp_season", "log_rate_ew", "avg_season", "ref_line", "own_missed",
                "mates_out_min"):
        assert np.allclose(first[col].to_numpy(dtype=float), second[col].to_numpy(dtype=float), equal_nan=True), col


def test_helpers_hold_values_within_a_day():
    frame = pd.DataFrame({"who": ["p"] * 5, "day": [1, 2, 2, 3, 4], "played": [True, True, True, True, False],
                          "x": [10.0, 20.0, 40.0, 60.0, np.nan]})
    prior, post = _ew(frame, "who", ["x"], halflife=1e9)                     # a plain mean
    assert np.isnan(prior["x"][0]) and prior["x"][1] == 10.0
    assert prior["x"][2] == 10.0                                             # the day's first game is not seen
    assert prior["x"][3] == pytest.approx((10 + 20 + 40) / 3) and prior["x"][4] == pytest.approx(32.5)
    rolled = _rolling_prior(frame, "who", "x", 2, 1, "mean")
    assert rolled[1] == 10.0 == rolled[2] and rolled[3] == 30.0 and rolled[4] == 50.0
    held = _hold_within_day(frame, "who", pd.Series([0.0, 1.0, 2.0, 3.0, 4.0]))
    assert list(held) == [0.0, 1.0, 1.0, 3.0, 4.0]


def test_reference_threshold_and_outcome_use_earlier_games_only(world):
    frame = build_prop_frame(world["store"], LEAGUE, "points", now=NOW)
    done = frame.completed
    box = world["store"].read_table(LEAGUE, "player_box")
    ev = world["store"].events(LEAGUE, with_results=False)[["event_id", "start_time_utc"]]
    row = done.iloc[len(done) // 2]
    mine = box[(box["player_id"] == row["player_id"]) & box["minutes"].notna()].merge(ev, on="event_id")
    before = mine[mine["start_time_utc"] < row["start_time_utc"]].sort_values("start_time_utc")
    assert row["ref_line"] == np.floor(before["pts"].tail(15).median()) + 0.5
    actual = float(mine.loc[mine["event_id"] == row["game_id"], "pts"].iloc[0])
    assert row["stat_value"] == actual and row["y"] == float(actual > row["ref_line"])
    assert row["line"] == row["ref_line"]
    # Every completed row is a player-game: unique, and the game is named separately.
    assert not done["event_id"].duplicated().any() and (done["event_id"] == done["game_id"] + "|" + done["player_id"]).all()


def test_a_game_changes_nothing_on_its_own_day(world, tmp_path):
    """Double one game's statistics: rows of that day keep their inputs, later rows move."""
    store = world["store"]
    frame = build_prop_frame(store, LEAGUE, "points", now=NOW)
    done = frame.completed
    target = done.iloc[len(done) // 2]
    day = done[done["start_time_utc"] == target["start_time_utc"]]
    other, _ = _clone(world, tmp_path)
    box = other.read_table(LEAGUE, "player_box")
    hit = box["event_id"] == target["game_id"]
    box.loc[hit, ["pts", "minutes"]] = box.loc[hit, ["pts", "minutes"]] * 2
    other.write_table(LEAGUE, "player_box", box, mode="replace")
    again = build_prop_frame(other, LEAGUE, "points", now=NOW).df.set_index("event_id")
    cols = list(frame.feature_cols)
    same_day = again.loc[day["event_id"], cols].to_numpy(dtype=float)
    assert np.allclose(same_day, day[cols].to_numpy(dtype=float), equal_nan=True)
    later = done[(done["start_time_utc"] > target["start_time_utc"])
                 & done["player_id"].isin(box.loc[hit, "player_id"])].head(8)
    moved = again.loc[later["event_id"], cols].to_numpy(dtype=float)
    assert not np.allclose(moved, later[cols].to_numpy(dtype=float), equal_nan=True)


def test_upcoming_rows_carry_no_threshold_until_one_is_imported(world):
    frame, up = _upcoming(world)
    assert len(up) and up["line"].isna().all() and up["y"].isna().all()
    assert up["stat_value"].isna().all() and up["minutes_played"].isna().all()
    assert (up["min_ew"] >= 15).all()                                   # only the modelled rotation
    assert predict_league(world["store"], LEAGUE, "player_prop:points", settings=world["settings"], now=NOW,
                          write=False) == []


def test_a_result_without_a_box_score_is_not_a_game_everyone_missed(world, tmp_path):
    """A result can be on file before its box score.  The game must not count as one the
    whole rotation sat out, the prediction must say its inputs are a game behind, and a
    statistic market on that game cannot be graded - least of all as "did not play"."""
    store, settings = _clone(world, tmp_path)
    frame, up = _upcoming(world, store=store)
    starts = sorted(up["start_time_utc"].unique())
    first, second = up[up["start_time_utc"] == starts[0]], up[up["start_time_utc"] == starts[1]]
    cols = ["mates_out_min", "mates_new_out_min", "own_missed", "min_ew", "log_rate_ew"]
    before = second.set_index("event_id")[cols]
    assert (second["n_pending_prior"] >= 1).all() and (first["n_pending_prior"] == 0).all()
    assert (up["n_box_missing"] == 0).all()

    pick = first.iloc[0]
    import_prop_lines(store, LEAGUE, _lines_csv(tmp_path, [_line_row(world, pick, 0.5)]), collected_at=NOW)
    made = predict_league(store, LEAGUE, "player_prop:points", settings=settings, now=NOW)
    assert [p["rec_status"] for p in made] == ["PROVISIONAL"]

    for game in first["game_id"].unique():
        _play(store, world, game, box=False)                                 # results only
    later = (starts[0] + pd.Timedelta(hours=12)).isoformat()
    after = build_prop_frame(store, LEAGUE, "points", now=later).upcoming.set_index("event_id")
    same = after.loc[before.index, cols]
    assert np.allclose(same.to_numpy(dtype=float), before.to_numpy(dtype=float), equal_nan=True)
    assert (after.loc[before.index, "n_box_missing"] >= 1).all()
    assert (after.loc[before.index, "n_pending_prior"] == 0).all()           # the earlier games are final now

    nxt = second.iloc[0]
    import_prop_lines(store, LEAGUE, _lines_csv(tmp_path, [_line_row(world, nxt, 0.5)], "next.csv"), collected_at=later)
    behind = predict_league(store, LEAGUE, "player_prop:points", settings=settings, now=later, write=False)
    mine = next(p for p in behind if p["display"]["player_id"] == nxt["player_id"])
    check = next(c for c in mine["checks"] if c["key"] == "box_scores_complete")
    assert check["state"] == "missing" and mine["rec_status"] == "PROVISIONAL"
    assert any("no box score" in r for r in mine["reasons"])

    grade_predictions(store, now="2027-12-30T00:00:00Z")
    assert made[0]["prediction_id"] not in set(store.grades()["prediction_id"])      # not graded, not void
    _play(store, world, pick["game_id"], {pick["player_id"]: (30.0, 12.0)})
    grade_predictions(store, now="2027-12-31T00:00:00Z")
    assert store.grades().set_index("prediction_id")["outcome"][made[0]["prediction_id"]] == "correct"


# ------------------------------------------------------------------ validation
def test_validation_runs_baselines_first_and_cannot_be_called_validated(world):
    report = world["report"]
    names = [c["name"] for c in report["candidates"]]
    assert names[:3] == ["base_rate", "season_average_poisson", "recent_form_nb"]
    dev = {c["name"]: c["dev"]["log_loss"] for c in report["candidates"]}
    assert dev["recent_form_nb"] < dev["base_rate"]                     # knowing the player beats knowing nothing
    gate = next(g for g in report["gates"] if g["gate"] == "market_thresholds")
    assert gate["passed"] is False and report["gates_passed"] is False
    assert coverage_status(world["store"], LEAGUE, "player_prop:points").value == "experimental"
    assert report["protocol"]["final_test_used_for_selection"] is False
    assert any("reference threshold" in text for text in report["limitations"])


def test_threshold_sweep_scores_shifted_and_whole_number_thresholds(world):
    sweep = world["report"]["supplement"]
    assert sweep["kind"] == "threshold_sweep" and set(sweep["folds"]) == {"last_development_fold", "final_test"}
    rows = {r["shift"]: r for r in sweep["folds"]["final_test"]["rows"]}
    assert rows[0.0]["push_rate"] == 0.0 == rows[0.0]["mean_predicted_push"] and rows[0.5]["push_rate"] > 0.0
    assert abs(rows[0.5]["push_rate"] - rows[0.5]["mean_predicted_push"]) < 0.03
    assert rows[-6.0]["over_rate"] > rows[0.0]["over_rate"] > rows[6.0]["over_rate"]
    for r in rows.values():
        assert abs(r["over_rate"] - r["mean_predicted_over"]) < 0.06


def test_threshold_sweep_does_not_need_a_count_model(world):
    """If the base rate were ever selected, the supplement must not stop the run from being saved."""
    frame = build_prop_frame(world["store"], LEAGUE, "points", now=NOW)
    plan = world["plugin"].split_plan(LEAGUE, "player_prop:points", frame)
    sweep = threshold_sweep(frame, PriorCandidate("base_rate"), plan)
    assert len(sweep["folds"]["final_test"]["rows"]) >= 5


# ---------------------------------------------------------------- import + run
def test_import_rejects_what_it_cannot_place(world, tmp_path):
    _, up = _upcoming(world)
    first = up.iloc[0]
    team = world["names"][first["side_a_id"]]
    bench = world["players"][(first["side_a_id"], N_PLAYERS - 1)]["name"]          # 12 minutes: not modelled
    bad = _lines_csv(tmp_path, [
        {"team": team, "player": first["player_name"], "stat": "points", "line": 12.25},
        {"team": team, "player": "Nobody Atall", "stat": "points", "line": 10.5},
        {"team": team, "player": first["player_name"], "stat": "steals", "line": 1.5},
        {"team": "Unknown Club", "player": first["player_name"], "stat": "points", "line": 10.5},
        {"team": team, "player": bench, "stat": "points", "line": 4.5},
        {"team": team, "player": first["player_name"], "stat": "points", "line": 0},
        {"team": team, "player": first["player_name"], "stat": "points", "line": 14.5},
        {"team": team, "player": first["player_name"], "stat": "points", "line": 17.5},
    ])
    with pytest.raises(CsvImportError) as err:
        import_prop_lines(world["store"], LEAGUE, bad, collected_at=NOW)
    text = str(err.value)
    assert "whole or half number" in text and "has not played for" in text and "'steals' is not one of" in text
    assert "unknown participant" in text and "outside the modelled rotation" in text
    assert "line 7" in text and "one half or more" in text                              # a threshold of zero
    assert "line 9" in text and "only one can be in force" in text                      # two thresholds, one market
    assert world["store"].prop_lines_as_of(LEAGUE, FAR).empty                           # nothing was stored
    for bad_line in (3.3, 0.0):
        with pytest.raises(StoreError):
            world["store"].add_prop_lines([{"league": LEAGUE, "event_id": first["game_id"], "player_id": "1",
                                            "player_name": "x", "stat": "points", "line": bad_line,
                                            "collected_at_utc": NOW}])


def test_a_threshold_read_after_tip_off_is_not_moved_to_the_next_game(world, tmp_path):
    store, _ = _clone(world, tmp_path)
    _, up = _upcoming(world, store=store)
    starts = sorted(up["start_time_utc"].unique())
    row = up[up["start_time_utc"] == starts[0]].iloc[0]
    during = (starts[0] + pd.Timedelta(minutes=20)).isoformat()
    path = _lines_csv(tmp_path, [_line_row(world, row, 9.5)])
    with pytest.raises(CsvImportError, match="started at"):
        import_prop_lines(store, LEAGUE, path, collected_at=during)
    # Naming the later game says which one is meant.
    events = store.events(LEAGUE, with_results=False)
    nxt = events[((events["home_id"] == row["side_a_id"]) | (events["away_id"] == row["side_a_id"]))
                 & (events["start_time_utc"] > starts[0])].sort_values("start_time_utc").iloc[0]
    named = _lines_csv(tmp_path, [_line_row(world, row, 9.5, event_id=nxt["provider_event_id"])], "named.csv")
    assert import_prop_lines(store, LEAGUE, named, collected_at=during)["thresholds"] == 1
    assert set(store.prop_lines_as_of(LEAGUE, FAR)["event_id"]) == {nxt["event_id"]}


def test_thresholds_are_priced_recorded_and_graded(world, tmp_path):
    store, settings = _clone(world, tmp_path)
    frame, up = _upcoming(world, store=store)
    slate = up[up["start_time_utc"] == up["start_time_utc"].min()]
    a, b, c = slate.iloc[0], slate.iloc[1], slate.iloc[2]
    # A whole number (it can be tied), far enough below his usual output for a clear "over".
    whole = float(max(np.floor(a["ref_line"]) - 4.0, 1.0))
    rows = [_line_row(world, a, whole), _line_row(world, b, 0.5), _line_row(world, c, float(c["ref_line"]) + 30.0)]
    path = _lines_csv(tmp_path, rows)
    assert import_prop_lines(store, LEAGUE, path, collected_at=NOW) == {"thresholds": 3, "players": 3}
    assert import_prop_lines(store, LEAGUE, path, collected_at=NOW)["thresholds"] == 0     # the same reading twice
    # A threshold read after the prediction time does not exist yet for that prediction.
    assert store.prop_lines_as_of(LEAGUE, "2027-11-19T00:00:00Z").empty

    preds = predict_league(store, LEAGUE, "player_prop:points", settings=settings, now=NOW)
    assert len(preds) == 3
    by_player = {p["display"]["player_id"]: p for p in preds}
    pa, pb, pc = by_player[a["player_id"]], by_player[b["player_id"]], by_player[c["player_id"]]
    assert pa["market"] == f"player_prop:points:{a['player_id']}" and pa["event_id"] == a["game_id"]
    assert pa["line"] == whole and set(pa["probs"]) == {"under", "over", "push"}
    assert abs(sum(pa["probs"].values()) - 1.0) < 1e-9 and pa["probs"]["push"] > 0.002
    assert pa["selection"] == "over" and pa["rec_status"] == "PROVISIONAL"       # PASS records are not graded
    # Recorded: the chance of winning.  Judged and rated: the chance given a decision.
    assert pa["prob"] == pytest.approx(pa["probs"]["over"])
    assert pa["confidence"]["prob_if_decided"] == pytest.approx(pa["prob"] / (1.0 - pa["probs"]["push"]))
    assert pa["selection_label"].startswith(a["player_name"]) and pa["selection_label"].endswith("points")
    assert pa["display"]["projection"]["q10"] <= pa["display"]["projection"]["q90"]
    assert pa["validation_status"] == "experimental"
    # The hit rates on file were measured against reference thresholds: no confidence label is given.
    assert pb["confidence"]["label"] == "Unrated" and "reference thresholds" in pb["confidence"]["basis"]
    assert pa["display"]["player_team"] == world["names"][a["side_a_id"]]
    # Far from the threshold the model is sure of the side - and still not FINAL: no injury report was checked.
    assert pb["selection"] == "over" and pb["prob"] > 0.9 and pb["rec_status"] == "PROVISIONAL"
    assert pc["selection"] == "under" and pc["prob"] > 0.9
    assert set(pb["probs"]) == {"under", "over"} == set(pc["probs"])              # no push off a whole number
    assert any("injury" in r for r in pb["reasons"])
    assert any("void when he does not play" in t for t in pb["factors"]["availability_assumptions"])
    ok, problems = store.verify_ledger()
    assert ok, problems
    # Unchanged inputs are not written twice; a new threshold supersedes the old prediction.
    again = predict_league(store, LEAGUE, "player_prop:points", settings=settings, now=NOW)
    assert all(p.get("unchanged") for p in again) and len(again) == 3
    later = "2027-11-20T01:00:00Z"
    import_prop_lines(store, LEAGUE, _lines_csv(tmp_path, [{**rows[1], "line": 1.5}], "newer.csv"), collected_at=later)
    moved = {p["display"]["player_id"]: p for p in
             predict_league(store, LEAGUE, "player_prop:points", settings=settings, now=later)}
    assert moved[b["player_id"]]["line"] == 1.5 and moved[b["player_id"]]["supersedes_id"] == pb["prediction_id"]

    # ---- the games are played: grade against the box score
    special = {a["player_id"]: (30.0, whole), b["player_id"]: (30.0, 20.0), c["player_id"]: (np.nan, np.nan)}
    for game in {a["game_id"], b["game_id"], c["game_id"]}:
        _play(store, world, game, special)
    grade_predictions(store, now="2027-12-30T00:00:00Z")
    grades = store.grades().set_index("prediction_id")["outcome"]
    assert grades[pa["prediction_id"]] == "push"                           # landed exactly on the whole number
    assert grades[moved[b["player_id"]]["prediction_id"]] == "correct"     # 20 points against 1.5
    assert grades[pc["prediction_id"]] == "void"                           # listed in a full box score, did not play
    # Results are grouped by the model's market, and compared with the probability given a decision.
    graded = graded_frame(store).set_index("prediction_id")
    assert set(graded["market"]) == {"player_prop:points"} and graded["ledger_market"].str.count(":").eq(2).all()
    assert graded.loc[pa["prediction_id"], "prob"] > graded.loc[pa["prediction_id"], "prob_recorded"]
    assert graded.loc[moved[b["player_id"]]["prediction_id"], "prob"] == moved[b["player_id"]]["prob"]


def test_a_whole_number_threshold_agrees_with_the_half_points_either_side(world):
    model = load_promoted(world["store"], LEAGUE, "player_prop:points")
    _, up = _upcoming(world)
    row = up.iloc[3].copy()
    whole = float(np.floor(row["ref_line"]))

    def at(line):
        r = row.copy()
        r["line"] = line
        return _outcome_probabilities(model, r, line, True)

    P, low, high, push = at(whole)
    above, below = at(whole + 0.5), at(whole - 0.5)
    assert P[1] == pytest.approx(above[0][1]) and P[0] == pytest.approx(below[0][0])
    assert push > 0 and P.sum() + push == pytest.approx(1.0)
    assert above[3] == 0.0 == below[3] and above[0].sum() == pytest.approx(1.0)
    assert (low <= P + 1e-12).all() and (P <= high + 1e-12).all()


def test_a_selection_about_a_player_who_left_the_frame_is_withdrawn(world, tmp_path, monkeypatch):
    store, settings = _clone(world, tmp_path)
    _, up = _upcoming(world, store=store)
    slate = up[up["start_time_utc"] == up["start_time_utc"].min()]
    gone, stays = slate.iloc[0], slate.iloc[1]
    import_prop_lines(store, LEAGUE, _lines_csv(tmp_path, [_line_row(world, gone, 0.5), _line_row(world, stays, 0.5)]),
                      collected_at=NOW)
    first = {p["display"]["player_id"]: p for p in
             predict_league(store, LEAGUE, "player_prop:points", settings=settings, now=NOW)}
    assert first[gone["player_id"]]["rec_status"] == "PROVISIONAL"

    plugin = world["plugin"]
    original = type(plugin).feature_frame

    def without_him(self, st, league, market, *, now=None):       # traded, or out of the modelled rotation
        frame = original(self, st, league, market, now=now)
        frame.df = frame.df[frame.df["player_id"] != gone["player_id"]].reset_index(drop=True)
        return frame

    monkeypatch.setattr(type(plugin), "feature_frame", without_him)
    later = "2027-11-20T01:00:00Z"
    second = predict_league(store, LEAGUE, "player_prop:points", settings=settings, now=later)
    withdrawn = [p for p in second if p.get("withdrawn")]
    assert len(withdrawn) == 1 and withdrawn[0]["rec_status"] == "PASS"
    assert withdrawn[0]["supersedes_id"] == first[gone["player_id"]]["prediction_id"]
    assert "withdrawn" in withdrawn[0]["reasons"][0]
    current = store.predictions(latest_only=True).set_index("market")["rec_status"]
    assert current[f"player_prop:points:{gone['player_id']}"] == "PASS"
    assert current[f"player_prop:points:{stays['player_id']}"] == "PROVISIONAL"
    third = predict_league(store, LEAGUE, "player_prop:points", settings=settings, now="2027-11-20T02:00:00Z")
    assert not [p for p in third if p.get("withdrawn")]                      # withdrawn once, not again
    assert store.verify_ledger()[0]


# ---------------------------------------------------------------- availability
def _statements(rows):
    return pd.DataFrame(rows, columns=["team_id", "player_name", "status"])


_REP = ("mates_out_min_rep", "mates_new_out_min_rep", "mates_out_share_rep", "mates_new_out_share_rep")


def _availability_case(world, reads="rep", report_read=True, value=37.0):
    """A row with report-based inputs (set to ``value``), a frame that says what the model
    reads (``rep``: the report-based absence inputs, ``prev``: the previous-game ones,
    ``none``: no absence input) and whether the team's report has been read for this game,
    the team, and its best other player."""
    frame, up = _upcoming(world)
    row = up.iloc[0].copy()
    for col in _REP:
        row[col] = value
    team = row["side_a_id"]
    frame.extra["model_features"] = ["min_ew"] + {"rep": ["mates_out_min_rep"], "prev": ["mates_out_min"],
                                                  "none": []}[reads]
    frame.extra["report_checked_teams"] = {team} if report_read else set()
    frame.extra["report_info_time"] = {team: NOW} if report_read else {}
    mates = [p for p in frame.extra["rotation"][team]["players"] if p["player_id"] != row["player_id"]]
    return frame, row, team, max(mates, key=lambda p: p["min"])


def _apply(frame, row, statements):
    return apply_prop_availability(row, frame, statements, AvailabilityEffect(row=row.copy()))


def _missed_last_game(frame, team, player):
    frame.extra["rotation"][team]["players"] = [{**p, "missed": 1} if p["player_id"] == player["player_id"] else p
                                                for p in frame.extra["rotation"][team]["players"]]


def test_the_players_own_status(world):
    frame, row, team, _ = _availability_case(world)
    out = _apply(frame, row, _statements([(team, row["player_name"], "out")]))
    assert out.blockers and "void" in out.blockers[0]
    doubt = _apply(frame, row, _statements([(team, row["player_name"], "questionable")]))
    assert not doubt.blockers and not doubt.worst_case_rows                  # cannot be put into a row
    assert any("conditional on" in text and "playing" in text for text in doubt.conditions)
    # Matched by id when the statement carries one, whatever the spelling.
    by_id = _statements([(team, "A Different Spelling", "out")]).assign(player_id=row["player_id"])
    assert _apply(frame, row, by_id).blockers
    # A name that matches nobody is said to match nobody - never dropped in silence.
    typo = _apply(frame, row, _statements([(team, "Nobody Atall", "out")]))
    assert not typo.blockers and any("matches nobody" in text for text in typo.unmatched)


def test_an_old_statement_is_out_of_date_unless_the_report_just_read_still_carries_it(world):
    """Listed out before his last game and has played since: out of date.  But a source may
    date an item before the game it is about - if the team's report read for this game,
    newer than his last game, still lists him, he is listed."""
    old = "2020-01-01T00:00:00Z"
    frame, row, team, star = _availability_case(world, report_read=False)
    stale = _statements([(team, row["player_name"], "out")]).assign(published_at_utc=old, collected_at_utc=old)
    gone = _apply(frame, row, stale)
    assert not gone.blockers and any("out of date" in text for text in gone.unmatched)
    frame, row, team, star = _availability_case(world, report_read=True)
    current = _statements([(team, row["player_name"], "out")]).assign(published_at_utc=old, collected_at_utc=NOW)
    assert _apply(frame, row, current).blockers
    mate = _statements([(team, star["name"], "out")]).assign(published_at_utc=old, collected_at_utc=NOW)
    assert _apply(frame, row, mate).row["mates_out_min_rep"] == pytest.approx(star["min"])


def test_team_mates_are_applied_only_when_the_model_reads_them_and_the_report_was_read(world):
    frame, row, team, star = _availability_case(world)
    listed = _apply(frame, row, _statements([(team, star["name"], "out")]))
    assert listed.row["mates_out_min_rep"] == pytest.approx(star["min"]) and listed.row["mates_out_share_rep"] > 0
    assert any(star["name"] in text for text in listed.applied) and not listed.conditions
    maybe = _apply(frame, row, _statements([(team, star["name"], "questionable")]))
    assert maybe.row["mates_out_min_rep"] == 0.0                                        # assumed to play ...
    assert maybe.worst_case_rows["a"]["mates_out_min_rep"] == pytest.approx(star["min"])  # ... tested as if he sits
    assert maybe.uncertain_a
    # Report read, nobody listed: that is information too - nobody is expected to be absent.
    nobody = _apply(frame, row, _statements([]))
    assert nobody.row["mates_out_min_rep"] == 0.0 and any("lists no rotation team-mate" in t for t in nobody.applied)
    # His own doubt and a team-mate's together: nothing is tested as a worst case, both are stated.
    both = _apply(frame, row, _statements([(team, row["player_name"], "questionable"), (team, star["name"], "questionable")]))
    assert not both.worst_case_rows and len(both.conditions) == 2


def test_a_team_mate_is_matched_by_id_then_name_then_joined_initials(world):
    frame, row, team, star = _availability_case(world)
    by_id = _statements([(team, "Another Spelling Entirely", "out")]).assign(player_id=star["player_id"])
    assert _apply(frame, row, by_id).row["mates_out_min_rep"] == pytest.approx(star["min"])
    # "O.G. Anunoby" on the report, "OG Anunoby" in the box score.
    frame.extra["rotation"][team]["players"] = [
        {**p, "key": "og anunoby", "name": "OG Anunoby"} if p["player_id"] == star["player_id"] else p
        for p in frame.extra["rotation"][team]["players"]]
    dotted = _apply(frame, row, _statements([(team, "O.G. Anunoby", "out")]))
    assert dotted.row["mates_out_min_rep"] == pytest.approx(star["min"])
    # Another first name, the same surname: it may be him - a condition, never an application.
    nick = _apply(frame, row, _statements([(team, "Ogugua Anunoby", "out")]))
    assert nick.row["mates_out_min_rep"] == 0.0
    assert any("may be OG Anunoby" in text for text in nick.conditions)
    # Someone from the end of the bench is recognised as such and changes nothing.
    frame.extra["rotation"][team]["others"] = [{"player_id": "77", "key": "deep bench"}]
    bench = _apply(frame, row, _statements([(team, "Deep Bench", "out")]))
    assert not bench.conditions and any("outside the team's recent rotation" in text for text in bench.unmatched)


def test_a_stray_statement_is_not_a_report(world):
    """Without a report on file for this game, single statements change no input: one
    'probable' line must not erase the absences the row already carries."""
    frame, row, team, star = _availability_case(world, report_read=False)
    other = next(p for p in frame.extra["rotation"][team]["players"]
                 if p["player_id"] not in (row["player_id"], star["player_id"]))
    probable = _apply(frame, row, _statements([(team, other["name"], "probable")]))
    assert probable.row["mates_out_min_rep"] == 37.0 and not probable.applied and not probable.conditions
    listed = _apply(frame, row, _statements([(team, star["name"], "out")]))
    assert listed.row["mates_out_min_rep"] == 37.0 and not listed.worst_case_rows and not listed.applied
    assert any("computed as if" in t and "plays, but he is listed out" in t for t in listed.conditions)


def test_conditions_say_which_way_the_inputs_differ_from_the_listing(world):
    """A model that takes absences from the previous game cannot use the report.  Each
    difference between what it assumed and what is listed is stated with its direction."""
    frame, row, team, star = _availability_case(world, reads="prev")
    assert star["missed"] == 0
    out = _apply(frame, row, _statements([(team, star["name"], "out")]))
    assert out.row["mates_out_min_rep"] == 37.0 and not out.applied and not out.worst_case_rows
    assert any("computed as if" in t and "plays, but he is listed out" in t for t in out.conditions)
    maybe = _apply(frame, row, _statements([(team, star["name"], "questionable")]))
    assert any("assumes" in t and "plays; the effect of him sitting" in t for t in maybe.conditions)

    _missed_last_game(frame, team, star)                 # now the inputs count him absent
    counted = _apply(frame, row, _statements([(team, star["name"], "out")]))
    assert not counted.conditions and any("already counted as absent" in text for text in counted.applied)
    doubt = _apply(frame, row, _statements([(team, star["name"], "questionable")]))
    assert any("computed as if" in t and "sits" in t and "effect of him playing" in t for t in doubt.conditions)
    back = _apply(frame, row, _statements([(team, star["name"], "available")]))
    assert any("sits" in t and "listed available" in t for t in back.conditions)
    unlisted = _apply(frame, row, _statements([]))        # the report was read and does not list him
    assert any("not on the team's report" in t for t in unlisted.conditions)


def test_a_model_without_absence_inputs_never_claims_to_have_counted_one(world):
    frame, row, team, star = _availability_case(world, reads="none")
    _missed_last_game(frame, team, star)
    listed = _apply(frame, row, _statements([(team, star["name"], "out")]))
    assert not listed.applied
    assert any("no input for team-mates' absences" in text for text in listed.conditions)


def test_only_a_report_on_file_for_this_game_counts_as_read(world, tmp_path, monkeypatch):
    """A check from another week, or one tied to another game, is not this game's report."""
    store, settings = _clone(world, tmp_path)
    _, up = _upcoming(world, store=store)
    row = up[up["start_time_utc"] == up["start_time_utc"].min()].iloc[0]
    team, start = row["side_a_id"], row["start_time_utc"]
    import_prop_lines(store, LEAGUE, _lines_csv(tmp_path, [_line_row(world, row, 0.5)]), collected_at=NOW)
    plugin = world["plugin"]
    original = type(plugin).apply_availability
    seen = []

    def spy(self, league, market, r, frame, availability):
        seen.append(set(frame.extra["report_checked_teams"]))
        return original(self, league, market, r, frame, availability)

    monkeypatch.setattr(type(plugin), "apply_availability", spy)
    other_game = up[up["game_id"] != row["game_id"]].iloc[0]["game_id"]
    store.log_check(LEAGUE, "injury_report", "nothing_listed", team_id=team,
                    checked_at=(start - pd.Timedelta(days=12)).isoformat())
    store.log_check(LEAGUE, "injury_report", "nothing_listed", team_id=team, event_id=other_game,
                    checked_at=(start - pd.Timedelta(hours=2)).isoformat())
    near = (start - pd.Timedelta(hours=1)).isoformat()
    predict_league(store, LEAGUE, "player_prop:points", settings=settings, now=near, write=False)
    assert seen and team not in seen[-1]
    store.log_check(LEAGUE, "injury_report", "nothing_listed", team_id=team,
                    checked_at=(start - pd.Timedelta(hours=2)).isoformat())
    predict_league(store, LEAGUE, "player_prop:points", settings=settings, now=near, write=False)
    assert team in seen[-1]


def test_a_listed_team_mate_makes_a_complete_selection_conditional_in_plain_words(world, tmp_path):
    """Every check complete: FINAL.  Then a rotation team-mate is listed questionable, and
    this model cannot put a number on it: CONDITIONAL, with the condition recorded as written."""
    store, settings = _clone(world, tmp_path)
    frame, up = _upcoming(world, store=store)
    row = up[up["start_time_utc"] == up["start_time_utc"].min()].iloc[0]
    team, start = row["side_a_id"], row["start_time_utc"]
    import_prop_lines(store, LEAGUE, _lines_csv(tmp_path, [_line_row(world, row, 0.5)]), collected_at=NOW)
    for side in (row["side_a_id"], row["side_b_id"]):
        store.log_check(LEAGUE, "injury_report", "reported", team_id=side,
                        checked_at=(start - pd.Timedelta(hours=2)).isoformat())
    near = (start - pd.Timedelta(hours=1)).isoformat()
    clear = predict_league(store, LEAGUE, "player_prop:points", settings=settings, now=near, write=False)[0]
    assert clear["rec_status"] == "FINAL", clear["reasons"]
    mate = next(p for p in frame.extra["rotation"][team]["players"] if p["player_id"] != row["player_id"]
                and p["missed"] == 0)
    store.add_availability(pd.DataFrame([{"league": LEAGUE, "team_id": team, "player_name": mate["name"],
                                          "status": "questionable",
                                          "collected_at_utc": (start - pd.Timedelta(hours=2)).isoformat()}]))
    doubt = predict_league(store, LEAGUE, "player_prop:points", settings=settings, now=near, write=False)[0]
    assert doubt["rec_status"] == "CONDITIONAL"
    assert any(mate["name"] in r and "effect of him sitting is not quantified" in r for r in doubt["reasons"])
    assert not any("holds only if" in r for r in doubt["reasons"])


def test_a_player_listed_out_gets_pass_not_a_selection(world, tmp_path):
    store, settings = _clone(world, tmp_path)
    _, up = _upcoming(world, store=store)
    row = up[up["start_time_utc"] == up["start_time_utc"].min()].iloc[-1]
    import_prop_lines(store, LEAGUE, _lines_csv(tmp_path, [_line_row(world, row, 0.5)]), collected_at=NOW)
    store.add_availability(pd.DataFrame([{
        "league": LEAGUE, "team_id": row["side_a_id"], "player_name": row["player_name"], "status": "out",
        "collected_at_utc": NOW}]))
    preds = predict_league(store, LEAGUE, "player_prop:points", settings=settings, now=NOW, write=False)
    mine = next(p for p in preds if p["display"]["player_id"] == row["player_id"])
    assert mine["rec_status"] == "PASS" and any("listed out" in r for r in mine["reasons"])


def test_projection_table_lists_players_without_recording_anything(world):
    from sportspred.cli.cmd_props import projection_table

    before = len(world["store"].predictions())
    table = projection_table(world["store"], LEAGUE, "points", settings=world["settings"], now=NOW)
    assert len(table) and {"player", "expected minutes", "projected points", "10th pct", "median", "90th pct"} <= set(table.columns)
    assert (table["10th pct"] <= table["median"]).all() and (table["median"] <= table["90th pct"]).all()
    assert table["expected minutes"].between(10, 48).all()
    one = projection_table(world["store"], LEAGUE, "points", settings=world["settings"], now=NOW, team="Prop Team A")
    assert len(one) and set(one["team"]) == {"Prop Team A"}
    assert len(world["store"].predictions()) == before


def test_a_threshold_dated_in_the_future_is_refused_when_the_clock_rule_is_on(world, tmp_path, monkeypatch):
    """This league's games are dated 2027, so with the clock rule in force (as it
    is outside the tests) a reading stamped then cannot have happened yet."""
    _, up = _upcoming(world)
    path = _lines_csv(tmp_path, [_line_row(world, up.iloc[0], 9.5, collected_at=NOW)], "future.csv")
    monkeypatch.setattr(Store, "wall_clock_tolerance_seconds", 300.0)
    with pytest.raises(CsvImportError, match="in the future"):
        import_prop_lines(world["store"], LEAGUE, path)
    assert world["store"].prop_lines_as_of(LEAGUE, FAR).empty


def test_two_team_mates_with_one_name_cannot_be_told_apart(world, tmp_path, monkeypatch):
    _, up = _upcoming(world)
    row = up.iloc[0]
    plugin = world["plugin"]
    pool = plugin.prop_players(world["store"], LEAGUE, now=NOW)
    twin = pool[(pool["event_id"] == row["game_id"]) & (pool["player_id"] == row["player_id"])].copy()
    twin["player_id"] = "999999"
    monkeypatch.setattr(type(plugin), "prop_players", lambda self, store, league, now=None: pd.concat([pool, twin]))
    path = _lines_csv(tmp_path, [_line_row(world, row, 9.5)], "twin.csv")
    with pytest.raises(CsvImportError, match="more than one"):
        import_prop_lines(world["store"], LEAGUE, path, collected_at=NOW)
    assert world["store"].prop_lines_as_of(LEAGUE, FAR).empty                 # the shared store is never written


def test_reports_published_within_the_hour_split_the_cache(world, tmp_path):
    """Most report versions come out at half past.  A frame built before one must not be
    served to a time after it, nor the other way round."""
    from sportspred.sports.basketball import props as P

    store, _ = _clone(world, tmp_path)
    _, up = _upcoming(world, store=store)
    row = up.iloc[0]
    team = row["side_a_id"]
    mate = next(p for p in build_prop_frame(store, LEAGUE, "points", now=NOW).extra["rotation"][team]["players"]
                if p["player_id"] != row["player_id"] and p["min"] >= 20)
    published, long_ago = "2027-11-20T00:30:00Z", "2027-11-01T00:00:00Z"
    earlier_game = build_prop_frame(store, LEAGUE, "points", now=NOW).completed.iloc[-1]["game_id"]
    store.write_table(LEAGUE, "pregame_report", pd.DataFrame([{
        "event_id": row["game_id"], "team_id": team, "player_name": mate["name"], "player_key": mate["key"],
        "status": "out", "published_at_utc": published, "cutoff_utc": published}]), mode="replace")
    store.write_table(LEAGUE, "pregame_report_games", pd.DataFrame([      # an older report exists at both times
        {"event_id": earlier_game, "published_at_utc": long_ago},
        {"event_id": row["game_id"], "published_at_utc": published}]), mode="replace")
    P._CACHE.clear()
    after = build_prop_frame(store, LEAGUE, "points", now="2027-11-20T00:50:00Z").df.set_index("event_id")
    before = build_prop_frame(store, LEAGUE, "points", now="2027-11-20T00:10:00Z").df.set_index("event_id")
    assert after.loc[row["event_id"], "mates_out_min_rep"] >= mate["min"] - 1e-9       # the report is in force
    assert before.loc[row["event_id"], "mates_out_min_rep"] == before.loc[row["event_id"], "mates_out_min"]
    assert before.loc[row["event_id"], "report_on_file"] == 0.0 and after.loc[row["event_id"], "report_on_file"] == 1.0


# ----------------------------------------------------- edge cases of the builder
def test_every_game_to_come_reports_the_latest_missing_box_score(world, tmp_path):
    """Scheduled games in between must not push a missing box score out of sight, and half
    a box score is a missing one for both teams."""
    store, _ = _clone(world, tmp_path)
    frame = build_prop_frame(store, LEAGUE, "points", now=NOW)
    last_game = frame.completed.sort_values("start_time_utc").iloc[-1]
    team, opponent = last_game["side_a_id"], last_game["side_b_id"]
    box = store.read_table(LEAGUE, "player_box")
    half = box[~((box["event_id"] == last_game["game_id"]) & (box["team_id"] == team))]      # one side only
    store.write_table(LEAGUE, "player_box", half, mode="replace")
    up = build_prop_frame(store, LEAGUE, "points", now=NOW).upcoming
    for side in (team, opponent):
        mine = up[(up["side_a_id"] == side) | (up["side_b_id"] == side)]
        assert mine["game_id"].nunique() >= 3 and (mine["n_box_missing"] >= 1).all(), side


def test_a_played_row_without_its_statistics_is_not_an_outcome(world, tmp_path):
    store, _ = _clone(world, tmp_path)
    frame = build_prop_frame(store, LEAGUE, "points", now=NOW)
    target = frame.completed.iloc[len(frame.completed) // 2]
    assert target["y"] in (0.0, 1.0)
    box = store.read_table(LEAGUE, "player_box")
    hit = (box["event_id"] == target["game_id"]) & (box["player_id"] == target["player_id"])
    box.loc[hit, "pts"] = np.nan
    store.write_table(LEAGUE, "player_box", box, mode="replace")
    again = build_prop_frame(store, LEAGUE, "points", now=NOW)
    assert target["event_id"] not in set(again.df["event_id"])               # not an "under", not a zero
    later = again.completed[(again.completed["player_id"] == target["player_id"])
                            & (again.completed["start_time_utc"] > target["start_time_utc"])
                            & (again.completed["block"] == target["block"])].iloc[0]
    played = box[(box["player_id"] == target["player_id"]) & box["minutes"].notna() & box["pts"].notna()]
    ev = store.events(LEAGUE, with_results=False)[["event_id", "start_time_utc", "season"]]
    played = played.merge(ev, on="event_id")
    before = played[(played["start_time_utc"] < later["start_time_utc"]) & (played["season"] == str(later["block"]))]
    assert later["avg_season"] == pytest.approx(before["pts"].mean())


def test_opening_night_is_served_from_last_seasons_rotation(world):
    """A team's first game of a season is in the history; it must also be predictable."""
    store = world["store"]
    full = build_prop_frame(store, LEAGUE, "points", now=FAR).df
    opener = full[full["block"] == 2024]["start_time_utc"].min()
    asof = build_prop_frame(store, LEAGUE, "points", now=opener.isoformat())
    first = asof.upcoming[asof.upcoming["start_time_utc"] == opener]
    assert len(first) >= 20
    result = asof_invariance(_Builder(), store, LEAGUE, "player_prop:points", cutoffs=[opener])
    assert result["n_events"].min() >= 20 and result["max_abs_diff"].max() == 0.0


def test_a_schedule_change_is_not_served_from_the_cache(world, tmp_path):
    store, _ = _clone(world, tmp_path)
    up = build_prop_frame(store, LEAGUE, "points", now=NOW).upcoming
    starts = sorted(up["start_time_utc"].unique())
    game = up[up["start_time_utc"] == starts[1]].iloc[0]
    events = store.events(event_ids=[game["game_id"]], with_results=False)
    moved = events.assign(start_time_utc=starts[1] + pd.Timedelta(hours=5), provider="synthetic")
    store.upsert_events(moved[["league", "season", "start_time_utc", "status", "provider", "provider_event_id",
                               "home_id", "away_id", "neutral_site"]], sport="basketball")
    again = build_prop_frame(store, LEAGUE, "points", now=NOW).upcoming
    assert again.loc[again["game_id"] == game["game_id"], "start_time_utc"].iloc[0] == starts[1] + pd.Timedelta(hours=5)
