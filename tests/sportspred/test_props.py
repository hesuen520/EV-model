"""Player-statistic markets: the count distribution, point-in-time features,
threshold import, the prediction path into the ledger, and grading."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from sportspred.adapters.csv_import import CsvImportError, import_prop_lines
from sportspred.config import Settings
from sportspred.coverage import coverage_status
from sportspred.ledger.grading import grade_predictions
from sportspred.models.counts import (ColumnMean, CountCandidate, ProductMean, TwoStageMean, count_probabilities,
                                      fit_dispersion, nb_alpha)
from sportspred.pipeline import train_league, validate_league
from sportspred.predict import predict_league
from sportspred.sports import get_league
from sportspred.sports.base import AvailabilityEffect
from sportspred.sports.basketball.props import (apply_prop_availability, build_prop_frame, prop_market,
                                                split_prop_market, threshold_sweep)
from sportspred.storage import Store
from sportspred.storage.store import StoreError
from sportspred.validation.backtest import _blind
from sportspred.validation.leakage import asof_invariance

LEAGUE = "nba"       # a temporary store; rows are not flagged synthetic so the default reads see them
N_TEAMS, N_PLAYERS = 6, 8
FIRST_SEASON, N_SEASONS = 2015, 13
NOW = "2027-11-20T00:00:00Z"            # after the last played round, before the scheduled one


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
def _build_league(store: Store) -> dict:
    """Six teams of eight players over thirteen seasons: persistent player
    minutes and per-minute rates, opponents that allow more or less, random
    absences.  The last season has played rounds and then a scheduled one."""
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
                day += pd.Timedelta(days=2)
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
    return {"teams": ids, "players": players, "names": dict(zip(ids, teams["name"])), "eid": eid}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    settings = Settings()
    settings.paths.data_dir = str(tmp_path_factory.mktemp("props"))
    settings.validation.n_boot, settings.validation.n_replicas = 30, 3
    settings.selection.horizon_hours = 240.0
    Store.wall_clock_tolerance_seconds = None
    store = Store(settings.paths.store)
    info = _build_league(store)
    plugin, _ = get_league(LEAGUE)
    report, _ = validate_league(store, LEAGUE, "player_prop:points", settings=settings, now=NOW)
    train_league(store, LEAGUE, "player_prop:points", settings=settings, now=NOW)
    return {"settings": settings, "store": store, "plugin": plugin, "report": report, **info}


def _upcoming(world, stat="points"):
    frame = build_prop_frame(world["store"], LEAGUE, stat, now=NOW)
    return frame, frame.upcoming.sort_values(["start_time_utc", "event_id"])


def _lines_csv(tmp_path, rows, name="lines.csv"):
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# -------------------------------------------------------------------- features
def test_features_do_not_change_when_the_future_is_hidden(world):
    class Builder:
        def feature_frame(self, store, league, market, *, now=None):
            return build_prop_frame(store, league, "points", now=now)

    result = asof_invariance(Builder(), world["store"], LEAGUE, "player_prop:points", n_cutoffs=3)
    assert len(result) and result["n_events"].min() >= 10
    assert result["max_abs_diff"].max() == 0.0


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
    box = store.read_table(LEAGUE, "player_box")
    changed = box.copy()
    hit = changed["event_id"] == target["game_id"]
    changed.loc[hit, ["pts", "minutes"]] = changed.loc[hit, ["pts", "minutes"]] * 2
    other = Store(tmp_path / "copy")
    _build_league(other)
    ids = dict(zip(store.events(LEAGUE, with_results=False)["provider_event_id"],
                   store.events(LEAGUE, with_results=False)["event_id"]))
    assert set(ids.values()) == set(other.events(LEAGUE, with_results=False)["event_id"])     # same stable ids
    other.write_table(LEAGUE, "player_box", changed, mode="replace")
    again = build_prop_frame(other, LEAGUE, "points", now=NOW).df.set_index("event_id")
    cols = [c for c in frame.feature_cols]
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
    assert rows[0.0]["push_rate"] == 0.0 and rows[0.5]["push_rate"] > 0.0
    assert abs(rows[0.5]["push_rate"] - rows[0.5]["mean_predicted_push"]) < 0.03
    assert rows[-6.0]["over_rate"] > rows[0.0]["over_rate"] > rows[6.0]["over_rate"]
    for r in rows.values():
        assert abs(r["over_rate"] - r["mean_predicted_over"]) < 0.06


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
    ])
    with pytest.raises(CsvImportError) as err:
        import_prop_lines(world["store"], LEAGUE, bad, collected_at=NOW)
    text = str(err.value)
    assert "whole or half number" in text and "has not played for" in text and "'steals' is not one of" in text
    assert "unknown participant" in text and "outside the modelled rotation" in text
    assert world["store"].prop_lines_as_of(LEAGUE, "2099-01-01T00:00:00Z").empty        # nothing was stored
    with pytest.raises(StoreError):
        world["store"].add_prop_lines([{"league": LEAGUE, "event_id": first["game_id"], "player_id": "1",
                                        "player_name": "x", "stat": "points", "line": 3.3,
                                        "collected_at_utc": NOW}])


def test_thresholds_are_priced_recorded_and_graded(world, tmp_path):
    store, settings = world["store"], world["settings"]
    frame, up = _upcoming(world)
    slate = up[up["start_time_utc"] == up["start_time_utc"].min()]
    a, b, c = slate.iloc[0], slate.iloc[1], slate.iloc[2]
    # A whole number (it can be tied), far enough below his usual output for a clear "over".
    whole = float(max(np.floor(a["ref_line"]) - 4.0, 1.0))
    rows = [
        {"team": world["names"][a["side_a_id"]], "player": a["player_name"], "stat": "points", "line": whole},
        {"team": world["names"][b["side_a_id"]], "player": b["player_name"], "stat": "points", "line": 0.5},
        {"team": world["names"][c["side_a_id"]], "player": c["player_name"], "stat": "points",
         "line": float(c["ref_line"]) + 30.0},
    ]
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
    assert pa["selection_label"].startswith(a["player_name"]) and pa["selection_label"].endswith("points")
    assert pa["display"]["projection"]["q10"] <= pa["display"]["projection"]["q90"]
    assert pa["validation_status"] == "experimental"
    # Far from the threshold the model is sure of the side - and still not FINAL: no injury report was checked.
    assert pb["selection"] == "over" and pb["prob"] > 0.9 and pb["rec_status"] == "PROVISIONAL"
    assert pc["selection"] == "under" and pc["prob"] > 0.9 and "push" not in pc["probs"]
    assert any("injury" in r for r in pb["reasons"])
    assert any("void when he does not play" in t for t in pb["factors"]["availability_assumptions"])
    ok, problems = store.verify_ledger()
    assert ok, problems
    # Unchanged inputs are not written twice; a new threshold supersedes the old prediction.
    again = predict_league(store, LEAGUE, "player_prop:points", settings=settings, now=NOW)
    assert all(p.get("unchanged") for p in again)
    later = "2027-11-20T01:00:00Z"
    newer = _lines_csv(tmp_path, [{**rows[1], "line": 1.5}], "newer.csv")
    import_prop_lines(store, LEAGUE, newer, collected_at=later)
    moved = {p["display"]["player_id"]: p for p in
             predict_league(store, LEAGUE, "player_prop:points", settings=settings, now=later)}
    assert moved[b["player_id"]]["line"] == 1.5 and moved[b["player_id"]]["supersedes_id"] == pb["prediction_id"]

    # ---- the games are played: grade against the box score
    box = store.read_table(LEAGUE, "player_box")
    template = box[box["minutes"].notna()].iloc[0].to_dict()
    new_rows, results = [], []
    for pick, pts, minutes in ((a, whole, 30.0), (b, 20.0, 30.0), (c, 5.0, np.nan)):
        row = {**template, "event_id": pick["game_id"], "team_id": pick["side_a_id"], "player_id": pick["player_id"],
               "player_name": pick["player_name"], "minutes": minutes, "pts": pts if minutes == minutes else np.nan}
        new_rows.append(row)
        ev = store.events(event_ids=[pick["game_id"]], with_results=False).iloc[0]
        results.append({"event_id": pick["game_id"], "home_score": 101, "away_score": 99})
    store.write_table(LEAGUE, "player_box", pd.concat([box, pd.DataFrame(new_rows)], ignore_index=True))
    store.upsert_results(pd.DataFrame(results).drop_duplicates("event_id"))
    grade_predictions(store, now="2027-12-30T00:00:00Z")
    grades = store.grades().set_index("prediction_id")["outcome"]
    assert grades[pa["prediction_id"]] == "push"                           # landed exactly on the whole number
    assert grades[moved[b["player_id"]]["prediction_id"]] == "correct"     # 20 points against 1.5
    assert grades[pc["prediction_id"]] == "void"                           # listed but did not play


# ---------------------------------------------------------------- availability
def _statements(rows):
    return pd.DataFrame(rows, columns=["team_id", "player_name", "status"])


def test_availability_of_the_player_and_his_team_mates(world):
    frame, up = _upcoming(world)
    row = up.iloc[0]
    team = row["side_a_id"]
    mates = [p for p in frame.extra["rotation"][team]["players"] if p["player_id"] != row["player_id"]]
    star = max(mates, key=lambda p: p["min"])

    out = apply_prop_availability(row, frame, _statements([(team, row["player_name"], "out")]),
                                  AvailabilityEffect(row=row.copy()))
    assert out.blockers and "void" in out.blockers[0]

    doubt = apply_prop_availability(row, frame, _statements([(team, row["player_name"], "questionable")]),
                                    AvailabilityEffect(row=row.copy()))
    assert not doubt.blockers and doubt.uncertain_a and not doubt.worst_case_rows      # cannot be put into a row

    assert "mates_out_min_rep" not in row.index            # this league has no report history: nothing to replace
    listed = apply_prop_availability(row, frame, _statements([(team, star["name"], "out")]),
                                     AvailabilityEffect(row=row.copy()))
    assert any("not applied" in text for text in listed.unmatched)

    # With report-based inputs in the frame, a listed team-mate replaces them.
    rep_row = row.copy()
    for col in ("mates_out_min_rep", "mates_new_out_min_rep", "mates_out_share_rep", "mates_new_out_share_rep"):
        rep_row[col] = 0.0
    applied = apply_prop_availability(rep_row, frame, _statements([(team, star["name"], "out")]),
                                      AvailabilityEffect(row=rep_row.copy()))
    assert applied.row["mates_out_min_rep"] == pytest.approx(star["min"])
    assert applied.row["mates_out_share_rep"] > 0 and any(star["name"] in t for t in applied.applied)
    maybe = apply_prop_availability(rep_row, frame, _statements([(team, star["name"], "questionable")]),
                                    AvailabilityEffect(row=rep_row.copy()))
    assert maybe.row["mates_out_min_rep"] == 0.0                              # assumed to play ...
    assert maybe.worst_case_rows["a"]["mates_out_min_rep"] == pytest.approx(star["min"])   # ... tested as if he sits
    assert maybe.uncertain_a


def test_a_player_listed_out_gets_pass_not_a_selection(world, tmp_path):
    store, settings = world["store"], world["settings"]
    _, up = _upcoming(world, "rebounds")
    validate_league(store, LEAGUE, "player_prop:rebounds", settings=settings, now=NOW)
    train_league(store, LEAGUE, "player_prop:rebounds", settings=settings, now=NOW)
    row = up[up["start_time_utc"] == up["start_time_utc"].min()].iloc[-1]
    path = _lines_csv(tmp_path, [{"team": world["names"][row["side_a_id"]], "player": row["player_name"],
                                  "stat": "rebounds", "line": 0.5}], "reb.csv")
    import_prop_lines(store, LEAGUE, path, collected_at=NOW)
    store.add_availability(pd.DataFrame([{
        "league": LEAGUE, "team_id": row["side_a_id"], "player_name": row["player_name"], "status": "out",
        "collected_at_utc": NOW}]))
    preds = predict_league(store, LEAGUE, "player_prop:rebounds", settings=settings, now=NOW, write=False)
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
