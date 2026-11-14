"""Formula 1 finishing markets: the field model, point-in-time inputs, the grid
import, the prediction path and grading - on a synthetic series."""

from __future__ import annotations

import copy
import shutil

import numpy as np
import pandas as pd
import pytest

from sportspred.adapters.csv_import import CsvImportError
from sportspred.config import Settings
from sportspred.coverage import coverage_status
from sportspred.ledger.grading import grade_predictions
from sportspred.models.field import FieldRankCandidate, rank_log_likelihood, simulate_field
from sportspred.pipeline import train_league, validate_league
from sportspred.predict import predict_league
from sportspred.sports import get_league
from sportspred.sports.motorsport.adapter import CONSTRUCTOR_SCOPE
from sportspred.sports.motorsport.features import build_finish_frame, finish_market, split_finish_market
from sportspred.sports.motorsport.grid import import_grid
from sportspred.storage import Store
from sportspred.validation.leakage import asof_invariance

LEAGUE = "f1"               # a temporary store; rows are not flagged synthetic so the default reads see them
N_TEAMS, FIRST, LAST = 8, 2000, 2026
FAR = "2099-01-01T00:00:00Z"
NOW = "2026-09-20T00:00:00Z"            # after the tenth race of the last season, before the eleventh


# ------------------------------------------------------------------ the field model
def test_rank_likelihood_matches_a_hand_calculation():
    s = np.array([[1.0, 0.0, -1.0]])
    by_hand = (1.0 - np.log(np.exp(1.0) + 1.0 + np.exp(-1.0))) + (0.0 - np.log(1.0 + np.exp(-1.0)))
    assert rank_log_likelihood(s, np.ones((1, 3), dtype=bool)) == pytest.approx(by_hand)
    # Padding after the finishers changes nothing.
    padded = np.array([[1.0, 0.0, -1.0, 9.0]])
    assert rank_log_likelihood(padded, np.array([[True, True, True, False]])) == pytest.approx(by_hand)


def test_simulated_race_is_a_proper_finish_distribution():
    strength = np.array([2.0, 1.0, 0.0, 0.0, -1.0])
    retire = np.array([0.1, 0.1, 0.3, 0.0, 0.2])
    pos = simulate_field(strength, retire, n_sims=20000, seed=7)
    done = pos > 0
    assert np.allclose(1.0 - done.mean(axis=0), retire, atol=0.01)            # retirements as asked
    wins = (pos == 1).mean(axis=0)
    assert wins.sum() == pytest.approx((done.any(axis=1)).mean(), abs=1e-9)    # one winner whenever anyone finishes
    assert wins[0] > wins[1] > wins[2] and wins[3] > wins[2]                   # stronger and more reliable wins more
    assert ((pos == 2).sum(axis=1) <= 1).all()                                 # one car per place
    again = simulate_field(strength, retire, n_sims=20000, seed=7)
    assert (pos == again).all()                                                # the same seed, the same race


def test_market_keys():
    assert finish_market("win") == "finish:win" and finish_market("podium", "p_1") == "finish:podium:p_1"
    assert split_finish_market("finish:points:p_9") == ("points", "p_9")
    assert split_finish_market("finish:lap") == (None, None) and split_finish_market("winner") == (None, None)


# ------------------------------------------------------------------ a synthetic series
def _build_series(store: Store) -> dict:
    """Eight two-car teams over twenty-seven seasons of twelve races.  Car pace
    changes each season, drivers keep theirs, qualifying is pace plus noise, the
    race is the grid plus noise, and cars retire at a rate that falls over time.
    The last season has ten races run and two to come."""
    rng = np.random.default_rng(21)
    teams = pd.DataFrame({"name": [f"Team {chr(65 + i)}" for i in range(N_TEAMS)]})
    team_ids = list(store.resolve_participants(teams, sport="motorsport", scope=CONSTRUCTOR_SCOPE, kind="team"))
    drivers = pd.DataFrame({"name": [f"Driver {chr(65 + i // 2)}{i % 2 + 1}" for i in range(2 * N_TEAMS)]})
    driver_ids = list(store.resolve_participants(drivers, sport="motorsport", scope=LEAGUE, kind="person"))
    skill = rng.normal(0.0, 0.6, 2 * N_TEAMS)
    events, entries = [], []
    for year in range(FIRST, LAST + 1):
        car = rng.normal(0.0, 1.0, N_TEAMS)
        for rnd in range(12):
            start = pd.Timestamp(f"{year}-03-01T13:00:00Z") + pd.Timedelta(days=21 * rnd)
            upcoming = year == LAST and rnd >= 10
            gid = f"{year}-{rnd + 1}"
            events.append({"league": LEAGUE, "season": str(year), "start_time_utc": start,
                           "status": "scheduled" if upcoming else "final", "provider": "synthetic",
                           "provider_event_id": gid, "participants": list(driver_ids), "neutral_site": 1,
                           "venue_name": f"Circuit {rnd + 1}",
                           "format": {"name": f"Grand Prix {rnd + 1} {year}", "circuit": f"c{rnd + 1}",
                                      "circuit_type": "STREET" if rnd % 4 == 0 else "RACE"}})
            if upcoming:
                continue
            pace = np.array([car[i // 2] + skill[i] for i in range(2 * N_TEAMS)])
            quali = pace + rng.normal(0.0, 0.5, 2 * N_TEAMS)
            grid = np.empty(2 * N_TEAMS)
            grid[np.argsort(-quali)] = np.arange(1, 2 * N_TEAMS + 1)
            race = pace - 0.08 * grid + rng.normal(0.0, 0.8, 2 * N_TEAMS)
            retired = rng.random(2 * N_TEAMS) < (0.25 - 0.005 * (year - FIRST))
            no_start = (rng.random(2 * N_TEAMS) < 0.01) & ~retired
            order = np.argsort(-np.where(retired | no_start, -np.inf, race), kind="stable")
            place = np.empty(2 * N_TEAMS)
            place[order] = np.arange(1, 2 * N_TEAMS + 1)
            for i, d in enumerate(driver_ids):
                classified = not (retired[i] or no_start[i])
                entries.append({"gid": gid, "driver_id": d, "constructor_id": team_ids[i // 2], "driver": d,
                                "constructor": team_ids[i // 2], "order": float(place[i]),
                                "position": float(place[i]) if classified else np.nan,
                                "position_text": str(int(place[i])) if classified else ("DNS" if no_start[i] else "DNF"),
                                "started": not no_start[i], "classified": classified, "grid": float(grid[i]),
                                "pit_lane": False, "quali_pos": float(grid[i]),
                                "quali_gap_pct": float(max(0.0, (quali.max() - quali[i]) * 0.8)),
                                "laps": 50.0, "reason": None})
    stored, _ = store.upsert_events(pd.DataFrame(events), sport="motorsport")
    eid = dict(zip(stored["provider_event_id"], stored["event_id"]))
    table = pd.DataFrame(entries)
    table["event_id"] = table["gid"].map(eid)
    table = table.drop(columns="gid")
    results = []
    for event_id, part in table.groupby("event_id", sort=False):
        ranked = part[part["classified"]].sort_values("position")
        results.append({"event_id": event_id, "winner_id": ranked["driver_id"].iloc[0],
                        "ranking": ranked["driver_id"].tolist()})
    store.upsert_results(pd.DataFrame(results))
    store.write_table(LEAGUE, "race_entries", table, mode="replace")
    return {"drivers": dict(zip(driver_ids, drivers["name"])), "teams": dict(zip(team_ids, teams["name"])),
            "eid": eid, "driver_ids": driver_ids, "team_ids": team_ids}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    settings = Settings()
    settings.paths.data_dir = str(tmp_path_factory.mktemp("f1"))
    settings.validation.n_boot, settings.validation.n_replicas = 30, 3
    settings.selection.horizon_hours = 24.0 * 9
    store = Store(settings.paths.store)
    info = _build_series(store)
    plugin, _ = get_league(LEAGUE)
    reports = {}
    for kind in ("win", "points"):
        reports[kind], _ = validate_league(store, LEAGUE, f"finish:{kind}", settings=settings, now=NOW)
        train_league(store, LEAGUE, f"finish:{kind}", settings=settings, now=NOW)
    return {"settings": settings, "store": store, "plugin": plugin, "reports": reports, **info}


def _clone(world, tmp_path):
    target = tmp_path / "data"
    shutil.copytree(world["settings"].paths.data_dir, target)
    settings = copy.deepcopy(world["settings"])
    settings.paths.data_dir = str(target)
    return Store(settings.paths.store), settings


def _grid_csv(tmp_path, world, order=None, name="grid.csv", **edits):
    ids = world["driver_ids"] if order is None else order
    rows = [{"driver": world["drivers"][d], "grid": str(i + 1)} for i, d in enumerate(ids)]
    for i, change in edits.items():
        rows[int(i)].update(change)
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class _Builder:
    def feature_frame(self, store, league, market, *, now=None):
        return build_finish_frame(store, league, "win", now=FAR if now is None else now)


# ------------------------------------------------------------------ inputs
def test_inputs_do_not_change_when_the_future_is_hidden(world):
    result = asof_invariance(_Builder(), world["store"], LEAGUE, "finish:win", n_cutoffs=3)
    assert len(result) and result["n_events"].min() >= 14 and result["max_abs_diff"].max() == 0.0


def test_a_race_has_no_rows_until_its_grid_is_known(world):
    store = world["store"]
    frame = build_finish_frame(store, LEAGUE, "win", now=NOW)
    assert frame.upcoming.empty                                               # two races to come, no grid on file
    done = frame.completed
    race = done[done["game_id"] == done["game_id"].iloc[len(done) // 2]]
    start = race["start_time_utc"].iloc[0]
    # A historical rebuild knows the official grid from two hours before a timed start - not before.
    def rows_at(hours_before):
        frame_then = build_finish_frame(store, LEAGUE, "win", now=(start - pd.Timedelta(hours=hours_before)).isoformat())
        return frame_then.df[frame_then.df["game_id"] == race["game_id"].iloc[0]]

    assert rows_at(24).empty and rows_at(2.01).empty
    known = rows_at(1.99)
    assert len(known) >= len(race) and known["y"].isna().all() and known["finish_position"].isna().all()


def test_a_race_with_only_a_date_does_not_know_its_grid_before_its_stored_instant(world, tmp_path):
    """A date-only race is stored at 00:00Z of its date, hours before it - or its qualifying -
    took place.  The two-hour lead of a timed race must not be applied to that placeholder."""
    store, _ = _clone(world, tmp_path)
    events = store.events(LEAGUE, with_results=False)
    race = events[events["status"] == "final"].sort_values("start_time_utc").iloc[150]
    midnight = race["start_time_utc"].normalize()
    moved = events[events["event_id"] == race["event_id"]].assign(start_time_utc=midnight, time_precision="date")
    with store.conn() as con:
        con.execute("UPDATE events SET start_time_utc=?, time_precision='date' WHERE event_id=?",
                    (midnight.strftime("%Y-%m-%dT%H:%M:%SZ"), race["event_id"]))
    assert len(moved) == 1
    before = build_finish_frame(store, LEAGUE, "win", now=(midnight - pd.Timedelta(minutes=30)).isoformat())
    assert race["event_id"] not in set(before.df["game_id"])                  # inside two hours, still unknown
    at = build_finish_frame(store, LEAGUE, "win", now=midnight.isoformat())
    assert race["event_id"] in set(at.upcoming["game_id"])


def test_ratings_come_from_earlier_races_only(world, tmp_path):
    """Reverse one race's classification: its own rows keep their inputs, later races move."""
    store = world["store"]
    frame = build_finish_frame(store, LEAGUE, "win", now=NOW)
    done = frame.completed
    target = done["game_id"].iloc[len(done) // 2]
    other, _ = _clone(world, tmp_path)
    table = other.read_table(LEAGUE, "race_entries")
    hit = table["event_id"] == target
    n = int(hit.sum())
    table.loc[hit, "order"] = n + 1 - table.loc[hit, "order"]
    # ... and who retired: every classified starter retires, every retirement is classified.
    flipped = ~table.loc[hit, "classified"].astype(bool) & table.loc[hit, "started"].astype(bool)
    table.loc[hit, "classified"] = flipped
    table.loc[hit, "position"] = np.where(flipped, table.loc[hit, "order"], np.nan)
    other.write_table(LEAGUE, "race_entries", table, mode="replace")
    again = build_finish_frame(other, LEAGUE, "win", now=NOW).df.set_index("event_id")
    cols = list(frame.feature_cols)
    own = done[done["game_id"] == target]
    assert np.allclose(again.loc[own["event_id"], cols].to_numpy(dtype=float), own[cols].to_numpy(dtype=float),
                       equal_nan=True)
    later = done[done["start_time_utc"] > own["start_time_utc"].iloc[0]].head(16)
    assert not np.allclose(again.loc[later["event_id"], cols].to_numpy(dtype=float),
                           later[cols].to_numpy(dtype=float), equal_nan=True)


def test_a_driver_who_does_not_start_has_no_row_but_stays_in_the_field(world):
    table = world["store"].read_table(LEAGUE, "race_entries")
    absent = table[~table["started"]].iloc[0]
    frame = build_finish_frame(world["store"], LEAGUE, "win", now=NOW)
    race = frame.df[frame.df["game_id"] == absent["event_id"]]
    assert absent["driver_id"] not in set(race["player_id"])                  # his market is void
    assert (race["n_field"] == 2 * N_TEAMS).all()                             # the others were measured against him too
    # The whole-race model simulates him with the others: they raced a field that had him in it.
    context = frame.extra["field_context"]
    mine = context[context["game_id"] == absent["event_id"]]
    assert absent["driver_id"] in set(mine["player_id"]) and "y" not in context.columns
    assert "finish_position" not in context.columns
    done = frame.completed.sort_values(["start_time_utc", "event_id"])
    cands = {c.name: c for c in world["plugin"].candidates(LEAGUE, "finish:points", build_finish_frame(
        world["store"], LEAGUE, "points", now=NOW))}
    with_him = cands["field_rank"].clone().fit(done.iloc[:3000], ("no", "yes"))
    alone = cands["field_rank"].clone()
    alone.context = None
    alone.fit(done.iloc[:3000], ("no", "yes"))
    assert with_him.predict_proba(race)[:, 1].sum() < alone.predict_proba(race)[:, 1].sum() - 0.05


# ------------------------------------------------------------------ validation
def test_validation_compares_baselines_and_counts_races_not_drivers(world):
    report = world["reports"]["win"]
    names = [c["name"] for c in report["candidates"]]
    assert names[:2] == ["base_rate", "grid_logit"] and "field_rank" in names
    dev = {c["name"]: c["dev"]["log_loss"] for c in report["candidates"]}
    assert dev["grid_logit"] < dev["base_rate"] and dev["full_logit"] < dev["form_logit"]
    gate = next(g for g in report["gates"] if g["gate"] == "independent_races")
    assert "races in the final-test seasons" in gate["detail"]
    # The baseline to beat is the starting grid alone, not the base rate.
    against_grid = next(g for g in report["gates"] if g["gate"] == "skill_over_grid_logit")
    if report["selected_model"] != "grid_logit":
        ref = report["final_test"]["skill_vs_reference"]
        assert ref["reference"] == "grid_logit" and ref["n_boot"] >= 20000    # not a verdict that moves with the seed
        final = report["final_test"]
        assert ref["diff"] == pytest.approx(final["baselines"]["grid_logit"]["log_loss"] - final["selected"]["log_loss"])
        assert against_grid["passed"] == (ref["low"] > 0)
    # What is offered is the "yes" side only: the tables and the calibration gate describe that side.
    offered = report["final_test"]["offered"]
    assert offered["classes"] == ["yes"] and offered["n"] < report["final_test"]["selected"]["n"] / 4
    calibration = next(g for g in report["gates"] if g["gate"] == "calibration")
    assert ("too few" in calibration["detail"]) == (offered["n"] < 300)
    assert all(r["prob_low"] >= 0.5 for r in report["confidence_table"])
    assert sum(r["n"] for r in report["confidence_table"]) < 0.3 * (
        report["final_test"]["selected"]["n"] + report["dev"]["selected"]["n"])
    per_race = report["supplement"]["folds"]["final_test"]
    assert per_race["k"] == 1 and 0.0 <= per_race["model_hits_per_race"] <= 1.0
    assert abs(per_race["mean_probability_sum_per_race"] - 1.0) < 0.25         # one winner per race
    points = world["reports"]["points"]["supplement"]["folds"]["final_test"]
    assert points["k"] == 10 and points["model_hits_per_race"] > 5.0
    assert report["protocol"]["final_test_used_for_selection"] is False


def test_the_field_model_gives_coherent_probabilities(world):
    frame = build_finish_frame(world["store"], LEAGUE, "win", now=NOW)
    done = frame.completed.sort_values(["start_time_utc", "event_id"])
    cand = next(c for c in world["plugin"].candidates(LEAGUE, "finish:win", frame) if c.name == "field_rank")
    model = cand.clone().fit(done.iloc[:-800], frame.classes)
    test = done.iloc[-800:]
    p = model.predict_proba(test)[:, 1]
    sums = pd.Series(p).groupby(test["game_id"].to_numpy()).sum()
    assert (sums <= 1.0001).all()                                             # never more than one winner
    # One winner, unless everyone retires - or part of the chance belongs to a driver who was
    # on the grid and did not start (simulated with the field, no row of his own).
    with_absentee = set(frame.extra["field_context"]["game_id"])
    assert sums[~sums.index.isin(with_absentee)].between(0.9, 1.0001).all()
    assert (sums[sums.index.isin(with_absentee)] < 1.0).all()
    proj = model.projection(test)
    assert (proj["win"] <= proj["podium"] + 1e-9).all() and (proj["podium"] <= proj["points"] + 1e-9).all()
    assert np.allclose(model.predict_proba(test)[:, 1], p)                    # seeded by the race: repeatable
    assert isinstance(cand, FieldRankCandidate) and cand.predicts_field
    # The order rows arrive in does not matter, and neither do the outcome columns.
    shuffled = test.sample(frac=1.0, random_state=3)
    assert np.allclose(model.predict_proba(shuffled)[:, 1], pd.Series(p, index=test.index).loc[shuffled.index])
    from sportspred.validation.backtest import _blind
    assert np.allclose(model.predict_proba(_blind(test, model, "block"))[:, 1], p)


# ------------------------------------------------------------------ grid import, prediction, grading
def test_grid_import_checks_the_whole_file(world, tmp_path):
    store, _ = _clone(world, tmp_path)
    # Too early: the next race is more than a week away, so the file has to say which race it is for.
    with pytest.raises(CsvImportError, match="no scheduled race within 7 days"):
        import_grid(store, LEAGUE, _grid_csv(tmp_path, world), collected_at=NOW)
    NOW_NEAR = (store.upcoming(LEAGUE, now=NOW)["start_time_utc"].min() - pd.Timedelta(hours=3)).isoformat()
    bad = _grid_csv(tmp_path, world, **{"0": {"driver": "Nobody Atall"}, "3": {"grid": "3"}, "5": {"grid": "x"}})
    with pytest.raises(CsvImportError) as err:
        import_grid(store, LEAGUE, bad, collected_at=NOW_NEAR)
    text = str(err.value)
    assert "is not on file" in text and "already taken" in text and "whole number from 1, or PL" in text
    assert not store.has_table(LEAGUE, "imported_grid")
    few = tmp_path / "few.csv"
    pd.DataFrame([{"driver": world["drivers"][d], "grid": i + 1} for i, d in enumerate(world["driver_ids"][:4])]).to_csv(few, index=False)
    with pytest.raises(CsvImportError, match="not a starting grid"):
        import_grid(store, LEAGUE, few, collected_at=NOW_NEAR)
    with pytest.raises(CsvImportError, match="not a scheduled race that starts after"):
        import_grid(store, LEAGUE, _grid_csv(tmp_path, world), collected_at=NOW_NEAR, event_id=f"{LAST}-3")
    assert not store.has_table(LEAGUE, "imported_grid")


def test_a_grid_is_priced_recorded_and_graded(world, tmp_path):
    store, settings = _clone(world, tmp_path)
    assert predict_league(store, LEAGUE, "finish:win", settings=settings, now=NOW, write=False) == []
    order = list(world["driver_ids"])
    start = store.upcoming(LEAGUE, now=NOW)["start_time_utc"].min()
    read_at = (start - pd.Timedelta(hours=3)).isoformat()
    path = _grid_csv(tmp_path, world, order, **{str(len(order) - 1): {"grid": "PL"}})
    assert import_grid(store, LEAGUE, path, collected_at=read_at) == {"drivers": 16, "pit_lane_starts": 1}
    near = (start - pd.Timedelta(hours=1)).isoformat()
    # A grid read after the prediction time does not exist for that prediction.
    assert predict_league(store, LEAGUE, "finish:win", settings=settings,
                          now=(start - pd.Timedelta(hours=5)).isoformat(), write=False) == []
    preds = predict_league(store, LEAGUE, "finish:win", settings=settings, now=near)
    assert len(preds) == 16
    by_driver = {p["display"]["player_id"]: p for p in preds}
    pole = by_driver[order[0]]
    assert pole["market"] == f"finish:win:{order[0]}" and pole["selection"] == "yes"
    assert pole["selection_label"] == f"{world['drivers'][order[0]]} to win"
    assert pole["display"]["event_name"].startswith("Grand Prix") and pole["display"]["player_team"] in world["teams"].values()
    assert set(pole["probs"]) == {"yes", "no"} and abs(sum(pole["probs"].values()) - 1.0) < 1e-9
    probs = {d: p["probs"]["yes"] for d, p in by_driver.items()}
    assert 0.5 < sum(probs.values()) < 1.5 and probs[order[0]] > probs[order[-1]]
    # Only "to win" is ever offered: nobody is selected "not to win", however likely that is.
    assert all(p["selection"] == "yes" for p in preds)
    assert all(p["rec_status"] == "PASS" for p in preds if p["prob"] < 0.6)
    # The grid check is complete; an earlier race without a result would hold it back.
    states = {c["key"]: c["state"] for c in pole["checks"]}
    assert states["starting_grid"] == "complete" and states["prior_games_complete"] == "complete"
    assert any("starts 1 of 16" in t for t in pole["factors"]["supporting"])
    assert store.verify_ledger()[0]
    again = predict_league(store, LEAGUE, "finish:win", settings=settings, now=near)
    assert all(p.get("unchanged") for p in again)

    # ---- points market on the same grid, then the race is run
    points = {p["display"]["player_id"]: p for p in
              predict_league(store, LEAGUE, "finish:points", settings=settings, now=near)}
    picked = [d for d, p in points.items() if p["rec_status"] != "PASS"]
    assert picked and all(points[d]["selection"] == "yes" for d in picked)
    race = store.upcoming(LEAGUE, now=NOW).sort_values("start_time_utc").iloc[0]["event_id"]
    table = store.read_table(LEAGUE, "race_entries")
    template = table.iloc[0].to_dict()
    finish = list(reversed(order))                                            # the pit-lane starter wins
    new = []
    for place, d in enumerate(finish, start=1):
        retired, no_start = place == 12, place == 16
        new.append({**template, "event_id": race, "driver_id": d, "driver": d, "order": float(place),
                    "position": np.nan if retired or no_start else float(place),
                    "position_text": "DNF" if retired else "DNS" if no_start else str(place),
                    "started": not no_start, "classified": not (retired or no_start)})
    store.write_table(LEAGUE, "race_entries", pd.concat([table, pd.DataFrame(new)], ignore_index=True), mode="replace")
    store.upsert_results(pd.DataFrame([{"event_id": race, "winner_id": finish[0], "ranking": finish[:11]}]))
    grade_predictions(store, now="2026-12-30T00:00:00Z")
    grades = store.grades().set_index("prediction_id")["outcome"]
    for d in picked:
        place = finish.index(d) + 1
        expected = "void" if place == 16 else "correct" if place <= 10 else "incorrect"
        assert grades[points[d]["prediction_id"]] == expected, (place, expected)
    assert coverage_status(store, LEAGUE, "finish:win").value in ("validated", "experimental")

    # ---- the grader itself, on every kind of entry
    from sportspred.ledger.grading import _finish

    ev = {"league": LEAGUE, "event_id": race}
    def grade(kind, driver, selection="yes"):
        return _finish(store, {"market": f"finish:{kind}:{driver}", "selection": selection}, ev)[0]

    assert grade("win", finish[0]) == "correct" and grade("win", finish[1]) == "incorrect"
    assert grade("podium", finish[2]) == "correct" and grade("podium", finish[3]) == "incorrect"
    assert grade("points", finish[9]) == "correct" and grade("points", finish[10]) == "incorrect"
    assert grade("points", finish[11]) == "incorrect"                         # retired: started, not classified
    assert grade("win", finish[15]) == "void" and grade("win", "p_nobody") == "void"
    assert grade("win", finish[1], "no") == "correct"
    # Disqualified from second place: he started and is not classified - a lost selection, not a void one.
    table = store.read_table(LEAGUE, "race_entries")
    dq = (table["event_id"] == race) & (table["driver_id"] == finish[1])
    table.loc[dq, ["position", "classified", "position_text"]] = [np.nan, False, "DSQ"]
    store.write_table(LEAGUE, "race_entries", table, mode="replace")
    assert grade("podium", finish[1]) == "incorrect"
    assert _finish(store, {"market": f"finish:win:{finish[0]}", "selection": "yes"},
                   {"league": LEAGUE, "event_id": "evt_not_run"})[0] == "no_result"


def test_a_later_grid_replaces_an_earlier_one_and_withdraws_a_driver_who_left_it(world, tmp_path):
    store, settings = _clone(world, tmp_path)
    order = list(world["driver_ids"])
    start = store.upcoming(LEAGUE, now=NOW)["start_time_utc"].min()
    first = (start - pd.Timedelta(hours=20)).isoformat()
    import_grid(store, LEAGUE, _grid_csv(tmp_path, world, order), collected_at=first)
    early = {p["display"]["player_id"]: p for p in
             predict_league(store, LEAGUE, "finish:points", settings=settings,
                            now=(start - pd.Timedelta(hours=19)).isoformat())}
    assert early[order[0]]["rec_status"] == "PROVISIONAL"                     # the grid must be re-read near the start
    assert any("re-checked" in r for r in early[order[0]]["reasons"])
    # Race morning: the pole-sitter is out and everyone moves up.
    later = (start - pd.Timedelta(hours=3)).isoformat()
    import_grid(store, LEAGUE, _grid_csv(tmp_path, world, order[1:], "later.csv"), collected_at=later)
    second = predict_league(store, LEAGUE, "finish:points", settings=settings,
                            now=(start - pd.Timedelta(hours=1)).isoformat())
    by_driver = {p["display"].get("player_id"): p for p in second if not p.get("withdrawn")}
    assert order[0] not in by_driver and len(by_driver) == 15
    assert any("starts 1 of 15" in t for t in by_driver[order[1]]["factors"]["supporting"])
    gone = [p for p in second if p.get("withdrawn")]
    assert len(gone) == 1 and gone[0]["market"] == f"finish:points:{order[0]}" and gone[0]["rec_status"] == "PASS"


# ------------------------------------------------------------------ findings of the independent review
def test_an_earlier_race_without_a_result_holds_the_next_one_back(world, tmp_path):
    """A race that should have been run but has no result on file: the next race's ratings
    are a race behind, and its selections cannot be FINAL."""
    store, settings = _clone(world, tmp_path)
    ahead = store.upcoming(LEAGUE, now=NOW).sort_values("start_time_utc")
    first, second = ahead.iloc[0], ahead.iloc[1]
    read_at = (second["start_time_utc"] - pd.Timedelta(hours=3)).isoformat()
    import_grid(store, LEAGUE, _grid_csv(tmp_path, world), collected_at=read_at,
                event_id=str(second["provider_event_id"]))
    near = (second["start_time_utc"] - pd.Timedelta(hours=1)).isoformat()
    frame = build_finish_frame(store, LEAGUE, "points", now=near)
    assert (frame.upcoming["n_pending_prior"] == 1).all()                     # the first race is in the past, unresolved
    preds = predict_league(store, LEAGUE, "finish:points", settings=settings, now=near, write=False)
    assert preds and all(p["rec_status"] in ("PROVISIONAL", "PASS") for p in preds)
    states = {c["key"]: c["state"] for c in preds[0]["checks"]}
    assert states["prior_games_complete"] == "missing" and states["results_current"] == "missing"
    assert first["event_id"] != second["event_id"]


def test_an_old_reading_of_the_grid_builds_no_rows(world, tmp_path):
    store, _ = _clone(world, tmp_path)
    race = store.upcoming(LEAGUE, now=NOW).sort_values("start_time_utc").iloc[0]
    start = race["start_time_utc"]
    import_grid(store, LEAGUE, _grid_csv(tmp_path, world), collected_at=(start - pd.Timedelta(hours=40)).isoformat(),
                event_id=str(race["provider_event_id"]))
    fresh = build_finish_frame(store, LEAGUE, "win", now=(start - pd.Timedelta(hours=30)).isoformat())
    assert len(fresh.upcoming) == 16
    stale = build_finish_frame(store, LEAGUE, "win", now=(start - pd.Timedelta(hours=1)).isoformat())
    assert stale.upcoming.empty                                               # 39 hours old: read it again


def test_grid_import_guards(world, tmp_path):
    store, _ = _clone(world, tmp_path)
    ahead = store.upcoming(LEAGUE, now=NOW).sort_values("start_time_utc")
    first, second = ahead.iloc[0], ahead.iloc[1]
    grid = _grid_csv(tmp_path, world)
    # Read half an hour after the first race started: it is not the next race's grid.
    during = (first["start_time_utc"] + pd.Timedelta(minutes=30)).isoformat()
    with pytest.raises(CsvImportError, match="a race started at"):
        import_grid(store, LEAGUE, grid, collected_at=during)
    # Before qualifying has been held there is no grid.
    with store.conn() as con:
        fmt = {"name": "Grand Prix", "circuit": "c11", "circuit_type": "RACE",
               "qualifying_utc": (first["start_time_utc"] - pd.Timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")}
        import json as _json
        con.execute("UPDATE events SET format_json=? WHERE event_id=?", (_json.dumps(fmt), first["event_id"]))
    with pytest.raises(CsvImportError, match="qualifying for this race"):
        import_grid(store, LEAGUE, grid, collected_at=(first["start_time_utc"] - pd.Timedelta(hours=30)).isoformat())
    near = (first["start_time_utc"] - pd.Timedelta(hours=3)).isoformat()
    with pytest.raises(CsvImportError, match="beyond a field"):
        import_grid(store, LEAGUE, _grid_csv(tmp_path, world, name="far.csv", **{"4": {"grid": "99"}}), collected_at=near)
    everyone_pl = tmp_path / "pl.csv"
    pd.DataFrame([{"driver": n, "grid": "PL"} for n in world["drivers"].values()]).to_csv(everyone_pl, index=False)
    with pytest.raises(CsvImportError, match="nobody is on pole"):
        import_grid(store, LEAGUE, everyone_pl, collected_at=near)
    assert not store.has_table(LEAGUE, "imported_grid")
    assert second["event_id"] != first["event_id"]


def test_a_three_letter_code_is_matched_among_current_drivers_only(world, tmp_path):
    """Codes are reused over the years.  One that fits a current driver and a retired one
    means the current driver; a retired driver has to be named, with his team."""
    store, _ = _clone(world, tmp_path)
    ids = world["driver_ids"]
    old = pd.DataFrame({"name": ["Retired Namesake"], "short": ["AA1"]})
    store.resolve_participants(old, sport="motorsport", scope=LEAGUE, kind="person", short_name_col="short")
    with store.conn() as con:
        con.execute("UPDATE participants SET short_name='AA1' WHERE participant_id=?", (ids[0],))
    race = store.upcoming(LEAGUE, now=NOW).sort_values("start_time_utc").iloc[0]
    near = (race["start_time_utc"] - pd.Timedelta(hours=3)).isoformat()
    import_grid(store, LEAGUE, _grid_csv(tmp_path, world, **{"0": {"driver": "aa1"}}), collected_at=near)
    stored = store.read_table(LEAGUE, "imported_grid")
    assert ids[0] in set(stored["driver_id"]) and len(stored) == 16           # the current driver, not the namesake
    # The retired driver by name, without a team: he has no recent start to take one from.
    path = _grid_csv(tmp_path, world, name="old.csv", **{"0": {"driver": "Retired Namesake"}})
    with pytest.raises(CsvImportError, match="has no race on file|has not started a race"):
        import_grid(store, LEAGUE, path, collected_at=(race["start_time_utc"] - pd.Timedelta(hours=2)).isoformat())


# ------------------------------------------------------------------ the adapter
def _f1db_zip(path, drop_round=None, n_years=2):
    """A tiny file in the source's format: two seasons of three races, four drivers in two teams."""
    import io, zipfile

    races, results, quali = [], [], []
    rid = 0
    for year in range(2024, 2024 + n_years):
        for rnd, gp in enumerate(("alpha", "beta", "gamma"), start=1):
            if drop_round == (year, rnd):
                continue
            rid += 1
            done = not (year == 2025 and rnd == 3)
            races.append({"id": rid, "year": year, "round": rnd, "date": f"{year}-0{rnd + 2}-10",
                          "time": "13:00" if year == 2025 else None, "grandPrixId": gp, "officialName": f"{gp} GP {year}",
                          "circuitId": f"c-{gp}", "circuitType": "STREET" if gp == "beta" else "RACE",
                          "qualifyingDate": f"{year}-0{rnd + 2}-09", "qualifyingTime": "14:00" if year == 2025 else None,
                          "sprintRaceDate": None})
            if not done:
                continue
            rows = [("d-a", "t-x", "1", 1, 1, "1"), ("d-b", "t-x", "2", 2, 2, "2"),
                    ("d-c", "t-y", "DNF", None, 3, "PL"), ("d-d", "t-y", "DNS", None, 4, "4")]
            for order, (drv, team, text, pos, qpos, grid) in enumerate(rows, start=1):
                results.append({"raceId": rid, "year": year, "round": rnd, "positionDisplayOrder": order,
                                "positionNumber": pos, "positionText": text, "driverId": drv, "constructorId": team,
                                "gridPositionNumber": None if grid == "PL" else int(grid), "gridPositionText": grid,
                                "qualificationPositionNumber": qpos, "laps": 50, "reasonRetired": None, "sharedCar": False})
                quali.append({"raceId": rid, "year": year, "driverId": drv, "timeMillis": None,
                              "q1Millis": 90000 + 300 * qpos, "q2Millis": None, "q3Millis": None})
    tables = {
        "f1db-races.csv": pd.DataFrame(races), "f1db-races-race-results.csv": pd.DataFrame(results),
        "f1db-races-qualifying-results.csv": pd.DataFrame(quali),
        "f1db-circuits.csv": pd.DataFrame({"id": ["c-alpha", "c-beta", "c-gamma"], "fullName": ["A", "B", "C"],
                                           "placeName": ["Aa", "Bb", "Cc"]}),
        "f1db-grands-prix.csv": pd.DataFrame({"id": ["alpha", "beta", "gamma"], "fullName": ["Alpha GP", "Beta GP", "Gamma GP"]}),
        "f1db-drivers.csv": pd.DataFrame({"id": ["d-a", "d-b", "d-c", "d-d"], "name": ["Ann A", "Bob B", "Cy C", "Di D"],
                                          "fullName": ["Ann Alpha A", "Bob Beta B", "Cy Gamma C", "Di Delta D"],
                                          "abbreviation": ["ANA", "BOB", "CYC", "DID"]}),
        "f1db-constructors.csv": pd.DataFrame({"id": ["t-x", "t-y"], "name": ["Team X", "Team Y"]}),
    }
    with zipfile.ZipFile(path, "w") as z:
        for name, table in tables.items():
            z.writestr(name, table.to_csv(index=False))
    return path


class _FileFetcher:
    """Stands in for the downloader: hands back a local file."""

    def __init__(self, path):
        self.path = path

    def fetch(self, url, **kwargs):
        outer = self

        class Got:
            path, stale, collected_at_utc = outer.path, False, "2025-06-01T00:00:00Z"

            def register(self, store, notes=""):
                return store.add_source("f1db", "file://" + str(outer.path), sha256="0" * 64, n_bytes=1,
                                        license="test", notes=notes)

        return Got()


def test_the_adapter_reads_the_source_as_it_is(tmp_path):
    from sportspred.sports.motorsport.adapter import ingest_f1

    store = Store(tmp_path / "store")
    fetcher = _FileFetcher(_f1db_zip(tmp_path / "f1db.zip"))
    report = ingest_f1(store, seasons=[2024, 2025], fetcher=fetcher)
    assert report.n_events == 6 and report.n_results == 5 and report.tables["race_entries"] == 20
    events = store.events(LEAGUE, with_results=True).sort_values("start_time_utc")
    assert list(events["status"]) == ["final"] * 5 + ["scheduled"]
    assert list(events["time_precision"]) == ["date"] * 3 + ["datetime"] * 3   # a time only where the source has one
    assert events.iloc[3]["start_time_utc"] == pd.Timestamp("2025-03-10T13:00:00Z")
    names = store.participant_names()
    assert names[events.iloc[0]["winner_id"]] == "Ann A"                      # the name he races under
    entries = store.read_table(LEAGUE, "race_entries")
    one = entries[entries["event_id"] == events.iloc[0]["event_id"]].set_index("driver")
    assert bool(one.loc["d-c", "pit_lane"]) and not bool(one.loc["d-c", "classified"]) and bool(one.loc["d-c", "started"])
    assert not bool(one.loc["d-d", "started"])                                # did not start
    assert one.loc["d-b", "quali_gap_pct"] == pytest.approx(100 * 300 / 90300)
    # A partial season list never narrows what is stored, and a second ingest changes nothing.
    again = ingest_f1(store, seasons=[2025], fetcher=fetcher)
    assert again.n_events == 6 and len(store.read_table(LEAGUE, "race_entries")) == 20


def test_a_renumbered_source_is_refused(tmp_path):
    from sportspred.sports import DataUnavailable
    from sportspred.sports.motorsport.adapter import ingest_f1

    store = Store(tmp_path / "store")
    ingest_f1(store, seasons=[2024, 2025], fetcher=_FileFetcher(_f1db_zip(tmp_path / "a.zip")))
    before = store.events(LEAGUE, with_results=True)[["event_id", "winner_id", "season"]]
    # The source drops a 2024 round: every later race id now names another Grand Prix.
    renumbered = _FileFetcher(_f1db_zip(tmp_path / "b.zip", drop_round=(2024, 2)))
    with pytest.raises(DataUnavailable, match="renumbered"):
        ingest_f1(store, seasons=[2024, 2025], fetcher=renumbered)
    after = store.events(LEAGUE, with_results=True)[["event_id", "winner_id", "season"]]
    assert before.equals(after)                                               # nothing was changed
