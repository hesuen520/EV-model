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
    # A historical rebuild knows the official grid from two hours before the start, not a day before.
    day_before = build_finish_frame(store, LEAGUE, "win", now=(start - pd.Timedelta(hours=24)).isoformat())
    assert race["game_id"].iloc[0] not in set(day_before.df["game_id"])
    hour_before = build_finish_frame(store, LEAGUE, "win", now=(start - pd.Timedelta(hours=1)).isoformat())
    known = hour_before.upcoming[hour_before.upcoming["game_id"] == race["game_id"].iloc[0]]
    assert len(known) >= len(race) and known["y"].isna().all() and known["finish_position"].isna().all()


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
    table.loc[hit, "position"] = np.where(table.loc[hit, "classified"], n + 1 - table.loc[hit, "position"], np.nan)
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


# ------------------------------------------------------------------ validation
def test_validation_compares_baselines_and_counts_races_not_drivers(world):
    report = world["reports"]["win"]
    names = [c["name"] for c in report["candidates"]]
    assert names[:2] == ["base_rate", "grid_logit"] and "field_rank" in names
    dev = {c["name"]: c["dev"]["log_loss"] for c in report["candidates"]}
    assert dev["grid_logit"] < dev["base_rate"] and dev["full_logit"] < dev["form_logit"]
    gate = next(g for g in report["gates"] if g["gate"] == "independent_races")
    assert "races in the final-test seasons" in gate["detail"]
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
    assert sums.between(0.9, 1.0001).all()                                    # one winner, unless everyone retires
    proj = model.projection(test)
    assert (proj["win"] <= proj["podium"] + 1e-9).all() and (proj["podium"] <= proj["points"] + 1e-9).all()
    assert np.allclose(model.predict_proba(test)[:, 1], p)                    # seeded by the race: repeatable
    assert isinstance(cand, FieldRankCandidate) and cand.predicts_field


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
