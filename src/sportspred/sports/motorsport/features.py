"""Point-in-time inputs for Formula 1 finishing markets.

One row per driver and race.  Three questions are asked of every starter -
does he **win**, finish on the **podium** (top three), finish in the
**points** (top ten) - and each is its own market with its own validation.

What a row knows before the start:

* **the starting grid** - the strongest pregame fact - and the driver's
  qualifying gap to pole where lap times are on file;
* **driver form**: his recent and longer-run results relative to the field,
  his qualifying form, and how he qualifies against his team-mates (which
  separates the driver from the car);
* **car form**: the constructor's recent and longer-run results and
  qualifying, with most of the weight dropped at a season change, when cars
  are new;
* **field strength**: each of those relative to the field of that race;
* **track fit**: how the driver has gone at this circuit before, and how the
  constructor goes at this type of circuit (street, road, permanent);
* **reliability**: how often the driver, the car and the field as a whole
  have failed to be classified.

Point-in-time rule: every rating of a race comes from races on earlier days.
The official starting grid of a race is taken as known from
``grid_lead_hours`` before its start (it is published after qualifying, with
penalties, well before that); before then a race has rows only if a grid has
been imported, read at or before ``now``.  Rows without a grid are not
predicted: this model starts from the grid.

Not in the data, so not in the model: weather and track conditions, tyre
allocation and strategy, upgrades, practice pace, sprint results, penalties
announced after the grid was read, and anything about the start itself.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...core.enums import OutcomeType
from ...core.timeutil import parse_ts, utcnow
from ..base import FeatureFrame

__all__ = ["F1Config", "KINDS", "KIND_LABELS", "finish_market", "split_finish_market", "build_finish_frame",
           "FORM_FEATURES", "GRID_FEATURES", "FULL_FEATURES", "DNF_FEATURES", "IMPORTED_GRID", "field_of"]

#: Market kind -> a classified finish at or inside this position.
KINDS: Dict[str, int] = {"win": 1, "podium": 3, "points": 10}
KIND_LABELS = {"win": ("to win", "not to win"), "podium": ("to finish on the podium", "not to finish on the podium"),
               "points": ("to finish in the points (top ten)", "not to finish in the points")}
IMPORTED_GRID = "imported_grid"


def finish_market(kind: str, driver_id: Optional[str] = None) -> str:
    """``finish:<kind>`` (the modelled market) or one driver's ledger key ``finish:<kind>:<driver id>``."""
    return f"finish:{kind}" + (f":{driver_id}" if driver_id is not None else "")


def split_finish_market(market: str) -> Tuple[Optional[str], Optional[str]]:
    parts = str(market).split(":")
    if parts[0] != "finish" or len(parts) < 2 or parts[1] not in KINDS:
        return None, None
    return parts[1], (parts[2] if len(parts) > 2 else None)


@dataclass(frozen=True)
class F1Config:
    first_season: int = 2000
    burn_in_seasons: int = 2
    form_halflife: float = 8.0          # in races the driver entered
    long_halflife: float = 30.0
    car_halflife: float = 5.0           # in races the constructor entered
    car_long_halflife: float = 20.0
    reliability_halflife: float = 20.0
    league_halflife: float = 40.0
    driver_season_carry: float = 0.7    # weight kept by a driver's history at a season change
    car_season_carry: float = 0.3       # ... and by a car's: cars are new each year
    circuit_prior_visits: float = 2.0   # earlier visits a circuit record is shrunk towards "no difference" by
    grid_lead_hours: float = 2.0        # the official grid counts as known this long before the start
    horizon_days: float = 10.0          # upcoming races further ahead get no rows


GRID_FEATURES = ["log_grid", "grid_rel"]
FORM_FEATURES = ["drv_perf_ew", "drv_perf_long", "drv_q_ew", "mate_gap_ew", "drv_starts", "con_perf_ew",
                 "con_perf_long", "con_q_ew", "drv_perf_rel", "con_perf_rel", "con_rank", "drv_circuit_delta",
                 "con_type_delta", "is_street"]
DNF_FEATURES = ["drv_dnf_ew", "con_dnf_ew", "league_dnf_ew", "grid_rel", "is_street"]
FULL_FEATURES = GRID_FEATURES + ["quali_gap_pct"] + FORM_FEATURES + ["drv_dnf_ew", "con_dnf_ew", "league_dnf_ew"]

FEATURE_NOTES = {
    "log_grid": "starting grid position (log); a pit-lane start counts as last",
    "grid_rel": "starting grid position as a share of the field (0 = pole, 1 = last)",
    "quali_gap_pct": "qualifying gap to pole, percent of the pole time (where lap times are on file)",
    "drv_perf_ew": "driver's recent results relative to the field (1 = win, 0 = last), exponentially weighted",
    "drv_perf_long": "the same over a longer run",
    "drv_q_ew": "driver's recent qualifying results relative to the field",
    "mate_gap_ew": "qualifying positions behind (+) or ahead of (-) his team-mates, recent",
    "drv_starts": "race starts on file (capped at 100)",
    "con_perf_ew": "constructor's recent results relative to the field",
    "con_perf_long": "the same over a longer run",
    "con_q_ew": "constructor's recent qualifying results",
    "drv_perf_rel": "driver's recent results minus the field's average",
    "con_perf_rel": "constructor's recent results minus the field's average",
    "con_rank": "constructor's rank in the field by recent results (0 = best)",
    "drv_circuit_delta": "how the driver has gone at this circuit before, relative to his form then",
    "con_type_delta": "how the constructor goes at this type of circuit, relative to its longer-run results",
    "is_street": "street circuit",
    "drv_dnf_ew": "share of the driver's recent starts not classified",
    "con_dnf_ew": "share of the constructor's recent starts not classified",
    "league_dnf_ew": "share of all recent starts not classified",
}


class _Ew:
    """Exponentially weighted mean updated one observation at a time."""

    __slots__ = ("num", "den", "decay")

    def __init__(self, halflife: float) -> None:
        self.num, self.den, self.decay = 0.0, 0.0, 0.5 ** (1.0 / halflife)

    def add(self, value: float) -> None:
        self.num = self.num * self.decay + value
        self.den = self.den * self.decay + 1.0

    def fade(self, keep: float) -> None:
        self.num *= keep
        self.den *= keep

    @property
    def mean(self) -> float:
        return self.num / self.den if self.den > 1e-9 else float("nan")


def field_of(entries: pd.DataFrame) -> pd.DataFrame:
    """The drivers with a place on the grid (a slot or a pit-lane start).  One
    who then fails to take the start is still part of the field the others
    were measured against; his own markets are void."""
    return entries[entries["grid"].notna() | entries["pit_lane"].fillna(False).astype(bool)]


def _imported_grid(store: Any, league: str, now_iso: str) -> Dict[str, pd.DataFrame]:
    """Per event, the newest imported grid read at or before ``now``."""
    if not store.has_table(league, IMPORTED_GRID):
        return {}
    grid = store.read_table(league, IMPORTED_GRID)
    grid = grid[grid["collected_at_utc"] <= now_iso]
    out: Dict[str, pd.DataFrame] = {}
    for event_id, part in grid.groupby("event_id", sort=False):
        newest = part["collected_at_utc"].max()
        out[event_id] = part[part["collected_at_utc"] == newest]
    return out


def build_finish_frame(store: Any, league: str, kind: str, *, now: Any = None,
                       cfg: Optional[F1Config] = None) -> FeatureFrame:
    if kind not in KINDS:
        raise KeyError(f"unknown finishing market {kind!r}; one of {sorted(KINDS)}")
    cfg = cfg or F1Config()
    now_ts = pd.Timestamp(utcnow() if now is None else parse_ts(now))
    now_iso = now_ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    if not store.has_table(league, "race_entries"):
        raise ValueError(f"no race classifications on file for {league}; run `sportspred ingest --league {league}`")
    ev = store.events(league, with_results=True)
    ev = ev[pd.to_numeric(ev["season"], errors="coerce") >= cfg.first_season]
    done_mask = (ev["status"] == "final") & ev["winner_id"].notna() & (ev["start_time_utc"] < now_ts)
    ahead_mask = (ev["status"].isin(["scheduled", "final"]) & (ev["start_time_utc"] >= now_ts)
                  & (ev["start_time_utc"] <= now_ts + pd.Timedelta(days=cfg.horizon_days)))
    ev = ev[done_mask | ahead_mask].sort_values(["start_time_utc", "event_id"]).reset_index(drop=True)
    if ev.empty:
        raise ValueError(f"no events on file for {league}; run `sportspred ingest --league {league}` first")
    ev["done"] = ((ev["status"] == "final") & ev["winner_id"].notna() & (ev["start_time_utc"] < now_ts)).to_numpy()
    all_entries = store.read_table(league, "race_entries")
    by_event = {e: part for e, part in all_entries.groupby("event_id", sort=False)}
    imported = _imported_grid(store, league, now_iso)
    names = store.participant_names()
    last_team: Dict[str, str] = {}

    drv_fast: Dict[str, _Ew] = {}
    drv_long: Dict[str, _Ew] = {}
    drv_q: Dict[str, _Ew] = {}
    drv_gap: Dict[str, _Ew] = {}
    drv_dnf: Dict[str, _Ew] = {}
    drv_starts: Dict[str, int] = {}
    con_fast: Dict[str, _Ew] = {}
    con_long: Dict[str, _Ew] = {}
    con_q: Dict[str, _Ew] = {}
    con_dnf: Dict[str, _Ew] = {}
    con_type: Dict[Tuple[str, str], _Ew] = {}
    circuit: Dict[Tuple[str, str], List[float]] = {}
    league_dnf = _Ew(cfg.league_halflife * 20.0)             # in starts, about forty races of twenty cars
    season_seen: Optional[str] = None
    pending_before = 0
    limit = KINDS[kind]
    rows: List[Dict[str, Any]] = []

    def mean_of(table: Dict[Any, _Ew], key: Any) -> float:
        item = table.get(key)
        return item.mean if item is not None else float("nan")

    for race in ev.itertuples():
        fmt = json.loads(race.format_json) if isinstance(race.format_json, str) else {}
        circuit_id, circuit_type = fmt.get("circuit"), fmt.get("circuit_type")
        if race.season != season_seen:                        # a new season: cars are new, drivers less so
            for table, keep in ((con_fast, cfg.car_season_carry), (con_long, cfg.car_season_carry),
                                (con_q, cfg.car_season_carry), (drv_fast, cfg.driver_season_carry),
                                (drv_q, cfg.driver_season_carry)):
                for item in table.values():
                    item.fade(keep)
            season_seen, pending_before = race.season, 0

        # ---- who is on the grid, as far as it is known at ``now``
        source, read_at = None, None
        on_file = by_event.get(race.event_id)
        official_known = race.done or (on_file is not None
                                       and now_ts >= race.start_time_utc - pd.Timedelta(hours=cfg.grid_lead_hours))
        if official_known and on_file is not None:
            field, source = field_of(on_file), "official starting grid"
        elif race.event_id in imported:
            field, source = imported[race.event_id], "imported grid"
            read_at = str(field["collected_at_utc"].iloc[0])
        else:
            field = None
        if field is not None and len(field) >= 2:
            n = len(field)
            drivers = field["driver_id"].tolist()
            teams = field["constructor_id"].tolist()
            grid = field["grid"].to_numpy(dtype=float)
            pit = field["pit_lane"].fillna(False).to_numpy(dtype=bool)
            grid_pos = np.where(pit | np.isnan(grid), float(n), grid)
            gap = field["quali_gap_pct"].to_numpy(dtype=float) if "quali_gap_pct" in field.columns else np.full(n, np.nan)
            d_fast = np.array([mean_of(drv_fast, d) for d in drivers])
            c_fast = np.array([mean_of(con_fast, t) for t in teams])
            c_rank = pd.Series(-c_fast).rank(method="average", na_option="keep").to_numpy()
            with np.errstate(invalid="ignore"):
                d_mean = np.nanmean(d_fast) if np.isfinite(d_fast).any() else float("nan")
                c_mean = np.nanmean(c_fast) if np.isfinite(c_fast).any() else float("nan")
            started = field["started"].to_numpy(dtype=bool) if (race.done and "started" in field.columns) \
                else np.ones(n, dtype=bool)
            position = field["position"].to_numpy(dtype=float) if (race.done and "position" in field.columns) \
                else np.full(n, np.nan)
            for i, (d, t) in enumerate(zip(drivers, teams)):
                if not started[i]:
                    continue                                   # did not take the start: his markets are void
                visits = circuit.get((d, circuit_id), [])
                long_car = mean_of(con_long, t)
                type_car = mean_of(con_type, (t, circuit_type))
                rows.append({
                    "event_id": f"{race.event_id}|{d}", "game_id": race.event_id,
                    "start_time_utc": race.start_time_utc, "block": int(race.season),
                    "y": (float(position[i] <= limit) if position[i] == position[i] else 0.0) if race.done else np.nan,
                    "side_a_id": d, "side_b_id": t, "player_id": d, "player_name": names.get(d, d),
                    "ledger_market": finish_market(kind, d), "constructor_name": names.get(t, t),
                    "grid_pos": float(grid_pos[i]), "pit_lane": bool(pit[i]), "n_field": float(n),
                    "finish_position": position[i] if race.done else np.nan,
                    "classified": (1.0 if position[i] == position[i] else 0.0) if race.done else np.nan,
                    "n_pending_prior": int(pending_before), "grid_source": source, "grid_read_at_utc": read_at,
                    "event_name": fmt.get("name") or race.venue_name,
                    "segment": ("grid 1 to 4" if grid_pos[i] <= 4 else "grid 5 to 10" if grid_pos[i] <= 10
                                else "grid 11 and back"),
                    "log_grid": math.log(grid_pos[i]), "grid_rel": (grid_pos[i] - 1.0) / max(n - 1.0, 1.0),
                    "quali_gap_pct": float(min(max(gap[i], 0.0), 6.0)) if gap[i] == gap[i] else np.nan,
                    "drv_perf_ew": d_fast[i], "drv_perf_long": mean_of(drv_long, d), "drv_q_ew": mean_of(drv_q, d),
                    "mate_gap_ew": mean_of(drv_gap, d), "drv_starts": min(drv_starts.get(d, 0), 100) / 100.0,
                    "con_perf_ew": c_fast[i], "con_perf_long": long_car, "con_q_ew": mean_of(con_q, t),
                    "drv_perf_rel": d_fast[i] - d_mean, "con_perf_rel": c_fast[i] - c_mean,
                    "con_rank": (c_rank[i] - 1.0) / max(n - 1.0, 1.0) if c_rank[i] == c_rank[i] else np.nan,
                    "drv_circuit_delta": sum(visits) / (len(visits) + cfg.circuit_prior_visits),
                    "con_type_delta": type_car - long_car,
                    "is_street": 1.0 if circuit_type == "STREET" else 0.0,
                    "drv_dnf_ew": mean_of(drv_dnf, d), "con_dnf_ew": mean_of(con_dnf, t),
                    "league_dnf_ew": league_dnf.mean,
                })
        if not race.done:
            pending_before += 1
            continue

        # ---- the race has been run: update every rating from its classification
        part = on_file
        if part is None:
            continue
        went = part[part["started"].astype(bool)]
        n_started = len(went)
        if n_started < 2:
            continue
        order = went["order"].rank(method="first").to_numpy()
        perf = 1.0 - (order - 1.0) / (n_started - 1.0)
        with_q = field_of(part)
        q_rank = with_q["quali_pos"].where(with_q["quali_pos"].notna(), with_q["grid"]).rank(method="average")
        q_perf = dict(zip(with_q["driver_id"], 1.0 - (q_rank - 1.0) / max(len(with_q) - 1.0, 1.0)))
        q_pos = dict(zip(with_q["driver_id"], with_q["quali_pos"]))
        by_team: Dict[str, List[str]] = {}
        for d, t in zip(with_q["driver_id"], with_q["constructor_id"]):
            by_team.setdefault(t, []).append(d)
        car_perf: Dict[str, List[float]] = {}
        car_q: Dict[str, List[float]] = {}
        for d, t, p, classified in zip(went["driver_id"], went["constructor_id"], perf, went["classified"]):
            before = mean_of(drv_fast, d)
            drv_fast.setdefault(d, _Ew(cfg.form_halflife)).add(float(p))
            drv_long.setdefault(d, _Ew(cfg.long_halflife)).add(float(p))
            failed = 0.0 if bool(classified) else 1.0
            drv_dnf.setdefault(d, _Ew(cfg.reliability_halflife)).add(failed)
            con_dnf.setdefault(t, _Ew(cfg.reliability_halflife)).add(failed)
            league_dnf.add(failed)
            drv_starts[d] = drv_starts.get(d, 0) + 1
            car_perf.setdefault(t, []).append(float(p))
            if before == before and circuit_id:
                circuit.setdefault((d, circuit_id), []).append(float(p) - before)
            last_team[d] = t
        for d, value in q_perf.items():
            if value == value:
                drv_q.setdefault(d, _Ew(cfg.form_halflife)).add(float(value))
        for t, members in by_team.items():
            placed = [q_pos[m] for m in members if q_pos.get(m) == q_pos.get(m)]
            for m in members:
                if q_pos.get(m) == q_pos.get(m) and len(placed) >= 2:
                    others = (sum(placed) - q_pos[m]) / (len(placed) - 1)
                    drv_gap.setdefault(m, _Ew(cfg.form_halflife)).add(float(q_pos[m] - others))
            values = [q_perf[m] for m in members if q_perf.get(m) == q_perf.get(m)]
            if values:
                car_q.setdefault(t, []).extend(values)
        for t, values in car_perf.items():
            car = float(np.mean(values))
            con_fast.setdefault(t, _Ew(cfg.car_halflife)).add(car)
            con_long.setdefault(t, _Ew(cfg.car_long_halflife)).add(car)
            if circuit_type:
                con_type.setdefault((t, circuit_type), _Ew(cfg.car_long_halflife)).add(car)
        for t, values in car_q.items():
            con_q.setdefault(t, _Ew(cfg.car_halflife)).add(float(np.mean(values)))

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"no race with a grid on file for {league}")
    df["start_time_utc"] = pd.to_datetime(df["start_time_utc"], utc=True)
    df = df.sort_values(["start_time_utc", "event_id"]).reset_index(drop=True)
    yes, no = KIND_LABELS[kind]
    return FeatureFrame(
        league=league, market=finish_market(kind), outcome=OutcomeType.BINARY, classes=("no", "yes"), df=df,
        feature_cols=list(dict.fromkeys(FULL_FEATURES + DNF_FEATURES)),
        feature_notes=dict(FEATURE_NOTES),
        limitations=[
            "One row per driver and race; the three markets (win, podium, points) are validated separately, as "
            "yes-or-no questions about each starter. The headline accuracy is dominated by the many drivers for "
            "whom the answer is plainly no - log loss against the base rate and the per-race figures are the ones "
            "to read.",
            "The model starts from the official starting grid. Before a grid is on file there is no prediction.",
            "The grid is the final one, with penalties, as recorded by the source; a change in the last two hours "
            "before the start (a pit-lane start decided late) is not point-in-time.",
            "Not in the data: weather and track conditions, tyres and strategy, upgrades, practice and sprint "
            "results. Lap times in qualifying are on file for part of the history only.",
            "Rule eras differ (points for the top six, eight, then ten; far fewer retirements now than in 2000). "
            "Ratings are relative to the field of each race, and retirement rates are tracked as they change, but "
            "older seasons are not the same sport in every respect.",
            "A driver who does not take the start has no row: his markets are void.",
        ],
        extra={"kind": kind, "stat": kind, "stat_label": None, "yes_label": yes, "no_label": no,
               "required_columns": ["grid_pos"], "selection_classes": ["yes"], "field_model": True,
               "burn_in_seasons": cfg.burn_in_seasons, "last_team": last_team,
               "config": {"first_season": cfg.first_season, "grid_lead_hours": cfg.grid_lead_hours}},
    )
