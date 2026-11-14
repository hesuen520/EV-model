"""Point-in-time baseball features for the game-winner market.

Every number in a point-in-time column is computed from games on **earlier
slates** only (a slate is a Pacific/Honolulu calendar day; for the date-only
Retrosheet data that means earlier dates), plus the public schedule.  The
second game of a doubleheader shares its slate with the first and therefore
never sees it.  A suspended game is listed on the date it began although its
result only existed when it was completed, so its result is never used to
update anything.  Side A is the home club.

Point-in-time feature families (their weights are learned in validation):

* opposition-adjusted team strength - Elo, and offence/defence ratings on runs
  per game where each game's surprise is measured against what the opponent
  normally allows, with a slow memory and a fast one ("recent form");
* the same slow rating on BaseRuns, a run estimate built from hits, walks and
  extra bases that is less noisy than the runs that actually scored;
* the run environment of the park, from the park's games in prior seasons only;
* schedule - day or night, days since the previous game, travel, time-zone
  change, second game of a doubleheader;
* bullpen usage - relief outs on the previous three days.

Retrospective columns (``sp_*``) describe the starting pitchers who **actually
started**: quality and workload from their prior starts only, days since the
last start, throwing hand.  The source does not record who was announced as
the probable starter, and announced starters change, so these columns are a
near-pregame reconstruction.  They are left empty for upcoming games and can
only feed candidates flagged ``uses_retrospective``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...core.enums import OutcomeType
from ...core.timeutil import DISPLAY_TZ_NAME, parse_ts, utcnow
from ...features.ratings import EloParams, elo_pregame, offdef_pregame, tune_elo
from ...features.schedule import haversine_km
from ..base import FeatureFrame
from .venues import city_coords

__all__ = [
    "BaseballConfig", "build_winner_frame", "base_runs", "park_factors", "rest_and_travel", "bullpen_load",
    "WINNER_FEATURES", "STARTER_FEATURES", "FEATURE_LABELS",
]

EXCLUDED_STAGES = ("preseason", "allstar", "offseason")
_BATTING = ("ab", "h", "d", "t", "hr", "hbp", "bb")
#: What a start contributes to a pitcher's record: outs, runs allowed, the
#: fielding-independent numerator 13*HR + 3*(BB+HBP) - 2*SO, strikeouts minus
#: walks, and batters faced.
_START_STATS = ("outs", "runs", "fip", "kbb", "bf")


@dataclass(frozen=True)
class BaseballConfig:
    burn_in_seasons: int = 3              # earliest seasons: rating warm-up and update-rate tuning only
    fast_lr: float = 0.12                 # "recent form": roughly the last 16 games
    slow_lr_grid: Tuple[float, ...] = (0.01, 0.02, 0.035)
    season_carry: float = 0.7
    elo_k_grid: Tuple[float, ...] = (2.0, 4.0, 6.0, 8.0)
    elo_home_grid: Tuple[float, ...] = (0.0, 20.0, 40.0)
    elo_carry_grid: Tuple[float, ...] = (0.5, 0.7, 0.9)
    elo_margin_grid: Tuple[Optional[float], ...] = (None, 3.0)
    park_seasons: int = 3                 # prior seasons a park factor looks back over
    park_prior_games: float = 81.0        # shrinkage towards a neutral park: one season of home games
    rest_cap_days: float = 4.0
    bullpen_days: int = 3
    starter_decay: float = 0.97           # per start; a starter's old starts fade
    starter_prior_starts: float = 10.0    # league-average starts mixed into every starter's record
    starter_rest_cap: float = 10.0
    league_lr: float = 0.02               # per slate; memory of the running league-average start


WINNER_FEATURES: List[str] = [
    "home_edge", "elo_diff", "net_slow_diff", "net_fast_diff", "form_diff", "bsr_slow_diff",
    "park_factor", "day_game", "rest_diff", "travel_km_a", "travel_km_b", "tz_shift_b", "dh_game2",
    "pen_outs_a", "pen_outs_b", "is_playoff",
]

STARTER_FEATURES: List[str] = [
    "sp_ra9_diff", "sp_fip_diff", "sp_kbb_diff", "sp_outs_diff", "sp_rest_a", "sp_rest_b",
    "sp_starts_a", "sp_starts_b", "sp_left_a", "sp_left_b",
]

FEATURE_LABELS: Dict[str, str] = {
    "home_edge": "home field",
    "elo_diff": "opponent-adjusted results (Elo)",
    "net_slow_diff": "long-term opponent-adjusted run differential",
    "net_fast_diff": "recent (about 16 games) opponent-adjusted run differential",
    "form_diff": "recent form relative to long-term level",
    "bsr_slow_diff": "long-term strength from hits, walks and power (BaseRuns)",
    "park_factor": "run environment of the park (prior seasons)",
    "day_game": "day game",
    "rest_diff": "days since the previous game",
    "travel_km_a": "home side travel",
    "travel_km_b": "away side travel",
    "tz_shift_b": "away side time-zone change",
    "dh_game2": "second game of a doubleheader",
    "pen_outs_a": "home bullpen workload (previous 3 days)",
    "pen_outs_b": "away bullpen workload (previous 3 days)",
    "is_playoff": "postseason game",
    "sp_ra9_diff": "starters' runs allowed per nine innings (retrospective)",
    "sp_fip_diff": "starters' fielding-independent pitching (retrospective)",
    "sp_kbb_diff": "starters' strikeouts minus walks per batter (retrospective)",
    "sp_outs_diff": "starters' outs per start (retrospective)",
    "sp_rest_a": "home starter's days since his last start (retrospective)",
    "sp_rest_b": "away starter's days since his last start (retrospective)",
    "sp_starts_a": "home starter's experience (retrospective)",
    "sp_starts_b": "away starter's experience (retrospective)",
    "sp_left_a": "home starter throws left-handed (retrospective)",
    "sp_left_b": "away starter throws left-handed (retrospective)",
}


def slate_days(start_time_utc: pd.Series) -> np.ndarray:
    """Pacific/Honolulu calendar day of each event as a day count, so that
    differences are whole days between slates."""
    local = start_time_utc.dt.tz_convert(DISPLAY_TZ_NAME).dt.tz_localize(None).dt.normalize()
    return ((local - pd.Timestamp("1970-01-01")) // pd.Timedelta(days=1)).to_numpy()


def base_runs(ab: np.ndarray, h: np.ndarray, d: np.ndarray, t: np.ndarray, hr: np.ndarray,
              walks: np.ndarray) -> np.ndarray:
    """BaseRuns (Smyth): runs a batting line would normally produce.

    A published run estimator; it only defines the input, and how much it
    matters is learned.  ``walks`` includes hit batters.
    """
    total_bases = h + d + 2.0 * t + 3.0 * hr
    runners = h + walks - hr
    advance = (1.4 * total_bases - 0.6 * h - 3.0 * hr + 0.1 * walks) * 1.02
    outs = ab - h
    with np.errstate(divide="ignore", invalid="ignore"):
        score_rate = advance / (advance + outs)
    return runners * np.where(np.isfinite(score_rate), score_rate, 0.0) + hr


def park_factors(season_idx: np.ndarray, park: np.ndarray, home: np.ndarray, away: np.ndarray,
                 runs: np.ndarray, *, n_seasons: int, prior_games: float) -> np.ndarray:
    """Run factor of each event's park, from the PRIOR seasons only.

    ``runs`` is a game's total runs (NaN where it may not be used).  In each
    earlier season the runs scored in the park are set against what its host
    clubs' games produced on the road that season; the ratio over the last
    ``n_seasons`` seasons is shrunk towards 1 as if ``prior_games`` neutral
    games had also been played there.  A park with no earlier games gets 1.
    """
    df = pd.DataFrame({"season": season_idx, "park": park, "home": home, "away": away, "runs": runs})
    known = df[df["runs"].notna()]
    road = known.groupby(["season", "away"])["runs"].mean()
    base = road.reindex(pd.MultiIndex.from_frame(known[["season", "home"]])).to_numpy()
    known = known.assign(base=base).dropna(subset=["base"])
    per = known.groupby(["park", "season"]).agg(runs=("runs", "sum"), base=("base", "sum"), n=("runs", "size"))
    totals = {key: row for key, row in zip(per.index, per.to_numpy())}
    out = np.ones(len(df))
    for (park_id, season), rows in df.groupby(["park", "season"]).indices.items():
        got = [totals[(park_id, s)] for s in range(season - n_seasons, season) if (park_id, s) in totals]
        if got:
            park_runs, road_runs, n = np.sum(got, axis=0)
            out[rows] = (n * park_runs / road_runs + prior_games) / (n + prior_games)
    return out


def rest_and_travel(a_idx: np.ndarray, b_idx: np.ndarray, season_idx: np.ndarray, day: np.ndarray,
                    lat: np.ndarray, lon: np.ndarray, *, n_teams: int, rest_cap: float) -> pd.DataFrame:
    """Schedule context per side from the list of games alone.

    Events must be in playing order (a doubleheader's first game first).
    ``rest_*`` is the number of slate days since the side's previous game this
    season (0 for the second game of a doubleheader; capped, and the cap for a
    season's first game); ``travel_km_*`` and ``tz_shift_*`` (hours, eastward
    positive) are measured from that previous game's city and are missing for
    a season's first game; ``season_games_*`` counts the side's earlier games.
    """
    n = len(a_idx)
    last_day = np.zeros(n_teams, dtype=np.int64)
    last_season = np.full(n_teams, -1)
    last_row = np.full(n_teams, -1)
    count = np.zeros(n_teams, dtype=int)
    rest = {s: np.full(n, rest_cap) for s in "ab"}
    prev = {s: np.full(n, -1) for s in "ab"}
    games = {s: np.zeros(n, dtype=int) for s in "ab"}
    for i in range(n):
        for side, team in (("a", a_idx[i]), ("b", b_idx[i])):
            if last_season[team] == season_idx[i]:
                rest[side][i] = min(float(day[i] - last_day[team]), rest_cap)
                prev[side][i] = last_row[team]
            else:
                count[team] = 0
            games[side][i] = count[team]
            last_day[team], last_season[team], last_row[team] = day[i], season_idx[i], i
            count[team] += 1
    out = pd.DataFrame({"rest_a": rest["a"], "rest_b": rest["b"],
                        "season_games_a": games["a"], "season_games_b": games["b"]})
    for side in "ab":
        known = prev[side] >= 0
        from_lat = np.where(known, lat[prev[side]], np.nan)
        from_lon = np.where(known, lon[prev[side]], np.nan)
        out[f"travel_km_{side}"] = haversine_km(from_lat, from_lon, lat, lon)
        out[f"tz_shift_{side}"] = (lon - from_lon) / 15.0
    return out


def bullpen_load(a_idx: np.ndarray, b_idx: np.ndarray, day: np.ndarray, relief_a: np.ndarray,
                 relief_b: np.ndarray, *, n_teams: int, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Relief outs each side's bullpen recorded on the previous ``window`` slate days.

    Games of the same slate (the first game of a doubleheader) are not known
    yet and are not counted.  The value is missing when a game inside the
    window has no pitching lines that may be used.
    """
    n = len(a_idx)
    history: List[List[Tuple[int, float]]] = [[] for _ in range(n_teams)]
    out_a, out_b = np.zeros(n), np.zeros(n)
    for i in range(n):
        for out, team in ((out_a, a_idx[i]), (out_b, b_idx[i])):
            total = 0.0
            for d, outs in reversed(history[team]):
                if day[i] - d > window:
                    break
                if d < day[i]:
                    total += outs
            out[i] = total
        history[a_idx[i]].append((day[i], relief_a[i]))
        history[b_idx[i]].append((day[i], relief_b[i]))
    return out_a, out_b


def _tune_slow_lr(a_idx: np.ndarray, b_idx: np.ndarray, va: np.ndarray, vb: np.ndarray, season_idx: np.ndarray,
                  batch: np.ndarray, home_edge: np.ndarray, tune_mask: np.ndarray, cfg: BaseballConfig,
                  n_teams: int) -> float:
    """Slow update rate with the smallest squared run-prediction error on the
    burn-in seasons (later events are never touched)."""
    end = int(np.flatnonzero(tune_mask).max()) + 1
    scored = tune_mask[:end] & np.isfinite(va[:end])
    best, best_lr = np.inf, cfg.slow_lr_grid[len(cfg.slow_lr_grid) // 2]
    for lr in cfg.slow_lr_grid:
        r = offdef_pregame(a_idx[:end], b_idx[:end], va[:end], vb[:end], season_idx[:end], batch[:end],
                           home_edge[:end], lr=lr, carry=cfg.season_carry, n_entities=n_teams)
        exp_a = r["mean"] + r["off_a"] + r["def_b"] + r["home"] * home_edge[:end] / 2
        exp_b = r["mean"] + r["off_b"] + r["def_a"] - r["home"] * home_edge[:end] / 2
        mse = float(np.mean((va[:end][scored] - exp_a[scored]) ** 2 + (vb[:end][scored] - exp_b[scored]) ** 2))
        if mse < best:
            best, best_lr = mse, lr
    return best_lr


def _net_diff(r: Dict[str, np.ndarray]) -> np.ndarray:
    """Side A's net rating (offence minus what it concedes) minus side B's;
    missing until both sides have a rated game."""
    net = (r["off_a"] - r["def_a"]) - (r["off_b"] - r["def_b"])
    return np.where((r["n_a"] == 0) | (r["n_b"] == 0), np.nan, net)


def _starter_lines(pitching: pd.DataFrame, event_id: pd.Series, starter: pd.Series) -> np.ndarray:
    """``(n_events, 5)`` record of each event's named starter (NaN when he has
    no starting line on file), in :data:`_START_STATS` order."""
    starts = pitching[pitching["is_start"] == 1]
    line = pd.DataFrame({"event_id": event_id.to_numpy(), "pitcher_id": starter.to_numpy()}).merge(
        starts.drop_duplicates(["event_id", "pitcher_id"]), on=["event_id", "pitcher_id"], how="left")
    walks = line["bb"] + line["hbp"]
    return np.column_stack([line["outs"], line["runs"], 13.0 * line["hr"] + 3.0 * walks - 2.0 * line["so"],
                            line["so"] - line["bb"], line["bf"]]).astype(float)


def _starter_features(code_a: np.ndarray, code_b: np.ndarray, line_a: np.ndarray, line_b: np.ndarray,
                      left: np.ndarray, day: np.ndarray, batch: np.ndarray, started: np.ndarray,
                      usable: np.ndarray, cfg: BaseballConfig) -> pd.DataFrame:
    """What each side's actual starter had done in his prior starts.

    ``code_*`` index the starters (-1 unknown).  ``started`` marks games whose
    starters are known as of now; ``usable`` those whose pitching lines may be
    learned from.  A slate's starts are applied only when the slate is over.
    """
    n, n_pitchers = len(code_a), len(left)
    sums = np.zeros((n_pitchers, len(_START_STATS)))
    weight = np.zeros(n_pitchers)           # decayed number of starts behind ``sums``
    total = np.zeros(n_pitchers)            # all prior starts
    last_day = np.full(n_pitchers, -1, dtype=np.int64)
    league: Optional[np.ndarray] = None     # running league-average start
    cols = ("ra9", "fip", "kbb", "outs", "rest", "starts", "left")
    out = {f"{c}_{s}": np.full(n, np.nan) for c in cols for s in "ab"}
    prior = cfg.starter_prior_starts
    i = 0
    while i < n:
        j = i
        while j < n and batch[j] == batch[i]:
            j += 1
        for t in range(i, j):
            if not started[t]:
                continue
            for side, p in (("a", code_a[t]), ("b", code_b[t])):
                if p < 0:
                    continue
                out[f"starts_{side}"][t] = np.log1p(total[p])
                out[f"left_{side}"][t] = left[p]
                if last_day[p] >= 0:
                    out[f"rest_{side}"][t] = min(float(day[t] - last_day[p]), cfg.starter_rest_cap)
                if league is not None:
                    outs, runs, fip, kbb, bf = sums[p] + prior * league
                    out[f"ra9_{side}"][t] = 27.0 * runs / outs
                    out[f"fip_{side}"][t] = 3.0 * fip / outs
                    out[f"kbb_{side}"][t] = kbb / bf
                    out[f"outs_{side}"][t] = outs / (weight[p] + prior)
        slate: List[np.ndarray] = []
        for t in range(i, j):
            if not started[t]:
                continue
            for p, line in ((code_a[t], line_a[t]), (code_b[t], line_b[t])):
                if p < 0:
                    continue
                total[p] += 1
                last_day[p] = day[t]
                if usable[t] and np.isfinite(line).all():
                    sums[p] = cfg.starter_decay * sums[p] + line
                    weight[p] = cfg.starter_decay * weight[p] + 1.0
                    slate.append(line)
        if slate:
            mean = np.mean(slate, axis=0)
            league = mean if league is None else league + cfg.league_lr * (mean - league)
        i = j
    return pd.DataFrame(out)


def _pitching_features(store: Any, league: str, ev: pd.DataFrame, stats: pd.DataFrame, a_idx: np.ndarray,
                       b_idx: np.ndarray, day: np.ndarray, batch: np.ndarray, completed: np.ndarray,
                       usable: np.ndarray, n_teams: int, cfg: BaseballConfig) -> pd.DataFrame:
    """Bullpen workload (point-in-time) and actual-starter record (retrospective)."""
    n = len(ev)
    if not store.has_table(league, "pitching"):
        cols = ["pen_outs_a", "pen_outs_b", *[f"{c}_{s}" for c in ("ra9", "fip", "kbb", "outs", "rest", "starts", "left")
                                             for s in "ab"]]
        return pd.DataFrame({c: np.full(n, np.nan) for c in cols})
    pitching = store.read_table(league, "pitching")
    relief = (pitching["outs"] * (1 - pitching["is_start"])).groupby(
        [pitching["event_id"], pitching["team_id"]]).sum()
    relief_a = relief.reindex(pd.MultiIndex.from_arrays([ev["event_id"], ev["home_id"]])).to_numpy(dtype=float)
    relief_b = relief.reindex(pd.MultiIndex.from_arrays([ev["event_id"], ev["away_id"]])).to_numpy(dtype=float)
    pen_a, pen_b = bullpen_load(a_idx, b_idx, day, np.where(usable, relief_a, np.nan),
                                np.where(usable, relief_b, np.nan), n_teams=n_teams, window=cfg.bullpen_days)

    ids = pd.Index(pd.unique(pd.concat([stats["home_sp"], stats["away_sp"]]).dropna()))
    throws = pd.Series(dtype=object)
    if store.has_table(league, "pitchers"):
        throws = store.read_table(league, "pitchers").set_index("pitcher_id")["throws"]
    left = throws.reindex(ids).map({"L": 1.0, "R": 0.0}).to_numpy(dtype=float)
    out = _starter_features(
        ids.get_indexer(stats["home_sp"]), ids.get_indexer(stats["away_sp"]),
        _starter_lines(pitching, ev["event_id"], stats["home_sp"]),
        _starter_lines(pitching, ev["event_id"], stats["away_sp"]),
        left, day, batch, completed, usable, cfg)
    out["pen_outs_a"], out["pen_outs_b"] = pen_a, pen_b
    return out


def build_winner_frame(store: Any, league: str, *, now: Any = None,
                       cfg: Optional[BaseballConfig] = None) -> FeatureFrame:
    cfg = cfg or BaseballConfig()
    now_ts = pd.Timestamp(utcnow() if now is None else parse_ts(now))
    ev = store.events(league, with_results=True)
    ev = ev[~ev["stage"].isin(EXCLUDED_STAGES)]
    # "As of now": an event that starts after ``now`` is upcoming whatever the
    # store knows about it today, and its result and statistics are ignored.
    ev = ev.assign(
        completed=(ev["status"] == "final") & ev["winner_id"].notna() & (ev["start_time_utc"] < now_ts))
    upcoming = ev["status"].isin(["scheduled", "final"]) & (ev["start_time_utc"] >= now_ts)
    ev = ev[ev["completed"] | upcoming].copy()
    if ev.empty:
        raise ValueError(f"no events on file for {league}; run `sportspred ingest --league {league}` first")
    fmt = [json.loads(f) if isinstance(f, str) else {} for f in ev["format_json"]]
    ev["game_number"] = [int(f.get("game_number") or 1) for f in fmt]
    ev["day_game"] = [{"day": 1.0, "night": 0.0}.get(f.get("day_night"), np.nan) for f in fmt]
    # Playing order: date-only games of one date are ordered by game number.
    ev = ev.sort_values(["start_time_utc", "game_number", "event_id"]).reset_index(drop=True)
    completed = ev["completed"].to_numpy()

    stat_cols = ["resumed_on", "home_sp", "away_sp", *[f"{s}_{c}" for s in ("home", "away") for c in _BATTING]]
    if store.has_table(league, "game_stats"):
        stats = store.read_table(league, "game_stats", columns=["event_id", *stat_cols])
        stats = stats.set_index("event_id").reindex(ev["event_id"]).reset_index(drop=True)
    else:
        stats = pd.DataFrame({c: [None] * len(ev) for c in stat_cols})
    # A suspended game's result did not exist until it was completed later:
    # it is a target like any other game but never updates a rating.
    usable = completed & stats["resumed_on"].isna().to_numpy()

    teams = pd.Index(pd.unique(pd.concat([ev["home_id"], ev["away_id"]])))
    a_idx, b_idx = teams.get_indexer(ev["home_id"]), teams.get_indexer(ev["away_id"])
    season_num = pd.to_numeric(ev["season"], errors="coerce").astype(int)
    seasons_sorted = np.sort(season_num.unique())
    season_idx = np.searchsorted(seasons_sorted, season_num.to_numpy())
    day = slate_days(ev["start_time_utc"])
    batch = pd.factorize(day)[0]
    home_edge = np.where(ev["neutral_site"].fillna(0).astype(int).to_numpy() == 1, 0.0, 1.0)
    runs_a, runs_b = ev["home_score"].to_numpy(dtype=float), ev["away_score"].to_numpy(dtype=float)
    y = np.where(completed, (ev["winner_id"] == ev["home_id"]).astype(float), np.nan)
    score = np.where(usable, y, np.nan)
    margin = np.where(usable, runs_a - runs_b, np.nan)
    va, vb = np.where(usable, runs_a, np.nan), np.where(usable, runs_b, np.nan)

    # ---- update rates are tuned on the burn-in seasons only (first one is warm-up)
    burn = seasons_sorted[: cfg.burn_in_seasons]
    tune_mask = np.isin(season_num.to_numpy(), burn[1:]) & usable
    slow_lr = cfg.slow_lr_grid[len(cfg.slow_lr_grid) // 2]
    if tune_mask.sum() < 300:
        elo_params, elo_grid = EloParams(k=4.0, home_adv=20.0, carry=0.7), []
    else:
        elo_params, elo_grid = tune_elo(
            a_idx, b_idx, score, season_idx, batch, home_edge, tune_mask=tune_mask, margin=margin,
            k_grid=cfg.elo_k_grid, home_grid=cfg.elo_home_grid, carry_grid=cfg.elo_carry_grid,
            margin_grid=cfg.elo_margin_grid)
        slow_lr = _tune_slow_lr(a_idx, b_idx, va, vb, season_idx, batch, home_edge, tune_mask, cfg, len(teams))
    elo_a, elo_b, _ = elo_pregame(a_idx, b_idx, score, season_idx, batch, home_edge, elo_params,
                                  margin=margin, n_entities=len(teams))

    def rated(val_a: np.ndarray, val_b: np.ndarray, lr: float) -> Dict[str, np.ndarray]:
        return offdef_pregame(a_idx, b_idx, val_a, val_b, season_idx, batch, home_edge, lr=lr,
                              carry=cfg.season_carry, n_entities=len(teams))

    slow, fast = rated(va, vb, slow_lr), rated(va, vb, cfg.fast_lr)
    bat = {s: {c: pd.to_numeric(stats[f"{s}_{c}"], errors="coerce").to_numpy(dtype=float) for c in _BATTING}
           for s in ("home", "away")}
    bsr = {s: base_runs(b["ab"], b["h"], b["d"], b["t"], b["hr"], b["bb"] + b["hbp"]) for s, b in bat.items()}
    bsr_slow = rated(np.where(usable, bsr["home"], np.nan), np.where(usable, bsr["away"], np.nan), slow_lr)

    # A park is identified by its venue id; without one the host's id stands in.
    park = ev["venue_id"].where(ev["venue_id"].notna(), "home:" + ev["home_id"].astype(str)).to_numpy()
    factor = park_factors(season_idx, park, a_idx, b_idx, va + vb, n_seasons=cfg.park_seasons,
                          prior_games=cfg.park_prior_games)
    coords = np.array([city_coords(c) for c in ev["venue_city"]], dtype=float).reshape(len(ev), 2)
    sched = rest_and_travel(a_idx, b_idx, season_idx, day, coords[:, 0], coords[:, 1], n_teams=len(teams),
                            rest_cap=cfg.rest_cap_days)
    pitch = _pitching_features(store, league, ev, stats, a_idx, b_idx, day, batch, completed, usable,
                               len(teams), cfg)

    # Latest information any feature can contain: the last event of earlier slates.
    last_of_batch = ev["start_time_utc"].groupby(batch).transform("max")
    prev_batch_last = pd.Series(pd.unique(last_of_batch)).shift().to_numpy()
    cutoff = pd.Series(prev_batch_last[batch], index=ev.index)

    # Unplayed scheduled games either side still has before an upcoming event.
    pending = np.zeros(len(ev), dtype=int)
    seen_pending: Dict[str, int] = {}
    for i in np.flatnonzero(~completed):
        h, a = ev.at[i, "home_id"], ev.at[i, "away_id"]
        pending[i] = seen_pending.get(h, 0) + seen_pending.get(a, 0)
        seen_pending[h] = seen_pending.get(h, 0) + 1
        seen_pending[a] = seen_pending.get(a, 0) + 1

    net_slow, net_fast = _net_diff(slow), _net_diff(fast)
    df = pd.DataFrame({
        "event_id": ev["event_id"],
        "start_time_utc": ev["start_time_utc"],
        "block": season_num.to_numpy(),
        "season": ev["season"],
        "status": ev["status"],
        "stage": ev["stage"],
        "y": y,
        "side_a_id": ev["home_id"],
        "side_b_id": ev["away_id"],
        "info_cutoff_utc": pd.to_datetime(cutoff, utc=True),
        "home_edge": home_edge,
        "elo_a": elo_a,
        "elo_b": elo_b,
        "elo_diff": elo_a - elo_b,
        "net_slow_diff": net_slow,
        "net_fast_diff": net_fast,
        "form_diff": net_fast - net_slow,
        "bsr_slow_diff": _net_diff(bsr_slow),
        "park_factor": factor,
        "day_game": ev["day_game"],
        "rest_diff": sched["rest_a"] - sched["rest_b"],
        "travel_km_a": np.log1p(sched["travel_km_a"]),
        "travel_km_b": np.log1p(sched["travel_km_b"]),
        "tz_shift_b": sched["tz_shift_b"],
        "dh_game2": (ev["game_number"] > 1).astype(float),
        "pen_outs_a": pitch["pen_outs_a"],
        "pen_outs_b": pitch["pen_outs_b"],
        "is_playoff": (ev["stage"] == "playoffs").astype(float),
        "sp_ra9_diff": pitch["ra9_a"] - pitch["ra9_b"],
        "sp_fip_diff": pitch["fip_a"] - pitch["fip_b"],
        "sp_kbb_diff": pitch["kbb_a"] - pitch["kbb_b"],
        "sp_outs_diff": pitch["outs_a"] - pitch["outs_b"],
        "sp_rest_a": pitch["rest_a"], "sp_rest_b": pitch["rest_b"],
        "sp_starts_a": pitch["starts_a"], "sp_starts_b": pitch["starts_b"],
        "sp_left_a": pitch["left_a"], "sp_left_b": pitch["left_b"],
        "rest_a": sched["rest_a"], "rest_b": sched["rest_b"],
        "season_games_a": sched["season_games_a"], "season_games_b": sched["season_games_b"],
        "margin": np.where(completed, runs_a - runs_b, np.nan),
        "total_runs": np.where(completed, runs_a + runs_b, np.nan),
        "n_pending_prior": pending,
        "dq_no_team_stats": np.isnan(net_slow).astype(float),
        "dq_no_bullpen": pitch["pen_outs_a"].isna().to_numpy().astype(float),
    })

    limitations = [
        "The source names the pitcher who actually started each game. Probable starters as announced "
        "beforehand are not recorded and can change, so starter features are a retrospective "
        "reconstruction used only by candidates flagged as such; the selectable models do not know who starts.",
        "Games are dated, not timed: features use results of earlier dates only, and the second game of a "
        "doubleheader does not see the first.",
        "No weather, announced lineups, injuries or betting odds exist in the source; there is no market "
        "benchmark.",
        "Travel distance is city-level and missing for a club's first game of a season.",
    ]
    if completed.any():
        limitations.append(
            f"Results on file end on {ev.loc[completed, 'start_time_utc'].max():%Y-%m-%d}; ratings know "
            "nothing later.")
    if not store.has_table(league, "pitching"):
        limitations.append("No pitching lines on file: bullpen and starter features are missing.")
    return FeatureFrame(
        league=league, market="winner", outcome=OutcomeType.BINARY, classes=("away", "home"), df=df,
        feature_cols=list(WINNER_FEATURES), retrospective_cols=list(STARTER_FEATURES),
        feature_notes=dict(FEATURE_LABELS), limitations=limitations, point_in_time=True,
        extra={
            "elo_params": elo_params.__dict__, "elo_tuning_grid_size": len(elo_grid), "slow_lr": slow_lr,
            "fast_lr": cfg.fast_lr, "burn_in_seasons": [int(s) for s in burn],
            "tuning_note": "Elo and run-rating update rates were tuned on the burn-in seasons only.",
        },
    )
