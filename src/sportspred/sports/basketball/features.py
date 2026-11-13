"""Point-in-time basketball features for the game-winner market.

Every number in a row is computed from games on **earlier slates** only
(a slate is a Pacific/Honolulu calendar day), plus the public schedule.
Side A is the home team.

Feature families (the weight of each is learned in validation, not set here):

* opposition-adjusted team strength - Elo, and offence/defence efficiency per
  100 possessions where each game's surprise is measured against what the
  opponent normally allows;
* recent form versus long-term level - the same efficiency rating with a fast
  memory (about the last 7 games) and a slow one;
* how the game is likely to be played - the four factors (shooting, turnovers,
  offensive rebounding, free-throw rate), each offence set against the
  opposing defence;
* roster - quality of the players expected to take the floor, from who has
  actually been playing recently, and how much quality has recently gone
  missing;
* schedule - rest, back-to-backs, congestion, travel distance, time zones.

One retrospective column, ``roster_q_actual_diff``, uses the minutes actually
played in the game being predicted.  That is not knowable pregame; it exists
only to measure how much perfect availability information would be worth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ...core.enums import OutcomeType
from ...core.timeutil import DISPLAY_TZ_NAME, parse_ts, utcnow
from ...features.ratings import EloParams, elo_pregame, offdef_pregame, tune_elo
from ...features.schedule import team_schedule_features
from ..base import FeatureFrame
from .venues import city_coords

__all__ = ["BasketballConfig", "build_winner_frame", "roster_quality", "WINNER_FEATURES", "FEATURE_LABELS"]

EXCLUDED_STAGES = ("preseason", "allstar", "offseason")


@dataclass(frozen=True)
class BasketballConfig:
    burn_in_seasons: int = 3              # earliest seasons: rating warm-up and update-rate tuning only
    fast_lr: float = 0.25                 # "recent form": roughly the last 7 games
    slow_lr_grid: Tuple[float, ...] = (0.02, 0.04, 0.07)
    season_carry: float = 0.7
    roster_fast_lr: float = 0.3
    roster_slow_lr: float = 0.05
    rating_decay: float = 0.985           # per game played; a player's old games fade
    rating_prior_minutes: float = 250.0   # shrinkage of a player's per-minute rate to the league rate
    regulation_minutes: float = 240.0     # team minutes in a regulation game (200 for 40-minute games)


WINNER_FEATURES: List[str] = [
    "home_edge", "elo_diff", "net_slow_diff", "net_fast_diff", "form_diff",
    "efg_edge", "tov_edge", "orb_edge", "ftr_edge",
    "roster_q_diff", "absence_diff",
    "rest_diff", "b2b_a", "b2b_b", "games_7d_diff", "travel_km_a", "travel_km_b", "tz_shift_b",
    "is_playoff",
]

#: Travel inputs need a venue location.  The built-in city table covers the NBA and
#: most of the WNBA; for the college leagues it locates under a tenth of games and
#: cannot tell same-named cities apart, so there these inputs are not offered at all.
TRAVEL_FEATURES = ("travel_km_a", "travel_km_b", "tz_shift_b")


def winner_features(league: str) -> List[str]:
    return [c for c in WINNER_FEATURES if not (league in THIN_HISTORY_LEAGUES and c in TRAVEL_FEATURES)]


FEATURE_LABELS: Dict[str, str] = {
    "home_edge": "home court",
    "elo_diff": "opponent-adjusted results (Elo)",
    "net_slow_diff": "long-term opponent-adjusted efficiency",
    "net_fast_diff": "recent (about 7 games) opponent-adjusted efficiency",
    "form_diff": "recent form relative to long-term level",
    "efg_edge": "shooting matchup (offence vs opposing defence)",
    "tov_edge": "turnover matchup",
    "orb_edge": "offensive-rebounding matchup",
    "ftr_edge": "free-throw-rate matchup",
    "roster_q_diff": "quality of the expected rotation",
    "absence_diff": "quality recently missing from the rotation",
    "rest_diff": "rest days",
    "b2b_a": "home side on a back-to-back",
    "b2b_b": "away side on a back-to-back",
    "games_7d_diff": "schedule congestion (games in 7 days)",
    "travel_km_a": "home side travel",
    "travel_km_b": "away side travel",
    "tz_shift_b": "away side time-zone change",
    "is_playoff": "playoff game",
    "roster_q_actual_diff": "rotation that actually played (retrospective)",
}


def slate_batches(start_time_utc: pd.Series) -> np.ndarray:
    """Batch label = Pacific/Honolulu calendar date, so games on one slate
    never inform each other's features."""
    days = start_time_utc.dt.tz_convert(DISPLAY_TZ_NAME).dt.strftime("%Y-%m-%d")
    return pd.factorize(days, sort=False)[0]


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    out[~np.isfinite(out)] = np.nan
    return out


def _game_stats(ev: pd.DataFrame, team_box: pd.DataFrame) -> pd.DataFrame:
    """Per event: efficiency and four factors for each side (NaN when no box score)."""
    cols = ["pts", "fgm", "fga", "fg3m", "fta", "oreb", "dreb", "tov"]
    tb = team_box[["event_id", "team_id", *cols]]
    home = ev[["event_id", "home_id"]].merge(
        tb, left_on=["event_id", "home_id"], right_on=["event_id", "team_id"], how="left")
    away = ev[["event_id", "away_id"]].merge(
        tb, left_on=["event_id", "away_id"], right_on=["event_id", "team_id"], how="left")
    h = {c: home[c].to_numpy(dtype=float) for c in cols}
    a = {c: away[c].to_numpy(dtype=float) for c in cols}
    poss_h = h["fga"] + 0.44 * h["fta"] - h["oreb"] + h["tov"]
    poss_a = a["fga"] + 0.44 * a["fta"] - a["oreb"] + a["tov"]
    poss = (poss_h + poss_a) / 2.0
    poss[~(poss > 30)] = np.nan
    out = pd.DataFrame({"event_id": ev["event_id"].to_numpy(), "poss": poss})
    out["eff_a"] = 100.0 * _safe_div(h["pts"], poss)
    out["eff_b"] = 100.0 * _safe_div(a["pts"], poss)
    out["efg_a"] = _safe_div(h["fgm"] + 0.5 * h["fg3m"], h["fga"])
    out["efg_b"] = _safe_div(a["fgm"] + 0.5 * a["fg3m"], a["fga"])
    out["tov_a"] = _safe_div(h["tov"], poss_h)
    out["tov_b"] = _safe_div(a["tov"], poss_a)
    out["orb_a"] = _safe_div(h["oreb"], h["oreb"] + a["dreb"])
    out["orb_b"] = _safe_div(a["oreb"], a["oreb"] + h["dreb"])
    out["ftr_a"] = _safe_div(h["fta"], h["fga"])
    out["ftr_b"] = _safe_div(a["fta"], a["fga"])
    return out


def _roster_features(ev: pd.DataFrame, player_box: Optional[pd.DataFrame], batch: np.ndarray,
                     season_idx: np.ndarray, completed: np.ndarray, cfg: BasketballConfig) -> pd.DataFrame:
    """Expected-rotation quality per side before each game, plus the
    retrospective quality of the rotation that actually played."""
    n = len(ev)
    cols = ("roster_q_a", "roster_q_b", "absence_a", "absence_b", "roster_q_actual_a", "roster_q_actual_b")
    out = {c: np.full(n, np.nan) for c in cols}
    if player_box is None or player_box.empty:
        empty = pd.DataFrame(out)
        empty.attrs["rosters"], empty.attrs["league_rate"] = {}, None
        return empty
    pb = player_box
    stats = {c: pb[c].fillna(0.0).to_numpy(dtype=float) for c in
             ("pts", "fgm", "fga", "fta", "ftm", "oreb", "dreb", "stl", "ast", "blk", "pf", "tov")}
    # Game Score (Hollinger): a published single-number box-score summary.
    # It only defines the input; how much roster quality matters is learned.
    gs = (stats["pts"] + 0.4 * stats["fgm"] - 0.7 * stats["fga"] - 0.4 * (stats["fta"] - stats["ftm"])
          + 0.7 * stats["oreb"] + 0.3 * stats["dreb"] + stats["stl"] + 0.7 * stats["ast"]
          + 0.7 * stats["blk"] - 0.4 * stats["pf"] - stats["tov"])
    minutes = pb["minutes"].fillna(0.0).to_numpy(dtype=float)
    pidx, _ = pd.factorize(pb["player_id"])
    groups = pb.groupby(["event_id", "team_id"], sort=False).indices
    gs_sum = np.zeros(pidx.max() + 1)
    min_sum = np.zeros(pidx.max() + 1)
    league_rate = 0.4                       # replaced by the running league rate after the first slate
    rate_num = rate_den = 0.0
    state: Dict[str, Dict[int, List[float]]] = {}
    team_season: Dict[str, int] = {}
    prior, decay = cfg.rating_prior_minutes, cfg.rating_decay
    lr_f, lr_s = cfg.roster_fast_lr, cfg.roster_slow_lr
    eids = ev["event_id"].to_numpy()
    homes, aways = ev["home_id"].to_numpy(), ev["away_id"].to_numpy()

    def rating(p: int) -> float:
        return (gs_sum[p] + prior * league_rate) / (min_sum[p] + prior)

    # Results of a slate are applied only when the slate is over.  A player's
    # rating is shared by every team that still carries him in its rotation
    # state (after a trade, say), so applying a game's box score at once would
    # let it reach other games of the same slate - including its own opponent.
    pending: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, Any]] = []
    last_played: Dict[int, Any] = {}        # player -> start of the last game he appeared in
    starts = ev["start_time_utc"].to_numpy()
    # The rotation behind each team's NEXT game: taken at its first row without a result.
    # (The loop goes on through the rest of the schedule, where a new season empties the
    # state; the state after the last row is therefore not the one a prediction needs.)
    snapshot: Dict[str, Tuple[int, Dict[int, List[float]]]] = {}

    def flush() -> None:
        nonlocal league_rate, rate_num, rate_den
        for team, players, mins, scores, when in pending:
            st = state[team]
            for p, m in zip(players.tolist(), mins.tolist()):
                if m > 0:
                    last_played[p] = when
            played = dict(zip(players.tolist(), mins.tolist()))
            for p in list(st):
                m = played.pop(p, 0.0)
                v = st[p]
                v[0] += lr_f * (m - v[0])
                v[1] += lr_s * (m - v[1])
                if m == 0.0 and v[0] < 0.2 and v[1] < 0.2:
                    del st[p]
            for p, m in played.items():
                if m > 0:
                    st[p] = [lr_f * m, lr_s * m]
            for p, m, g in zip(players, mins, scores):
                if m > 0:
                    gs_sum[p] = decay * gs_sum[p] + g
                    min_sum[p] = decay * min_sum[p] + m
                    rate_num += g
                    rate_den += m
        pending.clear()
        if rate_den > 0:
            league_rate += 0.05 * (rate_num / rate_den - league_rate)
            rate_num = rate_den = 0.0

    for i in range(n):
        if i and batch[i] != batch[i - 1]:
            flush()
        for side, team in (("a", homes[i]), ("b", aways[i])):
            if team_season.get(team) != season_idx[i]:
                state[team] = {}            # rosters change between seasons: start again
                team_season[team] = season_idx[i]
            st = state[team]
            if not completed[i] and team not in snapshot:
                snapshot[team] = (int(season_idx[i]), {p: list(v) for p, v in st.items()})
            weight = sum(v[0] for v in st.values())
            if weight >= 40.0:
                out[f"roster_q_{side}"][i] = sum(v[0] * rating(p) for p, v in st.items()) / weight
                out[f"absence_{side}"][i] = sum(
                    max(0.0, v[1] - v[0]) * (rating(p) - league_rate) for p, v in st.items()
                ) / cfg.regulation_minutes
            rows = groups.get((eids[i], team))
            if rows is None or not completed[i]:
                continue
            mins = minutes[rows]
            total = mins.sum()
            if total <= 0:
                continue
            players = pidx[rows]
            out[f"roster_q_actual_{side}"][i] = float(sum(m * rating(p) for p, m in zip(players, mins)) / total)
            pending.append((team, players, mins, gs[rows], starts[i]))
    flush()
    names = dict(zip(pidx.tolist(), pb["player_name"].tolist()))
    ids = dict(zip(pidx.tolist(), pb["player_id"].tolist()))

    def seen(p: int) -> Optional[str]:
        when = last_played.get(p)
        if when is None:
            return None
        ts = pd.Timestamp(when)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")

    rosters, roster_season = {}, {}
    for team, st in state.items():
        season, players = snapshot.get(team, (team_season.get(team, -1), st))
        roster_season[team] = int(season)
        rosters[team] = [{"player_id": ids[p], "name": names[p], "exp_min": round(v[0], 2),
                          "role_min": round(v[1], 2), "rating": round(float(rating(p)), 4),
                          "last_played_utc": seen(p)} for p, v in players.items()]
    frame = pd.DataFrame(out)
    frame.attrs["rosters"] = rosters
    frame.attrs["roster_season_idx"] = roster_season
    frame.attrs["league_rate"] = float(league_rate)
    return frame


def roster_quality(players: List[Dict[str, Any]], league_rate: float, regulation_minutes: float,
                   out: Sequence[str] = (), back: Sequence[str] = ()) -> Tuple[float, float]:
    """``(roster_q, absence)`` of a rotation, optionally with named players
    ruled out (their minutes go to the rest of the rotation in proportion) or
    confirmed back (they return to their longer-run role)."""
    out_set, back_set = set(out), set(back)
    weight = quality = absence = 0.0
    for p in players:
        w = 0.0 if p["player_id"] in out_set else (
            max(p["exp_min"], p["role_min"]) if p["player_id"] in back_set else p["exp_min"])
        weight += w
        quality += w * p["rating"]
        absence += max(0.0, p["role_min"] - w) * (p["rating"] - league_rate)
    if weight < 40.0:
        return float("nan"), float("nan")
    return quality / weight, absence / regulation_minutes


#: Leagues whose schedules mix in opponents from outside the league proper.
THIN_HISTORY_LEAGUES = frozenset({"ncaam", "ncaaw"})
THIN_HISTORY_GAMES = 10
SEGMENT_THIN = "a side with fewer than 10 games on file in the previous year (mostly non-Division-I opponents)"
SEGMENT_ESTABLISHED = "both sides established"


def _games_in_prior_year(ev: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """How many games each side of each event had on the schedule in the 365
    days before it (earlier days only), from the list of events alone."""
    day = (ev["start_time_utc"].dt.tz_convert(DISPLAY_TZ_NAME).dt.tz_localize(None).dt.normalize()
           - pd.Timestamp("1970-01-01")).dt.days.to_numpy()
    home, away = ev["home_id"].to_numpy(), ev["away_id"].to_numpy()
    days_of: Dict[str, List[int]] = {}
    for h, a, d in zip(home, away, day):
        days_of.setdefault(h, []).append(int(d))
        days_of.setdefault(a, []).append(int(d))
    sorted_days = {team: np.sort(np.asarray(v)) for team, v in days_of.items()}

    def count(teams: np.ndarray) -> np.ndarray:
        out = np.zeros(len(teams), dtype=int)
        for i, team in enumerate(teams):
            ds = sorted_days[team]
            out[i] = int(np.searchsorted(ds, day[i], side="left") - np.searchsorted(ds, day[i] - 365, side="left"))
        return out

    return count(home), count(away)


def build_winner_frame(store: Any, league: str, *, now: Any = None,
                       cfg: Optional[BasketballConfig] = None) -> FeatureFrame:
    cfg = cfg or BasketballConfig()
    now_ts = pd.Timestamp(utcnow() if now is None else parse_ts(now))
    ev = store.events(league, with_results=True)
    ev = ev[~ev["stage"].isin(EXCLUDED_STAGES)]
    # "As of now": an event that starts after ``now`` is upcoming whatever the
    # store knows about it today, and its result and statistics are ignored.
    # That makes a historical rebuild (now = some past instant) see exactly
    # what was knowable then, which the leakage tests rely on.
    completed_mask = (ev["status"] == "final") & ev["winner_id"].notna() & (ev["start_time_utc"] < now_ts)
    upcoming_mask = ev["status"].isin(["scheduled", "final"]) & (ev["start_time_utc"] >= now_ts)
    ev = ev[completed_mask | upcoming_mask].sort_values(["start_time_utc", "event_id"]).reset_index(drop=True)
    if ev.empty:
        raise ValueError(f"no events on file for {league}; run `sportspred ingest --league {league}` first")
    completed = ((ev["status"] == "final") & ev["winner_id"].notna() & (ev["start_time_utc"] < now_ts)).to_numpy()

    teams = pd.Index(pd.unique(pd.concat([ev["home_id"], ev["away_id"]])))
    a_idx = teams.get_indexer(ev["home_id"])
    b_idx = teams.get_indexer(ev["away_id"])
    season_num = pd.to_numeric(ev["season"], errors="coerce").astype(int)
    seasons_sorted = np.sort(season_num.unique())
    season_idx = np.searchsorted(seasons_sorted, season_num.to_numpy())
    batch = slate_batches(ev["start_time_utc"])
    home_edge = np.where(ev["neutral_site"].fillna(0).astype(int).to_numpy() == 1, 0.0, 1.0)
    y = np.where(completed, (ev["winner_id"] == ev["home_id"]).astype(float), np.nan)
    margin = np.where(completed, ev["home_score"].to_numpy(dtype=float) - ev["away_score"].to_numpy(dtype=float),
                      np.nan)

    # ---- update rates are tuned on the burn-in seasons only (first one is warm-up)
    burn = seasons_sorted[: cfg.burn_in_seasons]
    tune_mask = np.isin(season_num.to_numpy(), burn[1:]) & completed
    if tune_mask.sum() < 300:
        elo_params, elo_grid = EloParams(margin_scale=8.0), []
    else:
        elo_params, elo_grid = tune_elo(a_idx, b_idx, y, season_idx, batch, home_edge, tune_mask=tune_mask,
                                        margin=margin, margin_grid=(None, 6.0, 12.0))
    elo_a, elo_b, played_a = elo_pregame(a_idx, b_idx, y, season_idx, batch, home_edge, elo_params,
                                         margin=margin, n_entities=len(teams))

    has_box = store.has_table(league, "team_box")
    stats = _game_stats(ev, store.read_table(league, "team_box")) if has_box else None
    feats: Dict[str, np.ndarray] = {}
    slow_lr = cfg.slow_lr_grid[len(cfg.slow_lr_grid) // 2]
    if stats is not None:
        va = np.where(completed, stats["eff_a"].to_numpy(), np.nan)
        vb = np.where(completed, stats["eff_b"].to_numpy(), np.nan)
        if tune_mask.sum() >= 300:
            end = int(np.flatnonzero(tune_mask).max()) + 1
            best = np.inf
            for lr in cfg.slow_lr_grid:
                r = offdef_pregame(a_idx[:end], b_idx[:end], va[:end], vb[:end], season_idx[:end], batch[:end],
                                   home_edge[:end], lr=lr, carry=cfg.season_carry, n_entities=len(teams))
                exp_a = r["mean"] + r["off_a"] + r["def_b"] + r["home"] * home_edge[:end] / 2
                exp_b = r["mean"] + r["off_b"] + r["def_a"] - r["home"] * home_edge[:end] / 2
                m = tune_mask[:end] & np.isfinite(va[:end])
                mse = float(np.mean((va[:end][m] - exp_a[m]) ** 2 + (vb[:end][m] - exp_b[m]) ** 2))
                if mse < best:
                    best, slow_lr = mse, lr
        slow = offdef_pregame(a_idx, b_idx, va, vb, season_idx, batch, home_edge, lr=slow_lr,
                              carry=cfg.season_carry, n_entities=len(teams))
        fast = offdef_pregame(a_idx, b_idx, va, vb, season_idx, batch, home_edge, lr=cfg.fast_lr,
                              carry=cfg.season_carry, n_entities=len(teams))
        feats["net_slow_diff"] = (slow["off_a"] - slow["def_a"]) - (slow["off_b"] - slow["def_b"])
        feats["net_fast_diff"] = (fast["off_a"] - fast["def_a"]) - (fast["off_b"] - fast["def_b"])
        feats["form_diff"] = feats["net_fast_diff"] - feats["net_slow_diff"]
        no_stats = (slow["n_a"] == 0) | (slow["n_b"] == 0)
        for key in ("net_slow_diff", "net_fast_diff", "form_diff"):
            feats[key] = np.where(no_stats, np.nan, feats[key])
        for factor, sign in (("efg", 1.0), ("tov", -1.0), ("orb", 1.0), ("ftr", 1.0)):
            fa = np.where(completed, stats[f"{factor}_a"].to_numpy(), np.nan)
            fb = np.where(completed, stats[f"{factor}_b"].to_numpy(), np.nan)
            r = offdef_pregame(a_idx, b_idx, fa, fb, season_idx, batch, home_edge, lr=slow_lr,
                               carry=cfg.season_carry, n_entities=len(teams))
            # A's offence against B's defence, minus B's offence against A's.
            edge = (r["off_a"] + r["def_b"]) - (r["off_b"] + r["def_a"])
            feats[f"{factor}_edge"] = np.where(no_stats, np.nan, sign * edge)
        feats["dq_no_team_stats"] = no_stats.astype(float)
    else:
        for key in ("net_slow_diff", "net_fast_diff", "form_diff", "efg_edge", "tov_edge", "orb_edge", "ftr_edge"):
            feats[key] = np.full(len(ev), np.nan)
        feats["dq_no_team_stats"] = np.ones(len(ev))

    player_box = store.read_table(league, "player_box") if store.has_table(league, "player_box") else None
    roster = _roster_features(ev, player_box, batch, season_idx, completed, cfg)

    coords = [city_coords(c) for c in ev["venue_city"]]
    sched_in = ev[["event_id", "start_time_utc", "season", "home_id", "away_id", "neutral_site"]].copy()
    sched_in["lat"] = [c[0] for c in coords]
    sched_in["lon"] = [c[1] for c in coords]
    sched = team_schedule_features(sched_in).set_index("event_id").reindex(ev["event_id"]).reset_index()

    # Latest information any feature can contain: the last event of earlier slates.
    starts = ev["start_time_utc"]
    last_of_batch = starts.groupby(batch).transform("max")
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
        "games_played_a": played_a,
        "is_playoff": ev["stage"].isin(["playoffs", "play_in"]).astype(float),
        "margin": margin,
        "total_points": np.where(completed, ev["home_score"].to_numpy(dtype=float)
                                 + ev["away_score"].to_numpy(dtype=float), np.nan),
        "n_pending_prior": pending,
    })
    for key, val in feats.items():
        df[key] = val
    df["roster_q_diff"] = roster["roster_q_a"] - roster["roster_q_b"]
    df["absence_diff"] = roster["absence_a"] - roster["absence_b"]
    df["roster_q_actual_diff"] = roster["roster_q_actual_a"] - roster["roster_q_actual_b"]
    for col in ("roster_q_a", "roster_q_b", "absence_a", "absence_b"):
        df[col] = roster[col]
    df["rest_diff"] = sched["rest_days_a"] - sched["rest_days_b"]
    df["b2b_a"], df["b2b_b"] = sched["b2b_a"], sched["b2b_b"]
    df["games_7d_diff"] = sched["games_7d_a"] - sched["games_7d_b"]
    df["travel_km_a"] = np.log1p(sched["travel_km_a"])
    df["travel_km_b"] = np.log1p(sched["travel_km_b"])
    df["tz_shift_b"] = sched["tz_shift_b"]
    df["rest_days_a"], df["rest_days_b"] = sched["rest_days_a"], sched["rest_days_b"]
    df["dq_no_roster"] = df["roster_q_diff"].isna().astype(float)
    if league in THIN_HISTORY_LEAGUES:
        # College schedules include games against lower-division opponents, which are close
        # to certain and flatter a pooled accuracy figure.  This label - knowable before the
        # game, never a model input - lets validation report the two kinds separately.
        fewest = np.minimum(*_games_in_prior_year(ev))
        df["segment"] = np.where(fewest < THIN_HISTORY_GAMES, SEGMENT_THIN, SEGMENT_ESTABLISHED)

    limitations = [
        "No pregame injury reports or confirmed lineups exist in the historical data, so the backtest "
        "cannot include late availability news. Roster features infer availability only from who played "
        "in earlier games.",
        "Travel distance is city-level and missing for venues outside the built-in city table.",
        "Betting odds are not an input and no timestamped market benchmark is available from this source.",
    ]
    if player_box is None:
        limitations.append("No player box scores on file: roster features are missing.")
    return FeatureFrame(
        league=league, market="winner", outcome=OutcomeType.BINARY, classes=("away", "home"), df=df,
        feature_cols=winner_features(league), retrospective_cols=["roster_q_actual_diff"],
        feature_notes=dict(FEATURE_LABELS), limitations=limitations, point_in_time=True,
        extra={
            "elo_params": elo_params.__dict__, "elo_tuning_grid_size": len(elo_grid), "slow_lr": slow_lr,
            "fast_lr": cfg.fast_lr, "burn_in_seasons": [int(s) for s in burn],
            "tuning_note": "Elo and efficiency update rates were tuned on the burn-in seasons only.",
            "rosters": roster.attrs.get("rosters", {}), "league_rate": roster.attrs.get("league_rate"),
            # Season (block) of the game each team's rotation snapshot was taken for.
            "roster_block": {team: int(seasons_sorted[idx]) for team, idx in
                             (roster.attrs.get("roster_season_idx") or {}).items()
                             if 0 <= idx < len(seasons_sorted)},
            "regulation_minutes": cfg.regulation_minutes,
        },
    )
