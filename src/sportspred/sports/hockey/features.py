"""Point-in-time NHL features for the game-winner and regulation-result markets.

Every number in a row is computed from games on **earlier slates** only
(a slate is a Pacific/Honolulu calendar day, which is one NHL game night),
plus the public schedule.  Side A is the home team.

Feature families (the weight of each is learned in validation, not set here):

* opposition-adjusted results - Elo on wins, and goals for/against where each
  game's surprise is measured against what the opponent normally allows, with
  a slow memory and a fast one (about the last 9 games).  Shootout "goals" are
  not counted as goals;
* shots - opposition-adjusted shots on goal, and five-on-five expected goals
  from a shot-location model fitted on the burn-in seasons only;
* special teams - opposition-adjusted power-play goals for and against;
* goaltending - shrunk save percentage of each team's recent primary goalie;
* schedule - rest, back-to-backs, congestion, travel, home ice.

One retrospective column, ``goalie_actual_diff``, uses the goalies who really
started the game being predicted.  That is known only at puck drop; the column
exists to measure what a confirmed starter is worth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ...core.enums import OutcomeType
from ...core.timeutil import parse_ts, utcnow
from ...features.ratings import EloParams, elo_expected, elo_pregame, offdef_pregame, tune_elo
from ...features.schedule import team_schedule_features
from ...validation.backtest import slate_dates
from ..base import FeatureFrame
from .goalies import GoalieConfig, goalie_pregame
from .shots import game_shot_stats
from .teams import team_coords

__all__ = ["HockeyConfig", "build_frame", "WINNER_FEATURES", "THREE_WAY_FEATURES", "FEATURE_LABELS", "MARKETS"]

STAGES = ("regular", "playoffs")
MARKETS = ("winner", "winner_3way")


@dataclass(frozen=True)
class HockeyConfig:
    burn_in_seasons: int = 3              # earliest seasons: rating warm-up, update-rate tuning, shot model
    fast_lr: float = 0.2                  # "recent form": roughly the last 9 games
    lr_grid: Tuple[float, ...] = (0.01, 0.02, 0.04, 0.08)   # slow update rates tried on the burn-in seasons
    season_carry: float = 0.7
    stale_days: float = 4.0               # an unfinished game this recent still counts as pending
    min_shot_model_rows: int = 20000
    goalie: GoalieConfig = field(default_factory=GoalieConfig)


WINNER_FEATURES: List[str] = [
    "home_edge", "elo_diff", "gd_slow_diff", "gd_fast_diff", "form_diff",
    "xg5_diff", "sog_diff", "pp_diff", "goalie_diff",
    "rest_diff", "b2b_a", "b2b_b", "games_7d_diff", "travel_km_a", "travel_km_b", "tz_shift_b",
    "is_playoff",
]
#: The regulation-result market also needs to know how close and how
#: low-scoring a game is expected to be: that is what drives ties.
THREE_WAY_FEATURES: List[str] = WINNER_FEATURES + ["elo_gap_abs", "exp_total_goals"]

FEATURE_LABELS: Dict[str, str] = {
    "home_edge": "home ice",
    "elo_diff": "opponent-adjusted results (Elo)",
    "gd_slow_diff": "long-term opponent-adjusted goal difference",
    "gd_fast_diff": "recent (about 9 games) opponent-adjusted goal difference",
    "form_diff": "recent form relative to long-term level",
    "xg5_diff": "five-on-five expected-goal difference (shot quality)",
    "sog_diff": "shots-on-goal difference",
    "pp_diff": "special teams (power-play goals for and against)",
    "goalie_diff": "primary goalie's save percentage",
    "rest_diff": "rest days",
    "b2b_a": "home side on a back-to-back",
    "b2b_b": "away side on a back-to-back",
    "games_7d_diff": "schedule congestion (games in 7 days)",
    "travel_km_a": "home side travel",
    "travel_km_b": "away side travel",
    "tz_shift_b": "away side time-zone change",
    "is_playoff": "playoff game",
    "elo_gap_abs": "how lopsided the matchup is",
    "exp_total_goals": "expected goals in the game (both teams)",
    "goalie_actual_diff": "goalies who actually started (retrospective)",
}


@dataclass
class _Ratings:
    """Arrays parallel to the event rows that every rating run shares."""

    a_idx: np.ndarray
    b_idx: np.ndarray
    season_idx: np.ndarray
    batch: np.ndarray
    home_edge: np.ndarray
    tune_mask: np.ndarray
    n_teams: int
    carry: float

    def run(self, val_a: np.ndarray, val_b: np.ndarray, lr: float, end: Optional[int] = None) -> Dict[str, np.ndarray]:
        s = slice(0, end)
        return offdef_pregame(self.a_idx[s], self.b_idx[s], val_a[s], val_b[s], self.season_idx[s], self.batch[s],
                              self.home_edge[s], lr=lr, carry=self.carry, n_entities=self.n_teams)

    def tune_lr(self, val_a: np.ndarray, val_b: np.ndarray, grid: Sequence[float]) -> float:
        """Update rate with the smallest one-game-ahead squared error on the
        burn-in seasons (the middle of the grid when there is too little to
        tune on).  Later events are never touched."""
        scored = self.tune_mask & np.isfinite(val_a) & np.isfinite(val_b)
        best_lr, best = grid[len(grid) // 2], np.inf
        if scored.sum() < 300:
            return best_lr
        end = int(np.flatnonzero(self.tune_mask).max()) + 1
        m = scored[:end]
        for lr in grid:
            r = self.run(val_a, val_b, lr, end)
            home = r["home"] * self.home_edge[:end] / 2.0
            err_a = val_a[:end] - (r["mean"] + r["off_a"] + r["def_b"] + home)
            err_b = val_b[:end] - (r["mean"] + r["off_b"] + r["def_a"] - home)
            mse = float(np.mean(err_a[m] ** 2 + err_b[m] ** 2))
            if mse < best:
                best, best_lr = mse, lr
        return best_lr


def _net_diff(r: Dict[str, np.ndarray]) -> np.ndarray:
    """Side A's rating (produced minus conceded) minus side B's; NaN until
    both sides have a game with this statistic."""
    net = (r["off_a"] - r["def_a"]) - (r["off_b"] - r["def_b"])
    return np.where((r["n_a"] == 0) | (r["n_b"] == 0), np.nan, net)


def _events_as_of(store: Any, league: str, now_ts: pd.Timestamp, stale_days: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """``(events, stale)``: completed and upcoming events as of ``now``, and
    recent events that have started but have no result on file.

    An event that starts after ``now`` is upcoming whatever the store knows
    about it today; its result and statistics are ignored."""
    ev = store.events(league, with_results=True)
    ev = ev[ev["stage"].isin(STAGES)]
    start = ev["start_time_utc"]
    completed = (ev["status"] == "final") & ev["winner_id"].notna() & (start < now_ts)
    upcoming = ev["status"].isin(["scheduled", "final"]) & (start >= now_ts)
    stale = ev["status"].isin(["scheduled", "live"]) & (start <= now_ts) & (start > now_ts - pd.Timedelta(days=stale_days))
    frame = ev[completed | upcoming].sort_values(["start_time_utc", "event_id"]).reset_index(drop=True)
    return frame, ev[stale]


def _goals(ev: pd.DataFrame, completed: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Goals scored by each side, without the goal credited for winning a
    shootout.  NaN for unplayed games and for the few games whose shootout
    flag is disputed between sources (when the source has no such flag at
    all, the final score is used)."""
    home = ev["home_score"].to_numpy(dtype=float)
    away = ev["away_score"].to_numpy(dtype=float)
    shootout = ev["went_shootout"].to_numpy(dtype=float)
    if np.isnan(shootout[completed]).all():
        usable = completed
    else:
        usable = completed & ~np.isnan(shootout)
        level = np.minimum(home, away)
        home, away = np.where(shootout == 1, level, home), np.where(shootout == 1, level, away)
    return np.where(usable, home, np.nan), np.where(usable, away, np.nan)


def _schedule(store: Any, league: str, ev: pd.DataFrame) -> pd.DataFrame:
    """Rest and travel per side.  A game is located at the home club's arena
    city; games away from the regular arena have no known location."""
    sched = ev[["event_id", "start_time_utc", "season", "home_id", "away_id", "neutral_site"]].copy()
    sched["lat"] = sched["lon"] = np.nan
    if store.has_table(league, "games"):
        codes = store.read_table(league, "games").set_index("event_id")["home_abbr"].reindex(ev["event_id"])
        coords = [team_coords(code) for code in codes]
        at_home = ev["neutral_site"].fillna(0).to_numpy() != 1
        sched["lat"] = np.where(at_home, [c[0] for c in coords], np.nan)
        sched["lon"] = np.where(at_home, [c[1] for c in coords], np.nan)
    return team_schedule_features(sched).set_index("event_id").reindex(ev["event_id"]).reset_index()


def _pending_prior(ev: pd.DataFrame, completed: np.ndarray, stale: pd.DataFrame) -> np.ndarray:
    """Unfinished earlier games of either side before each upcoming event."""
    seen: Dict[str, int] = {}
    for team in [*stale["home_id"], *stale["away_id"]]:
        seen[team] = seen.get(team, 0) + 1
    pending = np.zeros(len(ev), dtype=int)
    home, away = ev["home_id"].to_numpy(), ev["away_id"].to_numpy()
    for i in np.flatnonzero(~completed):
        pending[i] = seen.get(home[i], 0) + seen.get(away[i], 0)
        for team in (home[i], away[i]):
            seen[team] = seen.get(team, 0) + 1
    return pending


def _info_cutoff(start: pd.Series, batch: np.ndarray) -> pd.Series:
    """Latest instant any feature of a row can contain: the last start of the
    slates before its own."""
    last_of_batch = start.groupby(batch).transform("max")
    previous = pd.Series(pd.unique(last_of_batch)).shift().to_numpy()
    return pd.to_datetime(pd.Series(previous[batch], index=start.index), utc=True)


def _target(market: str, ev: pd.DataFrame, completed: np.ndarray) -> np.ndarray:
    if market == "winner":
        return np.where(completed, (ev["winner_id"] == ev["home_id"]).astype(float), np.nan)
    reg_home = ev["reg_home_score"].to_numpy(dtype=float)
    reg_away = ev["reg_away_score"].to_numpy(dtype=float)
    result = np.where(reg_home > reg_away, 0.0, np.where(reg_home == reg_away, 1.0, 2.0))
    return np.where(completed & ~np.isnan(reg_home) & ~np.isnan(reg_away), result, np.nan)


def build_frame(store: Any, league: str, market: str = "winner", *, now: Any = None,
                cfg: Optional[HockeyConfig] = None) -> FeatureFrame:
    """Feature frame for ``market`` ("winner" includes overtime and shootout;
    "winner_3way" is the result after regulation: home, draw or away)."""
    if market not in MARKETS:
        raise NotImplementedError(f"hockey market {market!r} has no feature builder")
    cfg = cfg or HockeyConfig()
    now_ts = pd.Timestamp(utcnow() if now is None else parse_ts(now))
    ev, stale = _events_as_of(store, league, now_ts, cfg.stale_days)
    if ev.empty:
        raise ValueError(f"no events on file for {league}; run `sportspred ingest --league {league}` first")
    completed = ((ev["status"] == "final") & ev["winner_id"].notna() & (ev["start_time_utc"] < now_ts)).to_numpy()

    teams = pd.Index(pd.unique(pd.concat([ev["home_id"], ev["away_id"]])))
    season_num = pd.to_numeric(ev["season"], errors="coerce").astype(int).to_numpy()
    seasons_sorted = np.sort(np.unique(season_num))
    burn = seasons_sorted[: cfg.burn_in_seasons]
    home_edge = np.where(ev["neutral_site"].fillna(0).astype(int).to_numpy() == 1, 0.0, 1.0)
    rt = _Ratings(
        a_idx=teams.get_indexer(ev["home_id"]), b_idx=teams.get_indexer(ev["away_id"]),
        season_idx=np.searchsorted(seasons_sorted, season_num),
        batch=pd.factorize(slate_dates(ev["start_time_utc"]))[0], home_edge=home_edge,
        # Update rates are tuned on the burn-in seasons only (the first one is warm-up).
        tune_mask=np.isin(season_num, burn[1:]) & completed, n_teams=len(teams), carry=cfg.season_carry,
    )

    # ---- results: Elo on wins, opponent-adjusted goals
    won = np.where(completed, (ev["winner_id"] == ev["home_id"]).astype(float), np.nan)
    margin = np.where(completed, ev["home_score"].to_numpy(dtype=float) - ev["away_score"].to_numpy(dtype=float), np.nan)
    if rt.tune_mask.sum() < 300:
        elo_params, elo_grid = EloParams(k=8.0, home_adv=30.0), []
    else:
        elo_params, elo_grid = tune_elo(
            rt.a_idx, rt.b_idx, won, rt.season_idx, rt.batch, home_edge, tune_mask=rt.tune_mask, margin=margin,
            k_grid=(4, 6, 8, 12), home_grid=(0, 25, 50), carry_grid=(0.5, 0.7, 0.9), margin_grid=(None, 2.0))
    elo_a, elo_b, played_a = elo_pregame(rt.a_idx, rt.b_idx, won, rt.season_idx, rt.batch, home_edge, elo_params,
                                         margin=margin, n_entities=len(teams))
    goals_a, goals_b = _goals(ev, completed)
    lrs = {"goals": rt.tune_lr(goals_a, goals_b, cfg.lr_grid)}
    slow = rt.run(goals_a, goals_b, lrs["goals"])
    feats: Dict[str, np.ndarray] = {
        "gd_slow_diff": _net_diff(slow),
        "gd_fast_diff": _net_diff(rt.run(goals_a, goals_b, cfg.fast_lr)),
        "exp_total_goals": np.where((slow["n_a"] == 0) | (slow["n_b"] == 0), np.nan,
                                    2.0 * slow["mean"] + slow["off_a"] + slow["def_b"] + slow["off_b"] + slow["def_a"]),
    }
    feats["form_diff"] = feats["gd_fast_diff"] - feats["gd_slow_diff"]

    # ---- shots, expected goals, special teams
    has_shots = store.has_table(league, "shots")
    if has_shots:
        stats = game_shot_stats(ev["event_id"], store.read_table(league, "shots"), completed,
                                completed & np.isin(season_num, burn), min_fit_rows=cfg.min_shot_model_rows)
        for name, key in (("sog", "sog_diff"), ("xg5", "xg5_diff"), ("ppg", "pp_diff")):
            va, vb = stats[f"{name}_a"].to_numpy(), stats[f"{name}_b"].to_numpy()
            lrs[name] = rt.tune_lr(va, vb, cfg.lr_grid)
            feats[key] = _net_diff(rt.run(va, vb, lrs[name]))
    else:
        for key in ("sog_diff", "xg5_diff", "pp_diff"):
            feats[key] = np.full(len(ev), np.nan)

    # ---- goaltending
    goalie_box = store.read_table(league, "goalie_box") if store.has_table(league, "goalie_box") else None
    goalies = goalie_pregame(ev, goalie_box, rt.batch, completed, cfg.goalie)

    sched = _schedule(store, league, ev)
    is_playoff = (ev["stage"] == "playoffs").astype(float)
    df = pd.DataFrame({
        "event_id": ev["event_id"],
        "start_time_utc": ev["start_time_utc"],
        "block": season_num,
        "season": ev["season"],
        "status": ev["status"],
        "stage": ev["stage"],
        "y": _target(market, ev, completed),
        "side_a_id": ev["home_id"],
        "side_b_id": ev["away_id"],
        "info_cutoff_utc": _info_cutoff(ev["start_time_utc"], rt.batch),
        "home_edge": home_edge,
        "elo_a": elo_a,
        "elo_b": elo_b,
        "elo_diff": elo_a - elo_b,
        "elo_gap_abs": np.abs(elo_a - elo_b + elo_params.home_adv * home_edge),
        "elo_home_prob": elo_expected(elo_a - elo_b + elo_params.home_adv * home_edge, elo_params.scale),
        "games_played_a": played_a,
        "is_playoff": is_playoff,
        "n_pending_prior": _pending_prior(ev, completed, stale),
    })
    for key, val in feats.items():
        df[key] = val
    df["goalie_sv_a"], df["goalie_sv_b"] = goalies["goalie_sv_a"], goalies["goalie_sv_b"]
    df["goalie_diff"] = goalies["goalie_sv_a"] - goalies["goalie_sv_b"]
    df["goalie_actual_diff"] = goalies["goalie_actual_a"] - goalies["goalie_actual_b"]
    df["rest_diff"] = sched["rest_days_a"] - sched["rest_days_b"]
    df["rest_days_a"], df["rest_days_b"] = sched["rest_days_a"], sched["rest_days_b"]
    df["b2b_a"], df["b2b_b"] = sched["b2b_a"], sched["b2b_b"]
    df["games_7d_diff"] = sched["games_7d_a"] - sched["games_7d_b"]
    df["travel_km_a"] = np.log1p(sched["travel_km_a"])
    df["travel_km_b"] = np.log1p(sched["travel_km_b"])
    df["tz_shift_b"] = sched["tz_shift_b"]
    df["dq_no_shot_data"] = df["xg5_diff"].isna().astype(float)
    df["dq_no_goalie_history"] = df["goalie_diff"].isna().astype(float)
    if market == "winner_3way":
        # A played game whose regulation result is not trusted has no target;
        # it still fed the ratings above but is not a row of this market.
        df = df[~(completed & df["y"].isna().to_numpy())].reset_index(drop=True)

    limitations = [
        "No pregame injury reports, line-ups or confirmed starting goalies exist in the historical data, so the "
        "backtest cannot include late news. The goalie feature assumes each team's recent primary goalie starts.",
        "Skater availability is not modelled: a missing star skater only shows up through later results.",
        "Expected goals come from a simple shot-location model (distance, angle, shot type) at five-on-five; "
        "pre-shot movement, rebounds and rush chances are not in the data used.",
        "Travel is measured between home-arena cities; games away from a club's regular arena have no location.",
        "Betting odds are not an input and no timestamped market benchmark is available from this source.",
    ]
    if not has_shots:
        limitations.append("No play-by-play on file: shot, expected-goal and special-teams features are missing.")
    if goalie_box is None:
        limitations.append("No goalie box scores on file: the goalie feature is missing.")
    three_way = market == "winner_3way"
    return FeatureFrame(
        league=league, market=market,
        outcome=OutcomeType.THREE_WAY if three_way else OutcomeType.BINARY,
        classes=("home", "draw", "away") if three_way else ("away", "home"), df=df,
        feature_cols=list(THREE_WAY_FEATURES if three_way else WINNER_FEATURES),
        retrospective_cols=["goalie_actual_diff"], feature_notes=dict(FEATURE_LABELS),
        limitations=limitations, point_in_time=True,
        extra={
            "elo_params": elo_params.__dict__, "elo_tuning_grid_size": len(elo_grid),
            "slow_update_rates": lrs, "fast_lr": cfg.fast_lr, "burn_in_seasons": [int(s) for s in burn],
            "tuning_note": "Elo and rating update rates were tuned, and the shot model fitted, on the burn-in "
                           "seasons only.",
            "goalies": goalies.attrs["goalies"], "league_sv": goalies.attrs["league_sv"],
        },
    )
