"""Point-in-time soccer features for the three-way (home / draw / away) market.

Every number in a row comes from matches on **earlier slates** only (a slate
is a Pacific/Honolulu calendar day; for a date-only source that is the same
as "earlier dates") plus the public schedule.  Side A is the home side as
listed by the source - at a neutral venue the nominal home side - so it never
depends on the result.

Feature families (their weights are learned in validation, not set here):

* opponent-adjusted results - Elo, where a result moves a rating by how far
  it differed from what the opponent's rating implied;
* opponent-adjusted attack and defence - goals scored and conceded, each
  match's surprise measured against what the opponent usually concedes or
  scores, kept with a slow memory (long-term level) and a fast one (recent
  form);
* venue - home advantage, none at a neutral ground;
* schedule - rest days since each side's previous match in the data;
* international leagues: friendly or qualifier versus other competitions;
* club leagues: opponent-adjusted shots on target, and whether a side is new
  to the division this season.

The frame carries no retrospective columns: the sources hold nothing about
line-ups, so there is no "who actually played" experiment to run.  Expected
goals (xG) are not in any verified source and are not used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...core.enums import OutcomeType
from ...core.timeutil import DISPLAY_TZ_NAME, parse_ts, utcnow
from ...features.ratings import EloParams, elo_pregame, offdef_pregame, tune_elo
from ...features.schedule import team_schedule_features
from ..base import FeatureFrame

__all__ = ["SoccerConfig", "build_three_way_frame", "slate_batches", "CLASSES", "GOAL_CAP",
           "BASE_FEATURES", "INTL_FEATURES", "CLUB_FEATURES", "GOAL_MODEL_INPUTS", "FEATURE_LABELS"]

CLASSES: Tuple[str, ...] = ("home", "draw", "away")
#: Goals beyond this many in one match carry little extra information about
#: strength; ratings and the goals model count them as this many.
GOAL_CAP = 6.0
_MIN_TUNE_EVENTS = 300
_ELO_K_GRID = (10, 15, 20, 30, 40)
_ELO_HOME_GRID = (0, 40, 80, 120)
_ELO_MARGIN_GRID = (None, 1.0, 3.0)


@dataclass(frozen=True)
class SoccerConfig:
    international: bool = False           # calendar-year blocks, no season carry-over, competition-type features
    history_start: Optional[int] = None   # first block (season or year) read at all
    burn_in_blocks: int = 5               # earliest blocks: rating warm-up and update-rate tuning only
    warmup_blocks: int = 1                # of those, blocks before tuning starts
    min_train_blocks: int = 4
    max_dev_folds: Optional[int] = None
    fast_lr: float = 0.2                  # "recent form": roughly the last 9 matches
    slow_lr_grid: Tuple[float, ...] = (0.02, 0.04, 0.07)
    season_carry: float = 0.7             # share of an attack/defence rating kept across seasons (clubs)


BASE_FEATURES: List[str] = ["home_edge", "elo_diff", "goal_diff_exp", "goal_sum_exp", "form_diff", "rest_diff"]
INTL_FEATURES: List[str] = BASE_FEATURES + ["is_friendly", "is_qualifier"]
CLUB_FEATURES: List[str] = BASE_FEATURES + ["sot_edge", "promoted_a", "promoted_b"]
#: Per-side inputs of the goals model (see :mod:`.models`).
GOAL_MODEL_INPUTS: List[str] = ["home_edge", "att_a", "def_a", "att_b", "def_b",
                                "att_form_a", "def_form_a", "att_form_b", "def_form_b"]

FEATURE_LABELS: Dict[str, str] = {
    "home_edge": "home advantage (none at a neutral venue)",
    "elo_diff": "opponent-adjusted results (Elo)",
    "goal_diff_exp": "long-term opponent-adjusted goal difference",
    "goal_sum_exp": "how many goals these two sides' matches usually produce",
    "form_diff": "recent form relative to long-term level",
    "rest_diff": "rest days",
    "is_friendly": "friendly match",
    "is_qualifier": "qualifying match",
    "sot_edge": "opponent-adjusted shots on target",
    "promoted_a": "home side new to the division this season",
    "promoted_b": "away side new to the division this season",
    "att_a": "home attack rating (goals above average)",
    "def_a": "home defence rating (goals conceded above average)",
    "att_b": "away attack rating",
    "def_b": "away defence rating",
    "att_form_a": "home attack, recent minus long-term",
    "def_form_a": "home defence, recent minus long-term",
    "att_form_b": "away attack, recent minus long-term",
    "def_form_b": "away defence, recent minus long-term",
}


def slate_batches(start_time_utc: pd.Series) -> np.ndarray:
    """Batch label = Pacific/Honolulu calendar day, so matches of one slate
    never inform each other.  A date-only match is stored at 00:00Z, hence all
    matches of one calendar date share a batch."""
    days = start_time_utc.dt.tz_convert(DISPLAY_TZ_NAME).dt.strftime("%Y-%m-%d")
    return pd.factorize(days, sort=False)[0]


def _pending_prior(home: np.ndarray, away: np.ndarray, upcoming: np.ndarray, unplayed: np.ndarray) -> np.ndarray:
    """For each upcoming match: earlier matches of either side that have no
    result on file (still to be played, or played but not yet in the source)."""
    pending = np.zeros(len(home), dtype=int)
    seen: Dict[str, int] = {}
    for i in np.flatnonzero(unplayed):
        if upcoming[i]:
            pending[i] = seen.get(home[i], 0) + seen.get(away[i], 0)
        seen[home[i]] = seen.get(home[i], 0) + 1
        seen[away[i]] = seen.get(away[i], 0) + 1
    return pending


def _promoted(home: np.ndarray, away: np.ndarray, block: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """1.0 where a side did not play in the previous block on file (a promoted
    or otherwise new club); 0.0 in the first block, where that is unknowable."""
    members: Dict[int, set] = {}
    for b in np.unique(block):
        mask = block == b
        members[int(b)] = set(home[mask]) | set(away[mask])
    order = sorted(members)
    previous = {b: members[order[i - 1]] if i else None for i, b in enumerate(order)}
    flag_a, flag_b = np.zeros(len(home)), np.zeros(len(home))
    for i, b in enumerate(block):
        prev = previous[int(b)]
        if prev is not None:
            flag_a[i] = float(home[i] not in prev)
            flag_b[i] = float(away[i] not in prev)
    return flag_a, flag_b


def _matchup(r: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """``(difference, sum)`` of what each side is expected to produce against
    the other, relative to the league average and before home advantage."""
    a, b = r["off_a"] + r["def_b"], r["off_b"] + r["def_a"]
    return a - b, a + b


def _tune_slow_lr(cfg: SoccerConfig, tune_mask: np.ndarray, ratings_args: tuple, va: np.ndarray,
                  vb: np.ndarray, home_edge: np.ndarray, n_entities: int) -> float:
    """Slow update rate with the lowest squared goal-forecast error on the
    tuning blocks (events after them are never touched)."""
    default = cfg.slow_lr_grid[len(cfg.slow_lr_grid) // 2]
    if tune_mask.sum() < _MIN_TUNE_EVENTS:
        return default
    end = int(np.flatnonzero(tune_mask).max()) + 1
    a_idx, b_idx, season_idx, batch = (x[:end] for x in ratings_args)
    best, best_lr = np.inf, default
    for lr in cfg.slow_lr_grid:
        r = offdef_pregame(a_idx, b_idx, va[:end], vb[:end], season_idx, batch, home_edge[:end], lr=lr,
                           carry=cfg.season_carry, n_entities=n_entities)
        exp_a = r["mean"] + r["off_a"] + r["def_b"] + r["home"] * home_edge[:end] / 2
        exp_b = r["mean"] + r["off_b"] + r["def_a"] - r["home"] * home_edge[:end] / 2
        m = tune_mask[:end] & np.isfinite(va[:end])
        mse = float(np.mean((va[:end][m] - exp_a[m]) ** 2 + (vb[:end][m] - exp_b[m]) ** 2))
        if mse < best:
            best, best_lr = mse, lr
    return best_lr


def build_three_way_frame(store: Any, league: str, *, now: Any = None,
                          cfg: Optional[SoccerConfig] = None) -> FeatureFrame:
    cfg = cfg or SoccerConfig()
    now_ts = pd.Timestamp(utcnow() if now is None else parse_ts(now))
    ev = store.events(league, with_results=True)
    if cfg.history_start is not None:
        ev = ev[pd.to_numeric(ev["season"], errors="coerce") >= cfg.history_start]
    ev = ev.sort_values(["start_time_utc", "event_id"]).reset_index(drop=True)
    # "As of now": a match that starts after ``now`` is upcoming whatever the
    # store knows about it today, and its result and statistics are ignored.
    scored = ev["home_score"].notna() & ev["away_score"].notna()
    completed_all = ((ev["status"] == "final") & scored & (ev["start_time_utc"] < now_ts)).to_numpy()
    upcoming_all = (ev["status"].isin(["scheduled", "final"]) & (ev["start_time_utc"] >= now_ts)).to_numpy()
    # Scheduled for a date that has passed, but no result on file yet.
    overdue = ((ev["status"] == "scheduled") & (ev["start_time_utc"] <= now_ts)).to_numpy()
    if not (completed_all | upcoming_all).any():
        raise ValueError(f"no events on file for {league}; run `sportspred ingest --league {league}` first")
    pending_all = _pending_prior(ev["home_id"].to_numpy(), ev["away_id"].to_numpy(), upcoming_all,
                                 upcoming_all | overdue)
    listed = ev[completed_all | upcoming_all | overdue]
    rest = team_schedule_features(
        listed[["event_id", "start_time_utc", "season", "home_id", "away_id", "neutral_site"]]
    ).set_index("event_id")

    keep = completed_all | upcoming_all
    pending = pending_all[keep]
    ev = ev[keep].reset_index(drop=True)
    completed = completed_all[keep]
    n = len(ev)
    teams = pd.Index(pd.unique(pd.concat([ev["home_id"], ev["away_id"]])))
    a_idx, b_idx = teams.get_indexer(ev["home_id"]), teams.get_indexer(ev["away_id"])
    block = pd.to_numeric(ev["season"], errors="coerce").astype(int).to_numpy()
    blocks_sorted = np.unique(block)
    # International sides have no seasons: ratings never regress at a year end.
    season_idx = np.zeros(n, dtype=int) if cfg.international else np.searchsorted(blocks_sorted, block)
    batch = slate_batches(ev["start_time_utc"])
    home_edge = np.where(ev["neutral_site"].fillna(0).astype(int).to_numpy() == 1, 0.0, 1.0)
    goals_a = np.where(completed, ev["home_score"].to_numpy(dtype=float), np.nan)
    goals_b = np.where(completed, ev["away_score"].to_numpy(dtype=float), np.nan)
    diff = goals_a - goals_b
    y = np.where(completed, np.select([diff > 0, diff == 0], [0.0, 1.0], 2.0), np.nan)

    # ---- update rates are tuned on the burn-in blocks only, after a warm-up
    burn = blocks_sorted[: cfg.burn_in_blocks]
    tune_mask = np.isin(block, burn[cfg.warmup_blocks:]) & completed
    score_a = np.where(completed, np.select([diff > 0, diff == 0], [1.0, 0.5], 0.0), np.nan)
    # Goal difference plus one, so that a draw still moves the ratings when
    # the margin-of-victory multiplier is switched on.
    elo_margin = np.abs(diff) + 1.0
    if tune_mask.sum() < _MIN_TUNE_EVENTS:
        elo_params, grid_size = EloParams(k=20.0, home_adv=60.0, carry=0.8), 0
    else:
        elo_params, grid = tune_elo(
            a_idx, b_idx, score_a, season_idx, batch, home_edge, tune_mask=tune_mask, margin=elo_margin,
            k_grid=_ELO_K_GRID, home_grid=_ELO_HOME_GRID, margin_grid=_ELO_MARGIN_GRID,
            carry_grid=(1.0,) if cfg.international else (0.6, 0.8, 1.0))
        grid_size = len(grid)
    elo_a, elo_b, played_a = elo_pregame(a_idx, b_idx, score_a, season_idx, batch, home_edge, elo_params,
                                         margin=elo_margin, n_entities=len(teams))

    capped_a, capped_b = np.minimum(goals_a, GOAL_CAP), np.minimum(goals_b, GOAL_CAP)
    args = (a_idx, b_idx, season_idx, batch)
    slow_lr = _tune_slow_lr(cfg, tune_mask, args, capped_a, capped_b, home_edge, len(teams))
    slow = offdef_pregame(*args[:2], capped_a, capped_b, *args[2:], home_edge, lr=slow_lr,
                          carry=cfg.season_carry, n_entities=len(teams))
    fast = offdef_pregame(*args[:2], capped_a, capped_b, *args[2:], home_edge, lr=cfg.fast_lr,
                          carry=cfg.season_carry, n_entities=len(teams))
    new_a, new_b = slow["n_a"] == 0, slow["n_b"] == 0
    unrated = new_a | new_b
    goal_diff, goal_sum = _matchup(slow)
    fast_diff, _ = _matchup(fast)

    starts = ev["start_time_utc"]
    last_of_batch = starts.groupby(batch).transform("max")
    prev_batch_last = pd.Series(pd.unique(last_of_batch)).shift().to_numpy()
    rest_a = rest["rest_days_a"].reindex(ev["event_id"]).to_numpy()
    rest_b = rest["rest_days_b"].reindex(ev["event_id"]).to_numpy()

    df = pd.DataFrame({
        "event_id": ev["event_id"],
        "start_time_utc": starts,
        "block": block,
        "season": ev["season"],
        "status": ev["status"],
        "stage": ev["stage"],
        "y": y,
        "side_a_id": ev["home_id"],
        "side_b_id": ev["away_id"],
        "info_cutoff_utc": pd.to_datetime(pd.Series(prev_batch_last[batch], index=ev.index), utc=True),
        "goals_a": goals_a,
        "goals_b": goals_b,
        "n_pending_prior": pending,
        "home_edge": home_edge,
        "elo_a": elo_a,
        "elo_b": elo_b,
        "elo_diff": elo_a - elo_b,
        "matches_a": played_a,
        "goal_diff_exp": np.where(unrated, np.nan, goal_diff),
        "goal_sum_exp": np.where(unrated, np.nan, goal_sum),
        "form_diff": np.where(unrated, np.nan, fast_diff - goal_diff),
        "rest_days_a": rest_a,
        "rest_days_b": rest_b,
        "rest_diff": rest_a - rest_b,
        "att_a": np.where(new_a, np.nan, slow["off_a"]),
        "def_a": np.where(new_a, np.nan, slow["def_a"]),
        "att_b": np.where(new_b, np.nan, slow["off_b"]),
        "def_b": np.where(new_b, np.nan, slow["def_b"]),
        "att_form_a": np.where(new_a, np.nan, fast["off_a"] - slow["off_a"]),
        "def_form_a": np.where(new_a, np.nan, fast["def_a"] - slow["def_a"]),
        "att_form_b": np.where(new_b, np.nan, fast["off_b"] - slow["off_b"]),
        "def_form_b": np.where(new_b, np.nan, fast["def_b"] - slow["def_b"]),
        # What the running ratings alone expect each side to score (for explanations).
        "rating_goals_a": slow["mean"] + slow["off_a"] + slow["def_b"] + slow["home"] * home_edge / 2,
        "rating_goals_b": slow["mean"] + slow["off_b"] + slow["def_a"] - slow["home"] * home_edge / 2,
        "dq_unrated_team": unrated.astype(float),
    })
    limitations = [
        "No line-ups, injuries or suspensions exist in the historical data, so the backtest contains no "
        "team news at all; a late absence is invisible to the model.",
        "Expected-goals (xG) data is not available from any verified source; attack and defence are rated "
        "on goals" + (" only." if cfg.international else " and shots on target."),
        "Betting odds are never a model input.",
    ]
    if cfg.international:
        df["is_friendly"] = (ev["stage"] == "friendly").astype(float)
        df["is_qualifier"] = (ev["stage"] == "qualifying").astype(float)
        feature_cols = INTL_FEATURES
        limitations += [
            "The source's scores include extra time (never penalties) and do not mark knockout ties, so a "
            "match won in extra time counts as a win here although it was a draw after 90 minutes.",
            "Matches are date-only: every match of a calendar date is treated as simultaneous.",
        ]
    else:
        df["sot_edge"] = _shots_on_target_edge(store, league, ev, completed, args, home_edge, slow_lr, cfg,
                                               len(teams))
        df["promoted_a"], df["promoted_b"] = _promoted(ev["home_id"].to_numpy(), ev["away_id"].to_numpy(), block)
        feature_cols = CLUB_FEATURES
        limitations.append(
            "Only league matches are on file: rest days ignore cup and European fixtures, and a promoted "
            "club starts from an average rating plus a 'new to the division' flag.")
    inputs = list(dict.fromkeys(feature_cols + GOAL_MODEL_INPUTS))
    return FeatureFrame(
        league=league, market="winner_3way", outcome=OutcomeType.THREE_WAY, classes=CLASSES, df=df,
        feature_cols=inputs, retrospective_cols=[], feature_notes={c: FEATURE_LABELS[c] for c in inputs},
        limitations=limitations, point_in_time=True,
        extra={
            "elo_params": elo_params.__dict__, "elo_tuning_grid_size": grid_size, "slow_lr": slow_lr,
            "fast_lr": cfg.fast_lr, "goal_cap": GOAL_CAP, "burn_in_blocks": [int(b) for b in burn],
            "tuning_blocks": [int(b) for b in burn[cfg.warmup_blocks:]], "model_features": list(feature_cols),
            "tuning_note": "Elo and attack/defence update rates were tuned on the tuning blocks only.",
        },
    )


def _shots_on_target_edge(store: Any, league: str, ev: pd.DataFrame, completed: np.ndarray, args: tuple,
                          home_edge: np.ndarray, lr: float, cfg: SoccerConfig, n_entities: int) -> np.ndarray:
    """Opponent-adjusted shots-on-target matchup (NaN until both sides have a
    match with shot statistics; all NaN when the league has no such table)."""
    if not store.has_table(league, "match_stats"):
        return np.full(len(ev), np.nan)
    stats = store.read_table(league, "match_stats").set_index("event_id").reindex(ev["event_id"])
    va = np.where(completed, stats["sot_home"].to_numpy(dtype=float), np.nan)
    vb = np.where(completed, stats["sot_away"].to_numpy(dtype=float), np.nan)
    r = offdef_pregame(*args[:2], va, vb, *args[2:], home_edge, lr=lr, carry=cfg.season_carry,
                       n_entities=n_entities)
    edge, _ = _matchup(r)
    return np.where((r["n_a"] == 0) | (r["n_b"] == 0), np.nan, edge)
