"""Point-in-time features for any two-sided competition, from schedules and
results alone.

This is the builder behind every league that has no sport-specific plugin:
it needs only who played whom, when, and who won (scores when they exist).
Every feature of an event uses earlier slates only (a slate is a
Pacific/Honolulu calendar day), and an event starting after ``now`` is treated
as upcoming whatever the store knows about it.

Side A is the home side when the league has home advantage and the event is
not at a neutral site.  Otherwise side A is the participant whose ID sorts
first - a rule that cannot depend on who won, which matters because many
sources list the winner first.

Features (weights are learned in validation):

* ``elo_diff`` - opponent-adjusted results, update rates tuned on burn-in
  blocks only;
* ``net_slow_diff`` / ``net_fast_diff`` / ``form_diff`` - opponent-adjusted
  scoring margin with a slow memory and a fast one (about the last 7 events),
  only for leagues whose results carry scores;
* ``rest_diff`` - days since each side's previous event (capped);
* ``exp_diff`` - how many rated events each side has behind it (small-sample
  indicator);
* ``home_edge`` - 1 for a true home event, 0 otherwise.
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
from ...storage.store import is_demo_league
from ..base import FeatureFrame

__all__ = ["GenericOptions", "build_generic_frame", "FEATURE_LABELS", "oriented_events"]

FEATURE_LABELS: Dict[str, str] = {
    "home_edge": "home advantage",
    "elo_diff": "opponent-adjusted results (Elo)",
    "net_slow_diff": "long-term opponent-adjusted scoring margin",
    "net_fast_diff": "recent (about 7 events) opponent-adjusted scoring margin",
    "form_diff": "recent form relative to long-term level",
    "rest_diff": "days since the previous event",
    "exp_diff": "number of rated events behind each side",
}


@dataclass(frozen=True)
class GenericOptions:
    draws: bool = False                    # the market has a draw outcome (three-way)
    home_adv: bool = True                  # the league has a meaningful home side
    burn_in_blocks: int = 1                # earliest blocks: rating warm-up and update-rate tuning only
    fast_lr: float = 0.25                  # "recent form": roughly the last 7 events
    slow_lr: float = 0.05
    season_carry: float = 0.8
    rest_cap_days: float = 30.0
    min_score_share: float = 0.8           # scoring ratings need scores for at least this share of results
    excluded_stages: Tuple[str, ...] = ("preseason", "exhibition", "allstar", "offseason")
    k_grid: Tuple[float, ...] = (12, 20, 32, 48)
    carry_grid: Tuple[float, ...] = (0.6, 0.8, 1.0)
    margin_grid: Tuple[Optional[float], ...] = (None,)


def oriented_events(store: Any, league: str, now_ts: pd.Timestamp, opts: GenericOptions) -> pd.DataFrame:
    """Completed and upcoming events as of ``now``, oriented so that side A
    never depends on the result.  Adds ``a_id, b_id, completed, score_a``
    (1 / 0.5 / 0, NaN when undecided), ``val_a, val_b`` (scores) and
    ``home_edge``."""
    ev = store.events(league, with_results=True, include_synthetic=is_demo_league(league))
    ev = ev[~ev["stage"].isin(opts.excluded_stages) & ev["home_id"].notna() & ev["away_id"].notna()]
    decided = ev["winner_id"].notna() | (ev["is_draw"].fillna(0).astype(int) == 1)
    done = (ev["status"] == "final") & decided & (ev["start_time_utc"] < now_ts)
    upcoming = ev["status"].isin(["scheduled", "final"]) & (ev["start_time_utc"] > now_ts)
    ev = ev[done | upcoming].sort_values(["start_time_utc", "event_id"]).reset_index(drop=True)
    if ev.empty:
        raise ValueError(f"no events on file for {league}; import results first (see docs/CSV_IMPORT.md)")
    completed = ((ev["status"] == "final") & (ev["winner_id"].notna() | (ev["is_draw"].fillna(0).astype(int) == 1))
                 & (ev["start_time_utc"] < now_ts)).to_numpy()
    neutral = ev["neutral_site"].fillna(0).astype(int).to_numpy() == 1
    true_home = np.full(len(ev), bool(opts.home_adv)) & ~neutral
    home, away = ev["home_id"].to_numpy(), ev["away_id"].to_numpy()
    swap = ~true_home & (home > away)                       # no home side: smaller ID is side A
    ev["a_id"] = np.where(swap, away, home)
    ev["b_id"] = np.where(swap, home, away)
    ev["completed"] = completed
    ev["home_edge"] = true_home.astype(float)
    winner = ev["winner_id"].to_numpy()
    draw = ev["is_draw"].fillna(0).astype(int).to_numpy() == 1
    score = np.where(draw, 0.5, np.where(winner == ev["a_id"].to_numpy(), 1.0, 0.0))
    ev["score_a"] = np.where(completed, score, np.nan)
    hs, as_ = ev["home_score"].to_numpy(dtype=float), ev["away_score"].to_numpy(dtype=float)
    ev["val_a"] = np.where(completed, np.where(swap, as_, hs), np.nan)
    ev["val_b"] = np.where(completed, np.where(swap, hs, as_), np.nan)
    return ev


def _blocks(ev: pd.DataFrame) -> np.ndarray:
    season = pd.to_numeric(ev["season"], errors="coerce")
    if season.notna().all():
        return season.astype(int).to_numpy()
    return ev["start_time_utc"].dt.year.to_numpy()


def _rest_and_experience(ev: pd.DataFrame, cap: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Days since each side's previous event, and events already played."""
    n = len(ev)
    long = pd.concat([
        pd.DataFrame({"row": np.arange(n), "who": ev["a_id"].to_numpy(), "side": 0}),
        pd.DataFrame({"row": np.arange(n), "who": ev["b_id"].to_numpy(), "side": 1}),
    ], ignore_index=True)
    long["t"] = ev["start_time_utc"].to_numpy()[long["row"]]
    long = long.sort_values(["who", "t", "row"]).reset_index(drop=True)
    same = long["who"] == long["who"].shift()
    gap = (long["t"] - long["t"].shift()).dt.total_seconds() / 86400.0
    long["rest"] = gap.where(same).clip(upper=cap).fillna(cap)
    rest = np.zeros((n, 2))
    rest[long["row"].to_numpy(), long["side"].to_numpy()] = long["rest"].to_numpy()
    return rest[:, 0], rest[:, 1], rest[:, 0] - rest[:, 1]


def build_generic_frame(store: Any, league: str, market: str, *, now: Any = None,
                        opts: Optional[GenericOptions] = None) -> FeatureFrame:
    opts = opts or GenericOptions()
    now_ts = pd.Timestamp(utcnow() if now is None else parse_ts(now))
    ev = oriented_events(store, league, now_ts, opts)
    completed = ev["completed"].to_numpy()
    who = pd.Index(pd.unique(pd.concat([ev["a_id"], ev["b_id"]])))
    a_idx, b_idx = who.get_indexer(ev["a_id"]), who.get_indexer(ev["b_id"])
    block = _blocks(ev)
    blocks_sorted = np.sort(np.unique(block))
    season_idx = np.searchsorted(blocks_sorted, block)
    batch = pd.factorize(ev["start_time_utc"].dt.tz_convert(DISPLAY_TZ_NAME).dt.strftime("%Y-%m-%d"))[0]
    home_edge = ev["home_edge"].to_numpy()
    score = ev["score_a"].to_numpy(dtype=float)
    margin = ev["val_a"].to_numpy(dtype=float) - ev["val_b"].to_numpy(dtype=float)

    burn = blocks_sorted[: max(opts.burn_in_blocks, 1)]
    tune_mask = np.isin(block, burn) & completed
    first = int(np.flatnonzero(tune_mask)[len(np.flatnonzero(tune_mask)) // 3]) if tune_mask.any() else 0
    tune_mask[:first] = False                 # the first third of the burn-in is rating warm-up
    if tune_mask.sum() >= 150:
        elo_params, grid = tune_elo(
            a_idx, b_idx, score, season_idx, batch, home_edge, tune_mask=tune_mask,
            margin=margin if np.isfinite(margin[completed]).mean() >= opts.min_score_share else None,
            k_grid=opts.k_grid, home_grid=(0, 40, 80, 120) if opts.home_adv else (0,),
            carry_grid=opts.carry_grid, margin_grid=opts.margin_grid)
    else:
        elo_params, grid = EloParams(k=24.0, home_adv=60.0 if opts.home_adv else 0.0, carry=opts.season_carry), []
    elo_a, elo_b, played_a = elo_pregame(a_idx, b_idx, score, season_idx, batch, home_edge, elo_params,
                                         margin=margin, n_entities=len(who))
    # Events already rated for side B (side A's count comes from elo_pregame).
    seen = np.zeros(len(who), dtype=int)
    played_b = np.zeros(len(ev), dtype=int)
    order = np.arange(len(ev))
    for start in np.flatnonzero(np.r_[True, batch[1:] != batch[:-1]]):
        end = start
        while end < len(ev) and batch[end] == batch[start]:
            end += 1
        played_b[start:end] = seen[b_idx[start:end]]
        for i in order[start:end]:
            if completed[i]:
                seen[a_idx[i]] += 1
                seen[b_idx[i]] += 1

    df = pd.DataFrame({
        "event_id": ev["event_id"], "start_time_utc": ev["start_time_utc"], "block": block,
        "season": ev["season"], "status": ev["status"], "stage": ev["stage"],
        "side_a_id": ev["a_id"], "side_b_id": ev["b_id"], "home_edge": home_edge,
        "elo_a": elo_a, "elo_b": elo_b, "elo_diff": elo_a - elo_b, "games_played_a": played_a,
        "exp_diff": np.log1p(played_a) - np.log1p(played_b),
    })
    features = ["home_edge", "elo_diff", "rest_diff", "exp_diff"] if opts.home_adv else \
        ["elo_diff", "rest_diff", "exp_diff"]
    rest_a, rest_b, rest_diff = _rest_and_experience(ev, opts.rest_cap_days)
    df["rest_days_a"], df["rest_days_b"], df["rest_diff"] = rest_a, rest_b, rest_diff

    has_scores = bool(completed.any()) and float(np.isfinite(margin[completed]).mean()) >= opts.min_score_share
    if has_scores:
        va, vb = ev["val_a"].to_numpy(dtype=float), ev["val_b"].to_numpy(dtype=float)
        slow = offdef_pregame(a_idx, b_idx, va, vb, season_idx, batch, home_edge, lr=opts.slow_lr,
                              carry=opts.season_carry, n_entities=len(who))
        fast = offdef_pregame(a_idx, b_idx, va, vb, season_idx, batch, home_edge, lr=opts.fast_lr,
                              carry=opts.season_carry, n_entities=len(who))
        unseen = (slow["n_a"] == 0) | (slow["n_b"] == 0)
        net_slow = (slow["off_a"] - slow["def_a"]) - (slow["off_b"] - slow["def_b"])
        net_fast = (fast["off_a"] - fast["def_a"]) - (fast["off_b"] - fast["def_b"])
        df["net_slow_diff"] = np.where(unseen, np.nan, net_slow)
        df["net_fast_diff"] = np.where(unseen, np.nan, net_fast)
        df["form_diff"] = df["net_fast_diff"] - df["net_slow_diff"]
        features += ["net_slow_diff", "net_fast_diff", "form_diff"]

    three_way = bool(opts.draws) and market.endswith("3way")
    if three_way:
        y = np.where(completed, np.where(score == 1.0, 0.0, np.where(score == 0.5, 1.0, 2.0)), np.nan)
        classes, outcome = ("a", "draw", "b"), OutcomeType.THREE_WAY
    else:
        y = np.where(completed & (score != 0.5), score, np.nan)
        classes, outcome = ("b", "a"), OutcomeType.BINARY
        # A drawn event has no binary outcome: it fed the ratings, it is not a row.
        keep = ~(completed & np.isnan(y))
        df, y, batch, completed = df[keep].reset_index(drop=True), y[keep], batch[keep], completed[keep]
    df["y"] = y
    starts = df["start_time_utc"]
    last_of_batch = starts.groupby(batch).transform("max")
    prev = pd.Series(pd.unique(last_of_batch)).shift().to_numpy()
    df["info_cutoff_utc"] = pd.to_datetime(pd.Series(prev[pd.factorize(batch)[0]], index=df.index), utc=True)
    pending = np.zeros(len(df), dtype=int)
    counts: Dict[str, int] = {}
    for i in np.flatnonzero(~completed):
        a, b = df.at[i, "side_a_id"], df.at[i, "side_b_id"]
        pending[i] = counts.get(a, 0) + counts.get(b, 0)
        counts[a], counts[b] = counts.get(a, 0) + 1, counts.get(b, 0) + 1
    df["n_pending_prior"] = pending
    df["dq_few_events"] = ((df["games_played_a"] < 5)).astype(float)

    limitations = [
        "Built from schedules and results only: no line-ups, injuries, venues beyond the home flag, or "
        "sport-specific statistics are used.",
        "Betting odds are not an input.",
    ]
    if not has_scores:
        limitations.append("Results carry no usable scores, so only win/loss ratings are available.")
    return FeatureFrame(
        league=league, market=market, outcome=outcome, classes=classes, df=df, feature_cols=features,
        feature_notes={k: v for k, v in FEATURE_LABELS.items() if k in features}, limitations=limitations,
        point_in_time=True,
        extra={"elo_params": elo_params.__dict__, "elo_tuning_grid_size": len(grid),
               "burn_in_blocks": [int(b) for b in burn],
               "tuning_note": "Elo update rates were tuned on the burn-in blocks only."},
    )
