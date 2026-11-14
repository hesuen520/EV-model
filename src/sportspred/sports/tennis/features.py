"""Point-in-time tennis features for the match-winner market.

Ratings are built on the pooled results of a whole tour (tour-level main
draw, qualifying, Challenger / WTA 125 and ITF together), because a player's
strength does not reset when he or she changes level; the model for one
league is then fitted and validated on that league's matches only.

**Time ordering.**  The source dates a match by its tournament's start date.
All matches whose tournaments start in the same Monday-to-Sunday week form
one batch: a match never uses another match of its own tournament, not even
an earlier round, nor of any tournament starting that week.  That costs some
information and removes any chance of using a later round to "predict" an
earlier one.  For the same reason a tournament's rows count as results only
once its week is over: asked for features as of a Wednesday, the builder
does not know how that week's tournaments ended.  What remains: a two-week
event is treated as known after its first week, and a tournament dated on a
Sunday as known the next day; both reach other players' ratings only.

Side A is the player whose ID sorts first - never the winner.

Features (weights learned in validation):

* ``elo_diff`` - opponent-adjusted results on all surfaces;
* ``surf_elo_diff`` - the same on this match's surface only;
* ``serve_edge`` / ``serve_edge_fast`` - opponent-adjusted share of service
  points won, A's serve against B's return minus the reverse, with a slow and
  a fast memory;
* ``rest_diff``, ``load_14d_diff`` - days since the previous tournament and
  matches played in the two previous tournament weeks (fatigue, or a lay-off);
* ``exp_diff`` - rated matches behind each player;
* ``bo5``, ``elo_diff_bo5`` - best-of-five matches favour the stronger player.

Not used: official rankings and seedings (reputation is not an input) and
head-to-head records (tiny samples).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...core.enums import OutcomeType
from ...core.timeutil import parse_ts, utcnow
from ...features.ratings import EloParams, elo_pregame, offdef_pregame, tune_elo
from ..base import FeatureFrame
from ..generic.features import GenericOptions, _rest_and_experience, oriented_events
from .adapter import TENNIS_LEAGUES, TOUR_GROUP

__all__ = ["TennisConfig", "build_tennis_frame", "TENNIS_FEATURES", "FEATURE_LABELS"]

TENNIS_FEATURES: List[str] = ["elo_diff", "surf_elo_diff", "serve_edge", "serve_edge_fast", "rest_diff",
                              "load_14d_diff", "exp_diff", "bo5", "elo_diff_bo5"]
FEATURE_LABELS: Dict[str, str] = {
    "elo_diff": "opponent-adjusted results, all surfaces (Elo)",
    "surf_elo_diff": "opponent-adjusted results on this surface",
    "serve_edge": "serve against return, long-term (service points won, opponent-adjusted)",
    "serve_edge_fast": "serve against return, recent matches",
    "rest_diff": "days since the previous tournament",
    "load_14d_diff": "matches played in the two previous tournament weeks",
    "exp_diff": "rated matches behind each player",
    "bo5": "best-of-five match",
    "elo_diff_bo5": "rating gap in a best-of-five match",
}
SURFACES = ("hard", "clay", "grass", "carpet")


@dataclass(frozen=True)
class TennisConfig:
    burn_in_years: int = 3
    slow_lr: float = 0.04
    fast_lr: float = 0.2
    rest_cap_days: float = 60.0


def _pooled(store: Any, tour: str, now_ts: pd.Timestamp) -> pd.DataFrame:
    """Every league of the tour that has events on file, oriented and merged
    with its match statistics, in chronological order."""
    opts = GenericOptions(home_adv=False, excluded_stages=())
    parts = []
    for league in TOUR_GROUP[tour]:
        try:
            ev = oriented_events(store, league, now_ts, opts)
        except ValueError:
            continue                                      # that league has not been ingested
        if store.has_table(league, "match_stats"):
            ev = ev.merge(store.read_table(league, "match_stats"), on="event_id", how="left")
        parts.append(ev)
    if not parts:
        raise ValueError(f"no tennis events on file for the {tour}'s tour; run `sportspred ingest` first")
    df = pd.concat(parts, ignore_index=True)
    if "round_order" not in df:
        df["round_order"] = 5
    df["round_order"] = df["round_order"].fillna(5)
    # A fixture imported by CSV has no archive statistics row: take what the event itself says.
    fmt = df["format_json"].map(lambda v: json.loads(v) if isinstance(v, str) and v else {})
    for key in ("surface", "best_of"):
        from_event = fmt.map(lambda d: d.get(key))
        df[key] = df[key].where(df[key].notna(), from_event) if key in df else from_event
    df = df.sort_values(["start_time_utc", "round_order", "event_id"]).reset_index(drop=True)
    df["week_start"] = week_start(df["start_time_utc"])
    # Archive rows carry the tournament's start date, whatever day the match was played.
    # They are results only once that week is over; until then the builder must not know them.
    dated_by_tournament = df["time_precision"].ne("datetime")
    known = ~dated_by_tournament | (df["week_start"] + pd.Timedelta(days=7) <= now_ts)
    too_early = df["completed"] & ~known
    df.loc[too_early, ["score_a", "val_a", "val_b"]] = np.nan
    df["completed"] = df["completed"] & known
    return df


def week_start(start_time_utc: pd.Series) -> pd.Series:
    """Monday 00:00 UTC of the week an event's date falls in."""
    day = start_time_utc.dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    return (day - pd.to_timedelta(day.dt.dayofweek, unit="D")).dt.tz_localize("UTC")


def _load_prev_weeks(df: pd.DataFrame, completed: np.ndarray, weeks: int = 2) -> np.ndarray:
    """Completed matches of each side in the ``weeks`` tournament weeks before
    this one (side A minus side B).  Matches of this week are never counted."""
    week = ((df["week_start"] - pd.Timestamp("1970-01-05", tz="UTC")).dt.days // 7).to_numpy()
    counts: Dict[Tuple[str, int], int] = {}
    for col in ("a_id", "b_id"):
        who = df[col].to_numpy()
        for i in np.flatnonzero(completed):
            key = (who[i], int(week[i]))
            counts[key] = counts.get(key, 0) + 1
    out = np.zeros((len(df), 2))
    for side, col in enumerate(("a_id", "b_id")):
        who = df[col].to_numpy()
        for i in range(len(df)):
            out[i, side] = sum(counts.get((who[i], int(week[i]) - k), 0) for k in range(1, weeks + 1))
    return out[:, 0] - out[:, 1]


def build_tennis_frame(store: Any, league: str, market: str = "winner", *, now: Any = None,
                       cfg: Optional[TennisConfig] = None) -> FeatureFrame:
    cfg = cfg or TennisConfig()
    now_ts = pd.Timestamp(utcnow() if now is None else parse_ts(now))
    tour = TENNIS_LEAGUES[league].tour
    df = _pooled(store, tour, now_ts)
    completed = df["completed"].to_numpy()
    who = pd.Index(pd.unique(pd.concat([df["a_id"], df["b_id"]])))
    a_idx, b_idx = who.get_indexer(df["a_id"]), who.get_indexer(df["b_id"])
    year = df["start_time_utc"].dt.year.to_numpy()
    years = np.sort(np.unique(year))
    season_idx = np.searchsorted(years, year)
    batch = pd.factorize(df["week_start"])[0]             # one batch per tournament week
    zeros = np.zeros(len(df))
    score = df["score_a"].to_numpy(dtype=float)

    burn = years[: cfg.burn_in_years]
    tune_mask = np.isin(year, burn[1:]) & completed
    if tune_mask.sum() >= 500:
        params, grid = tune_elo(a_idx, b_idx, score, season_idx, batch, zeros, tune_mask=tune_mask,
                                k_grid=(16, 24, 32, 40, 48), home_grid=(0,), carry_grid=(1.0,))
    else:
        params, grid = EloParams(k=32.0, home_adv=0.0, carry=1.0), []
    elo_a, elo_b, played_a = elo_pregame(a_idx, b_idx, score, season_idx, batch, zeros, params, n_entities=len(who))

    surface = df.get("surface", pd.Series(None, index=df.index)).astype("string").str.lower().fillna("")
    surf_diff = np.full(len(df), np.nan)
    for name in SURFACES:
        m = (surface == name).to_numpy()
        if m.sum() < 50:
            continue
        sa, sb, _ = elo_pregame(a_idx[m], b_idx[m], score[m], season_idx[m], batch[m], zeros[m], params,
                                n_entities=len(who))
        surf_diff[m] = sa - sb

    def share(side: str) -> np.ndarray:
        won = df.get(f"{side}_1stWon", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float) \
            + df.get(f"{side}_2ndWon", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
        pts = df.get(f"{side}_svpt", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(pts >= 20, won / pts, np.nan)
        return np.where(completed, out, np.nan)

    spw_a, spw_b = share("home"), share("away")           # stored home is side A (ID order)
    edges: Dict[str, np.ndarray] = {}
    for key, lr in (("serve_edge", cfg.slow_lr), ("serve_edge_fast", cfg.fast_lr)):
        r = offdef_pregame(a_idx, b_idx, spw_a, spw_b, season_idx, batch, zeros, lr=lr, carry=1.0,
                           n_entities=len(who))
        edge = (r["off_a"] + r["def_b"]) - (r["off_b"] + r["def_a"])
        edges[key] = np.where((r["n_a"] == 0) | (r["n_b"] == 0), np.nan, edge)

    rest_a, rest_b, rest_diff = _rest_and_experience(df, cfg.rest_cap_days)
    seen = np.zeros(len(who), dtype=int)
    played_b = np.zeros(len(df), dtype=int)
    starts = np.flatnonzero(np.r_[True, batch[1:] != batch[:-1]])
    bounds = np.r_[starts, len(df)]
    for s, e in zip(bounds[:-1], bounds[1:]):
        played_b[s:e] = seen[b_idx[s:e]]
        done = np.flatnonzero(completed[s:e]) + s
        np.add.at(seen, a_idx[done], 1)
        np.add.at(seen, b_idx[done], 1)
    best_of = pd.to_numeric(df.get("best_of", pd.Series(np.nan, index=df.index)), errors="coerce").to_numpy(dtype=float)
    bo5 = (best_of == 5).astype(float)

    last_of_batch = df["start_time_utc"].groupby(batch).transform("max")
    prev = pd.Series(pd.unique(last_of_batch)).shift().to_numpy()
    out = pd.DataFrame({
        "event_id": df["event_id"], "start_time_utc": df["start_time_utc"], "block": year,
        "season": df["season"], "status": df["status"], "league": df["league"],
        "y": np.where(completed, score, np.nan), "side_a_id": df["a_id"], "side_b_id": df["b_id"],
        "info_cutoff_utc": pd.to_datetime(pd.Series(prev[batch], index=df.index), utc=True),
        "elo_a": elo_a, "elo_b": elo_b, "elo_diff": elo_a - elo_b, "surf_elo_diff": surf_diff,
        "serve_edge": edges["serve_edge"], "serve_edge_fast": edges["serve_edge_fast"],
        "rest_diff": rest_diff, "load_14d_diff": _load_prev_weeks(df, completed),
        "exp_diff": np.log1p(played_a) - np.log1p(played_b), "games_played_a": played_a,
        "bo5": bo5, "elo_diff_bo5": (elo_a - elo_b) * bo5,
        "surface": surface.replace("", pd.NA),
    })
    out["dq_few_events"] = (np.minimum(played_a, played_b) < 10).astype(float)
    out["n_pending_prior"] = 0
    out = out[out["league"] == league].reset_index(drop=True)
    return FeatureFrame(
        league=league, market=market, outcome=OutcomeType.BINARY, classes=("b", "a"), df=out,
        feature_cols=list(TENNIS_FEATURES), feature_notes=dict(FEATURE_LABELS), point_in_time=True,
        limitations=[
            "Matches are dated by tournament week, not by day: a match never uses results from its own "
            "tournament or from any tournament starting the same week, including earlier rounds, so "
            "in-tournament form and fatigue are invisible.",
            "A two-week event counts as finished after its first week, and a tournament dated on a Sunday "
            "after one day; this reaches other players' ratings only.",
            "No withdrawal, injury or fitness information; no court speed, altitude, indoor/outdoor or weather.",
            "Rankings, seedings and head-to-head records are deliberately not inputs.",
            "The source is a snapshot mirror ending in mid-2026 with no upcoming matches.",
            "Retirements count as wins for the player who advanced; walkovers are excluded.",
            "Betting odds are not an input and no market benchmark is available from this source.",
        ],
        extra={"elo_params": params.__dict__, "elo_tuning_grid_size": len(grid),
               "burn_in_years": [int(b) for b in burn], "pooled_leagues": sorted(df["league"].unique()),
               "tuning_note": "Elo update rate tuned on the burn-in years only."},
    )
