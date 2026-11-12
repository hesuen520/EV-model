"""Point-in-time tennis features for the match-winner market.

Ratings are built on the pooled results of a whole tour (tour-level main
draw, qualifying, Challenger / WTA 125 and ITF together), because a player's
strength does not reset when he or she changes level; the model for one
league is then fitted and validated on that league's matches only.

**Time ordering.**  The source dates a match by its tournament's start date.
All matches of one tournament week therefore form one batch: a match never
uses another match of its own tournament, not even an earlier round.  That
costs some information and removes any chance of using a later round to
"predict" an earlier one.  Two-week events overlap the following week's
tournaments; ratings are pairwise, so a player's rating still only contains
his or her own earlier matches.

Side A is the player whose ID sorts first - never the winner.

Features (weights learned in validation):

* ``elo_diff`` - opponent-adjusted results on all surfaces;
* ``surf_elo_diff`` - the same on this match's surface only;
* ``serve_edge`` / ``serve_edge_fast`` - opponent-adjusted share of service
  points won, A's serve against B's return minus the reverse, with a slow and
  a fast memory;
* ``rest_diff``, ``load_14d_diff`` - days since the previous tournament and
  matches played in the previous 14 days (fatigue, or a lay-off);
* ``exp_diff`` - rated matches behind each player;
* ``bo5``, ``elo_diff_bo5`` - best-of-five matches favour the stronger player.

Not used: official rankings and seedings (reputation is not an input) and
head-to-head records (tiny samples).
"""

from __future__ import annotations

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
    "load_14d_diff": "matches played in the previous 14 days",
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
    return df.sort_values(["start_time_utc", "round_order", "event_id"]).reset_index(drop=True)


def _load_14d(df: pd.DataFrame, completed: np.ndarray) -> np.ndarray:
    """Completed matches of each side dated in the 14 days before this
    tournament's start (same-date matches excluded)."""
    n = len(df)
    long = pd.concat([
        pd.DataFrame({"row": np.arange(n), "who": df["a_id"].to_numpy(), "side": 0}),
        pd.DataFrame({"row": np.arange(n), "who": df["b_id"].to_numpy(), "side": 1}),
    ], ignore_index=True)
    long["t"] = df["start_time_utc"].to_numpy()[long["row"]]
    long["done"] = completed[long["row"]]
    long = long.sort_values(["who", "t", "row"]).reset_index(drop=True)
    days = (long["t"].astype("int64") // (86400 * 10 ** (9 if str(long["t"].dtype).endswith("ns, UTC]") else 6))).to_numpy()
    who, done = long["who"].to_numpy(), long["done"].to_numpy()
    out = np.zeros(len(long))
    start = same = 0
    for i in range(len(long)):
        if i and who[i] != who[i - 1]:
            start = same = i
        while days[i] - days[start] > 14:
            start += 1
        if days[same] != days[i]:
            same = i
            while same > start and days[same - 1] == days[i]:
                same -= 1
        # rows start..same-1 are earlier dates within 14 days
        lo = start
        hi = i
        while hi > lo and days[hi - 1] == days[i]:
            hi -= 1
        out[i] = done[lo:hi].sum()
    load = np.zeros((n, 2))
    load[long["row"].to_numpy(), long["side"].to_numpy()] = out
    return load[:, 0] - load[:, 1]


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
    batch = pd.factorize(df["start_time_utc"].dt.strftime("%Y-%m-%d"))[0]
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
        "rest_diff": rest_diff, "load_14d_diff": _load_14d(df, completed),
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
            "tournament, including earlier rounds, so in-tournament form and fatigue are invisible.",
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
