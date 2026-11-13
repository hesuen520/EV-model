"""Point-in-time, opponent-adjusted rating systems shared by all sports.

Every function here walks events in chronological order and returns the
ratings as they stood *before* each event.  Events that share a start instant
(or, for date-only sources, a calendar date) form one batch: all of a batch's
pre-event ratings are read before any of its results are applied, so nothing
from an event that had not finished can reach another event's features.

Nothing in this module decides how much a rating matters - that weight is
learned by the models during validation.  The only tunable numbers are update
rates, and those are tuned on the earliest (burn-in) seasons alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "batch_ids",
    "EloParams",
    "elo_pregame",
    "tune_elo",
    "elo_expected",
    "offdef_pregame",
]


def batch_ids(start_time_utc: pd.Series, date_only: Optional[pd.Series] = None) -> np.ndarray:
    """Integer batch label per event (events must already be sorted by time).

    Events with the same start instant share a batch.  Where ``date_only`` is
    true the whole calendar date is one batch, because without start times we
    cannot know which of two same-day events finished first.
    """
    key = start_time_utc.dt.strftime("%Y-%m-%dT%H:%M:%S")
    if date_only is not None:
        key = key.where(~date_only.astype(bool), start_time_utc.dt.strftime("%Y-%m-%d"))
        # A timed event on a date that also has date-only events joins the date batch.
        dates = start_time_utc.dt.strftime("%Y-%m-%d")
        mixed = dates.isin(set(dates[date_only.astype(bool)]))
        key = key.where(~mixed, dates)
    return pd.factorize(key, sort=False)[0]


@dataclass(frozen=True)
class EloParams:
    k: float = 20.0
    home_adv: float = 60.0         # rating points added to the home side (0 at neutral sites)
    carry: float = 0.75            # share of a rating's distance from the mean kept across seasons
    margin_scale: Optional[float] = None   # None: ignore margin of victory
    scale: float = 400.0
    init: float = 1500.0


def elo_expected(diff: np.ndarray, scale: float = 400.0) -> np.ndarray:
    """Win expectancy of the side that is ``diff`` rating points ahead."""
    return 1.0 / (1.0 + 10.0 ** (-np.asarray(diff, dtype=float) / scale))


def elo_pregame(
    a_idx: np.ndarray,
    b_idx: np.ndarray,
    score_a: np.ndarray,
    season_idx: np.ndarray,
    batch: np.ndarray,
    home_edge: np.ndarray,
    params: EloParams,
    *,
    margin: Optional[np.ndarray] = None,
    n_entities: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-event Elo ratings.

    Parameters are parallel arrays over events sorted by time: entity indices
    of side A and side B; ``score_a`` in {1, 0.5, 0} (NaN for events without a
    result, which are rated but never update anything); the season index
    (ratings regress to the mean when an entity's season changes); the batch
    label; ``home_edge`` (+1 when side A is at home, -1 when side B is, 0 at a
    neutral site); optional ``margin`` (A's score minus B's).

    Returns ``(elo_a, elo_b, games_played_a)`` before each event, where the
    ratings do **not** include the home advantage.
    """
    n = len(a_idx)
    n_entities = int(max(a_idx.max(), b_idx.max()) + 1) if n_entities is None else n_entities
    rating = np.full(n_entities, params.init)
    last_season = np.full(n_entities, -1)
    played = np.zeros(n_entities, dtype=int)
    out_a, out_b, out_n = np.empty(n), np.empty(n), np.empty(n, dtype=int)
    use_margin = margin is not None and params.margin_scale is not None

    i = 0
    while i < n:
        j = i
        while j < n and batch[j] == batch[i]:
            j += 1
        # Season carry-over, then read every pre-event rating of the batch.
        for t in range(i, j):
            for e in (a_idx[t], b_idx[t]):
                if last_season[e] != season_idx[t]:
                    if last_season[e] >= 0:
                        rating[e] = params.init + params.carry * (rating[e] - params.init)
                    last_season[e] = season_idx[t]
        out_a[i:j] = rating[a_idx[i:j]]
        out_b[i:j] = rating[b_idx[i:j]]
        out_n[i:j] = played[a_idx[i:j]]
        # Apply the batch's results.
        deltas: List[Tuple[int, float]] = []
        for t in range(i, j):
            s = score_a[t]
            if s != s:  # no result: nothing to learn from
                continue
            diff = out_a[t] - out_b[t] + params.home_adv * home_edge[t]
            expected = 1.0 / (1.0 + 10.0 ** (-diff / params.scale))
            g = 1.0
            if use_margin and margin[t] == margin[t]:
                # Larger wins move ratings more, with diminishing returns; the
                # denominator damps the extra credit a favourite gets for
                # running up the score (autocorrelation correction).
                winner_diff = diff if s >= 0.5 else -diff
                g = np.log1p(abs(margin[t]) / params.margin_scale) * 2.2 / (winner_diff * 0.001 + 2.2)
            step = params.k * g * (s - expected)
            deltas.append((a_idx[t], step))
            deltas.append((b_idx[t], -step))
            played[a_idx[t]] += 1
            played[b_idx[t]] += 1
        for e, step in deltas:
            rating[e] += step
        i = j
    return out_a, out_b, out_n


def tune_elo(
    a_idx: np.ndarray,
    b_idx: np.ndarray,
    score_a: np.ndarray,
    season_idx: np.ndarray,
    batch: np.ndarray,
    home_edge: np.ndarray,
    *,
    tune_mask: np.ndarray,
    margin: Optional[np.ndarray] = None,
    k_grid: Sequence[float] = (8, 12, 16, 20, 26, 32),
    home_grid: Sequence[float] = (0, 30, 60, 90, 120),
    carry_grid: Sequence[float] = (0.5, 0.67, 0.8, 1.0),
    margin_grid: Sequence[Optional[float]] = (None,),
    scale: float = 400.0,
) -> Tuple[EloParams, List[Dict[str, float]]]:
    """Grid-search Elo update parameters by log loss on ``tune_mask`` events.

    Only events up to the last tuning event are ever touched, so later
    (development and final-test) events cannot influence the choice.  The mask
    should exclude the earliest stretch so ratings have warmed up.
    """
    tune_mask = np.asarray(tune_mask, dtype=bool)
    if not tune_mask.any():
        raise ValueError("no events in the tuning period")
    end = int(np.flatnonzero(tune_mask).max()) + 1
    sl = slice(0, end)
    rows: List[Dict[str, float]] = []
    best: Optional[EloParams] = None
    best_ll = np.inf
    scored = tune_mask[sl] & ~np.isnan(score_a[sl]) & (score_a[sl] != 0.5)
    for k, home, carry, mscale in product(k_grid, home_grid, carry_grid, margin_grid):
        params = EloParams(k=k, home_adv=home, carry=carry, margin_scale=mscale, scale=scale)
        ea, eb, _ = elo_pregame(a_idx[sl], b_idx[sl], score_a[sl], season_idx[sl], batch[sl], home_edge[sl],
                                params, margin=None if margin is None else margin[sl],
                                n_entities=int(max(a_idx.max(), b_idx.max()) + 1))
        p = np.clip(elo_expected(ea - eb + home * home_edge[sl], scale), 1e-9, 1 - 1e-9)
        y = score_a[sl]
        ll = float(-np.mean(np.where(y[scored] == 1.0, np.log(p[scored]), np.log(1 - p[scored]))))
        rows.append({"k": k, "home_adv": home, "carry": carry,
                     "margin_scale": -1.0 if mscale is None else mscale, "log_loss": ll})
        if ll < best_ll:
            best_ll, best = ll, params
    assert best is not None
    return best, rows


def offdef_pregame(
    a_idx: np.ndarray,
    b_idx: np.ndarray,
    val_a: np.ndarray,
    val_b: np.ndarray,
    season_idx: np.ndarray,
    batch: np.ndarray,
    home_edge: np.ndarray,
    *,
    lr: float,
    carry: float = 0.7,
    mean_lr: float = 0.002,
    n_entities: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Opponent-adjusted offence/defence ratings from a per-event quantity.

    ``val_a`` is what side A produced against side B (points per 100
    possessions, goals, runs ...) and ``val_b`` the reverse; NaN where the
    event has no result or no statistics.  The running model is

        expected(A vs B) = league_mean + off[A] + def[B] + home/2 * home_edge

    where ``def`` is what a side concedes above the league mean.  After each
    event the surprise (actual minus expected) moves A's offence and B's
    defence by ``lr`` times the surprise - so a big night against a strong
    defence counts for more than the same night against a weak one.  When a
    side has several events in one batch the updates are damped so the batch
    as a whole can never overshoot.  ``lr``
    sets the memory: roughly the last ``2/lr - 1`` events dominate, e.g.
    ``lr=0.25`` ~ the last 7 and ``lr=0.05`` ~ the last 39.

    Returns pre-event arrays: ``off_a, def_a, off_b, def_b, mean, home,
    n_a, n_b`` (events with statistics seen so far for each side).
    """
    n = len(a_idx)
    n_entities = int(max(a_idx.max(), b_idx.max()) + 1) if n_entities is None else n_entities
    off = np.zeros(n_entities)
    dfn = np.zeros(n_entities)
    seen = np.zeros(n_entities, dtype=int)
    last_season = np.full(n_entities, -1)
    finite = np.concatenate([val_a[np.isfinite(val_a)][:200], val_b[np.isfinite(val_b)][:200]])
    mean = float(finite.mean()) if len(finite) else 0.0
    home = 0.0
    out = {k: np.empty(n) for k in ("off_a", "def_a", "off_b", "def_b", "mean", "home")}
    out["n_a"] = np.empty(n, dtype=int)
    out["n_b"] = np.empty(n, dtype=int)

    i = 0
    while i < n:
        j = i
        while j < n and batch[j] == batch[i]:
            j += 1
        for t in range(i, j):
            for e in (a_idx[t], b_idx[t]):
                if last_season[e] != season_idx[t]:
                    if last_season[e] >= 0:
                        off[e] *= carry
                        dfn[e] *= carry
                    last_season[e] = season_idx[t]
        a, b = a_idx[i:j], b_idx[i:j]
        out["off_a"][i:j], out["def_a"][i:j] = off[a], dfn[a]
        out["off_b"][i:j], out["def_b"][i:j] = off[b], dfn[b]
        out["mean"][i:j], out["home"][i:j] = mean, home
        out["n_a"][i:j], out["n_b"][i:j] = seen[a], seen[b]
        updates: List[Tuple[int, int, float]] = []
        mean_new, home_new = mean, home
        for t in range(i, j):
            va, vb = val_a[t], val_b[t]
            if va != va or vb != vb:
                continue
            h = home * home_edge[t] / 2.0
            exp_a = mean + out["off_a"][t] + out["def_b"][t] + h
            exp_b = mean + out["off_b"][t] + out["def_a"][t] - h
            ea, eb = va - exp_a, vb - exp_b
            updates.append((a_idx[t], b_idx[t], ea))
            updates.append((b_idx[t], a_idx[t], eb))
            mean_new += mean_lr * ((va + vb) / 2.0 - mean_new)
            if home_edge[t] != 0:
                home_new += mean_lr * 2.0 * ((va - vb) * home_edge[t] - (
                    (out["off_a"][t] + out["def_b"][t] - out["off_b"][t] - out["def_a"][t]) * home_edge[t]
                ) - home_new)
            seen[a_idx[t]] += 1
            seen[b_idx[t]] += 1
        # An entity with several events in one batch (a pooled opponent, a
        # player through a tournament week) must not take the plain sum of the
        # updates: n updates at rate lr would have gain n * lr, which
        # overshoots and can diverge.  Each update is weighted so that the
        # batch has the gain of n successive updates, 1 - (1 - lr)^n <= 1.
        # With one event per batch this is exactly lr.
        n_off: Dict[int, int] = {}
        n_def: Dict[int, int] = {}
        for attacker, defender, _ in updates:
            n_off[attacker] = n_off.get(attacker, 0) + 1
            n_def[defender] = n_def.get(defender, 0) + 1
        for attacker, defender, err in updates:
            ka, kd = n_off[attacker], n_def[defender]
            off[attacker] += (lr if ka == 1 else (1.0 - (1.0 - lr) ** ka) / ka) * err
            dfn[defender] += (lr if kd == 1 else (1.0 - (1.0 - lr) ** kd) / kd) * err
        mean, home = mean_new, home_new
        i = j
    return out
