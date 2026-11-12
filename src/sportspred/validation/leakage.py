"""Generic leakage tests that every sport plugin must pass.

**As-of invariance.**  A plugin's ``feature_frame(store, league, market,
now=T)`` must treat every event starting after ``T`` as upcoming and ignore
anything the store knows about it.  If features are truly point-in-time, the
features of the first slate after ``T`` computed "as of T" are identical to
the features the full-history build gives those same events.  Any difference
means a feature depends on information from the event's own slate or later.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..core.timeutil import DISPLAY_TZ_NAME

__all__ = ["asof_invariance", "assert_asof_invariance", "default_cutoffs"]


def default_cutoffs(completed: pd.DataFrame, n: int = 4, seed: int = 0) -> List[pd.Timestamp]:
    """Cut-off instants spread over the later 60% of the history, each one
    second before the first event of a slate."""
    starts = completed["start_time_utc"].sort_values().reset_index(drop=True)
    days = starts.dt.tz_convert(DISPLAY_TZ_NAME).dt.strftime("%Y-%m-%d")
    first_of_day = starts[~days.duplicated()].reset_index(drop=True)
    lo = int(len(first_of_day) * 0.4)
    rng = np.random.default_rng(seed)
    picks = sorted(rng.choice(np.arange(lo, len(first_of_day)), size=min(n, len(first_of_day) - lo), replace=False))
    return [first_of_day[i] - pd.Timedelta(seconds=1) for i in picks]


def asof_invariance(plugin: Any, store: Any, league: str, market: str, *,
                    cutoffs: Optional[Sequence[pd.Timestamp]] = None, n_cutoffs: int = 4,
                    columns: Optional[Sequence[str]] = None, atol: float = 1e-9) -> pd.DataFrame:
    """Compare as-of features with full-history features.

    Returns one row per (cutoff, feature) with the number of compared events
    and the largest absolute difference; all differences must be ~0.
    """
    full = plugin.feature_frame(store, league, market)
    cols = list(columns) if columns is not None else list(full.feature_cols)
    cutoffs = list(cutoffs) if cutoffs is not None else default_cutoffs(full.completed, n_cutoffs)
    full_df = full.df.set_index("event_id")
    rows: List[Dict[str, Any]] = []
    for cutoff in cutoffs:
        asof = plugin.feature_frame(store, league, market, now=cutoff.to_pydatetime())
        up = asof.upcoming.sort_values("start_time_utc")
        if up.empty:
            continue
        first_day = up["start_time_utc"].dt.tz_convert(DISPLAY_TZ_NAME).dt.strftime("%Y-%m-%d")
        slate = up[first_day == first_day.iloc[0]].set_index("event_id")
        common = slate.index.intersection(full_df.index)
        if not len(common):
            continue
        for col in cols:
            a = slate.loc[common, col].to_numpy(dtype=float)
            b = full_df.loc[common, col].to_numpy(dtype=float)
            both_nan = np.isnan(a) & np.isnan(b)
            diff = np.where(both_nan, 0.0, np.abs(a - b))
            diff = np.where(np.isnan(diff), np.inf, diff)   # NaN on one side only is a mismatch
            rows.append({"cutoff": cutoff, "feature": col, "n_events": int(len(common)),
                         "max_abs_diff": float(diff.max()), "ok": bool(diff.max() <= atol)})
    return pd.DataFrame(rows)


def assert_asof_invariance(plugin: Any, store: Any, league: str, market: str, **kwargs: Any) -> pd.DataFrame:
    result = asof_invariance(plugin, store, league, market, **kwargs)
    if result.empty:
        raise AssertionError("as-of invariance could not be evaluated: no comparable events")
    bad = result[~result["ok"]]
    if len(bad):
        raise AssertionError(
            "features depend on information from their own slate or later:\n"
            + bad.sort_values("max_abs_diff", ascending=False).head(10).to_string(index=False))
    return result
