"""Validation of the complete combo selection process over historical slates.

For every historical Pacific/Honolulu slate the same rule used live is
applied to the out-of-sample predictions of the selected model: take the legs
whose probability clears the minimum, pick the ``k`` most probable, and
compare the estimated all-win probability with what happened.

Two things this measures that single-leg calibration cannot:

* **selection effect** - choosing the most confident legs of a slate also
  selects the legs whose probability is most likely to be overstated;
* **dependence** - if legs on a slate were positively dependent the product
  of their probabilities would understate (and negative dependence overstate)
  the all-win frequency.

The leg-probability minimum is a selection rule.  ``tune_min_leg_prob`` looks
at development slates only; the final-test slates are reported once for the
configured rule and never used to choose it.

Limitation: historical pregame check states do not exist, so this backtest
covers the probability and selection logic for all legs, not the FINAL-status
filter that is applied live.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..validation.backtest import slate_dates
from ..validation.metrics import wilson_interval

__all__ = ["combo_backtest", "tune_min_leg_prob"]


def _slate_table(oos: pd.DataFrame) -> pd.DataFrame:
    """One row per event of the selected model: slate, top probability, hit."""
    sel = oos[oos["selected"]].copy()
    cols = [c for c in sel.columns if c.startswith("p_")]
    P = sel[cols].to_numpy(dtype=float)
    sel["top"] = P.max(axis=1)
    sel["hit"] = (P.argmax(axis=1) == sel["y"].to_numpy().astype(int)).astype(float)
    sel["slate"] = slate_dates(sel["start_time_utc"])
    return sel[["event_id", "slate", "top", "hit", "is_final_test", "block"]]


def _evaluate(table: pd.DataFrame, sizes: Sequence[int], min_leg_prob: float) -> List[Dict[str, Any]]:
    n_slates = table["slate"].nunique()
    rows: List[Dict[str, Any]] = []
    eligible = table[table["top"] >= min_leg_prob]
    for k in sizes:
        predicted, won = [], []
        for _, slate in eligible.groupby("slate", sort=False):
            if len(slate) < k:
                continue
            best = slate.nlargest(k, "top")
            predicted.append(float(best["top"].prod()))
            won.append(float(best["hit"].all()))
        n = len(predicted)
        row: Dict[str, Any] = {"legs": k, "min_leg_prob": float(min_leg_prob), "n_slates": int(n_slates),
                               "n_combos": n, "coverage": float(n / n_slates) if n_slates else float("nan")}
        if n:
            lo, hi = wilson_interval(sum(won), n)
            pred, obs = float(np.mean(predicted)), float(np.mean(won))
            row.update({
                "mean_predicted_all_win": pred, "observed_all_win": obs, "observed_low": lo,
                "observed_high": hi, "observed_minus_predicted": obs - pred,
                "brier": float(np.mean((np.array(predicted) - np.array(won)) ** 2)),
                "predicted_inside_interval": bool(lo <= pred <= hi),
            })
        rows.append(row)
    return rows


def combo_backtest(oos: pd.DataFrame, *, min_leg_prob: float, sizes: Sequence[int] = (1, 2, 3, 4, 5, 6)
                   ) -> Dict[str, List[Dict[str, Any]]]:
    """Development-period and final-test results of the top-k-by-probability
    rule at the configured leg minimum."""
    table = _slate_table(oos)
    return {
        "development": _evaluate(table[~table["is_final_test"]], sizes, min_leg_prob),
        "final_test": _evaluate(table[table["is_final_test"]], sizes, min_leg_prob),
    }


def tune_min_leg_prob(oos: pd.DataFrame, *, size: int = 3,
                      grid: Sequence[float] = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80)) -> pd.DataFrame:
    """Coverage and all-win rate of the ``size``-leg rule for each candidate
    leg minimum, on **development slates only**."""
    table = _slate_table(oos)
    dev = table[~table["is_final_test"]]
    return pd.DataFrame([_evaluate(dev, [size], g)[0] for g in grid])
