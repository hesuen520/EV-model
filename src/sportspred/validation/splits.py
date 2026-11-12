"""Chronological splits: walk-forward development folds plus a reserved final test.

Blocks (usually seasons) are ordered by time.  Every fold is

    train blocks  <  calibration block(s)  <  test block

with strict time ordering asserted, and all rows of one event sit in one block
because a frame has one row per event and a block is a function of the event.
The last ``n_final_blocks`` full blocks (plus any trailing partial block) form
the final test period: no model selection, tuning or calibration may see it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import pandas as pd

from ..sports.base import SplitPlan

__all__ = ["Fold", "FoldSet", "make_folds", "LeakageError", "assert_chronological"]


class LeakageError(AssertionError):
    """A split or feature frame would let future information into the past."""


@dataclass
class Fold:
    name: str
    train_blocks: List[Any]
    calib_blocks: List[Any]
    test_blocks: List[Any]
    is_final: bool = False

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "train_blocks": [str(b) for b in self.train_blocks],
            "calib_blocks": [str(b) for b in self.calib_blocks],
            "test_blocks": [str(b) for b in self.test_blocks],
            "is_final": self.is_final,
        }


@dataclass
class FoldSet:
    blocks: List[Any]
    dev: List[Fold]
    final: Fold
    block_ranges: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def ordered_blocks(df: pd.DataFrame, block_col: str) -> pd.DataFrame:
    """Blocks ordered by their first event, with event counts and date range."""
    g = df.groupby(block_col, sort=False)["start_time_utc"].agg(["min", "max", "size"]).reset_index()
    g = g.sort_values(["min", block_col]).reset_index(drop=True)
    # A block must not interleave with the next one in time.
    for i in range(1, len(g)):
        if g.loc[i, "min"] <= g.loc[i - 1, "max"]:
            raise LeakageError(
                f"blocks {g.loc[i - 1, block_col]!r} and {g.loc[i, block_col]!r} overlap in time; "
                "fold boundaries must be calendar/event boundaries"
            )
    return g


def make_folds(completed: pd.DataFrame, plan: SplitPlan) -> FoldSet:
    """Build the fold set from completed events only."""
    if completed.empty:
        raise ValueError("no completed events to split")
    g = ordered_blocks(completed, plan.block_col)
    blocks = g[plan.block_col].tolist()
    sizes = dict(zip(blocks, g["size"]))

    # Final period: the last n_final full blocks plus any trailing partial block.
    final_blocks: List[Any] = []
    full = 0
    for b in reversed(blocks):
        final_blocks.insert(0, b)
        if sizes[b] >= plan.min_block_events:
            full += 1
        if full >= plan.n_final_blocks:
            break
    pre = [b for b in blocks if b not in final_blocks]
    usable = pre[plan.burn_in_blocks:]
    need = plan.min_train_blocks + plan.n_calib_blocks
    if len(usable) < need:
        raise ValueError(
            f"not enough history for an honest split: {len(blocks)} blocks, "
            f"{len(final_blocks)} reserved for the final test, {plan.burn_in_blocks} burn-in, "
            f"{len(usable)} left but {need} needed before any test block"
        )

    dev: List[Fold] = []
    for i in range(need, len(usable)):
        dev.append(Fold(
            name=f"dev:{usable[i]}",
            train_blocks=usable[: i - plan.n_calib_blocks],
            calib_blocks=usable[i - plan.n_calib_blocks: i],
            test_blocks=[usable[i]],
        ))
    if plan.max_dev_folds is not None:
        dev = dev[-plan.max_dev_folds:]
    final = Fold(
        name="final",
        train_blocks=usable[: len(usable) - plan.n_calib_blocks],
        calib_blocks=usable[len(usable) - plan.n_calib_blocks:],
        test_blocks=final_blocks,
        is_final=True,
    )
    ranges = {
        str(r[plan.block_col]): {"first": r["min"].isoformat(), "last": r["max"].isoformat(), "n": int(r["size"])}
        for _, r in g.iterrows()
    }
    return FoldSet(blocks=blocks, dev=dev, final=final, block_ranges=ranges)


def assert_chronological(df: pd.DataFrame, fold: Fold, block_col: str) -> None:
    """Train strictly before calibration strictly before test, by event time."""
    def span(blocks: Sequence[Any]):
        part = df[df[block_col].isin(blocks)]["start_time_utc"]
        return (part.min(), part.max()) if len(part) else (None, None)

    (_, train_max), (cal_min, cal_max), (test_min, _) = (
        span(fold.train_blocks), span(fold.calib_blocks), span(fold.test_blocks)
    )
    if train_max is not None and cal_min is not None and not train_max < cal_min:
        raise LeakageError(f"{fold.name}: training rows are not strictly before calibration rows")
    if cal_max is not None and test_min is not None and not cal_max < test_min:
        raise LeakageError(f"{fold.name}: calibration rows are not strictly before test rows")
    if train_max is not None and test_min is not None and not train_max < test_min:
        raise LeakageError(f"{fold.name}: training rows are not strictly before test rows")
    overlap = (set(fold.train_blocks) & set(fold.test_blocks)) | (set(fold.calib_blocks) & set(fold.test_blocks)) \
        | (set(fold.train_blocks) & set(fold.calib_blocks))
    if overlap:
        raise LeakageError(f"{fold.name}: blocks shared between train/calibration/test: {sorted(map(str, overlap))}")
