"""Daily-fantasy pick'em entries (PrizePicks, Underdog, ...).

Pick'em operators do not price each leg.  You pick more/less on several
players and the *entry* pays a fixed multiple depending on how many legs hit.
So a leg has no odds of its own; what it has is a **break-even probability**:
the win probability every leg would need for the entry to break even.  A leg
whose fair probability is above that number adds value.

This module computes entry EV exactly (the number of hits follows a
Poisson-binomial distribution when legs are independent) and derives
break-evens from payout tables.

.. warning::
   Operators change payout tables, vary them by state, and attach modified
   multipliers to individual legs ("demons", "goblins", boosted or reduced
   picks).  The tables in :data:`PAYOUTS` are illustrative defaults - check
   the payouts shown in your app and pass your own table if they differ.
   Legs are assumed independent; legs from the same game are correlated and
   the true EV of such an entry differs from the figure computed here.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from scipy.optimize import brentq

__all__ = [
    "PAYOUTS",
    "hit_distribution",
    "entry_ev",
    "entry_win_prob",
    "breakeven_leg_prob",
    "leg_equivalent_decimal",
    "breakeven_table",
    "build_entry",
]

#: ``PAYOUTS[operator][entry_type][n_legs] = {hits: multiple of stake returned}``.
#: Multiples are total returns (a 3x entry returns 3 units for 1 staked).
PAYOUTS: Dict[str, Dict[str, Dict[int, Dict[int, float]]]] = {
    "prizepicks": {
        "power": {
            2: {2: 3.0},
            3: {3: 5.0},
            4: {4: 10.0},
            5: {5: 20.0},
            6: {6: 37.5},
        },
        "flex": {
            3: {3: 2.25, 2: 1.25},
            4: {4: 5.0, 3: 1.5},
            5: {5: 10.0, 4: 2.0, 3: 0.4},
            6: {6: 25.0, 5: 2.0, 4: 0.4},
        },
    },
    "underdog": {
        "standard": {
            2: {2: 3.0},
            3: {3: 6.0},
            4: {4: 10.0},
            5: {5: 20.0},
        },
        "flex": {
            3: {3: 3.0, 2: 1.0},
            4: {4: 6.0, 3: 1.5},
            5: {5: 10.0, 4: 2.5},
        },
    },
}


def hit_distribution(probs: Sequence[float]) -> List[float]:
    """Distribution of the number of winning legs (independent legs).

    Returns ``d`` with ``d[k] = P(exactly k legs win)``.
    """
    dist = [1.0]
    for p in probs:
        p = float(p)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"leg probability must be in [0, 1], got {p!r}")
        nxt = [0.0] * (len(dist) + 1)
        for k, mass in enumerate(dist):
            nxt[k] += mass * (1.0 - p)
            nxt[k + 1] += mass * p
        dist = nxt
    return dist


def _check_payouts(n_legs: int, payouts: Mapping[int, float]) -> None:
    for hits, multiple in payouts.items():
        if not 0 <= int(hits) <= n_legs:
            raise ValueError(f"payout for {hits} hits is impossible in a {n_legs}-leg entry")
        if multiple < 0:
            raise ValueError("payout multiples must be >= 0")


def entry_ev(probs: Sequence[float], payouts: Mapping[int, float]) -> float:
    """Expected profit per unit staked on an entry with these leg probabilities."""
    _check_payouts(len(probs), payouts)
    dist = hit_distribution(probs)
    return sum(dist[int(hits)] * multiple for hits, multiple in payouts.items()) - 1.0


def entry_win_prob(probs: Sequence[float], payouts: Mapping[int, float]) -> float:
    """Probability the entry returns more than the stake."""
    _check_payouts(len(probs), payouts)
    dist = hit_distribution(probs)
    return sum(dist[int(hits)] for hits, multiple in payouts.items() if multiple > 1.0)


def breakeven_leg_prob(n_legs: int, payouts: Mapping[int, float]) -> float:
    """Per-leg win probability at which an entry of equal legs breaks even."""
    if n_legs < 1:
        raise ValueError("an entry needs at least one leg")
    _check_payouts(n_legs, payouts)

    def f(p: float) -> float:
        return entry_ev([p] * n_legs, payouts)

    if f(1.0) <= 0.0:
        raise ValueError("this payout table never breaks even")
    return float(brentq(f, 0.0, 1.0, xtol=1e-12))


def leg_equivalent_decimal(n_legs: int, payouts: Mapping[int, float]) -> float:
    """Decimal odds a single bet would need to match the entry's break-even."""
    return 1.0 / breakeven_leg_prob(n_legs, payouts)


def breakeven_table(payout_book: Optional[Mapping[str, Mapping[str, Mapping[int, Mapping[int, float]]]]] = None) -> pd.DataFrame:
    """Break-even leg probability of every entry type in a payout book."""
    payout_book = payout_book or PAYOUTS
    rows = []
    for operator, types in payout_book.items():
        for entry_type, by_legs in types.items():
            for n_legs, payouts in by_legs.items():
                breakeven = breakeven_leg_prob(int(n_legs), payouts)
                rows.append(
                    {
                        "operator": operator,
                        "entry_type": entry_type,
                        "legs": int(n_legs),
                        "breakeven": breakeven,
                        "equivalent_decimal": 1.0 / breakeven,
                        "top_payout": max(payouts.values()),
                    }
                )
    return pd.DataFrame(rows).sort_values(["operator", "breakeven"]).reset_index(drop=True)


def build_entry(
    legs: pd.DataFrame,
    n_legs: int,
    payouts: Mapping[int, float],
    prob_column: str = "fair_prob",
    one_per_event: bool = False,
) -> Tuple[pd.DataFrame, float]:
    """Assemble the highest-EV entry from candidate legs.

    ``legs`` is a slice of scanner output for one pick'em operator.  The
    ``n_legs`` most probable legs are taken, at most one per player (operators
    do not allow two picks on one player), and with ``one_per_event`` at most
    one per game, which keeps the legs close to independent.

    Returns ``(chosen_legs, expected_value_per_unit)``.  Raises ``ValueError``
    if there are not enough eligible legs.
    """
    ordered = legs.sort_values(prob_column, ascending=False, kind="stable")
    ordered = ordered.drop_duplicates("player_key", keep="first")
    if one_per_event:
        ordered = ordered.drop_duplicates("event_id", keep="first")
    if len(ordered) < n_legs:
        raise ValueError(f"need {n_legs} eligible legs, only {len(ordered)} available")
    chosen = ordered.head(n_legs)
    return chosen.reset_index(drop=True), entry_ev(chosen[prob_column].tolist(), payouts)
