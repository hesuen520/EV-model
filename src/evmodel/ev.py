"""Expected value and Kelly staking for a single bet.

Conventions: ``decimal`` is the decimal price (stake included), ``b = decimal
- 1`` the net profit per unit staked on a win.  A bet has three outcomes:
win with probability ``p_win``, push (stake returned) with ``p_push``, and
lose with ``p_lose = 1 - p_win - p_push``.
"""

from __future__ import annotations

import math
from typing import Optional

__all__ = [
    "expected_value",
    "edge",
    "kelly_fraction",
    "fractional_kelly",
    "shrink_prob",
    "bet_variance",
    "expected_log_growth",
]


def _check(p_win: float, p_push: float, decimal: float) -> float:
    if not decimal > 1.0 or not math.isfinite(decimal):
        raise ValueError(f"decimal odds must be finite and > 1, got {decimal!r}")
    if not 0.0 <= p_win <= 1.0:
        raise ValueError(f"p_win must be in [0, 1], got {p_win!r}")
    if not 0.0 <= p_push <= 1.0:
        raise ValueError(f"p_push must be in [0, 1], got {p_push!r}")
    p_lose = 1.0 - p_win - p_push
    if p_lose < -1e-9:
        raise ValueError("p_win + p_push must not exceed 1")
    return max(p_lose, 0.0)


def expected_value(p_win: float, decimal: float, p_push: float = 0.0) -> float:
    """Expected profit per unit staked: ``p_win * (d - 1) - p_lose``.

    ``0.03`` means the bet returns 3 cents of profit per dollar staked on
    average.  A push returns the stake and contributes nothing.
    """
    p_lose = _check(p_win, p_push, decimal)
    return p_win * (decimal - 1.0) - p_lose


def edge(p_win: float, decimal: float, p_push: float = 0.0) -> float:
    """Probability edge: win probability minus the price's break-even.

    The break-even accounts for pushes: ``(1 - p_push) / decimal``.
    """
    _check(p_win, p_push, decimal)
    return p_win - (1.0 - p_push) / decimal


def kelly_fraction(p_win: float, decimal: float, p_push: float = 0.0) -> float:
    """Full-Kelly stake as a fraction of bankroll (0 when the bet is not +EV).

    Maximises expected log growth ``p_win*log(1 + f*b) + p_lose*log(1 - f)``,
    which gives ``f* = (p_win*b - p_lose) / (b * (p_win + p_lose))``.  With no
    push this is the familiar ``(p*b - q) / b``.
    """
    p_lose = _check(p_win, p_push, decimal)
    b = decimal - 1.0
    denom = b * (p_win + p_lose)
    if denom <= 0.0:
        return 0.0
    f = (p_win * b - p_lose) / denom
    return min(max(f, 0.0), 1.0)


def shrink_prob(p_model: float, p_breakeven: float, confidence: float) -> float:
    """Shrink an estimated probability towards the price's break-even.

    ``confidence`` in [0, 1] is the weight on our estimate; the remainder is
    placed on the hypothesis that the market price is right.  Because Kelly
    staking is very sensitive to over-estimated edges, shrinking the
    probability first is a principled way to bet less when the estimate is
    uncertain.
    """
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    return confidence * p_model + (1.0 - confidence) * p_breakeven


def fractional_kelly(
    p_win: float,
    decimal: float,
    p_push: float = 0.0,
    fraction: float = 0.25,
    cap: Optional[float] = None,
) -> float:
    """Kelly stake scaled by ``fraction`` and limited to ``cap`` of bankroll.

    Fractional Kelly (a quarter is a common choice) gives up a little growth
    for a large reduction in variance and in sensitivity to estimation error.
    """
    if fraction < 0.0:
        raise ValueError("fraction must be >= 0")
    stake = fraction * kelly_fraction(p_win, decimal, p_push)
    if cap is not None:
        stake = min(stake, max(cap, 0.0))
    return stake


def bet_variance(p_win: float, decimal: float, p_push: float = 0.0) -> float:
    """Variance of profit per unit staked."""
    p_lose = _check(p_win, p_push, decimal)
    b = decimal - 1.0
    mean = p_win * b - p_lose
    return p_win * b * b + p_lose - mean * mean


def expected_log_growth(
    stake_fraction: float, p_win: float, decimal: float, p_push: float = 0.0
) -> float:
    """Expected log bankroll growth from staking ``stake_fraction``."""
    p_lose = _check(p_win, p_push, decimal)
    if not 0.0 <= stake_fraction < 1.0:
        raise ValueError("stake_fraction must be in [0, 1)")
    b = decimal - 1.0
    return p_win * math.log1p(stake_fraction * b) + p_lose * math.log1p(-stake_fraction)
