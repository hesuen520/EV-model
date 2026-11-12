"""Removing the bookmaker margin ("vig") from quoted prices.

A sportsbook's implied probabilities for all outcomes of a market sum to more
than 1; the excess is the *overround*.  De-vigging maps those implied
probabilities ``q_i`` to fair probabilities ``p_i`` that sum to exactly 1.
There is no single correct way to do this because it depends on how the book
spread its margin across outcomes, so several standard methods are provided:

``multiplicative``
    ``p_i = q_i / sum(q)``.  Margin proportional to each probability.
``additive``
    ``p_i = q_i - (sum(q) - 1) / n``.  Equal margin on every outcome.
``power``
    ``p_i = q_i ** k`` with ``k`` chosen so the result sums to 1.  Puts more
    margin on longshots (favourite-longshot bias).
``shin``
    Shin's (1992, 1993) insider-trading model.  For two outcomes it coincides
    with the additive method.
``odds_ratio``
    Constant odds ratio between quoted and fair odds for every outcome.
``probit``
    Constant shift on the normal-quantile scale: ``p_i = Phi(Phi^-1(q_i) - c)``.

For a balanced two-way market (for example -110 / -110) every method returns
50 / 50.  They differ as the market becomes lopsided.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from scipy.optimize import brentq
from scipy.stats import norm

__all__ = [
    "METHODS",
    "overround",
    "hold",
    "devig",
    "devig_decimal",
    "devig_two_way",
    "devig_all",
    "worst_case",
    "fair_prob_one_sided",
]

METHODS: Tuple[str, ...] = (
    "multiplicative",
    "additive",
    "power",
    "shin",
    "odds_ratio",
    "probit",
)

_EPS = 1e-12
_XTOL = 1e-14
_MAXITER = 200


def _validate(implied: Sequence[float]) -> List[float]:
    q = [float(x) for x in implied]
    if len(q) < 2:
        raise ValueError("de-vigging needs at least two outcomes")
    for x in q:
        if not (0.0 < x < 1.0) or math.isnan(x):
            raise ValueError(f"implied probabilities must be in (0, 1), got {x!r}")
    return q


def overround(implied: Sequence[float]) -> float:
    """Sum of implied probabilities minus one (``0.0476`` for -110/-110)."""
    return float(sum(float(x) for x in implied)) - 1.0


def hold(implied: Sequence[float]) -> float:
    """Theoretical book hold: ``1 - 1 / sum(q)`` (``0.0455`` for -110/-110)."""
    total = float(sum(float(x) for x in implied))
    if total <= 0:
        raise ValueError("implied probabilities must be positive")
    return 1.0 - 1.0 / total


def _multiplicative(q: List[float]) -> List[float]:
    total = sum(q)
    return [x / total for x in q]


def _additive(q: List[float]) -> Optional[List[float]]:
    shift = (sum(q) - 1.0) / len(q)
    p = [x - shift for x in q]
    if min(p) <= 0.0 or max(p) >= 1.0:
        return None  # additive breaks down for longshots with a large margin
    return p


def _power(q: List[float]) -> List[float]:
    total = sum(q)

    def f(k: float) -> float:
        return sum(x**k for x in q) - 1.0

    if total > 1.0:
        lo, hi = 1.0, 2.0
        while f(hi) > 0.0:
            hi *= 2.0
            if hi > 1e6:  # pragma: no cover - defensive
                raise RuntimeError("power de-vig failed to bracket a root")
    else:
        lo, hi = 1.0, 1.0
        while f(lo) < 0.0:
            lo /= 2.0
            if lo < 1e-9:  # pragma: no cover - defensive
                raise RuntimeError("power de-vig failed to bracket a root")
    k = brentq(f, lo, hi, xtol=_XTOL, maxiter=_MAXITER)
    return [x**k for x in q]


def _shin_probs(q: List[float], z: float) -> List[float]:
    total = sum(q)
    return [
        (math.sqrt(z * z + 4.0 * (1.0 - z) * x * x / total) - z) / (2.0 * (1.0 - z))
        for x in q
    ]


def _shin(q: List[float]) -> List[float]:
    def f(z: float) -> float:
        return sum(_shin_probs(q, z)) - 1.0

    # f(0) = sum(sqrt(q_i^2 / total)) - 1 = sqrt(total) - 1 > 0 when total > 1
    # and f decreases towards the proportion of insider money z.
    lo, hi = 0.0, 1.0 - 1e-9
    if f(hi) > 0.0:  # pragma: no cover - cannot happen for valid input
        raise RuntimeError("Shin de-vig failed to bracket a root")
    z = brentq(f, lo, hi, xtol=_XTOL, maxiter=_MAXITER)
    return _shin_probs(q, z)


def _odds_ratio(q: List[float]) -> List[float]:
    def probs(c: float) -> List[float]:
        return [x / (c * (1.0 - x) + x) for x in q]

    def f(c: float) -> float:
        return sum(probs(c)) - 1.0

    lo, hi = 1.0, 2.0
    while f(hi) > 0.0:
        hi *= 2.0
        if hi > 1e12:  # pragma: no cover - defensive
            raise RuntimeError("odds-ratio de-vig failed to bracket a root")
    c = brentq(f, lo, hi, xtol=_XTOL, maxiter=_MAXITER)
    return probs(c)


def _probit(q: List[float]) -> List[float]:
    zq = [float(norm.ppf(x)) for x in q]

    def probs(c: float) -> List[float]:
        return [float(norm.cdf(z - c)) for z in zq]

    def f(c: float) -> float:
        return sum(probs(c)) - 1.0

    lo, hi = 0.0, 1.0
    while f(hi) > 0.0:
        hi *= 2.0
        if hi > 64.0:  # pragma: no cover - defensive
            raise RuntimeError("probit de-vig failed to bracket a root")
    c = brentq(f, lo, hi, xtol=_XTOL, maxiter=_MAXITER)
    return probs(c)


_DISPATCH = {
    "multiplicative": _multiplicative,
    "additive": _additive,
    "power": _power,
    "shin": _shin,
    "odds_ratio": _odds_ratio,
    "probit": _probit,
}


def devig(implied: Sequence[float], method: str = "multiplicative") -> List[float]:
    """Remove the margin from a complete set of implied probabilities.

    Parameters
    ----------
    implied:
        Implied probabilities (``1 / decimal_odds``) of *every* outcome of one
        market at one book, e.g. ``[over, under]``.
    method:
        One of :data:`METHODS`.

    Returns
    -------
    list of float
        Fair probabilities in the same order, summing to 1.

    Notes
    -----
    If the quotes carry no margin (``sum(q) <= 1``, which can happen on
    exchanges or when sides come from different books) there is nothing to
    remove and the probabilities are simply normalised.  If the additive
    method would yield a probability outside (0, 1) the multiplicative result
    is returned instead.
    """
    if method not in _DISPATCH:
        raise ValueError(f"unknown de-vig method {method!r}; choose from {METHODS}")
    q = _validate(implied)
    if sum(q) <= 1.0 + _EPS:
        return _multiplicative(q)
    result = _DISPATCH[method](q)
    if result is None:
        result = _multiplicative(q)
    total = sum(result)
    return [x / total for x in result]  # wash out root-finder residue


def devig_decimal(decimal_odds: Sequence[float], method: str = "multiplicative") -> List[float]:
    """De-vig a market given decimal odds for every outcome."""
    implied = []
    for d in decimal_odds:
        d = float(d)
        if not d > 1.0:
            raise ValueError(f"decimal odds must be > 1, got {d!r}")
        implied.append(1.0 / d)
    return devig(implied, method)


def devig_two_way(
    decimal_a: float, decimal_b: float, method: str = "multiplicative"
) -> Tuple[float, float]:
    """Fair probabilities ``(p_a, p_b)`` of a two-way market from decimal odds."""
    p = devig_decimal([decimal_a, decimal_b], method)
    return p[0], p[1]


def devig_all(
    implied: Sequence[float], methods: Iterable[str] = METHODS
) -> Dict[str, List[float]]:
    """De-vig with several methods at once: ``{method: fair_probs}``."""
    return {m: devig(implied, m) for m in methods}


def worst_case(
    implied: Sequence[float], index: int, methods: Iterable[str] = METHODS
) -> float:
    """Most conservative fair probability for one outcome across methods.

    This is the lowest fair probability any method assigns to outcome
    ``index``; using it for the side you intend to bet guards against the
    choice of de-vig method manufacturing an edge.
    """
    return min(devig(implied, m)[index] for m in methods)


def fair_prob_one_sided(implied: float, assumed_hold: float = 0.07) -> float:
    """Estimate a fair probability when only one side of a market is quoted.

    Some props (anytime touchdown scorer, to record a hit, ...) are often
    offered as a single "Yes" price with no "No".  The margin cannot be
    measured, so it has to be assumed: ``assumed_hold`` is the overround
    (``sum(q) - 1``) the hidden two-way market is taken to carry, removed
    proportionally: ``p = q / (1 + assumed_hold)``.

    The result is necessarily rougher than a true two-way de-vig - books
    load far more margin onto longshots than onto favourites - so edges
    derived from it deserve extra caution.
    """
    q = float(implied)
    if not 0.0 < q < 1.0:
        raise ValueError(f"implied probability must be in (0, 1), got {implied!r}")
    if not (assumed_hold >= 0.0 and math.isfinite(assumed_hold)):
        raise ValueError("assumed_hold must be finite and >= 0")
    return q / (1.0 + assumed_hold)
