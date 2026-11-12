"""Predictive distributions for player-prop stats.

Every prop stat in this package (points, rebounds, receiving yards, ...) is an
integer, so every distribution here is a distribution over the integers and
answers the three questions a bet needs:

* ``prob_over(line)``  = P(X > line)
* ``prob_under(line)`` = P(X < line)
* ``prob_push(line)``  = P(X == line)  (non-zero only for whole-number lines)

A :class:`FamilySpec` describes a *family* of such distributions indexed by
their mean: the shape (Poisson, negative binomial, discretised normal or
gamma) plus a variance function of the mean.  That is what lets the market
engine translate a sharp book's price at one line into a fair probability at
another line: solve for the mean that reproduces the sharp price
(:meth:`FamilySpec.solve_mean`), then evaluate the same distribution at the
other line.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import special
from scipy.optimize import brentq

__all__ = [
    "Dist",
    "PoissonDist",
    "NegBinDist",
    "ScaledNegBinDist",
    "DiscretizedNormal",
    "DiscretizedGamma",
    "ZeroInflated",
    "MixtureDist",
    "FamilySpec",
    "FAMILIES",
    "is_integer_line",
]

_LINE_TOL = 1e-9
#: Relative excess of variance over mean below which a negative binomial is
#: numerically indistinguishable from a Poisson (and is treated as one).
_POISSON_EXCESS = 1e-7


def is_integer_line(line: float) -> bool:
    """True when a prop line is a whole number (a push is possible)."""
    return abs(line - round(line)) < _LINE_TOL


def _clip01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


class Dist:
    """Distribution of an integer-valued statistic.

    Subclasses implement :meth:`cdf` (``P(X <= k)`` for any real ``k``),
    :attr:`mean` and :attr:`var`, and should override :meth:`sf` with a
    direct upper-tail evaluation so small tail probabilities keep their
    relative accuracy.
    """

    # -- to be provided by subclasses -------------------------------------
    def cdf(self, k: float) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def mean(self) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def var(self) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- derived -----------------------------------------------------------
    def sf(self, k: float) -> float:
        """``P(X > k)``."""
        return _clip01(1.0 - self.cdf(k))

    @property
    def sd(self) -> float:
        return math.sqrt(max(self.var, 0.0))

    def pmf(self, k: int) -> float:
        """``P(X == k)``."""
        k = int(round(k))
        below = self.cdf(k - 1)
        if below > 0.5:  # upper tail: difference survival functions instead
            return max(self.sf(k - 1) - self.sf(k), 0.0)
        return max(self.cdf(k) - below, 0.0)

    def prob_over(self, line: float) -> float:
        """``P(X > line)``."""
        if is_integer_line(line):
            return _clip01(self.sf(round(line)))
        return _clip01(self.sf(math.floor(line)))

    def prob_under(self, line: float) -> float:
        """``P(X < line)``."""
        if is_integer_line(line):
            return _clip01(self.cdf(round(line) - 1))
        return _clip01(self.cdf(math.floor(line)))

    def prob_push(self, line: float) -> float:
        """``P(X == line)``; zero for half-point lines."""
        if is_integer_line(line):
            return self.pmf(int(round(line)))
        return 0.0

    def probs(self, line: float) -> Tuple[float, float, float]:
        """``(P(over), P(push), P(under))`` for a line; sums to 1."""
        return self.prob_over(line), self.prob_push(line), self.prob_under(line)

    def prob_over_no_push(self, line: float) -> float:
        """P(over) conditional on the bet not pushing.

        This is the quantity a de-vigged two-way price estimates, because a
        push refunds both sides.
        """
        over = self.prob_over(line)
        under = self.prob_under(line)
        if over + under <= 0.0:
            return 0.5
        return over / (over + under)

    def quantile(self, q: float) -> int:
        """Smallest integer ``k`` with ``P(X <= k) >= q``."""
        if not 0.0 < q < 1.0:
            raise ValueError("q must be in (0, 1)")
        width = max(10.0 * self.sd, 10.0)
        step = max(int(width), 1)
        lo = math.floor(self.mean - width) - 1
        hi = math.ceil(self.mean + width) + 1
        while self.cdf(lo) >= q:
            lo -= step
            step *= 2
        step = max(int(width), 1)
        while self.cdf(hi) < q:
            hi += step
            step *= 2
        while hi - lo > 1:  # invariant: cdf(lo) < q <= cdf(hi)
            mid = (lo + hi) // 2
            if self.cdf(mid) >= q:
                hi = mid
            else:
                lo = mid
        return hi

    @property
    def median(self) -> int:
        return self.quantile(0.5)

    def interval(self, coverage: float = 0.8) -> Tuple[int, int]:
        """Central prediction interval with (at least) the given coverage."""
        if not 0.0 < coverage < 1.0:
            raise ValueError("coverage must be in (0, 1)")
        tail = (1.0 - coverage) / 2.0
        return self.quantile(tail), self.quantile(1.0 - tail)


class PoissonDist(Dist):
    """Poisson distribution with the given mean."""

    def __init__(self, mean: float) -> None:
        if not mean > 0.0 or not math.isfinite(mean):
            raise ValueError(f"Poisson mean must be finite and > 0, got {mean!r}")
        self._mean = float(mean)

    def cdf(self, k: float) -> float:
        k = math.floor(k)
        if k < 0:
            return 0.0
        return float(special.gammaincc(k + 1.0, self._mean))

    def sf(self, k: float) -> float:
        k = math.floor(k)
        if k < 0:
            return 1.0
        return float(special.gammainc(k + 1.0, self._mean))

    def pmf(self, k: int) -> float:
        k = int(round(k))
        if k < 0:
            return 0.0
        return math.exp(k * math.log(self._mean) - self._mean - special.gammaln(k + 1.0))

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def var(self) -> float:
        return self._mean

    def __repr__(self) -> str:
        return f"PoissonDist(mean={self._mean:.4g})"


class NegBinDist(Dist):
    """Negative binomial parameterised by mean and variance (``var > mean``).

    Equivalent to a Poisson whose rate is gamma distributed; ``r = mean**2 /
    (var - mean)`` is the gamma shape ("size").
    """

    def __init__(self, mean: float, var: float) -> None:
        if not mean > 0.0 or not math.isfinite(mean):
            raise ValueError(f"negative binomial mean must be finite and > 0, got {mean!r}")
        if not var > mean or not math.isfinite(var):
            raise ValueError("negative binomial needs a finite var > mean; use PoissonDist")
        self._mean = float(mean)
        self._var = float(var)
        self.r = self._mean**2 / (self._var - self._mean)
        self.p = self.r / (self.r + self._mean)
        self._q = self._mean / (self.r + self._mean)  # 1 - p without cancellation

    def cdf(self, k: float) -> float:
        k = math.floor(k)
        if k < 0:
            return 0.0
        return float(special.betainc(self.r, k + 1.0, self.p))

    def sf(self, k: float) -> float:
        k = math.floor(k)
        if k < 0:
            return 1.0
        return float(special.betainc(k + 1.0, self.r, self._q))

    def pmf(self, k: int) -> float:
        k = int(round(k))
        if k < 0:
            return 0.0
        if self.r > 1e5:
            # gammaln(k + r) - gammaln(r) cancels badly for huge r; the
            # incomplete-beta differences stay accurate.
            return super().pmf(k)
        return math.exp(
            special.gammaln(k + self.r)
            - special.gammaln(self.r)
            - special.gammaln(k + 1.0)
            + self.r * math.log(self.p)
            + k * math.log(self._q)
        )

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def var(self) -> float:
        return self._var

    def __repr__(self) -> str:
        return f"NegBinDist(mean={self._mean:.4g}, var={self._var:.4g})"


class ScaledNegBinDist(Dist):
    """A negative binomial stretched by a ``scale`` factor.

    Some stats do not arrive one unit at a time: basketball points come in
    twos and threes.  If ``N`` scoring events are negative-binomial and each
    is worth about ``scale`` units, the total behaves like ``scale * N``: the
    same variance as an ordinary negative binomial then goes with visibly
    less right-skew, which is what real points distributions look like.

    The distribution is defined through the continuous extension of the
    negative-binomial cdf, ``G(x) = I_p(r, x + 1)``, as ``P(X <= k) =
    G((k + 0.5) / scale - 0.5)``.  With ``scale = 1`` this is exactly
    :class:`NegBinDist`.  The variance is floored at ``scale * mean`` (a
    scaled Poisson).  ``mean`` and ``var`` are the targets the distribution
    is built to match.  The exact mean is within about 1% of its target for
    means of a few units and converges quickly above that; the exact variance
    can be a few percent off for small means.
    """

    def __init__(self, mean: float, var: float, scale: float) -> None:
        if not mean > 0.0 or not math.isfinite(mean):
            raise ValueError(f"mean must be finite and > 0, got {mean!r}")
        if not scale >= 1.0 or not math.isfinite(scale):
            raise ValueError(f"scale must be finite and >= 1, got {scale!r}")
        self._mean = float(mean)
        self.scale = float(scale)
        # The event count cannot be under-dispersed, so the variance is at
        # least that of a scaled Poisson: scale * mean.
        self._var = max(float(var), self.scale * self._mean)
        self._event_mean = self._mean / self.scale
        event_var = self._var / (self.scale * self.scale)
        if event_var > self._event_mean * (1.0 + _POISSON_EXCESS):
            self.r: Optional[float] = self._event_mean**2 / (event_var - self._event_mean)
            self.p: Optional[float] = self.r / (self.r + self._event_mean)
            self._q = self._event_mean / (self.r + self._event_mean)
        else:  # no over-dispersion left for the event count: scaled Poisson
            self.r = None
            self.p = None
            self._q = 0.0

    def _x(self, k: int) -> float:
        return (k + 0.5) / self.scale + 0.5

    def cdf(self, k: float) -> float:
        k = math.floor(k)
        if k < 0:
            return 0.0
        if self.r is None:
            return float(special.gammaincc(self._x(k), self._event_mean))
        return float(special.betainc(self.r, self._x(k), self.p))

    def sf(self, k: float) -> float:
        k = math.floor(k)
        if k < 0:
            return 1.0
        if self.r is None:
            return float(special.gammainc(self._x(k), self._event_mean))
        return float(special.betainc(self._x(k), self.r, self._q))

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def var(self) -> float:
        return self._var

    def __repr__(self) -> str:
        return f"ScaledNegBinDist(mean={self._mean:.4g}, var={self._var:.4g}, scale={self.scale:.3g})"


class DiscretizedNormal(Dist):
    """Normal distribution rounded to the nearest integer.

    ``P(X <= k) = Phi((k + 0.5 - mu) / sd)`` where ``mu`` and ``sd`` are the
    parameters of the underlying ("latent") normal.  A half-point line falls
    exactly on a rounding boundary, and whole-number lines get a proper push
    probability.  Negative values are allowed (rushing yards can be
    negative).

    ``lower`` piles all mass below it onto ``lower`` (``0`` for stats that
    cannot be negative).  Without a clamp, :attr:`mean` and :attr:`var` are
    the latent normal's; with one they are the exact moments of the clamped
    integer variable, which can differ a lot when the clamp binds.
    """

    def __init__(self, mean: float, sd: float, lower: Optional[int] = None) -> None:
        if not sd > 0.0 or not math.isfinite(sd):
            raise ValueError(f"sd must be finite and > 0, got {sd!r}")
        if not math.isfinite(mean):
            raise ValueError(f"mean must be finite, got {mean!r}")
        self._mu = float(mean)
        self._sd = float(sd)
        self.lower = lower
        self._moments: Optional[Tuple[float, float]] = None

    def cdf(self, k: float) -> float:
        k = math.floor(k)
        if self.lower is not None and k < self.lower:
            return 0.0
        return float(special.ndtr((k + 0.5 - self._mu) / self._sd))

    def sf(self, k: float) -> float:
        k = math.floor(k)
        if self.lower is not None and k < self.lower:
            return 1.0
        return float(special.ndtr(-(k + 0.5 - self._mu) / self._sd))

    def _clamped_moments(self) -> Tuple[float, float]:
        if self._moments is None:
            lo = int(self.lower)  # type: ignore[arg-type]
            hi = max(lo, int(math.ceil(self._mu + 12.0 * self._sd))) + 1
            ks = np.arange(lo, hi + 1, dtype=float)
            # P(X > k) for k = lo .. hi: E[X] = lo + sum_{k >= lo} P(X > k).
            tail = special.ndtr(-(ks + 0.5 - self._mu) / self._sd)
            mean = lo + float(tail.sum())
            # E[(X - lo)^2] = sum_{j >= 0} (2j + 1) P(X - lo > j)
            second = float(((2.0 * (ks - lo) + 1.0) * tail).sum())
            self._moments = (mean, max(second - (mean - lo) ** 2, 0.0))
        return self._moments

    @property
    def mean(self) -> float:
        if self.lower is None:
            return self._mu
        return self._clamped_moments()[0]

    @property
    def var(self) -> float:
        if self.lower is None:
            return self._sd**2
        return self._clamped_moments()[1]

    def __repr__(self) -> str:
        clamp = "" if self.lower is None else f", lower={self.lower}"
        return f"DiscretizedNormal(mu={self._mu:.4g}, sd={self._sd:.4g}{clamp})"


class DiscretizedGamma(Dist):
    """Gamma distribution rounded to the nearest integer (support >= 0).

    Right-skewed, so its median sits below its mean - the shape yardage stats
    have.  Parameterised by, and reporting, the mean and variance of the
    underlying continuous gamma; rounding changes them slightly for means
    below a few units.
    """

    def __init__(self, mean: float, var: float) -> None:
        if not mean > 0.0 or not var > 0.0 or not (math.isfinite(mean) and math.isfinite(var)):
            raise ValueError("gamma mean and variance must be finite and > 0")
        self._mean = float(mean)
        self._var = float(var)
        self.shape = self._mean**2 / self._var
        self.scale = self._var / self._mean

    def cdf(self, k: float) -> float:
        k = math.floor(k)
        if k < 0:
            return 0.0
        return float(special.gammainc(self.shape, (k + 0.5) / self.scale))

    def sf(self, k: float) -> float:
        k = math.floor(k)
        if k < 0:
            return 1.0
        return float(special.gammaincc(self.shape, (k + 0.5) / self.scale))

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def var(self) -> float:
        return self._var

    def __repr__(self) -> str:
        return f"DiscretizedGamma(mean={self._mean:.4g}, var={self._var:.4g})"


class ZeroInflated(Dist):
    """Mixture of a point mass at zero (weight ``pi``) and a base distribution."""

    def __init__(self, base: Dist, pi: float) -> None:
        if not 0.0 <= pi < 1.0:
            raise ValueError("zero-inflation weight must be in [0, 1)")
        self.base = base
        self.pi = float(pi)

    def cdf(self, k: float) -> float:
        k = math.floor(k)
        zero = 1.0 if k >= 0 else 0.0
        return self.pi * zero + (1.0 - self.pi) * self.base.cdf(k)

    def sf(self, k: float) -> float:
        k = math.floor(k)
        zero = 0.0 if k >= 0 else 1.0
        return self.pi * zero + (1.0 - self.pi) * self.base.sf(k)

    @property
    def mean(self) -> float:
        return (1.0 - self.pi) * self.base.mean

    @property
    def var(self) -> float:
        m = self.base.mean
        return (1.0 - self.pi) * (self.base.var + self.pi * m * m)

    def __repr__(self) -> str:
        return f"ZeroInflated({self.base!r}, pi={self.pi:.3g})"


class MixtureDist(Dist):
    """Finite mixture of distributions (e.g. over scenarios for minutes)."""

    def __init__(self, components: Sequence[Dist], weights: Sequence[float]) -> None:
        if len(components) != len(weights) or not components:
            raise ValueError("need one weight per component")
        total = float(sum(weights))
        if not total > 0.0 or min(weights) < 0.0 or not math.isfinite(total):
            raise ValueError("weights must be non-negative, finite and sum to > 0")
        self.components = list(components)
        self.weights = [float(w) / total for w in weights]

    def cdf(self, k: float) -> float:
        return sum(w * c.cdf(k) for w, c in zip(self.weights, self.components))

    def sf(self, k: float) -> float:
        return sum(w * c.sf(k) for w, c in zip(self.weights, self.components))

    @property
    def mean(self) -> float:
        return sum(w * c.mean for w, c in zip(self.weights, self.components))

    @property
    def var(self) -> float:
        m = self.mean
        return sum(w * (c.var + (c.mean - m) ** 2) for w, c in zip(self.weights, self.components))

    def __repr__(self) -> str:
        return f"MixtureDist(n={len(self.components)}, mean={self.mean:.4g})"


def _clamped_latent_mean(target: float, sd: float, lower: int) -> float:
    """Latent normal mean whose rounded-and-clamped variable has mean ``target``.

    With ``T = lower + 0.5`` and ``t = (T - m) / sd`` the clamped variable's
    mean is, to rounding accuracy, ``lower*Phi(t) + m*(1 - Phi(t)) +
    sd*phi(t)``, which increases with ``m``.
    """
    threshold = lower + 0.5

    def gap(m: float) -> float:
        t = (threshold - m) / sd
        tail = float(special.ndtr(-t))
        density = math.exp(-0.5 * t * t) / math.sqrt(2.0 * math.pi)
        return lower * (1.0 - tail) + m * tail + sd * density - target

    hi = target + 1.0  # the clamped mean is never below the latent mean
    lo = threshold - 40.0 * sd
    if gap(lo) >= 0.0:
        return lo
    return float(brentq(gap, lo, hi, xtol=1e-12, rtol=1e-14, maxiter=200))


FAMILIES: Tuple[str, ...] = ("poisson", "negbin", "normal", "gamma")


@dataclass(frozen=True)
class FamilySpec:
    """A family of predictive distributions indexed by their mean.

    The variance at mean ``m`` is ``c*m/(m + 1) + phi*m + alpha*m**2``.  The
    first term is an intercept that fades out below a mean of about one, so
    the spread of a non-negative stat shrinks to nothing as its mean does.
    (The unclamped normal, which may have a negative mean, keeps a plain
    intercept: ``c + phi*m + alpha*m**2`` with ``m`` floored at zero.)

    * ``poisson``: variance equals the mean (the variance terms are ignored).
    * ``negbin``: over-dispersed count.  ``phi=1, alpha>0`` is the classic
      NB2 form ``var = mean + alpha*mean**2``.  The variance is floored at
      the mean (a Poisson).  ``scale > 1`` stretches the count (see
      :class:`ScaledNegBinDist`) for stats such as points that accumulate in
      chunks; the floor is then ``scale * mean``.
    * ``normal``: discretised normal whose *latent* variance is given by the
      variance function.  ``lower`` clamps the support from below (``0`` for
      stats that cannot be negative; ``None`` for no clamp, which allows
      negative means).
    * ``gamma``: discretised gamma (right-skewed, non-negative).

    ``zero_inflation`` adds a point mass at zero (a player who barely plays).
    """

    family: str = "negbin"
    phi: float = 1.0
    alpha: float = 0.0
    c: float = 0.0
    lower: Optional[int] = 0
    zero_inflation: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family {self.family!r}; choose from {FAMILIES}")
        for name in ("phi", "alpha", "c"):
            value = getattr(self, name)
            if not (value >= 0.0 and math.isfinite(value)):
                raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
        if not 0.0 <= self.zero_inflation < 1.0:
            raise ValueError("zero_inflation must be in [0, 1)")
        if not (self.scale >= 1.0 and math.isfinite(self.scale)):
            raise ValueError("scale must be finite and >= 1")
        if self.scale != 1.0 and self.family != "negbin":
            raise ValueError("scale is only supported for the negbin family")

    # -- construction --------------------------------------------------------
    def variance(self, mean: float) -> float:
        """Value of the variance function at ``mean``.

        This is the variance of the base distribution built for ``mean``,
        except for a clamped normal, where it is the variance of the latent
        normal (use ``dist(mean).var`` for the clamped variable's own).
        """
        m = max(mean, 0.0)
        if self.family == "poisson":
            return m
        if self.family == "normal" and self.lower is None:
            return self.c + self.phi * m + self.alpha * m * m
        var = self.c * m / (m + 1.0) + self.phi * m + self.alpha * m * m
        if self.family == "negbin":
            var = max(var, self.scale * m)
        return var

    def dist(self, mean: float) -> Dist:
        """The member of the family whose overall mean is ``mean``.

        For the rounded continuous families (normal, gamma) and the scaled
        negative binomial the mean is matched to within rounding effects:
        closely for means of a few units and above, less so below one.
        """
        if not math.isfinite(mean):
            raise ValueError(f"mean must be finite, got {mean!r}")
        base_mean = mean / (1.0 - self.zero_inflation) if self.zero_inflation else mean
        base = self._base(base_mean)
        if self.zero_inflation:
            return ZeroInflated(base, self.zero_inflation)
        return base

    def _base(self, mean: float) -> Dist:
        if self.family == "normal":
            sd = math.sqrt(max(self.variance(mean), 1e-9))
            if self.lower is None:
                return DiscretizedNormal(mean, sd)
            target = max(mean, self.lower + 1e-9)
            return DiscretizedNormal(_clamped_latent_mean(target, sd, self.lower), sd, lower=self.lower)
        mean = max(mean, 1e-9)
        if self.family == "poisson":
            return PoissonDist(mean)
        var = self.variance(mean)
        if self.family == "negbin":
            if self.scale != 1.0:
                return ScaledNegBinDist(mean, var, self.scale)
            if var <= mean * (1.0 + _POISSON_EXCESS):
                return PoissonDist(mean)
            return NegBinDist(mean, var)
        return DiscretizedGamma(mean, max(var, 1e-9))

    # -- inversion -----------------------------------------------------------
    def solve_mean(self, line: float, p_over: float, exclude_push: bool = True) -> float:
        """Mean at which the family reproduces a fair over probability.

        ``p_over`` is the fair probability of the over at ``line``.  With
        ``exclude_push`` (default) it is interpreted as conditional on no
        push, which is what a de-vigged two-way price means on a
        whole-number line.  Raises ``ValueError`` when no member of the
        family can produce that probability at that line.
        """
        if not 0.0 < p_over < 1.0:
            raise ValueError(f"p_over must be in (0, 1), got {p_over!r}")
        if not math.isfinite(line):
            raise ValueError(f"line must be finite, got {line!r}")

        def f(mean: float) -> float:
            d = self.dist(mean)
            value = d.prob_over_no_push(line) if exclude_push else d.prob_over(line)
            return value - p_over

        span = math.sqrt(max(self.variance(line), 1.0)) + max(abs(line), 1.0)
        if self.family == "normal" and self.lower is None:
            lo, step = line - span, span
            for _ in range(80):
                if f(lo) < 0.0:
                    break
                lo -= step
                step *= 2.0
            else:
                raise ValueError("cannot reach the requested probability at this line")
        else:
            floor = float(self.lower) if (self.family == "normal" and self.lower is not None) else 0.0
            lo = floor + 1e-9
            if f(lo) >= 0.0:
                raise ValueError("cannot reach the requested probability at this line")
        hi, step = max(line, lo) + span, span
        for _ in range(80):
            if f(hi) > 0.0:
                break
            hi += step
            step *= 2.0
        else:
            raise ValueError("cannot reach the requested probability at this line")
        return float(brentq(f, lo, hi, xtol=1e-10, rtol=1e-12, maxiter=300))

    def translate(
        self, from_line: float, p_over: float, to_line: float
    ) -> Tuple[float, float, float]:
        """Fair ``(over, push, under)`` at ``to_line`` implied by a price at ``from_line``."""
        return self.dist(self.solve_mean(from_line, p_over)).probs(to_line)

    # -- (de)serialisation ---------------------------------------------------
    def to_dict(self) -> Dict[str, object]:
        return {
            "family": self.family,
            "phi": self.phi,
            "alpha": self.alpha,
            "c": self.c,
            "lower": self.lower,
            "zero_inflation": self.zero_inflation,
            "scale": self.scale,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "FamilySpec":
        fields_ = ("family", "phi", "alpha", "c", "lower", "zero_inflation", "scale")
        known = {k: data[k] for k in fields_ if k in data}
        return cls(**known)  # type: ignore[arg-type]
