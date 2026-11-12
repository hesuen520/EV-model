"""Independent verification of :mod:`evmodel.distributions`.

Reference values come from four sources that share no code with the module:

* brute-force summation over the probability mass function;
* ``scipy.stats`` (poisson, nbinom, norm, gamma) frozen distributions;
* 50-digit constants produced with mpmath (pasted below with their recipe);
* seeded Monte-Carlo simulation of the generative story in each docstring,
  with tolerances set to five standard errors.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import FrozenInstanceError, dataclass
from functools import lru_cache
from typing import Callable, Dict, List, Tuple

import numpy as np
import pytest
from scipy import stats
from scipy.optimize import brentq

from evmodel.distributions import (
    FAMILIES,
    DiscretizedGamma,
    DiscretizedNormal,
    Dist,
    FamilySpec,
    MixtureDist,
    NegBinDist,
    PoissonDist,
    ScaledNegBinDist,
    ZeroInflated,
    is_integer_line,
)

NAN = float("nan")
INF = float("inf")
SHEPPARD = 1.0 / 12.0  # variance added by rounding a smooth density to integers


# --------------------------------------------------------------------------
# Catalogue of distributions under test
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Case:
    """A distribution with an integer window holding all but ~1e-15 of its mass.

    ``moments`` says how the ``mean``/``var`` attributes relate to the moments
    of the integer variable: ``"exact"`` (they are its moments), ``"sheppard"``
    (mean exact, variance is that of the latent continuous variable, i.e. up
    to 1/12 below the true one), ``"latent"`` (documented as the underlying
    continuous distribution's, visibly different at small means) or
    ``"target"`` (documented as approximate targets).
    """

    dist: Dist
    lo: int
    hi: int
    moments: str


def _build_cases() -> Dict[str, Case]:
    pois4 = PoissonDist(4.0)
    nb10 = NegBinDist(10.0, 30.0)
    return {
        "poisson_0.3": Case(PoissonDist(0.3), -3, 60, "exact"),
        "poisson_2.5": Case(PoissonDist(2.5), -3, 90, "exact"),
        "poisson_24.7": Case(PoissonDist(24.7), -3, 220, "exact"),
        "poisson_310.2": Case(PoissonDist(310.2), -3, 900, "exact"),
        "negbin_0.5_0.75": Case(NegBinDist(0.5, 0.75), -3, 160, "exact"),
        "negbin_2.5_4": Case(NegBinDist(2.5, 4.0), -3, 220, "exact"),
        "negbin_10_30": Case(nb10, -3, 520, "exact"),
        "negbin_25_60": Case(NegBinDist(25.0, 60.0), -3, 650, "exact"),
        "negbin_150_3000": Case(NegBinDist(150.0, 3000.0), -3, 3200, "exact"),
        "negbin_heavy_5_500": Case(NegBinDist(5.0, 500.0), -3, 6500, "exact"),
        "normal_20_5": Case(DiscretizedNormal(20.0, 5.0), -60, 100, "sheppard"),
        "normal_neg_-3.2_2.5": Case(DiscretizedNormal(-3.2, 2.5), -45, 35, "sheppard"),
        "normal_0.4_1.3": Case(DiscretizedNormal(0.4, 1.3), -20, 20, "sheppard"),
        "normal_67.3_24.1": Case(DiscretizedNormal(67.3, 24.1), -220, 360, "sheppard"),
        "gamma_60_900": Case(DiscretizedGamma(60.0, 900.0), -3, 1000, "sheppard"),
        "gamma_250_8000": Case(DiscretizedGamma(250.0, 8000.0), -3, 1900, "sheppard"),
        "zi_poisson": Case(ZeroInflated(pois4, 0.3), -3, 90, "exact"),
        "zi_negbin": Case(ZeroInflated(nb10, 0.15), -3, 520, "exact"),
        "zi_zero_weight": Case(ZeroInflated(pois4, 0.0), -3, 90, "exact"),
        "zi_normal_negative_support": Case(
            ZeroInflated(DiscretizedNormal(3.0, 6.0), 0.3), -70, 80, "sheppard"
        ),
        "mixture_counts": Case(MixtureDist([pois4, nb10], [1.0, 3.0]), -3, 520, "exact"),
        "mixture_single": Case(MixtureDist([nb10], [7.0]), -3, 520, "exact"),
        "mixture_nested": Case(
            MixtureDist([ZeroInflated(PoissonDist(6.0), 0.2), PoissonDist(1.5)], [2.0, 1.0]),
            -3,
            110,
            "exact",
        ),
        "mixture_normals": Case(
            MixtureDist([DiscretizedNormal(-3.0, 2.0), DiscretizedNormal(12.0, 3.0)], [0.4, 0.6]),
            -40,
            60,
            "sheppard",
        ),
    }


CASES = _build_cases()
CASE_IDS = sorted(CASES)

#: Clamped normals (exact moments of the clamped variable are documented) and
#: small-mean gammas (documented to report the latent gamma's moments).
LATENT_CASES: Dict[str, Case] = {
    "normal_clamped_20_5": Case(DiscretizedNormal(20.0, 5.0, lower=0), -5, 100, "exact"),
    "normal_clamped_2_5": Case(DiscretizedNormal(2.0, 5.0, lower=0), -5, 70, "exact"),
    "normal_clamped_-3_2": Case(DiscretizedNormal(-3.0, 2.0, lower=0), -5, 30, "exact"),
    "normal_clamped_at_2": Case(DiscretizedNormal(5.0, 3.0, lower=2), -5, 45, "exact"),
    "normal_clamped_at_-6": Case(DiscretizedNormal(-4.0, 3.0, lower=-6), -10, 40, "exact"),
    "gamma_5_25": Case(DiscretizedGamma(5.0, 25.0), -3, 300, "latent"),
    "gamma_0.5_0.5": Case(DiscretizedGamma(0.5, 0.5), -3, 70, "latent"),
    "zi_normal_clamped": Case(
        ZeroInflated(DiscretizedNormal(8.0, 4.0, lower=0), 0.2), -5, 70, "exact"
    ),
}

#: Scaled negative binomials: ``mean``/``var`` are documented as *targets* that the
#: exact moments only approximate, so they too have dedicated moment tests.
SCALED_CASES: Dict[str, Case] = {
    "scaled_negbin_24_60_x2.2": Case(ScaledNegBinDist(24.0, 60.0, 2.2), -3, 260, "target"),
    "scaled_negbin_100_400_x2.2": Case(ScaledNegBinDist(100.0, 400.0, 2.2), -3, 700, "target"),
    "scaled_negbin_8_20_x2": Case(ScaledNegBinDist(8.0, 20.0, 2.0), -3, 160, "target"),
    "scaled_negbin_3_8_x2": Case(ScaledNegBinDist(3.0, 8.0, 2.0), -3, 130, "target"),
    "scaled_poisson_20_x2": Case(ScaledNegBinDist(20.0, 10.0, 2.0), -3, 160, "target"),
    "zi_scaled_negbin": Case(ZeroInflated(ScaledNegBinDist(24.0, 60.0, 2.2), 0.1), -3, 260, "target"),
}

ALL_CASES: Dict[str, Case] = {**CASES, **LATENT_CASES, **SCALED_CASES}
ALL_IDS = sorted(ALL_CASES)


@lru_cache(maxsize=None)
def _table(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Integer grid and brute-force pmf values for a catalogued case."""
    case = ALL_CASES[name]
    ks = np.arange(case.lo, case.hi + 1)
    pmf = np.array([case.dist.pmf(int(k)) for k in ks])
    return ks, pmf


def _lines(name: str) -> List[float]:
    """Whole and half lines spread across (and beyond) a case's support."""
    ks, pmf = _table(name)
    cum = np.cumsum(pmf)
    picks = sorted({int(ks[np.searchsorted(cum, level)]) for level in (0.03, 0.3, 0.5, 0.8, 0.97)})
    lines: List[float] = [-0.5, 0.0, 0.5]
    for k in picks:
        lines.extend([float(k), k + 0.5, k - 0.5])
    lines.extend([float(ks[0]) - 2.0, float(ks[0]) - 1.5, float(ks[-1]) + 2.0, float(ks[-1]) + 2.5])
    return lines


# --------------------------------------------------------------------------
# scipy.stats references for the four base families
# --------------------------------------------------------------------------
def _scipy_poisson(mean: float) -> Tuple[Callable, Callable]:
    ref = stats.poisson(mean)
    return ref.cdf, ref.sf


def _scipy_negbin(mean: float, var: float) -> Tuple[Callable, Callable]:
    """nbinom with size r = mean^2/(var-mean) and success probability mean/var."""
    ref = stats.nbinom(mean * mean / (var - mean), mean / var)
    return ref.cdf, ref.sf


def _scipy_normal(mean: float, sd: float) -> Tuple[Callable, Callable]:
    ref = stats.norm(loc=mean, scale=sd)
    return (lambda k: ref.cdf(np.asarray(k) + 0.5)), (lambda k: ref.sf(np.asarray(k) + 0.5))


def _scipy_gamma(mean: float, var: float) -> Tuple[Callable, Callable]:
    ref = stats.gamma(a=mean * mean / var, scale=var / mean)
    return (
        lambda k: np.where(np.asarray(k) < 0, 0.0, ref.cdf(np.asarray(k) + 0.5)),
        lambda k: np.where(np.asarray(k) < 0, 1.0, ref.sf(np.asarray(k) + 0.5)),
    )


def _scipy_scaled_negbin(mean: float, var: float, scale: float) -> Tuple[Callable, Callable]:
    """P(X <= k) = I_p(r, (k + 1/2)/scale + 1/2), written as a Beta(r, x) cdf at p.

    The event count has mean ``mean/scale`` and variance ``var/scale**2``.
    """
    event_mean, event_var = mean / scale, var / (scale * scale)
    r = event_mean * event_mean / (event_var - event_mean)
    p = event_mean / event_var

    def shape(k: np.ndarray) -> np.ndarray:
        return (np.maximum(np.asarray(k, dtype=float), 0.0) + 0.5) / scale + 0.5

    return (
        lambda k: np.where(np.asarray(k) < 0, 0.0, stats.beta.cdf(p, r, shape(k))),
        lambda k: np.where(np.asarray(k) < 0, 1.0, stats.beta.sf(p, r, shape(k))),
    )


def _scipy_scaled_poisson(mean: float, scale: float) -> Tuple[Callable, Callable]:
    """P(X <= k) = Q((k + 1/2)/scale + 1/2, mean/scale) = P(Gamma(shape) > mean/scale)."""
    event_mean = mean / scale

    def shape(k: np.ndarray) -> np.ndarray:
        return (np.maximum(np.asarray(k, dtype=float), 0.0) + 0.5) / scale + 0.5

    return (
        lambda k: np.where(np.asarray(k) < 0, 0.0, stats.gamma.sf(event_mean, a=shape(k))),
        lambda k: np.where(np.asarray(k) < 0, 1.0, stats.gamma.cdf(event_mean, a=shape(k))),
    )


SCIPY_REFERENCES: Dict[str, Tuple[Callable, Callable]] = {
    "poisson_0.3": _scipy_poisson(0.3),
    "poisson_2.5": _scipy_poisson(2.5),
    "poisson_24.7": _scipy_poisson(24.7),
    "poisson_310.2": _scipy_poisson(310.2),
    "negbin_0.5_0.75": _scipy_negbin(0.5, 0.75),
    "negbin_2.5_4": _scipy_negbin(2.5, 4.0),
    "negbin_10_30": _scipy_negbin(10.0, 30.0),
    "negbin_25_60": _scipy_negbin(25.0, 60.0),
    "negbin_150_3000": _scipy_negbin(150.0, 3000.0),
    "negbin_heavy_5_500": _scipy_negbin(5.0, 500.0),
    "normal_20_5": _scipy_normal(20.0, 5.0),
    "normal_neg_-3.2_2.5": _scipy_normal(-3.2, 2.5),
    "normal_0.4_1.3": _scipy_normal(0.4, 1.3),
    "normal_67.3_24.1": _scipy_normal(67.3, 24.1),
    "gamma_60_900": _scipy_gamma(60.0, 900.0),
    "gamma_250_8000": _scipy_gamma(250.0, 8000.0),
    "gamma_5_25": _scipy_gamma(5.0, 25.0),
    "gamma_0.5_0.5": _scipy_gamma(0.5, 0.5),
    "scaled_negbin_24_60_x2.2": _scipy_scaled_negbin(24.0, 60.0, 2.2),
    "scaled_negbin_100_400_x2.2": _scipy_scaled_negbin(100.0, 400.0, 2.2),
    "scaled_negbin_8_20_x2": _scipy_scaled_negbin(8.0, 20.0, 2.0),
    "scaled_negbin_3_8_x2": _scipy_scaled_negbin(3.0, 8.0, 2.0),
    "scaled_poisson_20_x2": _scipy_scaled_poisson(20.0, 2.0),
}


# --------------------------------------------------------------------------
# pmf / cdf / sf consistency
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", ALL_IDS)
class TestMassFunction:
    def test_pmf_sums_to_one(self, name: str) -> None:
        _, pmf = _table(name)
        assert np.all(pmf >= 0.0)
        assert math.fsum(pmf) == pytest.approx(1.0, abs=1e-12)

    def test_pmf_equals_cdf_differences(self, name: str) -> None:
        dist = ALL_CASES[name].dist
        ks, pmf = _table(name)
        cdf = np.array([dist.cdf(int(k)) for k in ks])
        below = dist.cdf(int(ks[0]) - 1)
        assert pmf == pytest.approx(np.diff(np.concatenate([[below], cdf])), abs=1e-12)

    def test_cdf_is_the_running_sum_of_the_pmf(self, name: str) -> None:
        dist = ALL_CASES[name].dist
        ks, pmf = _table(name)
        cdf = np.array([dist.cdf(int(k)) for k in ks])
        assert cdf == pytest.approx(np.cumsum(pmf), abs=1e-11)

    def test_cdf_is_a_distribution_function(self, name: str) -> None:
        dist = ALL_CASES[name].dist
        ks, _ = _table(name)
        cdf = np.array([dist.cdf(int(k)) for k in ks])
        assert np.all((cdf >= 0.0) & (cdf <= 1.0))
        assert np.all(np.diff(cdf) >= -1e-15)
        assert dist.cdf(int(ks[0]) - 1) == pytest.approx(0.0, abs=1e-13)
        assert dist.cdf(int(ks[-1])) == pytest.approx(1.0, abs=1e-13)

    def test_cdf_is_a_right_continuous_step_function(self, name: str) -> None:
        dist = ALL_CASES[name].dist
        ks, _ = _table(name)
        for k in (int(ks[len(ks) // 3]), int(ks[len(ks) // 2]), 0, 1, -1):
            assert dist.cdf(k + 0.25) == dist.cdf(k)
            assert dist.cdf(k + 0.999) == dist.cdf(k)
            assert dist.cdf(k - 1e-9) == dist.cdf(k - 1)
            assert dist.cdf(float(k)) == dist.cdf(k)

    def test_sf_is_the_complement_of_cdf(self, name: str) -> None:
        dist = ALL_CASES[name].dist
        ks, pmf = _table(name)
        tail = np.cumsum(pmf[::-1])[::-1]  # P(X >= k)
        for i in range(0, len(ks) - 1, max(1, len(ks) // 40)):
            k = int(ks[i])
            assert dist.sf(k) == pytest.approx(1.0 - dist.cdf(k), abs=1e-13)
            assert dist.sf(k) == pytest.approx(tail[i + 1], abs=1e-11)
            assert 0.0 <= dist.sf(k) <= 1.0

    def test_pmf_rounds_its_argument(self, name: str) -> None:
        dist = ALL_CASES[name].dist
        ks, pmf = _table(name)
        i = int(np.argmax(pmf))
        k = int(ks[i])
        assert dist.pmf(k + 0.2) == pmf[i]
        assert dist.pmf(k - 0.2) == pmf[i]
        assert dist.pmf(float(k)) == pmf[i]

    def test_sd_is_the_root_of_var(self, name: str) -> None:
        dist = ALL_CASES[name].dist
        assert dist.sd == pytest.approx(math.sqrt(dist.var), rel=1e-15)


# --------------------------------------------------------------------------
# Moments
# --------------------------------------------------------------------------
def _brute_moments(name: str) -> Tuple[float, float]:
    ks, pmf = _table(name)
    mean = math.fsum(ks * pmf)
    var = math.fsum((ks - mean) ** 2 * pmf)
    return mean, var


class TestMoments:
    @pytest.mark.parametrize("name", [n for n in ALL_IDS if ALL_CASES[n].moments == "exact"])
    def test_mean_and_var_match_brute_force(self, name: str) -> None:
        """Includes clamped normals, whose moments are those of the clamped variable."""
        dist = ALL_CASES[name].dist
        mean, var = _brute_moments(name)
        assert dist.mean == pytest.approx(mean, rel=1e-10, abs=1e-12)
        assert dist.var == pytest.approx(var, rel=1e-9, abs=1e-11)

    @pytest.mark.parametrize("name", [n for n in CASE_IDS if CASES[n].moments == "sheppard"])
    def test_discretised_mean_matches_and_var_is_within_sheppard_correction(self, name: str) -> None:
        """Rounding keeps the mean and adds 1/12 to the variance.

        Without a clamp the ``var`` attribute of the discretised families is
        documented as the latent variance; either convention (latent or
        integer-variable variance) is accepted here, anything outside that
        band is an error.
        """
        dist = CASES[name].dist
        mean, var = _brute_moments(name)
        assert dist.mean == pytest.approx(mean, rel=1e-6, abs=1e-7)
        assert -1e-4 <= var - dist.var <= SHEPPARD + 1e-4

    @pytest.mark.parametrize("name", ["normal_20_5", "normal_neg_-3.2_2.5", "normal_67.3_24.1"])
    def test_rounded_normal_variance_is_latent_plus_one_twelfth(self, name: str) -> None:
        _, var = _brute_moments(name)
        latent_sd = {"normal_20_5": 5.0, "normal_neg_-3.2_2.5": 2.5, "normal_67.3_24.1": 24.1}[name]
        assert var == pytest.approx(latent_sd**2 + SHEPPARD, abs=1e-9)

    @pytest.mark.parametrize(
        "mu, sd, lower, mean, var",
        [
            # brute-force values, independently reproduced by Monte Carlo below
            (2.0, 5.0, 0, 3.149122251532, 12.827622613857),
            (-3.0, 2.0, 0, 0.055940621415, 0.093468985270),
            (5.0, 3.0, 2, 5.246585772542, 6.851932811832),
        ],
    )
    def test_clamped_normal_moments_describe_the_clamped_variable(
        self, mu: float, sd: float, lower: int, mean: float, var: float
    ) -> None:
        """A non-negative variable must not report a negative mean (latent mu = -3)."""
        dist = DiscretizedNormal(mu, sd, lower=lower)
        assert dist.mean == pytest.approx(mean, abs=1e-10)
        assert dist.var == pytest.approx(var, abs=1e-10)
        assert dist.mean >= lower
        assert dist.sd == pytest.approx(math.sqrt(var), abs=1e-10)

    def test_clamped_normal_moments_by_monte_carlo(self, rng: np.random.Generator) -> None:
        """max(round(N(2, 5)), 0): n = 400,000, sd 3.58, so SE(mean) = 0.0057 (5 SE = 0.028).

        SE of the sample variance is var * sqrt((kurtosis - 1) / n) < 0.03 for
        this distribution (kurtosis ~ 3.3); 5 SE = 0.15.
        """
        n = 400_000
        draws = np.maximum(np.floor(rng.normal(2.0, 5.0, n) + 0.5), 0.0)
        dist = DiscretizedNormal(2.0, 5.0, lower=0)
        assert float(draws.mean()) == pytest.approx(dist.mean, abs=0.028)
        assert float(draws.var()) == pytest.approx(dist.var, abs=0.15)

    def test_clamp_that_never_binds_leaves_the_latent_moments(self) -> None:
        dist = DiscretizedNormal(20.0, 5.0, lower=0)
        assert dist.mean == pytest.approx(20.0, abs=1e-4)
        assert dist.var == pytest.approx(25.0 + SHEPPARD, abs=2e-3)

    def test_clamp_holding_all_the_mass(self) -> None:
        dist = DiscretizedNormal(-50.0, 1.0, lower=0)
        assert dist.mean == pytest.approx(0.0, abs=1e-12)
        assert dist.var == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("name, tolerance", [("gamma_5_25", 0.01), ("gamma_0.5_0.5", 0.15)])
    def test_small_gamma_reports_the_latent_moments(self, name: str, tolerance: float) -> None:
        """Documented: mean/var are the continuous gamma's; rounding shifts small means.

        DiscretizedGamma(0.5, 0.5) has a true mean of 0.438 (-12%) and
        DiscretizedGamma(5, 25) of 4.992 (-0.2%).
        """
        dist = LATENT_CASES[name].dist
        mean, _ = _brute_moments(name)
        assert mean < dist.mean
        assert mean == pytest.approx(dist.mean, rel=tolerance)


# --------------------------------------------------------------------------
# Agreement with scipy.stats and with high-precision constants
# --------------------------------------------------------------------------
class TestAgainstScipy:
    @pytest.mark.parametrize("name", sorted(SCIPY_REFERENCES))
    def test_cdf_sf_pmf_agree_to_1e_10(self, name: str) -> None:
        dist = ALL_CASES[name].dist
        ref_cdf, ref_sf = SCIPY_REFERENCES[name]
        ks, pmf = _table(name)
        cdf = np.array([dist.cdf(int(k)) for k in ks])
        sf = np.array([dist.sf(int(k)) for k in ks])
        expected_cdf = np.asarray(ref_cdf(ks), dtype=float)
        assert cdf == pytest.approx(expected_cdf, abs=1e-10)
        assert sf == pytest.approx(np.asarray(ref_sf(ks), dtype=float), abs=1e-10)
        expected_pmf = np.diff(np.concatenate([[float(ref_cdf(int(ks[0]) - 1))], expected_cdf]))
        assert pmf == pytest.approx(expected_pmf, abs=1e-10)

    @pytest.mark.parametrize("mean", [0.3, 2.5, 24.7, 310.2, 5000.0])
    def test_poisson_pmf_relative_accuracy(self, mean: float) -> None:
        dist = PoissonDist(mean)
        ref = stats.poisson(mean)
        sd = math.sqrt(mean)
        for k in sorted({0, 1, 2, max(0, int(mean - 6 * sd)), int(mean), int(mean + 6 * sd) + 1}):
            assert dist.pmf(k) == pytest.approx(float(ref.pmf(k)), rel=1e-10, abs=1e-300)

    @pytest.mark.parametrize("mean, var", [(0.5, 0.75), (2.5, 4.0), (10.0, 30.0), (25.0, 60.0), (150.0, 3000.0), (5.0, 500.0)])
    def test_negbin_pmf_relative_accuracy(self, mean: float, var: float) -> None:
        dist = NegBinDist(mean, var)
        ref = stats.nbinom(mean * mean / (var - mean), mean / var)
        sd = math.sqrt(var)
        for k in sorted({0, 1, 2, int(mean), int(mean + 3 * sd), int(mean + 8 * sd)}):
            assert dist.pmf(k) == pytest.approx(float(ref.pmf(k)), rel=1e-10, abs=1e-300)

    def test_negbin_parameters_follow_the_docstring(self) -> None:
        dist = NegBinDist(10.0, 30.0)
        assert dist.r == pytest.approx(5.0, rel=1e-15)  # 100 / (30 - 10)
        assert dist.p == pytest.approx(1.0 / 3.0, rel=1e-15)  # r / (r + mean)


# Reference constants produced with mpmath (mp.dps = 50).
#   Poisson:  pmf = exp(k ln m - m - lgamma(k+1)); cdf = fsum of pmf(0..k).
#   NegBin:   r = m^2/(v-m), p = r/(r+m),
#             pmf = exp(lgamma(k+r) - lgamma(r) - lgamma(k+1) + r ln p + k ln(1-p)),
#             cdf = fsum of pmf(0..k).
#   Normal:   cdf = mp.ncdf((k + 1/2 - mu) / sd).
#   Gamma:    cdf = mp.gammainc(shape, 0, (k + 1/2)/scale, regularized=True)
#             with shape = m^2/v and scale = v/m.
# Parameters were entered as exact decimal strings.
MP_POISSON = [  # (mean, k, pmf, cdf)
    (2.5, 3, 0.21376301724973644575, 0.75757613313306596375),
    (0.01, 0, 0.99004983374916805357, 0.99004983374916805357),
    (0.01, 2, 0.000049502491687458402679, 0.99999983457834719251),
    (24.7, 24, 0.080342900258303193051, 0.49738512720069008883),
    (24.7, 40, 0.001172625567689035435, 0.99835060191260947157),
    (300.0, 280, 0.012050767388696130412, 0.1295252242678801498),
    (300.0, 330, 0.0051379725593361333602, 0.95922367252001928142),
    (1e-6, 0, 0.99999900000049999983, 0.99999900000049999983),
    (1e-6, 1, 9.9999900000049999983e-7, 0.99999999999950000033),
]
MP_NEGBIN = [  # (mean, var, k, pmf, cdf)
    (2.5, 4.0, 3, 0.16462367803006047703, 0.7397358513636032267),
    (10.0, 30.0, 12, 0.05772580602984061589, 0.71860255434244728342),
    (10.0, 30.0, 0, 0.0041152263374485596708, 0.0041152263374485596708),
    (25.0, 60.0, 40, 0.0082840361481199478183, 0.96590770826869320906),
    (0.5, 0.75, 1, 0.22222222222222222222, 0.88888888888888888889),
    (150.0, 3000.0, 100, 0.0062420175976181748314, 0.18359171329162063533),
    (150.0, 3000.0, 400, 0.00001504643768375485184, 0.99958076976874939891),
    (5.0, 5000.0, 0, 0.96601747995226455777, 0.96601747995226455777),
    (5.0, 5000.0, 50, 0.000094063535663129547267, 0.98775447620223549929),
]
MP_NORMAL = [  # (mu, sd, k, cdf)
    (20.0, 5.0, 22, 0.69146246127401310364),
    (20.0, 5.0, 3, 0.00048342414238377720111),
    (-3.2, 2.5, -6, 0.17878637961437171282),
    (0.0, 1.0, 5, 0.99999998101043753411),
    (67.3, 24.1, 100, 0.91583556406848271364),
    (0.0, 1.0, -7, 4.0160005838591178083e-11),
]
MP_GAMMA = [  # (mean, var, k, cdf)
    (60.0, 900.0, 55, 0.50584675584958153147),
    (60.0, 900.0, 150, 0.98991324994731436373),
    (5.0, 25.0, 0, 0.095162581964040426836),
    (5.0, 25.0, 9, 0.85043138077736494736),
    (0.5, 0.5, 0, 0.68268949213708589717),
    (250.0, 8000.0, 200, 0.31803935506564047197),
]
# NegBin close to the Poisson limit, binary-exact parameters so that the float
# inputs equal the mpmath inputs: mean = 16, var = 16 + 2**-20 (r = 2**28) and
# var = 16 + 2**-26 (r = 2**34).  (k, pmf, cdf)
MP_NEGBIN_R_2_28 = [
    (10, 0.034097701479801005881, 0.077396021867500892321),
    (16, 0.099217528665243086301, 0.56596242300987659894),
    (25, 0.0091969160536960716507, 0.98688143464561225548),
]
MP_NEGBIN_R_2_34 = [
    (10, 0.034097699854292994412, 0.077396015865624957757),
    (16, 0.099217531575954056401, 0.56596242300987656957),
    (25, 0.009196915109372365623, 0.98688143707387335177),
]
# Far upper tails, also mpmath at 50 digits (1 - cdf evaluated without
# cancellation).  (description, distribution factory, k, P(X > k))
MP_FAR_TAILS: List[Tuple[str, Callable[[], Dist], int, float]] = [
    ("poisson(2.5) > 30", lambda: PoissonDist(2.5), 30, 2.3475600844315837074e-23),
    ("poisson(0.01) > 4", lambda: PoissonDist(0.01), 4, 8.2641856418064977585e-13),
    ("negbin(2.5, 4) > 60", lambda: NegBinDist(2.5, 4.0), 60, 1.6283496468934604614e-22),
    ("normal(0, 1) > 9", lambda: DiscretizedNormal(0.0, 1.0), 9, 1.0494515075362607493e-21),
    ("normal(20, 5) > 70", lambda: DiscretizedNormal(20.0, 5.0), 70, 2.7621094717645067829e-24),
    ("gamma(60, 900) > 600", lambda: DiscretizedGamma(60.0, 900.0), 600, 4.7401168974010545525e-14),
]


class TestAgainstHighPrecision:
    @pytest.mark.parametrize("mean, k, pmf, cdf", MP_POISSON)
    def test_poisson(self, mean: float, k: int, pmf: float, cdf: float) -> None:
        dist = PoissonDist(mean)
        assert dist.pmf(k) == pytest.approx(pmf, rel=1e-12, abs=0.0)
        assert dist.cdf(k) == pytest.approx(cdf, abs=1e-14)
        assert dist.sf(k) == pytest.approx(1.0 - cdf, abs=1e-14)

    @pytest.mark.parametrize("mean, var, k, pmf, cdf", MP_NEGBIN)
    def test_negbin(self, mean: float, var: float, k: int, pmf: float, cdf: float) -> None:
        dist = NegBinDist(mean, var)
        assert dist.pmf(k) == pytest.approx(pmf, rel=1e-11, abs=0.0)
        assert dist.cdf(k) == pytest.approx(cdf, abs=1e-13)

    @pytest.mark.parametrize("mu, sd, k, cdf", MP_NORMAL)
    def test_discretized_normal(self, mu: float, sd: float, k: int, cdf: float) -> None:
        assert DiscretizedNormal(mu, sd).cdf(k) == pytest.approx(cdf, rel=1e-12, abs=1e-15)

    @pytest.mark.parametrize("mean, var, k, cdf", MP_GAMMA)
    def test_discretized_gamma(self, mean: float, var: float, k: int, cdf: float) -> None:
        assert DiscretizedGamma(mean, var).cdf(k) == pytest.approx(cdf, rel=1e-12, abs=0.0)

    @pytest.mark.parametrize("k, pmf, cdf", MP_NEGBIN_R_2_28)
    def test_negbin_cdf_stays_accurate_near_the_poisson_limit(self, k: int, pmf: float, cdf: float) -> None:
        dist = NegBinDist(16.0, 16.0 + 2.0**-20)
        assert dist.cdf(k) == pytest.approx(cdf, abs=1e-12)
        assert dist.cdf(k) - dist.cdf(k - 1) == pytest.approx(pmf, rel=1e-11, abs=0.0)

    @pytest.mark.parametrize(
        "excess, table, tolerance",
        [(2.0**-20, MP_NEGBIN_R_2_28, 1e-9), (2.0**-26, MP_NEGBIN_R_2_34, 1e-6)],
    )
    def test_negbin_pmf_stays_accurate_near_the_poisson_limit(
        self, excess: float, table: List[Tuple[int, float, float]], tolerance: float
    ) -> None:
        """gammaln(k + r) - gammaln(r) cancels for huge r; the pmf must not inherit that.

        At r = 2**34 the incomplete beta function itself is only good to ~1e-8,
        hence the looser tolerance; FamilySpec never builds such an r (it
        switches to Poisson below a relative excess of 1e-7).
        """
        dist = NegBinDist(16.0, 16.0 + excess)
        for k, pmf, _ in table:
            assert dist.pmf(k) == pytest.approx(pmf, rel=tolerance, abs=0.0)
            assert dist.prob_push(float(k)) == pytest.approx(pmf, rel=tolerance, abs=0.0)

    @pytest.mark.parametrize("label, factory, k, tail", MP_FAR_TAILS, ids=[t[0] for t in MP_FAR_TAILS])
    def test_far_upper_tail_keeps_relative_accuracy(
        self, label: str, factory: Callable[[], Dist], k: int, tail: float
    ) -> None:
        """1 - cdf would collapse to 0.0 here; a direct survival function does not."""
        dist = factory()
        assert dist.sf(k) == pytest.approx(tail, rel=1e-9, abs=0.0)
        assert dist.prob_over(k + 0.5) == pytest.approx(tail, rel=1e-9, abs=0.0)
        assert dist.prob_over(float(k)) == pytest.approx(tail, rel=1e-9, abs=0.0)

    @pytest.mark.parametrize(
        "dist, k, expected",
        [
            # P(X = k) deep in the upper tail, from scipy survival functions
            (DiscretizedNormal(0.0, 1.0), 9, float(stats.norm.sf(8.5) - stats.norm.sf(9.5))),
            (DiscretizedNormal(20.0, 5.0), 70, float(stats.norm.sf(9.9) - stats.norm.sf(10.1))),
            (DiscretizedGamma(60.0, 900.0), 600, float(stats.gamma.sf(599.5, 4.0, scale=15.0) - stats.gamma.sf(600.5, 4.0, scale=15.0))),
            (ZeroInflated(DiscretizedNormal(0.0, 1.0), 0.25), 9, 0.75 * float(stats.norm.sf(8.5) - stats.norm.sf(9.5))),
        ],
    )
    def test_generic_pmf_keeps_relative_accuracy_in_the_upper_tail(self, dist: Dist, k: int, expected: float) -> None:
        assert expected < 1e-13  # guard: cdf(k) - cdf(k-1) would be pure rounding noise
        assert dist.pmf(k) == pytest.approx(expected, rel=1e-8, abs=0.0)
        assert dist.prob_push(float(k)) == pytest.approx(expected, rel=1e-8, abs=0.0)

    @pytest.mark.parametrize("label, factory, k, tail", MP_FAR_TAILS, ids=[t[0] for t in MP_FAR_TAILS])
    def test_far_upper_tail_is_accurate_in_absolute_terms(
        self, label: str, factory: Callable[[], Dist], k: int, tail: float
    ) -> None:
        dist = factory()
        assert 0.0 <= dist.sf(k) <= 1.0
        assert dist.sf(k) == pytest.approx(tail, abs=5e-16)


# --------------------------------------------------------------------------
# Over / under / push
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", ALL_IDS)
class TestLineProbabilities:
    def test_over_push_under_match_brute_force(self, name: str) -> None:
        """P(X > L), P(X == L), P(X < L) summed directly from the pmf."""
        dist = ALL_CASES[name].dist
        ks, pmf = _table(name)
        for line in _lines(name):
            over = math.fsum(pmf[ks > line])
            under = math.fsum(pmf[ks < line])
            push = math.fsum(pmf[ks == line])
            assert dist.prob_over(line) == pytest.approx(over, abs=1e-11)
            assert dist.prob_under(line) == pytest.approx(under, abs=1e-11)
            assert dist.prob_push(line) == pytest.approx(push, abs=1e-11)
            assert dist.probs(line) == pytest.approx((over, push, under), abs=1e-11)

    def test_probs_sum_to_one_for_half_and_whole_lines(self, name: str) -> None:
        dist = ALL_CASES[name].dist
        for line in _lines(name):
            over, push, under = dist.probs(line)
            assert over + push + under == pytest.approx(1.0, abs=1e-12)
            assert min(over, push, under) >= 0.0
            assert max(over, push, under) <= 1.0
            assert dist.prob_over(line) + dist.prob_push(line) + dist.prob_under(line) == pytest.approx(
                1.0, abs=1e-11
            )

    def test_half_lines_cannot_push(self, name: str) -> None:
        dist = ALL_CASES[name].dist
        for line in _lines(name):
            if line != math.floor(line):
                assert dist.prob_push(line) == 0.0
                assert dist.probs(line)[1] == 0.0
                assert dist.prob_over_no_push(line) == pytest.approx(dist.prob_over(line), abs=1e-13)

    def test_whole_line_brackets_the_adjacent_half_lines(self, name: str) -> None:
        """P(X > k) = P(X > k + 1/2) and P(X < k) = P(X < k - 1/2)."""
        dist = ALL_CASES[name].dist
        for line in _lines(name):
            if line == math.floor(line):
                assert dist.prob_over(line) == pytest.approx(dist.prob_over(line + 0.5), abs=1e-15)
                assert dist.prob_under(line) == pytest.approx(dist.prob_under(line - 0.5), abs=1e-15)
                assert dist.prob_push(line) == pytest.approx(
                    dist.prob_over(line - 0.5) - dist.prob_over(line + 0.5), abs=1e-11
                )

    def test_prob_over_no_push_is_conditional_on_no_push(self, name: str) -> None:
        dist = ALL_CASES[name].dist
        ks, pmf = _table(name)
        for line in _lines(name):
            over = math.fsum(pmf[ks > line])
            under = math.fsum(pmf[ks < line])
            if over + under > 1e-6:
                assert dist.prob_over_no_push(line) == pytest.approx(over / (over + under), abs=1e-9)

    def test_prob_over_is_non_increasing_in_the_line(self, name: str) -> None:
        dist = ALL_CASES[name].dist
        lines = sorted(_lines(name))
        overs = [dist.prob_over(line) for line in lines]
        unders = [dist.prob_under(line) for line in lines]
        assert all(a >= b - 1e-15 for a, b in zip(overs, overs[1:]))
        assert all(a <= b + 1e-15 for a, b in zip(unders, unders[1:]))


class TestLineSemantics:
    def test_is_integer_line(self) -> None:
        assert is_integer_line(24.0)
        assert is_integer_line(24)
        assert is_integer_line(0.0)
        assert is_integer_line(-3.0)
        assert not is_integer_line(24.5)
        assert not is_integer_line(-0.5)
        assert not is_integer_line(24.001)
        assert not is_integer_line(24.25)

    def test_is_integer_line_tolerates_float_noise(self) -> None:
        assert is_integer_line(0.1 + 0.2 + 24.7)  # 25.000000000000004
        assert is_integer_line(25.0 - 1e-12)
        assert not is_integer_line(25.0 - 1e-6)

    def test_poisson_whole_line_by_hand(self) -> None:
        """Poisson(3) at line 3: P(X>3), P(X=3), P(X<3) from the series."""
        e3 = math.exp(-3.0)
        under = e3 * (1 + 3 + 4.5)
        push = e3 * 4.5
        over = 1.0 - under - push
        assert PoissonDist(3.0).probs(3) == pytest.approx((over, push, under), abs=1e-14)
        assert PoissonDist(3.0).probs(3.0) == pytest.approx((over, push, under), abs=1e-14)
        assert PoissonDist(3.0).probs(2.5) == pytest.approx((over + push, 0.0, under), abs=1e-14)
        assert PoissonDist(3.0).probs(3.5) == pytest.approx((over, 0.0, under + push), abs=1e-14)
        assert PoissonDist(3.0).prob_over_no_push(3) == pytest.approx(over / (over + under), abs=1e-14)

    def test_count_distributions_below_zero(self) -> None:
        for dist in (PoissonDist(3.0), NegBinDist(3.0, 5.0), DiscretizedGamma(3.0, 5.0)):
            assert dist.probs(-0.5) == pytest.approx((1.0, 0.0, 0.0), abs=1e-15)
            assert dist.probs(-1) == pytest.approx((1.0, 0.0, 0.0), abs=1e-15)
            assert dist.prob_under(0) == 0.0
            assert dist.prob_push(0) == dist.pmf(0) > 0.0
            assert dist.pmf(-1) == 0.0
            assert dist.cdf(-1) == 0.0
            assert dist.cdf(-0.5) == 0.0

    def test_negative_lines_on_a_signed_distribution(self) -> None:
        """Rushing yards can be negative: N(0, 3) rounded, line -2 and -2.5."""
        dist = DiscretizedNormal(0.0, 3.0)
        phi = stats.norm.cdf
        over = 1.0 - phi(-1.5 / 3.0)   # X >= -1  <=>  N > -1.5
        push = phi(-1.5 / 3.0) - phi(-2.5 / 3.0)
        under = phi(-2.5 / 3.0)        # X <= -3  <=>  N < -2.5
        assert dist.probs(-2) == pytest.approx((over, push, under), abs=1e-14)
        assert dist.probs(-2.5) == pytest.approx((over + push, 0.0, under), abs=1e-14)
        assert dist.prob_under(-2.5) == pytest.approx(under, abs=1e-14)
        assert dist.prob_over(-1.5) == pytest.approx(over, abs=1e-14)

    def test_degenerate_push_returns_one_half(self) -> None:
        """All mass on the line: the conditional over probability is defined as 0.5."""
        dist = DiscretizedNormal(5.0, 1e-9)
        assert dist.probs(5) == pytest.approx((0.0, 1.0, 0.0), abs=1e-15)
        assert dist.prob_over_no_push(5) == 0.5

    def test_nan_line_raises(self) -> None:
        with pytest.raises(ValueError):
            PoissonDist(3.0).prob_over(NAN)
        with pytest.raises(ValueError):
            PoissonDist(3.0).probs(NAN)


# --------------------------------------------------------------------------
# Quantiles, median, intervals
# --------------------------------------------------------------------------
QUANTILE_LEVELS = (1e-9, 0.0013, 0.02, 0.1, 0.25, 0.5, 0.731, 0.9, 0.975, 0.9999, 1 - 1e-9)


@pytest.mark.parametrize("name", ALL_IDS)
class TestQuantile:
    def test_is_the_smallest_k_with_cdf_at_least_q(self, name: str) -> None:
        dist = ALL_CASES[name].dist
        for q in QUANTILE_LEVELS:
            k = dist.quantile(q)
            assert isinstance(k, int)
            assert dist.cdf(k) >= q
            assert dist.cdf(k - 1) < q

    def test_matches_a_brute_force_scan(self, name: str) -> None:
        dist = ALL_CASES[name].dist
        ks, pmf = _table(name)
        cum = np.cumsum(pmf)
        for q in QUANTILE_LEVELS:
            if np.min(np.abs(cum - q)) < 1e-12:
                continue  # q sits on a step: the scan would depend on rounding
            assert dist.quantile(q) == int(ks[np.searchsorted(cum, q, side="left")])

    def test_is_monotone_in_q(self, name: str) -> None:
        dist = ALL_CASES[name].dist
        values = [dist.quantile(q) for q in QUANTILE_LEVELS]
        assert values == sorted(values)

    def test_median(self, name: str) -> None:
        dist = ALL_CASES[name].dist
        assert dist.median == dist.quantile(0.5)
        assert dist.cdf(dist.median) >= 0.5 > dist.cdf(dist.median - 1)

    @pytest.mark.parametrize("coverage", [0.5, 0.8, 0.95, 0.999])
    def test_interval_has_at_least_the_requested_coverage(self, name: str, coverage: float) -> None:
        dist = ALL_CASES[name].dist
        ks, pmf = _table(name)
        lo, hi = dist.interval(coverage)
        tail = (1.0 - coverage) / 2.0
        assert (lo, hi) == (dist.quantile(tail), dist.quantile(1.0 - tail))
        assert lo <= dist.median <= hi
        assert math.fsum(pmf[(ks >= lo) & (ks <= hi)]) >= coverage - 1e-12
        # central: neither tail outside the interval exceeds its share
        assert math.fsum(pmf[ks < lo]) <= tail + 1e-12
        assert math.fsum(pmf[ks > hi]) <= tail + 1e-12


class TestQuantileSpecialCases:
    @pytest.mark.parametrize("q", [0.0, 1.0, -0.1, 1.1, NAN])
    def test_invalid_levels_raise(self, q: float) -> None:
        with pytest.raises(ValueError):
            PoissonDist(3.0).quantile(q)

    def test_default_interval_is_80_percent(self) -> None:
        dist = NegBinDist(10.0, 30.0)
        assert dist.interval() == (dist.quantile(0.1), dist.quantile(0.9))
        assert dist.interval() == dist.interval(0.8)

    @pytest.mark.parametrize("coverage", [1.0, 0.0, -0.1, 1.5, NAN])
    def test_interval_rejects_coverage_outside_the_open_unit_interval(self, coverage: float) -> None:
        with pytest.raises(ValueError):
            PoissonDist(3.0).interval(coverage)

    def test_narrow_interval_collapses_onto_the_median(self) -> None:
        dist = NegBinDist(10.0, 30.0)
        assert dist.interval(1e-9) == (dist.median, dist.median)

    @pytest.mark.parametrize("mean", [0.3, 2.5, 24.7, 310.2, 1e4])
    def test_poisson_matches_scipy_ppf(self, mean: float) -> None:
        dist = PoissonDist(mean)
        for q in (1e-6, 0.01, 0.2, 0.5, 0.8, 0.99, 1 - 1e-6):
            assert dist.quantile(q) == int(stats.poisson.ppf(q, mean))

    def test_huge_poisson_mean_satisfies_the_definition_under_scipy_cdf(self) -> None:
        """At mean 1e7 scipy's own ppf is off by ~20 in the tails, so use its cdf."""
        mean = 1e7
        dist = PoissonDist(mean)
        for q in (1e-6, 0.01, 0.2, 0.5, 0.8, 0.99, 1 - 1e-6):
            k = dist.quantile(q)
            assert float(stats.poisson.cdf(k, mean)) >= q > float(stats.poisson.cdf(k - 1, mean))
        assert dist.median == 10_000_000

    @pytest.mark.parametrize("mean, var", [(0.5, 0.75), (10.0, 30.0), (150.0, 3000.0), (5.0, 5000.0), (1000.0, 1e7)])
    def test_negbin_matches_scipy_ppf_including_heavy_tails(self, mean: float, var: float) -> None:
        dist = NegBinDist(mean, var)
        ref = stats.nbinom(mean * mean / (var - mean), mean / var)
        for q in (1e-6, 0.01, 0.2, 0.5, 0.8, 0.99, 0.9999, 1 - 1e-6):
            assert dist.quantile(q) == int(ref.ppf(q))

    def test_heavy_tailed_negbin_has_median_far_below_its_mean(self) -> None:
        dist = NegBinDist(1000.0, 1e7)  # r = 0.1: most mass near zero, mean 1000
        assert dist.median < 20
        lo, hi = dist.interval(0.999)
        assert lo == 0 and hi > 20000

    @pytest.mark.parametrize("mean", [1e-300, 1e-12, 1e-9, 1e-4])
    def test_tiny_poisson_means(self, mean: float) -> None:
        dist = PoissonDist(mean)
        assert dist.median == 0
        assert dist.interval(0.8) == (0, 0)
        assert dist.quantile(1e-12) == 0
        if mean >= 1e-12:
            # P(X = 0) = exp(-mean) ~ 1 - mean separates these two levels
            assert dist.quantile(1.0 - 2.0 * mean) == 0
            assert dist.quantile(1.0 - 0.5 * mean) == 1

    def test_tiny_negbin_and_gamma_means(self) -> None:
        assert NegBinDist(0.001, 10.0).interval(0.999) == (0, 0)
        assert DiscretizedGamma(1e-6, 1e-3).interval(0.999) == (0, 0)

    def test_exponential_gamma_quantile_in_closed_form(self) -> None:
        """shape 1, scale 1000: cdf(k) = 1 - exp(-(k + 1/2)/1000)."""
        dist = DiscretizedGamma(1000.0, 1e6)
        for q in (0.001, 0.1, 0.5, 0.9, 0.999, 1 - 1e-9):
            assert dist.quantile(q) == max(math.ceil(-1000.0 * math.log1p(-q) - 0.5), 0)

    def test_gamma_median_sits_below_its_mean(self) -> None:
        """Documented shape property of the right-skewed family."""
        for mean, var in ((60.0, 900.0), (250.0, 8000.0), (40.0, 1600.0)):
            assert DiscretizedGamma(mean, var).median < mean

    def test_normal_quantiles_follow_the_normal_quantile_function(self) -> None:
        dist = DiscretizedNormal(67.3, 24.1)
        for q in (0.01, 0.1, 0.5, 0.9, 0.99):
            expected = math.ceil(67.3 + 24.1 * float(stats.norm.ppf(q)) - 0.5)
            assert dist.quantile(q) == expected

    def test_far_negative_normals(self) -> None:
        dist = DiscretizedNormal(-1e6, 1e3)
        assert dist.median == -1000000
        lo, hi = dist.interval(0.8)
        assert lo == math.ceil(-1e6 + 1e3 * float(stats.norm.ppf(0.1)) - 0.5)
        assert hi == math.ceil(-1e6 + 1e3 * float(stats.norm.ppf(0.9)) - 0.5)

    def test_clamped_normal_with_all_mass_on_the_clamp(self) -> None:
        dist = DiscretizedNormal(-50.0, 1.0, lower=0)
        for q in (1e-12, 0.5, 1 - 1e-12):
            assert dist.quantile(q) == 0
        assert dist.interval(0.999) == (0, 0)
        assert dist.probs(0) == pytest.approx((0.0, 1.0, 0.0), abs=1e-15)

    def test_spike_normal(self) -> None:
        dist = DiscretizedNormal(1e6, 1e-3)
        assert dist.interval(0.999) == (1000000, 1000000)


# --------------------------------------------------------------------------
# Constructors
# --------------------------------------------------------------------------
class TestConstructors:
    @pytest.mark.parametrize("mean", [0.0, -1.0, NAN, INF])
    def test_poisson_rejects_non_positive_or_non_finite_mean(self, mean: float) -> None:
        with pytest.raises(ValueError):
            PoissonDist(mean)

    @pytest.mark.parametrize(
        "mean, var", [(0.0, 1.0), (-1.0, 1.0), (1.0, 1.0), (1.0, 0.5), (2.0, NAN), (NAN, 2.0), (2.0, INF), (INF, INF)]
    )
    def test_negbin_requires_positive_mean_and_overdispersion(self, mean: float, var: float) -> None:
        with pytest.raises(ValueError):
            NegBinDist(mean, var)

    @pytest.mark.parametrize(
        "mean, var, scale",
        [(0.0, 5.0, 2.0), (-1.0, 5.0, 2.0), (NAN, 5.0, 2.0), (INF, 5.0, 2.0), (5.0, 9.0, 0.99), (5.0, 9.0, 0.0),
         (5.0, 9.0, -2.0), (5.0, 9.0, NAN), (5.0, 9.0, INF)],
    )
    def test_scaled_negbin_requires_positive_mean_and_scale_of_at_least_one(
        self, mean: float, var: float, scale: float
    ) -> None:
        with pytest.raises(ValueError):
            ScaledNegBinDist(mean, var, scale)

    @pytest.mark.parametrize("mean, sd", [(1.0, 0.0), (1.0, -1.0), (1.0, NAN), (1.0, INF), (NAN, 1.0), (INF, 1.0), (-INF, 1.0)])
    def test_normal_rejects_invalid_parameters(self, mean: float, sd: float) -> None:
        with pytest.raises(ValueError):
            DiscretizedNormal(mean, sd)
        with pytest.raises(ValueError):
            DiscretizedNormal(mean, sd, lower=0)

    @pytest.mark.parametrize(
        "mean, var", [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -2.0), (NAN, 1.0), (1.0, NAN), (INF, 1.0), (1.0, INF)]
    )
    def test_gamma_rejects_non_positive_parameters(self, mean: float, var: float) -> None:
        with pytest.raises(ValueError):
            DiscretizedGamma(mean, var)

    @pytest.mark.parametrize("pi", [1.0, 1.5, -0.01, NAN])
    def test_zero_inflation_weight_must_be_in_half_open_unit_interval(self, pi: float) -> None:
        with pytest.raises(ValueError):
            ZeroInflated(PoissonDist(1.0), pi)

    def test_mixture_validation(self) -> None:
        one, two = PoissonDist(1.0), PoissonDist(2.0)
        with pytest.raises(ValueError):
            MixtureDist([], [])
        with pytest.raises(ValueError):
            MixtureDist([one], [1.0, 2.0])
        with pytest.raises(ValueError):
            MixtureDist([one, two], [1.0])
        with pytest.raises(ValueError):
            MixtureDist([one, two], [2.0, -1.0])
        with pytest.raises(ValueError):
            MixtureDist([one, two], [0.0, 0.0])
        with pytest.raises(ValueError):
            MixtureDist([one, two], [1.0, INF])

    @pytest.mark.parametrize(
        "dist, fragments",
        [
            (PoissonDist(2.5), ["PoissonDist", "2.5"]),
            (NegBinDist(10.0, 30.0), ["NegBinDist", "10", "30"]),
            (ScaledNegBinDist(24.0, 60.0, 2.2), ["ScaledNegBinDist", "24", "60", "2.2"]),
            (DiscretizedNormal(20.0, 5.0), ["DiscretizedNormal", "20", "5"]),
            (DiscretizedNormal(20.0, 5.0, lower=3), ["DiscretizedNormal", "20", "5", "lower=3"]),
            (DiscretizedGamma(60.0, 900.0), ["DiscretizedGamma", "60", "900"]),
            (ZeroInflated(PoissonDist(4.0), 0.3), ["ZeroInflated", "PoissonDist", "0.3"]),
            (MixtureDist([PoissonDist(4.0), PoissonDist(1.0)], [1, 1]), ["MixtureDist", "2"]),
        ],
    )
    def test_reprs_name_the_class_and_parameters(self, dist: Dist, fragments: List[str]) -> None:
        text = repr(dist)
        for fragment in fragments:
            assert fragment in text

    def test_base_class_is_abstract(self) -> None:
        with pytest.raises(NotImplementedError):
            Dist().cdf(1)


# --------------------------------------------------------------------------
# Family-specific behaviour
# --------------------------------------------------------------------------
class TestDiscretizedNormal:
    def test_is_symmetric_about_an_integer_mean(self) -> None:
        dist = DiscretizedNormal(4.0, 2.7)
        for j in range(0, 15):
            assert dist.pmf(4 + j) == pytest.approx(dist.pmf(4 - j), abs=1e-15)
        assert dist.pmf(-3) > 0.0  # negative values carry mass without a clamp

    def test_half_lines_fall_on_rounding_boundaries(self) -> None:
        """P(X > n + 1/2) = P(N > n + 1/2): no continuity correction is needed."""
        dist = DiscretizedNormal(20.0, 5.0)
        for line in (12.5, 19.5, 20.5, 31.5):
            assert dist.prob_over(line) == pytest.approx(float(stats.norm.sf(line, 20.0, 5.0)), abs=1e-14)
        assert dist.prob_over(19.5) + dist.prob_over(20.5) == pytest.approx(1.0, abs=1e-14)

    @pytest.mark.parametrize("mu, sd, lower", [(2.0, 5.0, 0), (-3.0, 2.0, 0), (5.0, 3.0, 2), (20.0, 5.0, 0)])
    def test_clamp_piles_the_lower_tail_onto_the_bound(self, mu: float, sd: float, lower: int) -> None:
        free = DiscretizedNormal(mu, sd)
        clamped = DiscretizedNormal(mu, sd, lower=lower)
        assert clamped.lower == lower and free.lower is None
        for k in range(lower - 6, lower):
            assert clamped.pmf(k) == 0.0
            assert clamped.cdf(k) == 0.0
        assert clamped.pmf(lower) == pytest.approx(free.cdf(lower), abs=1e-15)
        assert clamped.pmf(lower) == pytest.approx(
            float(stats.norm.cdf((lower + 0.5 - mu) / sd)), abs=1e-14
        )
        for k in range(lower + 1, lower + 40):
            assert clamped.pmf(k) == pytest.approx(free.pmf(k), abs=1e-15)
            assert clamped.cdf(k) == free.cdf(k)

    def test_clamp_does_not_change_over_probabilities_above_the_bound(self) -> None:
        free = DiscretizedNormal(2.0, 5.0)
        clamped = DiscretizedNormal(2.0, 5.0, lower=0)
        for line in (0.5, 1, 4.5, 12):
            assert clamped.prob_over(line) == free.prob_over(line)
        assert clamped.prob_under(0.5) == pytest.approx(free.prob_under(0.5), abs=1e-15)
        assert clamped.prob_under(0) == 0.0 < free.prob_under(0)
        assert clamped.prob_over(-0.5) == 1.0 > free.prob_over(-0.5)

    def test_monte_carlo_rounding_semantics(self, rng: np.random.Generator) -> None:
        """Simulate round(N(mu, sd)) with and without the clamp.

        n = 400,000 draws; a binomial proportion has standard error
        sqrt(p(1-p)/n) <= 0.0008, so 5 SE = 0.004 at most.  An off-by-half
        rounding convention would move these cdf values by ~0.04.
        """
        n = 400_000
        draws = np.floor(rng.normal(3.3, 4.2, n) + 0.5)
        free = DiscretizedNormal(3.3, 4.2)
        clamped = DiscretizedNormal(3.3, 4.2, lower=0)
        for k in (-6, -1, 0, 3, 4, 9):
            for dist, sample in ((free, draws), (clamped, np.maximum(draws, 0.0))):
                p = dist.cdf(k)
                tol = 5.0 * math.sqrt(max(p * (1.0 - p), 1e-12) / n) + 1e-9
                assert float(np.mean(sample <= k)) == pytest.approx(p, abs=tol)
        p0 = clamped.pmf(0)
        assert float(np.mean(np.maximum(draws, 0.0) == 0)) == pytest.approx(
            p0, abs=5.0 * math.sqrt(p0 * (1.0 - p0) / n)
        )


class TestDiscretizedGamma:
    def test_shape_and_scale(self) -> None:
        dist = DiscretizedGamma(60.0, 900.0)
        assert dist.shape == pytest.approx(4.0, rel=1e-15)
        assert dist.scale == pytest.approx(15.0, rel=1e-15)

    def test_zero_collects_the_mass_below_one_half(self) -> None:
        dist = DiscretizedGamma(5.0, 25.0)  # exponential with scale 5
        assert dist.pmf(0) == pytest.approx(1.0 - math.exp(-0.1), abs=1e-15)
        assert dist.pmf(3) == pytest.approx(math.exp(-0.5) - math.exp(-0.7), abs=1e-15)
        assert dist.prob_over(9.5) == pytest.approx(math.exp(-1.9), abs=1e-15)

    def test_monte_carlo_rounding_semantics(self, rng: np.random.Generator) -> None:
        """Simulate round(Gamma(shape, scale)); n = 400,000, tolerance 5 SE (<= 0.004)."""
        n = 400_000
        mean, var = 12.0, 60.0
        draws = np.floor(rng.gamma(mean * mean / var, var / mean, n) + 0.5)
        dist = DiscretizedGamma(mean, var)
        for k in (0, 3, 8, 12, 20, 35):
            p = dist.cdf(k)
            assert float(np.mean(draws <= k)) == pytest.approx(
                p, abs=5.0 * math.sqrt(p * (1.0 - p) / n) + 1e-9
            )


class TestNegBinAsGammaPoisson:
    def test_monte_carlo_gamma_mixed_poisson(self, rng: np.random.Generator) -> None:
        """Docstring: a Poisson whose rate is gamma distributed with shape r.

        n = 400,000 draws.  cdf tolerance is 5 binomial SE (<= 0.004); the
        sample mean has SE sqrt(var/n) = 0.0087, tolerance 5 SE.
        """
        n = 400_000
        mean, var = 10.0, 30.0
        r = mean * mean / (var - mean)
        draws = rng.poisson(rng.gamma(r, mean / r, n))
        dist = NegBinDist(mean, var)
        for k in (2, 6, 10, 15, 25):
            p = dist.cdf(k)
            assert float(np.mean(draws <= k)) == pytest.approx(p, abs=5.0 * math.sqrt(p * (1.0 - p) / n))
        assert float(draws.mean()) == pytest.approx(dist.mean, abs=5.0 * math.sqrt(var / n))
        # SE of a sample variance ~ var * sqrt((kurtosis - 1) / n); kurtosis < 6 here
        assert float(draws.var()) == pytest.approx(dist.var, abs=5.0 * var * math.sqrt(5.0 / n))

    def test_converges_to_poisson_as_overdispersion_vanishes(self) -> None:
        poisson = PoissonDist(7.0)
        previous = math.inf
        for excess in (1.0, 0.1, 0.01, 0.001):
            negbin = NegBinDist(7.0, 7.0 + excess)
            gap = max(abs(negbin.cdf(k) - poisson.cdf(k)) for k in range(0, 40))
            assert gap < previous
            previous = gap
        assert previous < 1e-4


def _brute(dist: Dist, hi: int) -> Tuple[float, float, float]:
    """Mean, variance and skewness of a non-negative distribution by summation."""
    ks = np.arange(0, hi + 1)
    pmf = np.array([dist.pmf(int(k)) for k in ks])
    mean = math.fsum(ks * pmf)
    var = math.fsum((ks - mean) ** 2 * pmf)
    skew = math.fsum((ks - mean) ** 3 * pmf) / var**1.5
    return mean, var, skew


class TestScaledNegBin:
    @pytest.mark.parametrize("mean, var", [(0.5, 0.75), (10.0, 30.0), (24.0, 60.0), (150.0, 3000.0)])
    def test_scale_one_is_exactly_the_negative_binomial(self, mean: float, var: float) -> None:
        scaled, plain = ScaledNegBinDist(mean, var, 1.0), NegBinDist(mean, var)
        assert (scaled.mean, scaled.var, scaled.scale) == (plain.mean, plain.var, 1.0)
        for k in range(-2, int(mean + 12 * math.sqrt(var))):
            assert scaled.cdf(k) == plain.cdf(k)

    @pytest.mark.parametrize("var", [10.0, 7.0, 0.0])
    def test_scale_one_without_overdispersion_is_poisson(self, var: float) -> None:
        scaled, poisson = ScaledNegBinDist(10.0, var, 1.0), PoissonDist(10.0)
        assert scaled.var == 10.0  # floored at scale * mean
        for k in range(-2, 60):
            assert scaled.cdf(k) == poisson.cdf(k)

    @pytest.mark.parametrize(
        "mean, var, scale, floored",
        [(24.0, 60.0, 2.2, 60.0), (24.0, 30.0, 2.2, 52.8), (20.0, 10.0, 2.0, 40.0), (5.0, 30.0, 5.0, 30.0), (5.0, 1.0, 5.0, 25.0)],
    )
    def test_variance_is_floored_at_scale_times_mean(
        self, mean: float, var: float, scale: float, floored: float
    ) -> None:
        dist = ScaledNegBinDist(mean, var, scale)
        assert dist.var == pytest.approx(floored, rel=1e-12)
        assert dist.mean == mean

    @pytest.mark.parametrize("mean, var", [(24.0, 100.0), (9.0, 40.0), (60.0, 400.0)])
    def test_integer_scale_reproduces_the_event_count_on_its_lattice(self, mean: float, var: float) -> None:
        """With scale 3, (k + 1/2)/3 - 1/2 is the integer n at k = 3n + 1.

        There the documented cdf G((k + 1/2)/scale - 1/2) is the ordinary
        negative-binomial cdf of the event count N (mean/3, var/9) at n.
        """
        dist = ScaledNegBinDist(mean, var, 3.0)
        event_mean, event_var = mean / 3.0, var / 9.0
        events = stats.nbinom(event_mean**2 / (event_var - event_mean), event_mean / event_var)
        for n in range(0, 60):
            assert dist.cdf(3 * n + 1) == pytest.approx(float(events.cdf(n)), abs=1e-12)

    def test_integer_scale_poisson_floor_on_its_lattice(self) -> None:
        dist = ScaledNegBinDist(21.0, 10.0, 3.0)  # var floored to 63: scaled Poisson(7)
        for n in range(0, 40):
            assert dist.cdf(3 * n + 1) == pytest.approx(float(stats.poisson.cdf(n, 7.0)), abs=1e-12)

    def test_is_continuous_in_the_scale_at_one(self) -> None:
        plain = NegBinDist(10.0, 30.0)
        nearly = ScaledNegBinDist(10.0, 30.0, 1.0 + 1e-7)
        assert max(abs(nearly.cdf(k) - plain.cdf(k)) for k in range(0, 80)) < 1e-7

    @pytest.mark.parametrize(
        "mean, var, scale", [(24.0, 60.0, 2.2), (24.0, 60.0, 1.5), (30.0, 500.0, 2.2), (100.0, 400.0, 2.2), (300.0, 2000.0, 2.2)]
    )
    def test_targets_are_met_closely_for_realistic_means(self, mean: float, var: float, scale: float) -> None:
        """Documented: exact moments differ 'far less' than 1% above a few units."""
        dist = ScaledNegBinDist(mean, var, scale)
        true_mean, true_var, _ = _brute(dist, int(mean + 40 * math.sqrt(var)))
        assert true_mean == pytest.approx(mean, rel=5e-4)
        assert true_var == pytest.approx(dist.var, rel=1e-2)

    @pytest.mark.parametrize("mean, var, scale", [(3.0, 5.0, 1.5), (3.0, 8.0, 2.0), (5.0, 12.0, 2.0), (8.0, 20.0, 2.0)])
    def test_mean_target_is_within_one_percent_for_means_of_a_few_units(
        self, mean: float, var: float, scale: float
    ) -> None:
        true_mean, _, _ = _brute(ScaledNegBinDist(mean, var, scale), 400)
        assert true_mean == pytest.approx(mean, rel=1e-2)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "BUG (minor, documentation): the docstring says the exact moments differ by "
            "'about 1% for means of a few units', but the variance falls well short of its "
            "target there: -2.7% for (mean 3, var 5, scale 1.5), -4.2% for (3, 8, 2), -8.4% "
            "for (5, 30, 5) and -9.1% for (2, 4.4, 2.2). The shortfall is about "
            "(scale**2 - 1)/12 plus a small-mean term"
        ),
    )
    @pytest.mark.parametrize("mean, var, scale", [(3.0, 5.0, 1.5), (3.0, 8.0, 2.0), (5.0, 30.0, 5.0), (2.0, 4.4, 2.2)])
    def test_variance_target_is_within_about_one_percent_for_means_of_a_few_units(
        self, mean: float, var: float, scale: float
    ) -> None:
        dist = ScaledNegBinDist(mean, var, scale)
        _, true_var, _ = _brute(dist, 600)
        assert true_var == pytest.approx(dist.var, rel=2e-2)

    def test_is_less_right_skewed_than_a_negative_binomial_with_the_same_moments(self) -> None:
        """The reason the class exists: same mean and variance, visibly less skew."""
        _, _, plain = _brute(NegBinDist(24.0, 60.0), 400)
        skews = [plain] + [_brute(ScaledNegBinDist(24.0, 60.0, s), 400)[2] for s in (1.5, 2.0, 2.4)]
        assert all(a > b for a, b in zip(skews, skews[1:]))
        assert plain == pytest.approx((2 * 60.0 / 24.0 - 1) / math.sqrt(60.0), rel=1e-6)  # (2v/m - 1)/sd
        assert skews[-1] < 0.75 * plain

    def test_support_starts_at_zero(self) -> None:
        dist = ScaledNegBinDist(24.0, 60.0, 2.2)
        assert dist.cdf(-1) == 0.0 and dist.pmf(-1) == 0.0
        assert dist.pmf(0) > 0.0
        assert dist.probs(-0.5) == pytest.approx((1.0, 0.0, 0.0), abs=1e-15)


class TestZeroInflated:
    def test_cdf_and_pmf_formula(self) -> None:
        base = NegBinDist(10.0, 30.0)
        dist = ZeroInflated(base, 0.15)
        assert dist.base is base and dist.pi == 0.15
        assert dist.pmf(0) == pytest.approx(0.15 + 0.85 * base.pmf(0), abs=1e-15)
        for k in (1, 5, 10, 40):
            assert dist.pmf(k) == pytest.approx(0.85 * base.pmf(k), abs=1e-15)
            assert dist.cdf(k) == pytest.approx(0.15 + 0.85 * base.cdf(k), abs=1e-15)
            assert dist.prob_over(k + 0.5) == pytest.approx(0.85 * base.prob_over(k + 0.5), abs=1e-15)
        assert dist.cdf(-1) == 0.0

    def test_moments_in_closed_form(self) -> None:
        """mean = (1-pi) m; var = (1-pi) (s2 + pi m^2)."""
        dist = ZeroInflated(PoissonDist(4.0), 0.3)
        assert dist.mean == pytest.approx(2.8, rel=1e-15)
        assert dist.var == pytest.approx(0.7 * (4.0 + 0.3 * 16.0), rel=1e-15)

    def test_zero_weight_is_the_base_distribution(self) -> None:
        base = NegBinDist(10.0, 30.0)
        dist = ZeroInflated(base, 0.0)
        assert (dist.mean, dist.var) == (base.mean, base.var)
        for k in range(-2, 60):
            assert dist.cdf(k) == base.cdf(k)

    def test_point_mass_sits_at_zero_even_with_negative_support(self) -> None:
        base = DiscretizedNormal(3.0, 6.0)
        dist = ZeroInflated(base, 0.3)
        assert dist.pmf(-4) == pytest.approx(0.7 * base.pmf(-4), abs=1e-15)
        assert dist.pmf(0) == pytest.approx(0.3 + 0.7 * base.pmf(0), abs=1e-15)
        assert dist.cdf(-1) == pytest.approx(0.7 * base.cdf(-1), abs=1e-15)

    def test_monte_carlo(self, rng: np.random.Generator) -> None:
        """n = 400,000; mean tolerance 5 * sd / sqrt(n) = 0.02, cdf tolerance 5 SE."""
        n = 400_000
        draws = np.where(rng.random(n) < 0.3, 0, rng.poisson(4.0, n))
        dist = ZeroInflated(PoissonDist(4.0), 0.3)
        assert float(draws.mean()) == pytest.approx(dist.mean, abs=5.0 * dist.sd / math.sqrt(n))
        for k in (0, 2, 5, 9):
            p = dist.cdf(k)
            assert float(np.mean(draws <= k)) == pytest.approx(p, abs=5.0 * math.sqrt(p * (1.0 - p) / n))


class TestMixture:
    def test_weights_are_normalised(self) -> None:
        mix = MixtureDist([PoissonDist(1.0), PoissonDist(2.0)], [2, 6])
        assert mix.weights == pytest.approx([0.25, 0.75], rel=1e-15)
        assert MixtureDist([PoissonDist(1.0), PoissonDist(2.0)], np.array([2, 6])).weights == pytest.approx(
            [0.25, 0.75], rel=1e-15
        )

    def test_cdf_is_the_weighted_average(self) -> None:
        a, b, c = PoissonDist(4.0), NegBinDist(10.0, 30.0), DiscretizedNormal(-3.0, 2.0)
        mix = MixtureDist([a, b, c], [1.0, 2.0, 1.0])
        for k in (-6, -1, 0, 3, 9, 30):
            expected = 0.25 * a.cdf(k) + 0.5 * b.cdf(k) + 0.25 * c.cdf(k)
            assert mix.cdf(k) == pytest.approx(expected, abs=1e-15)
            assert mix.pmf(k) == pytest.approx(0.25 * a.pmf(k) + 0.5 * b.pmf(k) + 0.25 * c.pmf(k), abs=1e-14)

    def test_moments_follow_the_law_of_total_variance(self) -> None:
        a, b = PoissonDist(4.0), NegBinDist(10.0, 30.0)
        mix = MixtureDist([a, b], [1.0, 3.0])
        mean = 0.25 * 4.0 + 0.75 * 10.0
        within = 0.25 * 4.0 + 0.75 * 30.0
        between = 0.25 * (4.0 - mean) ** 2 + 0.75 * (10.0 - mean) ** 2
        assert mix.mean == pytest.approx(mean, rel=1e-15)
        assert mix.var == pytest.approx(within + between, rel=1e-15)

    def test_zero_weight_component_is_ignored(self) -> None:
        a, b = PoissonDist(4.0), PoissonDist(40.0)
        mix = MixtureDist([a, b], [1.0, 0.0])
        assert (mix.mean, mix.var) == (4.0, 4.0)
        for k in range(0, 30):
            assert mix.cdf(k) == a.cdf(k)

    def test_two_point_zero_inflation_equals_zero_inflated(self) -> None:
        """A mixture with a (nearly) degenerate spike at zero is a ZeroInflated."""
        base = PoissonDist(5.0)
        spike = DiscretizedNormal(0.0, 1e-6)
        mix = MixtureDist([spike, base], [0.2, 0.8])
        zi = ZeroInflated(base, 0.2)
        for k in range(-2, 25):
            assert mix.cdf(k) == pytest.approx(zi.cdf(k), abs=1e-15)
        assert mix.mean == pytest.approx(zi.mean, rel=1e-12)
        assert mix.var == pytest.approx(zi.var, rel=1e-9)

    def test_monte_carlo(self, rng: np.random.Generator) -> None:
        """n = 400,000 draws of a two-scenario minutes mixture; tolerances are 5 SE."""
        n = 400_000
        pick = rng.random(n) < 0.35
        draws = np.where(pick, rng.poisson(6.0, n), rng.poisson(18.0, n))
        mix = MixtureDist([PoissonDist(6.0), PoissonDist(18.0)], [0.35, 0.65])
        assert float(draws.mean()) == pytest.approx(mix.mean, abs=5.0 * mix.sd / math.sqrt(n))
        for k in (4, 9, 14, 20, 26):
            p = mix.cdf(k)
            assert float(np.mean(draws <= k)) == pytest.approx(p, abs=5.0 * math.sqrt(p * (1.0 - p) / n))


# --------------------------------------------------------------------------
# FamilySpec: construction
# --------------------------------------------------------------------------
SPECS: Dict[str, FamilySpec] = {
    "poisson": FamilySpec("poisson"),
    "negbin_default": FamilySpec(),
    "negbin_nb2": FamilySpec("negbin", phi=1.0, alpha=0.15),
    "negbin_quasi": FamilySpec("negbin", phi=1.8, alpha=0.0),
    "negbin_offset": FamilySpec("negbin", phi=1.0, alpha=0.05, c=2.0),
    "negbin_underdispersed": FamilySpec("negbin", phi=0.6, alpha=0.0),
    "negbin_heavy": FamilySpec("negbin", phi=1.0, alpha=2.0),
    "normal_clamped": FamilySpec("normal", phi=1.5, alpha=0.04, c=0.0, lower=0),
    "normal_free": FamilySpec("normal", phi=0.0, alpha=0.05, c=30.0, lower=None),
    "normal_free_cv": FamilySpec("normal", phi=2.0, alpha=0.1, c=0.0, lower=None),
    "gamma": FamilySpec("gamma", phi=2.0, alpha=0.2),
    "gamma_cv": FamilySpec("gamma", phi=0.0, alpha=0.3),
    "gamma_offset": FamilySpec("gamma", phi=1.0, alpha=0.1, c=5.0),
    "zi_poisson": FamilySpec("poisson", zero_inflation=0.1),
    "zi_negbin": FamilySpec("negbin", phi=1.0, alpha=0.2, zero_inflation=0.05),
    "zi_normal": FamilySpec("normal", phi=1.0, alpha=0.05, c=0.0, lower=0, zero_inflation=0.05),
    "zi_gamma": FamilySpec("gamma", phi=2.0, alpha=0.2, zero_inflation=0.08),
    "negbin_scaled": FamilySpec("negbin", phi=1.0, alpha=0.03, scale=2.2),
    "negbin_scaled_floor": FamilySpec("negbin", phi=1.0, alpha=0.0, scale=2.2),
    "negbin_scaled_wide": FamilySpec("negbin", phi=2.5, alpha=0.05, c=1.0, scale=3.0),
    "zi_negbin_scaled": FamilySpec("negbin", phi=1.0, alpha=0.03, scale=2.2, zero_inflation=0.05),
}
SPEC_IDS = sorted(SPECS)


def _documented_variance(spec: FamilySpec, mean: float) -> float:
    """The variance function exactly as the FamilySpec docstring states it."""
    m = abs(mean) if spec.family == "normal" else max(mean, 0.0)
    if spec.family == "poisson":
        return m
    var = spec.c * m / (m + 1.0) + spec.phi * m + spec.alpha * m * m
    if spec.family == "negbin":
        var = max(var, spec.scale * m)  # floored at a (scaled) Poisson
    return var


def _latent_sd(dist: Dist, k: int) -> float:
    """Recover a discretised normal's latent sd from two adjacent cdf values.

    ``cdf(k) = Phi((k + 1/2 - mu) / sd)``, so consecutive z-scores differ by ``1 / sd``.
    """
    return 1.0 / float(stats.norm.ppf(dist.cdf(k + 1)) - stats.norm.ppf(dist.cdf(k)))


def _brute_mean(dist: Dist, lo: int, hi: int) -> float:
    ks = np.arange(lo, hi + 1)
    return math.fsum(ks * np.array([dist.pmf(int(k)) for k in ks]))


class TestFamilySpecConstruction:
    def test_defaults(self) -> None:
        spec = FamilySpec()
        assert (spec.family, spec.phi, spec.alpha, spec.c, spec.lower, spec.zero_inflation, spec.scale) == (
            "negbin", 1.0, 0.0, 0.0, 0, 0.0, 1.0,
        )
        assert FAMILIES == ("poisson", "negbin", "normal", "gamma")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"family": "lognormal"},
            {"family": ""},
            {"family": "Poisson"},
            {"phi": -0.1},
            {"alpha": -0.1},
            {"c": -1.0},
            {"phi": NAN},
            {"alpha": NAN},
            {"c": NAN},
            {"phi": INF},
            {"alpha": INF},
            {"c": INF},
            {"zero_inflation": 1.0},
            {"zero_inflation": -0.1},
            {"zero_inflation": 1.5},
            {"zero_inflation": NAN},
            {"scale": 0.99},
            {"scale": 0.0},
            {"scale": -2.0},
            {"scale": NAN},
            {"scale": INF},
            {"family": "poisson", "scale": 2.0},
            {"family": "normal", "scale": 2.0},
            {"family": "gamma", "scale": 1.5},
        ],
    )
    def test_invalid_specs_raise(self, kwargs: Dict[str, object]) -> None:
        with pytest.raises(ValueError):
            FamilySpec(**kwargs)  # type: ignore[arg-type]

    def test_is_frozen_and_hashable(self) -> None:
        spec = FamilySpec("gamma", phi=2.0)
        with pytest.raises(FrozenInstanceError):
            spec.phi = 3.0  # type: ignore[misc]
        assert spec == FamilySpec("gamma", phi=2.0)
        assert len({spec, FamilySpec("gamma", phi=2.0), FamilySpec("gamma", phi=2.5)}) == 2

    # -- variance function ---------------------------------------------------
    @pytest.mark.parametrize("mean", [0.0, 0.2, 1.0, 7.5, 60.0, 250.0])
    def test_variance_function_is_c_m_over_m_plus_1_plus_phi_m_plus_alpha_m2(self, mean: float) -> None:
        for family in ("normal", "gamma"):
            spec = FamilySpec(family, phi=1.3, alpha=0.07, c=2.5)
            expected = 2.5 * mean / (mean + 1.0) + 1.3 * mean + 0.07 * mean * mean
            assert spec.variance(mean) == pytest.approx(expected, rel=1e-14, abs=1e-15)
        nb2 = FamilySpec("negbin", phi=1.0, alpha=0.2)
        assert nb2.variance(mean) == pytest.approx(mean + 0.2 * mean * mean, rel=1e-14, abs=1e-15)
        assert FamilySpec("poisson", phi=3.0, alpha=2.0, c=9.0).variance(mean) == mean

    def test_intercept_fades_out_below_a_mean_of_one(self) -> None:
        """Documented: the spread shrinks to nothing as the mean does, and c is the limit."""
        spec = FamilySpec("gamma", phi=0.0, alpha=0.0, c=9.0)
        assert spec.variance(0.0) == 0.0
        assert spec.variance(1.0) == pytest.approx(4.5, rel=1e-15)      # half the intercept at mean 1
        assert spec.variance(1e-6) == pytest.approx(9e-6, rel=1e-5)
        assert spec.variance(1e6) == pytest.approx(9.0, rel=1e-5)
        values = [spec.variance(m) for m in np.geomspace(1e-4, 1e4, 60)]
        assert all(a < b for a, b in zip(values, values[1:]))

    @pytest.mark.parametrize("mean", [0.3, 4.0, 40.0])
    def test_negbin_variance_is_floored_at_the_mean(self, mean: float) -> None:
        assert FamilySpec("negbin", phi=0.5).variance(mean) == mean
        assert FamilySpec("negbin", phi=0.0, alpha=0.0).variance(mean) == mean
        assert FamilySpec("negbin", phi=1.0, alpha=0.0, scale=2.5).variance(mean) == pytest.approx(2.5 * mean, rel=1e-15)
        assert FamilySpec("gamma", phi=0.5).variance(mean) == pytest.approx(0.5 * mean, rel=1e-15)  # no floor

    def test_normal_variance_uses_the_absolute_mean(self) -> None:
        spec = FamilySpec("normal", phi=1.0, alpha=0.1, c=4.0, lower=None)
        for mean in (0.5, 3.0, 25.0):
            assert spec.variance(-mean) == spec.variance(mean) > 0.0

    @pytest.mark.parametrize("name", sorted(SPECS))
    def test_variance_matches_the_documented_formula_for_every_catalogued_spec(self, name: str) -> None:
        spec = SPECS[name]
        for mean in (0.0, 0.05, 0.7, 4.0, 31.0, 220.0):
            assert spec.variance(mean) == pytest.approx(_documented_variance(spec, mean), rel=1e-14, abs=1e-15)

    # -- members of each family ----------------------------------------------
    @pytest.mark.parametrize("mean", [0.2, 3.0, 24.5, 180.0])
    def test_poisson_family(self, mean: float) -> None:
        dist = FamilySpec("poisson", phi=3.0, alpha=2.0, c=9.0).dist(mean)
        assert isinstance(dist, PoissonDist)
        assert (dist.mean, dist.var) == (mean, mean)

    @pytest.mark.parametrize("mean", [0.2, 3.0, 24.5, 180.0])
    def test_negbin_family(self, mean: float) -> None:
        spec = FamilySpec("negbin", phi=1.0, alpha=0.15)
        dist = spec.dist(mean)
        assert isinstance(dist, NegBinDist)
        assert dist.mean == mean
        assert dist.var == pytest.approx(mean + 0.15 * mean * mean, rel=1e-15)
        assert dist.r == pytest.approx(1.0 / 0.15, rel=1e-12)  # NB2: r = 1 / alpha
        offset = FamilySpec("negbin", phi=1.0, alpha=0.0, c=2.0).dist(mean)
        assert isinstance(offset, NegBinDist)
        assert offset.var == pytest.approx(mean + 2.0 * mean / (mean + 1.0), rel=1e-15)

    @pytest.mark.parametrize("mean", [0.2, 3.0, 24.5, 180.0])
    def test_scaled_negbin_family(self, mean: float) -> None:
        spec = FamilySpec("negbin", phi=1.0, alpha=0.03, scale=2.2)
        dist = spec.dist(mean)
        assert isinstance(dist, ScaledNegBinDist)
        assert (dist.mean, dist.scale) == (mean, 2.2)
        expected_var = max(mean + 0.03 * mean * mean, 2.2 * mean)  # floored at scale * mean
        assert dist.var == pytest.approx(expected_var, rel=1e-12)
        assert spec.variance(mean) == pytest.approx(expected_var, rel=1e-12)

    def test_scale_one_keeps_the_plain_count_distributions(self) -> None:
        assert isinstance(FamilySpec("negbin", alpha=0.1, scale=1.0).dist(8.0), NegBinDist)
        assert isinstance(FamilySpec("negbin", scale=1.0).dist(8.0), PoissonDist)

    def test_scaled_family_without_overdispersion_is_a_scaled_poisson(self) -> None:
        dist = FamilySpec("negbin", phi=1.0, alpha=0.0, scale=3.0).dist(21.0)
        assert isinstance(dist, ScaledNegBinDist)
        assert dist.var == pytest.approx(63.0, rel=1e-12)
        for n in range(0, 30):
            assert dist.cdf(3 * n + 1) == pytest.approx(float(stats.poisson.cdf(n, 7.0)), abs=1e-12)

    @pytest.mark.parametrize(
        "spec",
        [
            FamilySpec(),                                  # var == mean exactly
            FamilySpec("negbin", phi=0.6),                 # under-dispersed
            FamilySpec("negbin", phi=0.0, alpha=0.0),      # zero variance function
            FamilySpec("negbin", phi=1.0, alpha=1e-13),    # numerically indistinguishable
        ],
    )
    @pytest.mark.parametrize("mean", [0.2, 3.0, 24.5, 180.0])
    def test_negbin_falls_back_to_poisson_without_overdispersion(self, spec: FamilySpec, mean: float) -> None:
        dist = spec.dist(mean)
        assert isinstance(dist, PoissonDist)
        assert (dist.mean, dist.var) == (mean, mean)
        assert spec.variance(mean) == pytest.approx(dist.var, rel=1e-9)

    def test_negbin_fallback_is_continuous(self) -> None:
        """Probabilities must not jump where the family switches to Poisson."""
        poisson = FamilySpec("poisson").dist(12.0)
        for alpha in (1e-4, 1e-6):
            nearly = FamilySpec("negbin", phi=1.0, alpha=alpha).dist(12.0)
            assert isinstance(nearly, NegBinDist)
            for line in (8.5, 12, 15.5):
                # first-order effect of alpha at mean 12 is below 3 * alpha here
                assert nearly.prob_over(line) == pytest.approx(poisson.prob_over(line), abs=3 * alpha)
                assert nearly.prob_push(12) == pytest.approx(poisson.prob_push(12), abs=3 * alpha)

    @pytest.mark.parametrize("alpha", [1e-10, 5e-9, 2e-8, 1e-7])
    def test_no_jump_at_the_poisson_switch_for_negligible_alpha(self, alpha: float) -> None:
        """Either side of the switch must agree to ~alpha * mean^2 in every probability."""
        spec, poisson = FamilySpec("negbin", phi=1.0, alpha=alpha), PoissonDist(25.0)
        dist = spec.dist(25.0)
        for line in (20.5, 25.0, 31.5):
            assert dist.probs(line) == pytest.approx(poisson.probs(line), abs=1e-6)
        over, push, under = spec.translate(25.0, 0.55, 25.0)
        assert over / (over + under) == pytest.approx(0.55, abs=1e-9)

    @pytest.mark.parametrize("mean", [-12.0, -0.5, 0.5, 3.0, 67.3])
    def test_unclamped_normal_family(self, mean: float) -> None:
        spec = FamilySpec("normal", phi=1.5, alpha=0.04, c=3.0, lower=None)
        dist = spec.dist(mean)
        assert isinstance(dist, DiscretizedNormal)
        assert dist.lower is None
        assert dist.mean == mean
        m = abs(mean)
        expected_var = 3.0 * m / (m + 1.0) + 1.5 * m + 0.04 * m * m
        assert dist.var == pytest.approx(expected_var, rel=1e-12)
        assert _latent_sd(dist, round(mean)) == pytest.approx(math.sqrt(expected_var), rel=1e-6)

    @pytest.mark.parametrize("lower", [0, 3, -5])
    @pytest.mark.parametrize("offset", [2.0, 6.0, 40.0])
    def test_clamped_normal_family(self, lower: int, offset: float) -> None:
        """Clamped members keep the latent sd of the variance function and hit the mean."""
        spec = FamilySpec("normal", phi=1.5, alpha=0.04, c=3.0, lower=lower)
        mean = lower + offset
        dist = spec.dist(mean)
        assert isinstance(dist, DiscretizedNormal)
        assert dist.lower == lower
        assert dist.cdf(lower - 1) == 0.0
        assert _latent_sd(dist, round(mean)) == pytest.approx(math.sqrt(spec.variance(mean)), rel=1e-6)
        assert dist.mean == pytest.approx(mean, abs=0.02)
        assert _brute_mean(dist, lower - 2, lower + 600) == pytest.approx(mean, abs=0.02)

    @pytest.mark.parametrize("mean", [0.0, -3.0, -1e6])
    def test_clamped_normal_below_its_clamp_is_a_point_mass_on_the_clamp(self, mean: float) -> None:
        dist = FamilySpec("normal", phi=1.0, alpha=0.05, c=4.0, lower=0).dist(mean)
        assert dist.pmf(0) == pytest.approx(1.0, abs=1e-6)
        assert dist.prob_over(0.5) == pytest.approx(0.0, abs=1e-6)
        assert dist.mean == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("mean", [0.2, 3.0, 24.5, 180.0])
    def test_gamma_family(self, mean: float) -> None:
        spec = FamilySpec("gamma", phi=2.0, alpha=0.2, c=1.0)
        dist = spec.dist(mean)
        assert isinstance(dist, DiscretizedGamma)
        assert dist.mean == mean
        assert dist.var == pytest.approx(mean / (mean + 1.0) + 2.0 * mean + 0.2 * mean * mean, rel=1e-15)

    @pytest.mark.parametrize(
        "name",
        [n for n in SPEC_IDS if not SPECS[n].zero_inflation and not (SPECS[n].family == "normal" and SPECS[n].lower is not None)],
    )
    def test_dist_var_matches_the_variance_function(self, name: str) -> None:
        """Holds for every spec without zero inflation or a clamp (incl. the Poisson floor)."""
        spec = SPECS[name]
        for mean in (0.7, 4.0, 31.0, 220.0):
            assert spec.dist(mean).var == pytest.approx(spec.variance(mean), rel=1e-12)

    @pytest.mark.parametrize(
        "spec, mean",
        [
            (FamilySpec("negbin", phi=0.5), 10.0),                          # Poisson floor
            (FamilySpec("normal", phi=1.0, c=4.0, lower=None), -5.0),       # negative mean
            (FamilySpec("negbin", phi=1.0, alpha=0.0, scale=2.0), 10.0),    # scaled Poisson floor
        ],
    )
    def test_variance_agrees_with_the_distribution_it_describes(self, spec: FamilySpec, mean: float) -> None:
        assert spec.variance(mean) > 0.0
        assert spec.variance(mean) == pytest.approx(spec.dist(mean).var, rel=1e-12)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "BUG (minor, documentation): FamilySpec.variance() is documented as the "
            "'variance of the base distribution built for mean', but for a clamped normal "
            "it is the latent variance: FamilySpec('normal', c=25).variance(2.0) = 16.7 "
            "while dist(2.0).var = 6.95 (the clamp at 0 removes most of the spread)"
        ),
    )
    def test_variance_agrees_with_a_clamped_normal(self) -> None:
        spec = FamilySpec("normal", phi=0.0, alpha=0.0, c=25.0)
        assert spec.variance(2.0) == pytest.approx(spec.dist(2.0).var, rel=0.05)

    # -- zero inflation and degenerate means ---------------------------------
    @pytest.mark.parametrize("family", FAMILIES)
    @pytest.mark.parametrize("pi", [0.05, 0.3, 0.9])
    def test_zero_inflation_preserves_the_requested_mean(self, family: str, pi: float) -> None:
        spec = FamilySpec(family, phi=1.2, alpha=0.1, c=1.0, lower=None, zero_inflation=pi)
        bare = FamilySpec(family, phi=1.2, alpha=0.1, c=1.0, lower=None)
        for mean in (0.4, 2.0, 11.0, 95.0):
            dist = spec.dist(mean)
            assert isinstance(dist, ZeroInflated)
            assert dist.pi == pi
            assert dist.mean == pytest.approx(mean, rel=1e-12)
            assert dist.base.mean == pytest.approx(mean / (1.0 - pi), rel=1e-12)
            assert dist.base.var == pytest.approx(bare.dist(mean / (1.0 - pi)).var, rel=1e-12)
            assert dist.pmf(0) >= pi

    @pytest.mark.parametrize("family", ["poisson", "negbin"])
    def test_zero_inflated_count_mean_by_brute_force(self, family: str) -> None:
        spec = FamilySpec(family, phi=1.0, alpha=0.2, zero_inflation=0.25)
        dist = spec.dist(6.0)
        ks = np.arange(0, 600)
        pmf = np.array([dist.pmf(int(k)) for k in ks])
        assert math.fsum(pmf) == pytest.approx(1.0, abs=1e-12)
        assert math.fsum(ks * pmf) == pytest.approx(6.0, rel=1e-10)

    def test_no_zero_inflation_returns_the_bare_base(self) -> None:
        for family in FAMILIES:
            assert not isinstance(FamilySpec(family, alpha=0.1, c=1.0).dist(5.0), ZeroInflated)

    @pytest.mark.parametrize("family", FAMILIES)
    @pytest.mark.parametrize("mean", [NAN, INF, -INF])
    def test_non_finite_mean_raises(self, family: str, mean: float) -> None:
        with pytest.raises(ValueError):
            FamilySpec(family, c=1.0).dist(mean)

    @pytest.mark.parametrize("family", ["poisson", "negbin", "gamma"])
    def test_non_positive_mean_degenerates_to_a_point_mass_at_zero(self, family: str) -> None:
        for mean in (0.0, -3.0):
            dist = FamilySpec(family, alpha=0.1, c=2.0).dist(mean)
            assert dist.pmf(0) == pytest.approx(1.0, abs=1e-8)
            assert dist.prob_over(0.5) == pytest.approx(0.0, abs=1e-8)

    # -- "the member of the family whose overall mean is `mean`" ---------------
    @pytest.mark.parametrize("name", [n for n in SPEC_IDS if SPECS[n].family in ("poisson", "negbin") and SPECS[n].scale == 1.0])
    @pytest.mark.parametrize("mean", [0.3, 2.0, 9.0, 40.0])
    def test_count_families_have_exactly_the_requested_mean(self, name: str, mean: float) -> None:
        dist = SPECS[name].dist(mean)
        assert dist.mean == pytest.approx(mean, rel=1e-12)
        assert _brute_mean(dist, 0, 4000) == pytest.approx(mean, rel=1e-8)

    @pytest.mark.parametrize("name", ["normal_clamped", "zi_normal", "gamma", "gamma_cv", "gamma_offset", "negbin_scaled"])
    @pytest.mark.parametrize("mean", [5.0, 20.0, 80.0])
    def test_continuous_shaped_families_have_the_requested_mean_above_a_few_units(self, name: str, mean: float) -> None:
        dist = SPECS[name].dist(mean)
        assert _brute_mean(dist, 0, 3000) == pytest.approx(mean, rel=2e-3)

    def test_default_normal_family_accounts_for_its_clamp(self) -> None:
        """A latent N(2, sd) clamped at 0 would have mean 3.15; the family corrects for that."""
        dist = FamilySpec("normal", phi=0.0, alpha=0.0, c=25.0).dist(2.0)
        assert _brute_mean(dist, -5, 80) == pytest.approx(2.0, abs=0.01)
        assert dist.mean == pytest.approx(2.0, abs=0.01)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "BUG (minor): dist(mean) is documented as 'the member of the family whose "
            "overall mean is mean', but for means below ~1 the clamped normal and the gamma "
            "families miss it because the latent mean is chosen from a continuous "
            "approximation that ignores rounding: FamilySpec('normal', phi=1).dist(0.5) has "
            "true mean 0.549 (+10%), dist(0.2) has 0.262 (+31%); "
            "FamilySpec('gamma', phi=1).dist(0.5) has 0.438 (-12%), dist(0.2) has 0.156 (-22%)"
        ),
    )
    @pytest.mark.parametrize("family", ["normal", "gamma"])
    @pytest.mark.parametrize("mean", [0.2, 0.5])
    def test_small_means_are_reproduced_too(self, family: str, mean: float) -> None:
        dist = FamilySpec(family, phi=1.0, alpha=0.0, c=0.0).dist(mean)
        assert _brute_mean(dist, 0, 200) == pytest.approx(mean, rel=0.03)


# --------------------------------------------------------------------------
# FamilySpec: solve_mean / translate
# --------------------------------------------------------------------------
HALF_LINES = (0.5, 1.5, 2.5, 5.5, 10.5, 24.5, 49.5, 99.5, 150.5, 300.5)
WHOLE_LINES = (1.0, 2.0, 3.0, 7.0, 25.0, 100.0, 300.0)
PROBS = (0.02, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.98)


def _over(spec: FamilySpec, mean: float, line: float, exclude_push: bool) -> float:
    dist = spec.dist(mean)
    return dist.prob_over_no_push(line) if exclude_push else dist.prob_over(line)


@pytest.mark.parametrize("name", SPEC_IDS)
class TestSolveMeanRoundTrip:
    @pytest.mark.parametrize("exclude_push", [True, False])
    def test_round_trip_over_lines_and_probabilities(self, name: str, exclude_push: bool) -> None:
        spec = SPECS[name]
        ceiling = 1.0 - spec.zero_inflation  # a zero-inflated over can never exceed 1 - pi
        for line in HALF_LINES + WHOLE_LINES:
            for p in PROBS:
                if p >= ceiling - 1e-9:
                    continue
                mean = spec.solve_mean(line, p, exclude_push=exclude_push)
                assert math.isfinite(mean)
                assert _over(spec, mean, line, exclude_push) == pytest.approx(p, abs=1e-8)

    def test_half_lines_ignore_exclude_push(self, name: str) -> None:
        spec = SPECS[name]
        for line in (0.5, 5.5, 49.5):
            for p in (0.1, 0.5, 0.8):
                assert spec.solve_mean(line, p, exclude_push=True) == pytest.approx(
                    spec.solve_mean(line, p, exclude_push=False), rel=1e-9
                )

    def test_whole_lines_need_a_higher_mean_without_push_exclusion(self, name: str) -> None:
        """P(over) < P(over | no push), so matching the unconditional needs more mean."""
        spec = SPECS[name]
        for line in (3.0, 25.0):
            for p in (0.3, 0.5, 0.7):
                assert spec.solve_mean(line, p, exclude_push=False) > spec.solve_mean(line, p, exclude_push=True)

    def test_default_excludes_pushes(self, name: str) -> None:
        spec = SPECS[name]
        assert spec.solve_mean(7.0, 0.45) == spec.solve_mean(7.0, 0.45, exclude_push=True)

    def test_mean_is_increasing_in_probability_and_line(self, name: str) -> None:
        spec = SPECS[name]
        probs = [p for p in PROBS if p < 1.0 - spec.zero_inflation - 1e-9]
        for line in (2.5, 25.0, 99.5):
            means = [spec.solve_mean(line, p) for p in probs]
            assert all(a < b for a, b in zip(means, means[1:]))
        for p in (0.25, 0.6):
            means = [spec.solve_mean(line, p) for line in HALF_LINES]
            assert all(a < b for a, b in zip(means, means[1:]))

    def test_prob_over_is_monotone_in_a_positive_mean(self, name: str) -> None:
        """Stochastic ordering in the mean is what makes solve_mean well posed.

        Checked wherever the over probability is at least 0.1%: below that,
        count families with an intercept (``c > 0``) have dips of relative
        size ~1e-2 at levels under 1e-4, which no price can express.
        """
        spec = SPECS[name]
        for line in (0.5, 3.0, 3.5, 24.5, 25.0, 100.5):
            means = np.linspace(0.01, 3.0 * line + 20.0, 400)
            for exclude_push in (True, False):
                values = np.array([_over(spec, float(m), line, exclude_push) for m in means])
                values = values[int(np.argmax(values >= 1e-3)):]
                assert np.all(np.diff(values) >= -1e-12)
                assert values[-1] > values[0]


UNCLAMPED_NORMALS = {
    "intercept": FamilySpec("normal", phi=0.0, alpha=0.0, c=64.0, lower=None),
    "intercept_and_cv": FamilySpec("normal", phi=0.0, alpha=0.05, c=30.0, lower=None),
    "poisson_like": FamilySpec("normal", phi=2.0, alpha=0.1, c=0.0, lower=None),
}


class TestUnclampedNormalAroundZero:
    """Signed stats (``lower=None``): the family must stay ordered through mean 0."""

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "BUG: for the unclamped normal family the variance function is applied to "
            "abs(mean) and vanishes at 0, so every member near mean 0 is a spike and "
            "P(X > line) is not monotone in the mean. FamilySpec('normal', c=64, lower=None): "
            "P(X > 2.5) is 0.268 at mean -1, 0.000 at mean 0 and 0.300 at mean 0.35. A signed "
            "stat with a mean near zero and a real spread cannot be represented"
        ),
    )
    @pytest.mark.parametrize("name", sorted(UNCLAMPED_NORMALS))
    @pytest.mark.parametrize("line", [0.5, 2.5])
    def test_prob_over_is_monotone_in_the_mean_across_zero(self, name: str, line: float) -> None:
        spec = UNCLAMPED_NORMALS[name]
        values = [spec.dist(float(m)).prob_over(line) for m in np.linspace(-15.0, 15.0, 601)]
        assert all(b >= a - 1e-12 for a, b in zip(values, values[1:]))

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "BUG: consequence of the above - the mean reproducing a price is not unique. "
            "With FamilySpec('normal', c=64, lower=None) three means (-1.85, -0.41 and +0.22) "
            "all give P(X > 2.5) = 0.25, yet imply P(X > 10.5) of 0.0277, 0.0057 and 0.0012; "
            "solve_mean/translate return whichever root the bracketing happens to find"
        ),
    )
    @pytest.mark.parametrize("name", sorted(UNCLAMPED_NORMALS))
    def test_the_mean_reproducing_a_price_is_unique(self, name: str) -> None:
        spec = UNCLAMPED_NORMALS[name]
        target = 0.6 * max(spec.dist(-m).prob_over(2.5) for m in (0.5, 1.0, 2.0))
        gaps = np.array([spec.dist(float(m)).prob_over(2.5) - target for m in np.linspace(-40.0, 40.0, 16001)])
        crossings = int(np.sum(np.sign(gaps[1:]) != np.sign(gaps[:-1])))
        assert crossings == 1

    @pytest.mark.parametrize("name", sorted(UNCLAMPED_NORMALS))
    def test_round_trip_still_holds_on_whichever_branch_is_found(self, name: str) -> None:
        spec = UNCLAMPED_NORMALS[name]
        for line in (-5.5, -0.5, 0.5, 2.5, 5.5):
            for p in (0.05, 0.2, 0.45, 0.6, 0.9):
                mean = spec.solve_mean(line, p)
                assert spec.dist(mean).prob_over(line) == pytest.approx(p, abs=1e-8)


class TestSolveMeanAgainstIndependentSolutions:
    @pytest.mark.parametrize("p", PROBS)
    def test_poisson_line_one_half_in_closed_form(self, p: float) -> None:
        # P(X >= 1) = 1 - exp(-m) = p
        assert FamilySpec("poisson").solve_mean(0.5, p) == pytest.approx(-math.log1p(-p), rel=1e-8)
        assert FamilySpec().solve_mean(0.5, p) == pytest.approx(-math.log1p(-p), rel=1e-8)

    @pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
    def test_poisson_whole_line_zero_without_push_exclusion(self, p: float) -> None:
        """P(X > 0) = p is the same event as the 0.5 line."""
        assert FamilySpec("poisson").solve_mean(0.0, p, exclude_push=False) == pytest.approx(
            -math.log1p(-p), rel=1e-8
        )

    @pytest.mark.parametrize("line", [1.5, 5.5, 24.5, 150.5])
    @pytest.mark.parametrize("p", [0.05, 0.4, 0.5, 0.9])
    def test_poisson_half_lines_against_scipy(self, line: float, p: float) -> None:
        k = math.floor(line)
        expected = brentq(lambda m: float(stats.poisson.sf(k, m)) - p, 1e-6, 5 * line + 50, xtol=1e-12)
        assert FamilySpec("poisson").solve_mean(line, p) == pytest.approx(expected, rel=1e-8)

    @pytest.mark.parametrize("line", [2.0, 7.0, 25.0])
    @pytest.mark.parametrize("p", [0.2, 0.5, 0.8])
    def test_poisson_whole_lines_against_scipy(self, line: float, p: float) -> None:
        k = int(line)

        def conditional(m: float) -> float:
            return float(stats.poisson.sf(k, m) / (1.0 - stats.poisson.pmf(k, m)))

        expected = brentq(lambda m: conditional(m) - p, 1e-6, 5 * line + 50, xtol=1e-12)
        assert FamilySpec("poisson").solve_mean(line, p) == pytest.approx(expected, rel=1e-8)
        unconditional = brentq(lambda m: float(stats.poisson.sf(k, m)) - p, 1e-6, 5 * line + 50, xtol=1e-12)
        assert FamilySpec("poisson").solve_mean(line, p, exclude_push=False) == pytest.approx(
            unconditional, rel=1e-8
        )

    @pytest.mark.parametrize("alpha", [0.05, 0.15, 1.0])
    @pytest.mark.parametrize("line", [0.5, 5.5, 24.5, 99.5])
    @pytest.mark.parametrize("p", [0.1, 0.5, 0.85])
    def test_nb2_half_lines_against_scipy(self, alpha: float, line: float, p: float) -> None:
        """NB2 (var = m + alpha m^2) is nbinom(n = 1/alpha, p = n / (n + m))."""
        n, k = 1.0 / alpha, math.floor(line)
        expected = brentq(
            lambda m: float(stats.nbinom.sf(k, n, n / (n + m))) - p, 1e-6, 1e5, xtol=1e-12
        )
        spec = FamilySpec("negbin", phi=1.0, alpha=alpha)
        assert spec.solve_mean(line, p) == pytest.approx(expected, rel=1e-7)

    @pytest.mark.parametrize("name", sorted(UNCLAMPED_NORMALS))
    @pytest.mark.parametrize("line", [0.5, 24.5, 80.5])
    @pytest.mark.parametrize("p", [0.5, 0.75, 0.95])
    def test_unclamped_normal_against_an_independent_root(self, name: str, line: float, p: float) -> None:
        """P(X > n + 1/2) = Phi((m - line) / sd(m)) with the documented sd(m).

        For a positive line and p >= 1/2 the root lies at m >= line, where the
        function is increasing, so it is unique.
        """
        spec = UNCLAMPED_NORMALS[name]

        def over(m: float) -> float:
            return float(stats.norm.cdf((m - line) / math.sqrt(_documented_variance(spec, m))))

        expected = brentq(lambda m: over(m) - p, line - 1e-9, line + 2000.0, xtol=1e-12)
        assert spec.solve_mean(line, p) == pytest.approx(expected, abs=1e-6)
        # whole line n = floor(line), pushes not excluded: P(X > n) = Phi((m - n - 1/2) / sd(m)),
        # which is the same event as the half line above it
        assert spec.solve_mean(float(math.floor(line)), p, exclude_push=False) == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize("name", sorted(UNCLAMPED_NORMALS))
    def test_symmetric_normal_at_one_half_sits_on_the_line(self, name: str) -> None:
        """z = 0 whatever the sd; by symmetry a whole line behaves the same."""
        spec = UNCLAMPED_NORMALS[name]
        for line in (-7.5, 0.5, 30.5, 12.0, -4.0):
            assert spec.solve_mean(line, 0.5) == pytest.approx(line, abs=1e-6)

    @pytest.mark.parametrize("alpha", [0.1, 0.3])
    @pytest.mark.parametrize("line", [10.5, 60.5, 250.5])
    @pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
    def test_constant_cv_gamma_in_closed_form(self, alpha: float, line: float, p: float) -> None:
        """var = alpha m^2: shape 1/alpha, scale alpha*m; P(X > n) = Q(shape, (n+1/2)/scale)."""
        spec = FamilySpec("gamma", phi=0.0, alpha=alpha)
        x = float(stats.gamma(a=1.0 / alpha).isf(p))  # Q(shape, x) = p
        expected = line / (alpha * x)  # floor(line) + 1/2 == line for half lines
        assert spec.solve_mean(line, p) == pytest.approx(expected, rel=1e-8)

    @pytest.mark.parametrize("c", [25.0, 100.0])
    @pytest.mark.parametrize("line, p", [(0.5, 0.3), (5.5, 0.2), (10.5, 0.1), (24.5, 0.6)])
    def test_clamped_normal_solution_has_the_documented_spread_and_mean(self, c: float, line: float, p: float) -> None:
        """The solved member is a clamped normal with sd(mean) whose overall mean is `mean`."""
        spec = FamilySpec("normal", phi=0.0, alpha=0.0, c=c, lower=0)
        mean = spec.solve_mean(line, p)
        dist = spec.dist(mean)
        assert _latent_sd(dist, math.floor(line)) == pytest.approx(math.sqrt(c * mean / (mean + 1.0)), rel=1e-6)
        assert _brute_mean(dist, 0, 400) == pytest.approx(mean, abs=0.02)
        assert dist.prob_over(line) == pytest.approx(p, abs=1e-8)


class TestSolveMeanEdges:
    @pytest.mark.parametrize("name", SPEC_IDS)
    @pytest.mark.parametrize("p", [0.0, 1.0, -0.2, 1.2, NAN])
    def test_probability_outside_open_interval_raises(self, name: str, p: float) -> None:
        with pytest.raises(ValueError):
            SPECS[name].solve_mean(5.5, p)

    @pytest.mark.parametrize("name", ["poisson", "normal_free", "gamma", "zi_negbin"])
    @pytest.mark.parametrize("line", [NAN, INF, -INF])
    def test_non_finite_line_raises(self, name: str, line: float) -> None:
        with pytest.raises(ValueError):
            SPECS[name].solve_mean(line, 0.5)

    @pytest.mark.parametrize("name", [n for n in SPEC_IDS if SPECS[n].zero_inflation])
    def test_unreachable_probability_under_zero_inflation_raises(self, name: str) -> None:
        """P(over) <= 1 - zero_inflation: asking for more must not return a number."""
        spec = SPECS[name]
        target = 1.0 - spec.zero_inflation + 0.02
        for line in (0.5, 5.5, 3.0):
            with pytest.raises(ValueError):
                spec.solve_mean(line, target)
            with pytest.raises(ValueError):
                spec.solve_mean(line, target, exclude_push=False)

    @pytest.mark.parametrize("name", ["poisson", "negbin_nb2", "gamma", "normal_clamped", "negbin_scaled"])
    def test_lines_no_member_can_price_raise(self, name: str) -> None:
        """Below the support every member has P(over) = 1; a silent answer would be wrong."""
        spec = SPECS[name]
        for line in (-0.5, -3.0):
            with pytest.raises(ValueError):
                spec.solve_mean(line, 0.5)
            with pytest.raises(ValueError):
                spec.solve_mean(line, 0.5, exclude_push=False)
        with pytest.raises(ValueError):
            spec.solve_mean(0.0, 0.5)  # P(over | no push) = 1 at the bottom of the support

    def test_line_below_a_raised_clamp_raises(self) -> None:
        spec = FamilySpec("normal", phi=1.0, alpha=0.02, c=4.0, lower=3)
        with pytest.raises(ValueError):
            spec.solve_mean(2.5, 0.5)
        mean = spec.solve_mean(3.5, 0.5)
        assert mean > 3.0
        assert spec.dist(mean).prob_over(3.5) == pytest.approx(0.5, abs=1e-8)

    @pytest.mark.parametrize("name", ["poisson", "negbin_nb2", "gamma"])
    def test_probability_below_the_smallest_representable_mean_raises(self, name: str) -> None:
        with pytest.raises(ValueError):
            SPECS[name].solve_mean(0.5, 1e-12)

    @pytest.mark.parametrize("name", ["normal_free", "normal_free_cv"])
    def test_probability_beyond_the_family_limit_raises(self, name: str) -> None:
        """With a constant coefficient of variation P(over) tends to Phi(1/cv) < 1."""
        with pytest.raises(ValueError):
            SPECS[name].solve_mean(5.5, 1.0 - 1e-9)

    @pytest.mark.parametrize("name", ["poisson", "negbin_nb2", "gamma", "negbin_scaled"])
    @pytest.mark.parametrize("line", [24.5, 300.5, 5000.5])
    def test_probabilities_near_zero_and_one(self, name: str, line: float) -> None:
        """Tail targets of 1e-6: the reproduced tail must be right to 0.1%."""
        spec = SPECS[name]
        for p in (1e-6, 1e-3, 1 - 1e-3, 1 - 1e-6):
            mean = spec.solve_mean(line, p)
            dist = spec.dist(mean)
            assert dist.prob_over(line) == pytest.approx(p, rel=1e-3 * min(p, 1.0 - p) / p, abs=0.0)
            assert dist.prob_under(line) == pytest.approx(1.0 - p, rel=1e-3 * min(p, 1.0 - p) / (1.0 - p), abs=0.0)

    @pytest.mark.parametrize("line", [24.5, 300.5, 5000.5])
    def test_heavy_tailed_negbin_near_zero_and_one(self, line: float) -> None:
        """alpha = 2 (r = 1/2): a 99.9% over already needs a mean ~1e6 times the line."""
        spec = SPECS["negbin_heavy"]
        for p in (1e-6, 1e-3, 1 - 1e-3):
            mean = spec.solve_mean(line, p)
            assert abs(spec.dist(mean).prob_over(line) - p) <= 1e-3 * min(p, 1.0 - p)

    @pytest.mark.parametrize("name", ["poisson", "negbin_nb2", "gamma", "normal_clamped", "negbin_scaled"])
    def test_very_large_lines(self, name: str) -> None:
        spec = SPECS[name]
        for line in (1000.5, 20000.5):
            for p in (0.05, 0.5, 0.95):
                mean = spec.solve_mean(line, p)
                assert spec.dist(mean).prob_over(line) == pytest.approx(p, abs=1e-7)

    def test_poisson_mean_tracks_a_large_line(self) -> None:
        """P(X <= n) = 1/2 when the mean is the median of Gamma(n + 1), about n + 2/3.

        With n = line - 1/2 that is line + 1/6.
        """
        mean = FamilySpec("poisson").solve_mean(20000.5, 0.5)
        assert mean == pytest.approx(20000.5 + 1.0 / 6.0, abs=1e-3)

    @pytest.mark.parametrize("name", ["poisson", "negbin_nb2", "gamma"])
    def test_tiny_probability_at_the_lowest_line(self, name: str) -> None:
        spec = SPECS[name]
        mean = spec.solve_mean(0.5, 1e-4)
        assert 0.0 < mean < 1e-3
        assert spec.dist(mean).prob_over(0.5) == pytest.approx(1e-4, rel=1e-4, abs=0.0)

    @pytest.mark.parametrize("line, p", [(0.5, 0.3), (0.5, 0.02), (5.5, 0.2), (10.5, 0.1), (10.5, 0.01)])
    def test_clamped_normal_reaches_low_probabilities_at_low_lines(self, line: float, p: float) -> None:
        """Needs a negative latent mean: the solver must not stop at a positive one."""
        spec = FamilySpec("normal", phi=0.0, alpha=0.0, c=100.0, lower=0)
        mean = spec.solve_mean(line, p)
        assert mean > 0.0
        assert spec.dist(mean).prob_over(line) == pytest.approx(p, abs=1e-8)

    @pytest.mark.parametrize("line, p", [(0.5, 0.05), (1.5, 0.05), (0.5, 0.02), (0.5, 0.001)])
    def test_unclamped_normal_with_large_sd_reaches_low_probabilities_at_small_lines(self, line: float, p: float) -> None:
        """The lower bracket has to scale with the spread, not with the line."""
        spec = FamilySpec("normal", phi=0.0, alpha=0.0, c=100.0, lower=None)
        mean = spec.solve_mean(line, p)
        assert spec.dist(mean).prob_over(line) == pytest.approx(p, abs=1e-8)

    @pytest.mark.parametrize("name", SPEC_IDS)
    def test_solved_mean_is_never_silently_wrong(self, name: str, rng: np.random.Generator) -> None:
        """Random lines and probabilities: either the price is reproduced or ValueError."""
        spec = SPECS[name]
        for _ in range(150):
            line = float(rng.choice([0.5, 1.0, 1.5, 2.0, 3.5, 6.5, 12.0, 18.5, 33.5, 61.0, 140.5]))
            p = float(rng.uniform(0.005, 0.995))
            exclude_push = bool(rng.integers(0, 2))
            try:
                mean = spec.solve_mean(line, p, exclude_push=exclude_push)
            except ValueError:
                assert p > 1.0 - spec.zero_inflation - 0.01  # only the documented ceiling may refuse
                continue
            assert _over(spec, mean, line, exclude_push) == pytest.approx(p, abs=1e-8)


@pytest.mark.parametrize("name", SPEC_IDS)
class TestTranslate:
    def test_same_half_line_returns_the_input_probability(self, name: str) -> None:
        spec = SPECS[name]
        for line, p in ((0.5, 0.3), (24.5, 0.55), (99.5, 0.8)):
            over, push, under = spec.translate(line, p, line)
            assert (over, push, under) == pytest.approx((p, 0.0, 1.0 - p), abs=1e-8)

    def test_same_whole_line_reproduces_the_no_push_probability(self, name: str) -> None:
        spec = SPECS[name]
        for line, p in ((3.0, 0.4), (25.0, 0.55)):
            over, push, under = spec.translate(line, p, line)
            assert over + push + under == pytest.approx(1.0, abs=1e-12)
            assert push > 0.0
            assert over / (over + under) == pytest.approx(p, abs=1e-8)

    def test_equals_the_solved_distribution_evaluated_at_the_new_line(self, name: str) -> None:
        spec = SPECS[name]
        for from_line, p, to_line in ((24.5, 0.55, 26.5), (24.5, 0.55, 25.0), (7.0, 0.4, 5.5), (2.5, 0.7, 0.5)):
            expected = spec.dist(spec.solve_mean(from_line, p, exclude_push=True)).probs(to_line)
            assert spec.translate(from_line, p, to_line) == pytest.approx(expected, abs=1e-12)

    def test_result_is_a_probability_triple(self, name: str) -> None:
        spec = SPECS[name]
        for to_line in (0.5, 12.0, 24.5, 25.0, 31.5, 80.0):
            over, push, under = spec.translate(24.5, 0.55, to_line)
            assert min(over, push, under) >= 0.0
            assert over + push + under == pytest.approx(1.0, abs=1e-12)
            assert (push > 0.0) == (to_line == math.floor(to_line))

    def test_moving_the_line_up_lowers_the_over(self, name: str) -> None:
        spec = SPECS[name]
        overs = [spec.translate(24.5, 0.55, to)[0] for to in (18.5, 22.5, 24.5, 25.0, 26.5, 30.5, 40.5)]
        assert all(a > b for a, b in zip(overs, overs[1:]))
        unders = [spec.translate(24.5, 0.55, to)[2] for to in (18.5, 22.5, 24.5, 26.5, 30.5, 40.5)]
        assert all(a < b for a, b in zip(unders, unders[1:]))

    def test_half_line_translation_is_reversible(self, name: str) -> None:
        spec = SPECS[name]
        there = spec.translate(24.5, 0.55, 27.5)[0]
        back = spec.translate(27.5, there, 24.5)[0]
        assert back == pytest.approx(0.55, abs=1e-7)


class TestTranslateClosedForm:
    def test_poisson_from_half_to_one_and_a_half(self) -> None:
        """P(X>=1) = p gives m = -ln(1-p); then P(X>=2) = 1 - (1-p)(1+m)."""
        for p in (0.2, 0.5, 0.9):
            m = -math.log1p(-p)
            over, push, under = FamilySpec("poisson").translate(0.5, p, 1.5)
            assert over == pytest.approx(1.0 - (1.0 - p) * (1.0 + m), abs=1e-9)
            assert push == 0.0
            assert under == pytest.approx((1.0 - p) * (1.0 + m), abs=1e-9)

    def test_poisson_to_a_whole_line(self) -> None:
        p = 0.6
        m = -math.log1p(-p)
        over, push, under = FamilySpec("poisson").translate(0.5, p, 1.0)
        assert push == pytest.approx(m * math.exp(-m), abs=1e-9)
        assert under == pytest.approx(math.exp(-m), abs=1e-9)
        assert over == pytest.approx(1.0 - math.exp(-m) * (1.0 + m), abs=1e-9)

    @pytest.mark.parametrize("name", sorted(UNCLAMPED_NORMALS))
    def test_unclamped_normal_moves_along_its_own_z_scale(self, name: str) -> None:
        """Solve the mean independently, then a line move of d shifts z by d / sd(mean)."""
        spec = UNCLAMPED_NORMALS[name]

        def over(m: float, line: float) -> float:
            return float(stats.norm.cdf((m - line) / math.sqrt(_documented_variance(spec, m))))

        mean = brentq(lambda m: over(m, 24.5) - 0.55, 24.5, 200.0, xtol=1e-12)
        sd = math.sqrt(_documented_variance(spec, mean))
        z = float(stats.norm.ppf(0.55))
        translated, push, _ = spec.translate(24.5, 0.55, 28.5)
        assert push == 0.0
        assert translated == pytest.approx(float(stats.norm.cdf(z - 4.0 / sd)), abs=1e-8)
        assert translated == pytest.approx(over(mean, 28.5), abs=1e-8)


# --------------------------------------------------------------------------
# FamilySpec: (de)serialisation
# --------------------------------------------------------------------------
class TestSerialisation:
    @pytest.mark.parametrize("name", SPEC_IDS)
    def test_round_trip(self, name: str) -> None:
        spec = SPECS[name]
        data = spec.to_dict()
        assert data == dataclasses.asdict(spec)  # every field is serialised, nothing else
        assert FamilySpec.from_dict(data) == spec
        assert FamilySpec.from_dict(json.loads(json.dumps(data))) == spec

    def test_to_dict_values(self) -> None:
        data = FamilySpec("normal", 1.2, 0.3, 4.0, None, 0.1).to_dict()
        assert {k: data[k] for k in ("family", "phi", "alpha", "c", "lower", "zero_inflation")} == {
            "family": "normal", "phi": 1.2, "alpha": 0.3, "c": 4.0, "lower": None, "zero_inflation": 0.1,
        }
        assert FamilySpec("negbin", alpha=0.03, scale=2.2).to_dict()["scale"] == 2.2

    def test_from_dict_fills_defaults_and_ignores_unknown_keys(self) -> None:
        assert FamilySpec.from_dict({}) == FamilySpec()
        assert FamilySpec.from_dict({"family": "gamma", "phi": 2, "fitted_on": "2025-01-01"}) == FamilySpec(
            "gamma", phi=2
        )

    def test_from_dict_validates(self) -> None:
        with pytest.raises(ValueError):
            FamilySpec.from_dict({"family": "weibull"})
        with pytest.raises(ValueError):
            FamilySpec.from_dict({"alpha": -1.0})
        with pytest.raises(ValueError):
            FamilySpec.from_dict({"family": "gamma", "scale": 2.0})

    def test_round_tripped_spec_behaves_identically(self) -> None:
        spec = SPECS["zi_negbin"]
        clone = FamilySpec.from_dict(json.loads(json.dumps(spec.to_dict())))
        assert clone.solve_mean(24.5, 0.55) == spec.solve_mean(24.5, 0.55)
        assert clone.translate(24.5, 0.55, 26.5) == spec.translate(24.5, 0.55, 26.5)


def test_spec_catalogue_covers_every_family_and_both_clamp_modes() -> None:
    """Guards the parametrised suites above against a silently shrinking catalogue."""
    assert {spec.family for spec in SPECS.values()} == set(FAMILIES)
    assert {spec.lower for spec in SPECS.values() if spec.family == "normal"} == {0, None}
    assert {spec.family for spec in SPECS.values() if spec.zero_inflation} == set(FAMILIES)
