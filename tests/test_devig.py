"""Independent verification of :mod:`evmodel.devig`.

Every method is checked three ways:

* against its *defining equation* (a single exponent / odds ratio / probit
  shift / Shin ``z`` recovered from the returned probabilities);
* against an independent root-finder written here;
* against 50-digit reference values produced with mpmath (constants pasted
  below, see the comment next to them).
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.stats import norm

from evmodel.devig import (
    METHODS,
    devig,
    devig_all,
    devig_decimal,
    devig_two_way,
    fair_prob_one_sided,
    hold,
    overround,
    worst_case,
)

NAN = float("nan")


def _implied(american: float) -> float:
    """Implied probability of an American price (independent of evmodel.odds)."""
    if american < 0:
        return -american / (-american + 100.0)
    return 100.0 / (american + 100.0)


#: Markets with a positive margin used by the property tests.
MARKETS: Dict[str, List[float]] = {
    "balanced_-110": [_implied(-110), _implied(-110)],
    "mild_-150_+125": [_implied(-150), _implied(125)],
    "dog_first_+150_-190": [_implied(150), _implied(-190)],
    "heavy_-2000_+1200": [_implied(-2000), _implied(1200)],
    "extreme_-100000_+5000": [_implied(-100000), _implied(5000)],
    "three_way": [1 / 2.1, 1 / 3.4, 1 / 3.6],
    "nine_way": [1 / 3.0, 1 / 4.5, 1 / 6, 1 / 8, 1 / 10, 1 / 15, 1 / 26, 1 / 51, 1 / 101],
    "longshots": [0.98, 0.03, 0.02, 0.001],
    "twenty_equal": [0.07] * 20,
    "big_margin": [0.6, 0.5, 0.4],
}

MARKET_IDS = sorted(MARKETS)


def _random_markets(rng: np.random.Generator, count: int) -> List[List[float]]:
    """Random markets with 2-16 outcomes and 0.1%-50% overround."""
    markets: List[List[float]] = []
    while len(markets) < count:
        n = int(rng.choice([2, 3, 5, 9, 16]))
        fair = rng.dirichlet(np.ones(n) * rng.uniform(0.3, 3.0))
        q = np.clip(fair * rng.uniform(1.001, 1.5), 1e-4, 0.995)
        if q.sum() > 1.0005:
            markets.append([float(x) for x in q])
    return markets


# --------------------------------------------------------------------------
# overround / hold
# --------------------------------------------------------------------------
class TestMargin:
    def test_overround_of_minus_110_both_sides(self) -> None:
        # 2 * 110/210 - 1 = 1/21
        assert overround([_implied(-110)] * 2) == pytest.approx(1 / 21, abs=1e-15)
        assert overround([_implied(-110)] * 2) == pytest.approx(0.0476, abs=5e-5)

    def test_hold_of_minus_110_both_sides(self) -> None:
        # 1 - 21/22 = 1/22
        assert hold([_implied(-110)] * 2) == pytest.approx(1 / 22, abs=1e-15)
        assert hold([_implied(-110)] * 2) == pytest.approx(0.0455, abs=5e-5)

    def test_hold_and_overround_are_consistent(self) -> None:
        for q in MARKETS.values():
            over = overround(q)
            assert over == pytest.approx(math.fsum(q) - 1.0, abs=1e-14)
            assert hold(q) == pytest.approx(over / (1.0 + over), abs=1e-14)

    def test_zero_and_negative_margin(self) -> None:
        assert overround([0.6, 0.4]) == pytest.approx(0.0, abs=1e-15)
        assert hold([0.6, 0.4]) == pytest.approx(0.0, abs=1e-15)
        assert overround([0.55, 0.40]) == pytest.approx(-0.05, abs=1e-15)
        assert hold([0.55, 0.40]) < 0.0

    def test_hold_rejects_non_positive_total(self) -> None:
        with pytest.raises(ValueError):
            hold([])
        with pytest.raises(ValueError):
            hold([0.0, 0.0])


# --------------------------------------------------------------------------
# Properties every method must satisfy
# --------------------------------------------------------------------------
@pytest.mark.parametrize("method", METHODS)
class TestCommonProperties:
    @pytest.mark.parametrize("market", MARKET_IDS)
    def test_sums_to_one_and_stays_inside_unit_interval(self, method: str, market: str) -> None:
        q = MARKETS[market]
        p = devig(q, method)
        assert len(p) == len(q)
        assert math.fsum(p) == pytest.approx(1.0, abs=1e-12)
        assert all(0.0 < x < 1.0 for x in p)
        assert all(isinstance(x, float) for x in p)

    @pytest.mark.parametrize("market", MARKET_IDS)
    def test_preserves_order_and_ties(self, method: str, market: str) -> None:
        q = MARKETS[market]
        p = devig(q, method)
        for i in range(len(q)):
            for j in range(len(q)):
                if q[i] > q[j]:
                    assert p[i] > p[j]
                elif q[i] == q[j]:
                    assert p[i] == pytest.approx(p[j], abs=1e-15)

    @pytest.mark.parametrize("market", MARKET_IDS)
    def test_never_raises_a_probability(self, method: str, market: str) -> None:
        """Removing a positive margin can only lower each implied probability."""
        q = MARKETS[market]
        p = devig(q, method)
        assert all(fair <= quoted + 1e-15 for fair, quoted in zip(p, q))

    def test_random_markets(self, method: str, rng: np.random.Generator) -> None:
        for q in _random_markets(rng, 300):
            p = np.array(devig(q, method))
            assert p.sum() == pytest.approx(1.0, abs=1e-12)
            assert p.min() > 0.0 and p.max() < 1.0
            assert np.all(p <= np.array(q) + 1e-15)
            order = np.argsort(q, kind="stable")
            assert np.all(np.diff(p[order]) >= -1e-15)

    def test_is_permutation_equivariant(self, method: str, rng: np.random.Generator) -> None:
        for q in _random_markets(rng, 60):
            perm = rng.permutation(len(q))
            base = np.array(devig(q, method))
            shuffled = np.array(devig([q[i] for i in perm], method))
            assert shuffled == pytest.approx(base[perm], abs=1e-12)

    @pytest.mark.parametrize("n", [2, 3, 8, 20])
    @pytest.mark.parametrize("margin", [0.02, 0.1, 0.8])
    def test_symmetric_market_is_uniform(self, method: str, n: int, margin: float) -> None:
        p = devig([(1.0 + margin) / n] * n, method)
        assert p == pytest.approx([1.0 / n] * n, abs=1e-12)

    def test_zero_margin_is_returned_unchanged(self, method: str) -> None:
        for q in ([0.6, 0.4], [0.5, 0.3, 0.2], [0.999, 0.001], [0.125] * 8):
            assert devig(q, method) == pytest.approx([x / sum(q) for x in q], abs=1e-15)

    def test_negative_margin_is_simply_normalised(self, method: str) -> None:
        """Documented: when sum(q) <= 1 there is nothing to remove."""
        for q in ([0.55, 0.40], [0.3, 0.3, 0.3], [0.9, 0.02], [0.01, 0.02]):
            total = math.fsum(q)
            assert devig(q, method) == pytest.approx([x / total for x in q], abs=1e-15)

    def test_tiny_margin_barely_moves_the_quotes(self, method: str) -> None:
        for margin in (1e-11, 1e-9, 1e-6):
            q = [0.7 + margin, 0.3]
            p = devig(q, method)
            assert math.fsum(p) == pytest.approx(1.0, abs=1e-12)
            assert p[0] == pytest.approx(0.7, abs=2 * margin)
            assert p[1] == pytest.approx(0.3, abs=2 * margin)
            assert 0.3 - margin - 1e-15 <= p[1] <= 0.3 + 1e-15

    def test_converges_to_the_quotes_as_margin_vanishes(self, method: str) -> None:
        fair = np.array([0.62, 0.27, 0.11])
        previous = math.inf
        for margin in (0.08, 0.02, 0.005, 0.001, 0.0001):
            error = float(np.abs(np.array(devig(list(fair * (1 + margin)), method)) - fair).max())
            assert error <= previous + 1e-15
            assert error <= margin
            previous = error

    def test_is_idempotent(self, method: str) -> None:
        for q in MARKETS.values():
            once = devig(q, method)
            assert devig(once, method) == pytest.approx(once, abs=1e-14)

    def test_accepts_any_sequence_and_does_not_mutate(self, method: str) -> None:
        q = [0.6, 0.45, 0.05]
        original = list(q)
        expected = devig(q, method)
        assert q == original
        assert devig(tuple(q), method) == expected
        assert devig(np.array(q), method) == pytest.approx(expected, abs=1e-15)

    def test_heavy_favourite(self, method: str) -> None:
        """-2000 / +1200: fair favourite between the multiplicative value and its quote."""
        q = MARKETS["heavy_-2000_+1200"]
        p = devig(q, method)
        assert q[0] / sum(q) - 1e-12 <= p[0] < q[0]
        assert 0.0 < p[1] <= q[1] / sum(q) + 1e-12


# --------------------------------------------------------------------------
# Closed forms and defining equations
# --------------------------------------------------------------------------
class TestMultiplicative:
    @pytest.mark.parametrize("market", MARKET_IDS)
    def test_closed_form(self, market: str) -> None:
        q = MARKETS[market]
        total = math.fsum(q)
        assert devig(q, "multiplicative") == pytest.approx([x / total for x in q], abs=1e-15)

    def test_is_the_default_method(self) -> None:
        q = MARKETS["three_way"]
        assert devig(q) == devig(q, "multiplicative")

    def test_minus_2000_plus_1200_by_hand(self) -> None:
        # q = (20/21, 1/13); sum = 281/273; fair = (260/281, 21/281)
        p = devig(MARKETS["heavy_-2000_+1200"], "multiplicative")
        assert p == pytest.approx([260 / 281, 21 / 281], abs=1e-15)


class TestAdditive:
    @pytest.mark.parametrize(
        "market",
        ["balanced_-110", "mild_-150_+125", "dog_first_+150_-190", "heavy_-2000_+1200",
         "extreme_-100000_+5000", "three_way", "nine_way", "big_margin", "twenty_equal"],
    )
    def test_closed_form(self, market: str) -> None:
        q = MARKETS[market]
        shift = (math.fsum(q) - 1.0) / len(q)
        expected = [x - shift for x in q]
        assert min(expected) > 0.0  # guard: these markets do not need the fallback
        assert devig(q, "additive") == pytest.approx(expected, abs=1e-15)

    def test_minus_2000_plus_1200_by_hand(self) -> None:
        # overround = 8/273, so each side loses 4/273: (256/273, 17/273)
        p = devig(MARKETS["heavy_-2000_+1200"], "additive")
        assert p == pytest.approx([256 / 273, 17 / 273], abs=1e-15)

    @pytest.mark.parametrize(
        "q",
        [
            [0.8, 0.2, 0.05, 0.01],         # 0.01 - 0.06/4 < 0
            [0.98, 0.03, 0.02, 0.001],      # 0.001 - 0.031/4 < 0
            [0.9, 0.1, 0.02, 0.02, 0.02],   # 0.02 - 0.06/5 > 0 would be fine ...
            [0.9, 0.1, 0.02, 0.02, 0.005],  # ... but 0.005 - 0.045/5 < 0 is not
        ],
    )
    def test_falls_back_to_multiplicative_only_when_a_probability_would_be_non_positive(
        self, q: List[float]
    ) -> None:
        shift = (math.fsum(q) - 1.0) / len(q)
        total = math.fsum(q)
        if min(q) - shift > 0.0:
            expected = [x - shift for x in q]
        else:
            expected = [x / total for x in q]
        assert devig(q, "additive") == pytest.approx(expected, abs=1e-15)

    def test_fallback_when_the_longshot_lands_exactly_on_zero(self) -> None:
        """Dyadic quotes make the arithmetic exact: shift = 0.1875 / 3 = 0.0625 = min(q)."""
        q = [0.75, 0.375, 0.0625]
        assert devig(q, "additive") == pytest.approx([x / 1.1875 for x in q], abs=1e-15)

    def test_two_way_never_needs_the_fallback(self, rng: np.random.Generator) -> None:
        """With two outcomes q_min - (q1+q2-1)/2 > 0 whenever both quotes are < 1."""
        for _ in range(2000):
            a, b = (float(x) for x in rng.uniform(0.001, 0.999, 2))
            if a + b <= 1.0 + 1e-9:
                continue
            shift = (a + b - 1.0) / 2.0
            assert devig([a, b], "additive") == pytest.approx([a - shift, b - shift], abs=1e-14)


class TestPower:
    @pytest.mark.parametrize("market", MARKET_IDS)
    def test_single_exponent(self, market: str) -> None:
        """p_i = q_i ** k for one k > 1 shared by every outcome."""
        q = MARKETS[market]
        p = devig(q, "power")
        exponents = [math.log(fair) / math.log(quoted) for fair, quoted in zip(p, q)]
        assert max(exponents) - min(exponents) < 1e-11
        assert min(exponents) > 1.0

    @pytest.mark.parametrize("market", MARKET_IDS)
    def test_matches_independent_root_finder(self, market: str) -> None:
        q = MARKETS[market]
        k = brentq(lambda e: math.fsum(x**e for x in q) - 1.0, 1.0, 50.0, xtol=1e-15)
        assert devig(q, "power") == pytest.approx([x**k for x in q], abs=1e-12)

    def test_balanced_two_way_exponent_in_closed_form(self) -> None:
        # 2 * q**k = 1  =>  k = ln(1/2) / ln(q)
        q = _implied(-110)
        k = math.log(0.5) / math.log(q)
        assert devig([q, q], "power") == pytest.approx([q**k, q**k], abs=1e-13)


class TestOddsRatio:
    @pytest.mark.parametrize("market", MARKET_IDS)
    def test_single_odds_ratio(self, market: str) -> None:
        """odds(q_i) / odds(p_i) is one constant > 1 for every outcome."""
        q = MARKETS[market]
        p = devig(q, "odds_ratio")
        ratios = [(b / (1 - b)) / (a / (1 - a)) for a, b in zip(p, q)]
        assert max(ratios) - min(ratios) < 1e-10 * max(ratios)
        assert min(ratios) > 1.0

    @pytest.mark.parametrize("market", MARKET_IDS)
    def test_matches_independent_root_finder(self, market: str) -> None:
        q = MARKETS[market]

        def fair(c: float) -> List[float]:
            return [1.0 / (1.0 + c * (1.0 - x) / x) for x in q]

        c = brentq(lambda v: math.fsum(fair(v)) - 1.0, 1.0, 1e6, xtol=1e-15)
        assert devig(q, "odds_ratio") == pytest.approx(fair(c), abs=1e-12)

    def test_two_way_closed_form(self) -> None:
        """For two outcomes the odds ratio solves c**2 = odds(q1) * odds(q2)."""
        for market in ("mild_-150_+125", "heavy_-2000_+1200", "dog_first_+150_-190"):
            q1, q2 = MARKETS[market]
            c = math.sqrt((q1 / (1 - q1)) * (q2 / (1 - q2)))
            expected = [q1 / (c * (1 - q1) + q1), q2 / (c * (1 - q2) + q2)]
            assert devig([q1, q2], "odds_ratio") == pytest.approx(expected, abs=1e-12)


class TestProbit:
    @pytest.mark.parametrize("market", MARKET_IDS)
    def test_single_shift_on_the_normal_quantile_scale(self, market: str) -> None:
        q = MARKETS[market]
        p = devig(q, "probit")
        shifts = [float(norm.ppf(b) - norm.ppf(a)) for a, b in zip(p, q)]
        assert max(shifts) - min(shifts) < 1e-9
        assert min(shifts) > 0.0

    @pytest.mark.parametrize("market", MARKET_IDS)
    def test_matches_independent_root_finder(self, market: str) -> None:
        q = MARKETS[market]
        z = [float(norm.isf(1.0 - x)) for x in q]  # isf: a different code path to ppf
        c = brentq(lambda v: math.fsum(float(norm.sf(v - zi)) for zi in z) - 1.0, 0.0, 20.0, xtol=1e-15)
        expected = [float(norm.sf(c - zi)) for zi in z]
        assert devig(q, "probit") == pytest.approx(expected, abs=1e-11)

    def test_two_way_closed_form(self) -> None:
        """Two outcomes: Phi(z1 - c) + Phi(z2 - c) = 1  =>  c = (z1 + z2) / 2."""
        for market in ("mild_-150_+125", "heavy_-2000_+1200", "dog_first_+150_-190"):
            q1, q2 = MARKETS[market]
            z1, z2 = float(norm.ppf(q1)), float(norm.ppf(q2))
            c = (z1 + z2) / 2.0
            expected = [float(norm.cdf(z1 - c)), float(norm.cdf(z2 - c))]
            assert devig([q1, q2], "probit") == pytest.approx(expected, abs=1e-12)


def _shin_z(p: Sequence[float], q: Sequence[float]) -> List[float]:
    """Invert Shin's formula outcome by outcome.

    ``p = (sqrt(z^2 + 4(1-z) q^2/T) - z) / (2(1-z))`` rearranges to
    ``z = (q^2/T - p^2) / (p (1 - p))``.
    """
    total = math.fsum(q)
    return [(b * b / total - a * a) / (a * (1.0 - a)) for a, b in zip(p, q)]


class TestShin:
    @pytest.mark.parametrize("market", MARKET_IDS)
    def test_published_formula_with_a_single_z(self, market: str) -> None:
        q = MARKETS[market]
        p = devig(q, "shin")
        zs = _shin_z(p, q)
        assert max(zs) - min(zs) < 1e-10
        assert 0.0 <= min(zs) and max(zs) < 1.0

    @pytest.mark.parametrize("market", MARKET_IDS)
    def test_matches_independent_root_finder(self, market: str) -> None:
        q = MARKETS[market]
        total = math.fsum(q)

        def fair(z: float) -> List[float]:
            return [
                (math.sqrt(z * z + 4.0 * (1.0 - z) * x * x / total) - z) / (2.0 * (1.0 - z))
                for x in q
            ]

        z = brentq(lambda v: math.fsum(fair(v)) - 1.0, 0.0, 0.5, xtol=1e-16)
        assert devig(q, "shin") == pytest.approx(fair(z), abs=1e-12)

    def test_equals_additive_for_two_outcomes(self, rng: np.random.Generator) -> None:
        checked = 0
        for _ in range(4000):
            a, b = (float(x) for x in rng.uniform(0.002, 0.998, 2))
            if a + b <= 1.0005:
                continue
            checked += 1
            shift = (a + b - 1.0) / 2.0
            assert devig([a, b], "shin") == pytest.approx([a - shift, b - shift], abs=1e-12)
            assert devig([a, b], "shin") == pytest.approx(devig([a, b], "additive"), abs=1e-12)
        assert checked > 1500

    def test_differs_from_additive_beyond_two_outcomes(self) -> None:
        q = MARKETS["nine_way"]
        shin, additive = devig(q, "shin"), devig(q, "additive")
        assert max(abs(a - b) for a, b in zip(shin, additive)) > 1e-3

    def test_insider_share_grows_with_the_margin(self) -> None:
        fair = np.array([0.5, 0.3, 0.2])
        zs = []
        for margin in (0.02, 0.05, 0.10, 0.20):
            q = list(fair * (1 + margin))
            zs.append(_shin_z(devig(q, "shin"), q)[0])
        assert all(a < b for a, b in zip(zs, zs[1:]))


# --------------------------------------------------------------------------
# High-precision references
# --------------------------------------------------------------------------
# Produced with mpmath at 50 significant digits (mp.dps = 50): for each market
# the single parameter of the method (power exponent k, Shin z, odds ratio c,
# probit shift d) was found with mp.findroot on "sum of fair probabilities = 1"
# and the fair probabilities evaluated from the method's formula.  Quotes are
# exact rationals: three-way = 1/2.1, 1/3.4, 1/3.6; heavy = 2000/2100, 100/1300;
# nine-way = 1/3, 1/4.5, 1/6, 1/8, 1/10, 1/15, 1/26, 1/51, 1/101.
MPMATH_REFERENCE: Dict[str, Dict[str, List[float]]] = {
    "three_way": {
        "power": [0.460174457860487741, 0.277979964076388685, 0.261845578063123574],
        "shin": [0.458665715961335662, 0.278737637132790297, 0.262596646905874041],
        "odds_ratio": [0.457794969319540405, 0.279009253661322137, 0.263195777019137458],
        "probit": [0.458301318434390702, 0.27880601546344855, 0.262892666102160748],
    },
    "heavy_-2000_+1200": {
        "power": [0.945994456766449705, 0.0540055432335502948],
        "shin": [0.937728937728937729, 0.0622710622710622711],
        "odds_ratio": [0.939364295460963734, 0.0606357045390362659],
        "probit": [0.939096588925520944, 0.0609034110744790556],
    },
    "nine_way": {
        "power": [
            0.31737676900300388, 0.207788422820300054, 0.153852322664061265,
            0.113916535232558239, 0.0902297372209255409, 0.0590739355231409228,
            0.0332542882551673738, 0.0164507722755772678, 0.00805721700526545637,
        ],
        "shin": [
            0.316807461028089267, 0.209424356520401796, 0.155744346692619355,
            0.115497793967658425, 0.0913621483954253055, 0.0592170222602579769,
            0.0321201045387663222, 0.0142666901925931881, 0.00556007640418836422,
        ],
        "odds_ratio": [
            0.311931305236453868, 0.205752443700430137, 0.153501716579855955,
            0.114673386611583851, 0.0915226820084961804, 0.0608241488754493966,
            0.0349981661852358027, 0.0178107549738475166, 0.00898539582864729255,
        ],
        "probit": [
            0.314201162616921562, 0.206690246077058982, 0.153705150475870459,
            0.114372323659733031, 0.0909707301621526777, 0.060052209753624067,
            0.0342218372291654429, 0.0172079002241370974, 0.00857843980133668075,
        ],
    },
}

#: Method parameters from the same mpmath run (three-way market).
MPMATH_THREE_WAY_PARAMETERS = {
    "power_k": 1.04611206147482403,
    "shin_z": 0.0240828367317284362,
    "odds_ratio_c": 1.07671271483743643,
    "probit_d": 0.0449970461168403718,
}


class TestHighPrecisionReference:
    @pytest.mark.parametrize("market", sorted(MPMATH_REFERENCE))
    @pytest.mark.parametrize("method", ["power", "shin", "odds_ratio", "probit"])
    def test_fair_probabilities(self, market: str, method: str) -> None:
        expected = MPMATH_REFERENCE[market][method]
        assert devig(MARKETS[market], method) == pytest.approx(expected, abs=1e-12)

    def test_recovered_parameters(self) -> None:
        q = MARKETS["three_way"]
        ref = MPMATH_THREE_WAY_PARAMETERS
        p = devig(q, "power")
        assert math.log(p[0]) / math.log(q[0]) == pytest.approx(ref["power_k"], abs=1e-10)
        p = devig(q, "shin")
        assert _shin_z(p, q)[0] == pytest.approx(ref["shin_z"], abs=1e-10)
        p = devig(q, "odds_ratio")
        assert (q[0] / (1 - q[0])) / (p[0] / (1 - p[0])) == pytest.approx(ref["odds_ratio_c"], abs=1e-10)
        p = devig(q, "probit")
        assert float(norm.ppf(q[0]) - norm.ppf(p[0])) == pytest.approx(ref["probit_d"], abs=1e-10)


# --------------------------------------------------------------------------
# How the methods relate to one another
# --------------------------------------------------------------------------
class TestMethodRelations:
    @pytest.mark.parametrize("quoted", [0.501, 0.5238, 0.6, 0.9, 0.99])
    def test_all_methods_agree_on_symmetric_two_way(self, quoted: float) -> None:
        results = devig_all([quoted, quoted])
        assert set(results) == set(METHODS)
        for probs in results.values():
            assert probs == pytest.approx([0.5, 0.5], abs=1e-12)

    @pytest.mark.parametrize(
        "market", ["mild_-150_+125", "dog_first_+150_-190", "heavy_-2000_+1200", "extreme_-100000_+5000"]
    )
    def test_power_takes_more_from_the_longshot_than_multiplicative(self, market: str) -> None:
        q = MARKETS[market]
        dog = int(np.argmin(q))
        favourite = 1 - dog
        power, mult = devig(q, "power"), devig(q, "multiplicative")
        assert power[dog] < mult[dog]
        assert power[favourite] > mult[favourite]
        # relative margin removed: none of the others is proportional
        assert q[dog] / power[dog] > q[dog] / mult[dog] > 1.0

    @pytest.mark.parametrize("market", ["three_way", "nine_way", "longshots"])
    def test_longshot_bias_methods_shrink_the_rank_outsider_most(self, market: str) -> None:
        """Relative to multiplicative, the non-proportional methods cut the outsider."""
        q = MARKETS[market]
        outsider = int(np.argmin(q))
        mult = devig(q, "multiplicative")
        for method in ("power", "shin", "odds_ratio", "probit"):
            assert devig(q, method)[outsider] < mult[outsider]

    def test_two_way_longshot_ordering_on_a_heavy_favourite(self) -> None:
        """-2000/+1200: power < odds-ratio < probit < additive = shin < multiplicative."""
        q = MARKETS["heavy_-2000_+1200"]
        dog = {m: devig(q, m)[1] for m in METHODS}
        assert dog["power"] < dog["odds_ratio"] < dog["probit"] < dog["additive"] < dog["multiplicative"]
        assert dog["shin"] == pytest.approx(dog["additive"], abs=1e-12)

    def test_methods_coincide_to_first_order_near_even_money(self) -> None:
        q = [0.53, 0.52]
        results = devig_all(q)
        spread = max(p[0] for p in results.values()) - min(p[0] for p in results.values())
        assert spread < 5e-4


# --------------------------------------------------------------------------
# Helpers built on devig
# --------------------------------------------------------------------------
class TestHelpers:
    def test_devig_decimal_matches_devig_on_reciprocals(self) -> None:
        decimals = [2.1, 3.4, 3.6]
        for method in METHODS:
            assert devig_decimal(decimals, method) == devig([1 / d for d in decimals], method)

    def test_devig_decimal_default_is_multiplicative(self) -> None:
        assert devig_decimal([1.5, 2.6]) == devig([1 / 1.5, 1 / 2.6], "multiplicative")

    def test_devig_two_way_returns_a_tuple(self) -> None:
        for method in METHODS:
            result = devig_two_way(1.5, 2.6, method)
            assert isinstance(result, tuple) and len(result) == 2
            assert list(result) == devig([1 / 1.5, 1 / 2.6], method)

    def test_devig_two_way_minus_110_is_a_coin_flip(self) -> None:
        assert devig_two_way(21 / 11, 21 / 11) == pytest.approx((0.5, 0.5), abs=1e-15)

    @pytest.mark.parametrize("bad", [1.0, 0.5, 0.0, -2.0, NAN, float("inf")])
    def test_devig_decimal_rejects_invalid_prices(self, bad: float) -> None:
        with pytest.raises(ValueError):
            devig_decimal([1.9, bad])
        with pytest.raises(ValueError):
            devig_two_way(bad, 1.9)

    def test_devig_all_covers_every_method_in_order(self) -> None:
        q = MARKETS["three_way"]
        results = devig_all(q)
        assert tuple(results) == METHODS
        for method in METHODS:
            assert results[method] == devig(q, method)

    def test_devig_all_with_a_subset(self) -> None:
        q = MARKETS["three_way"]
        results = devig_all(q, methods=["shin", "power"])
        assert list(results) == ["shin", "power"]
        with pytest.raises(ValueError):
            devig_all(q, methods=["shin", "nope"])

    def test_methods_constant(self) -> None:
        assert METHODS == ("multiplicative", "additive", "power", "shin", "odds_ratio", "probit")

    @pytest.mark.parametrize("market", MARKET_IDS)
    def test_worst_case_is_the_minimum_across_methods(self, market: str) -> None:
        q = MARKETS[market]
        for index in range(len(q)):
            expected = min(devig(q, m)[index] for m in METHODS)
            assert worst_case(q, index) == expected

    def test_worst_case_known_argmin_on_a_heavy_favourite(self) -> None:
        q = MARKETS["heavy_-2000_+1200"]
        assert worst_case(q, 0) == devig(q, "multiplicative")[0]  # favourite: 260/281
        assert worst_case(q, 0) == pytest.approx(260 / 281, abs=1e-15)
        assert worst_case(q, 1) == devig(q, "power")[1]  # longshot
        assert worst_case(q, 0) + worst_case(q, 1) < 1.0

    def test_worst_case_with_a_subset_of_methods(self) -> None:
        q = MARKETS["heavy_-2000_+1200"]
        assert worst_case(q, 1, methods=["multiplicative"]) == pytest.approx(21 / 281, abs=1e-15)
        assert worst_case(q, 1, methods=["multiplicative", "additive"]) == pytest.approx(
            17 / 273, abs=1e-15
        )

    def test_worst_case_is_uniform_on_balanced_markets(self) -> None:
        assert worst_case(MARKETS["balanced_-110"], 0) == pytest.approx(0.5, abs=1e-12)

    def test_worst_case_bad_index(self) -> None:
        with pytest.raises(IndexError):
            worst_case(MARKETS["balanced_-110"], 2)


class TestValidation:
    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize(
        "q",
        [[], [0.5], [0.5, 0.0], [0.5, 1.0], [0.5, -0.1], [0.5, 1.2], [0.5, NAN], [NAN, NAN], [0.6, 0.6, 0.0]],
    )
    def test_invalid_probabilities_raise(self, method: str, q: List[float]) -> None:
        with pytest.raises(ValueError):
            devig(q, method)

    @pytest.mark.parametrize("method", ["", "Multiplicative", "balanced", "logit", None])
    def test_unknown_method_raises(self, method: str) -> None:
        with pytest.raises(ValueError):
            devig([0.55, 0.5], method)


# --------------------------------------------------------------------------
# fair_prob_one_sided
# --------------------------------------------------------------------------
class TestOneSided:
    @pytest.mark.parametrize("q", [0.02, 0.1, 0.3, 0.5, 0.75, 0.95, 0.999])
    @pytest.mark.parametrize("assumed", [0.01, 0.07, 0.25, 1.5])
    def test_closed_form_and_margin_round_trip(self, q: float, assumed: float) -> None:
        p = fair_prob_one_sided(q, assumed)
        assert p == pytest.approx(q / (1.0 + assumed), rel=1e-15)
        assert p * (1.0 + assumed) == pytest.approx(q, rel=1e-15)  # re-applying the margin
        assert 0.0 < p < q

    def test_defaults_to_a_seven_percent_overround(self) -> None:
        assert fair_prob_one_sided(0.3) == pytest.approx(0.3 / 1.07, rel=1e-15)
        assert fair_prob_one_sided(0.3) == fair_prob_one_sided(0.3, 0.07)
        assert fair_prob_one_sided(0.3, assumed_hold=0.05) == pytest.approx(0.3 / 1.05, rel=1e-15)

    def test_zero_assumed_hold_returns_the_quote(self) -> None:
        for q in (0.01, 0.3, 0.99):
            assert fair_prob_one_sided(q, 0.0) == q

    @pytest.mark.parametrize(
        "prices", [(-110, -110), (-150, 125), (150, -190), (-400, 300), (-2000, 1200)]
    )
    def test_agrees_with_two_way_multiplicative_when_the_true_overround_is_assumed(
        self, prices: Tuple[int, int]
    ) -> None:
        q = [_implied(prices[0]), _implied(prices[1])]
        margin = q[0] + q[1] - 1.0
        two_way = devig(q, "multiplicative")
        assert fair_prob_one_sided(q[0], margin) == pytest.approx(two_way[0], abs=1e-14)
        assert fair_prob_one_sided(q[1], margin) == pytest.approx(two_way[1], abs=1e-14)

    def test_is_monotone_in_the_quote(self) -> None:
        """A shorter price must never map to a smaller fair probability."""
        for assumed in (0.03, 0.07, 0.25):
            values = [fair_prob_one_sided(float(q), assumed) for q in np.linspace(0.005, 0.995, 199)]
            assert all(a < b for a, b in zip(values, values[1:]))

    def test_larger_assumed_hold_gives_a_lower_fair_probability(self) -> None:
        values = [fair_prob_one_sided(0.6, h) for h in (0.0, 0.01, 0.03, 0.07, 0.15, 0.3)]
        assert all(a > b for a, b in zip(values, values[1:]))

    def test_is_at_least_as_generous_as_any_two_way_method_on_a_longshot(self) -> None:
        """Documented caveat: books load more margin on longshots than this removes."""
        q = MARKETS["heavy_-2000_+1200"]
        margin = q[0] + q[1] - 1.0
        one_sided = fair_prob_one_sided(q[1], margin)
        for method in METHODS:
            assert one_sided >= devig(q, method)[1] - 1e-15

    @pytest.mark.parametrize("q", [0.0, 1.0, -0.2, 1.3, NAN, float("inf")])
    def test_invalid_quote_raises(self, q: float) -> None:
        with pytest.raises(ValueError):
            fair_prob_one_sided(q, 0.07)

    @pytest.mark.parametrize("assumed", [-0.01, -1.0, NAN, float("inf")])
    def test_invalid_hold_raises(self, assumed: float) -> None:
        with pytest.raises(ValueError):
            fair_prob_one_sided(0.3, assumed)
