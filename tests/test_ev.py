"""Independent verification of :mod:`evmodel.ev`.

Expected values are derived by direct enumeration of the three outcomes of a
bet (win ``+b``, push ``0``, lose ``-1``), by numerically maximising expected
log growth with scipy, and by seeded Monte-Carlo simulation.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from evmodel.ev import (
    bet_variance,
    edge,
    expected_log_growth,
    expected_value,
    fractional_kelly,
    kelly_fraction,
    shrink_prob,
)

NAN = float("nan")
INF = float("inf")

#: (p_win, decimal, p_push) triples spanning favourites, dogs and pushes.
BETS: List[Tuple[float, float, float]] = [
    (0.55, 2.0, 0.0),
    (0.5, 21 / 11, 0.0),
    (0.6, 21 / 11, 0.0),
    (0.3, 4.2, 0.0),
    (0.08, 15.0, 0.0),
    (0.92, 1.12, 0.0),
    (0.5, 2.1, 0.1),
    (0.45, 1.91, 0.1),
    (0.48, 2.05, 0.06),
    (0.2, 6.5, 0.15),
    (0.7, 1.3, 0.25),
    (0.35, 2.0, 0.3),
    (0.05, 1.5, 0.9),
    (0.0, 2.0, 0.0),
    (1.0, 2.0, 0.0),
    (0.4, 3.0, 0.6),
    (0.0, 2.0, 1.0),
]


def _moments(p_win: float, decimal: float, p_push: float) -> Tuple[float, float]:
    """Mean and variance of profit per unit by enumerating the three outcomes."""
    outcomes = [(decimal - 1.0, p_win), (0.0, p_push), (-1.0, 1.0 - p_win - p_push)]
    mean = math.fsum(x * p for x, p in outcomes)
    var = math.fsum((x - mean) ** 2 * p for x, p in outcomes)
    return mean, var


def _log_growth(f: float, p_win: float, decimal: float, p_push: float) -> float:
    """Expected log bankroll growth, written out outcome by outcome."""
    p_lose = 1.0 - p_win - p_push
    return p_win * math.log(1.0 + f * (decimal - 1.0)) + p_lose * math.log(1.0 - f)


# --------------------------------------------------------------------------
# expected_value / edge
# --------------------------------------------------------------------------
class TestExpectedValue:
    @pytest.mark.parametrize(
        "p_win, decimal, p_push, expected",
        [
            (0.55, 2.0, 0.0, 0.10),
            (0.5, 2.0, 0.0, 0.0),
            (0.5, 21 / 11, 0.0, -1 / 22),   # -110 coin flip loses the hold
            (0.25, 4.0, 0.0, 0.0),
            (0.5, 1.9, 0.1, 0.05),          # 0.5*0.9 - 0.4
            (0.0, 3.0, 0.0, -1.0),
            (1.0, 3.0, 0.0, 2.0),
            (0.0, 3.0, 1.0, 0.0),           # certain push
            (0.3, 2.5, 0.7, 0.45),          # cannot lose
        ],
    )
    def test_hand_worked(self, p_win: float, decimal: float, p_push: float, expected: float) -> None:
        assert expected_value(p_win, decimal, p_push) == pytest.approx(expected, abs=1e-14)

    @pytest.mark.parametrize("p_win, decimal, p_push", BETS)
    def test_matches_outcome_enumeration(self, p_win: float, decimal: float, p_push: float) -> None:
        mean, _ = _moments(p_win, decimal, p_push)
        assert expected_value(p_win, decimal, p_push) == pytest.approx(mean, abs=1e-14)

    def test_push_defaults_to_zero(self) -> None:
        assert expected_value(0.55, 2.0) == expected_value(0.55, 2.0, 0.0)
        assert edge(0.55, 2.0) == edge(0.55, 2.0, 0.0)

    def test_fair_price_has_zero_ev(self) -> None:
        for p in np.linspace(0.02, 0.98, 49):
            assert expected_value(float(p), 1.0 / float(p)) == pytest.approx(0.0, abs=1e-13)

    def test_fair_price_with_push_has_zero_ev(self) -> None:
        """Fair odds given a push probability are (1 - push) / p_win."""
        for p_win, p_push in ((0.4, 0.1), (0.2, 0.3), (0.55, 0.05)):
            fair = (1.0 - p_push) / p_win
            assert expected_value(p_win, fair, p_push) == pytest.approx(0.0, abs=1e-14)
            assert edge(p_win, fair, p_push) == pytest.approx(0.0, abs=1e-14)

    def test_is_linear_in_win_probability_and_price(self) -> None:
        assert expected_value(0.6, 2.0) - expected_value(0.5, 2.0) == pytest.approx(0.2, abs=1e-14)
        assert expected_value(0.5, 2.4) - expected_value(0.5, 2.0) == pytest.approx(0.2, abs=1e-14)

    def test_a_push_only_replaces_losses(self) -> None:
        """Moving probability from 'lose' to 'push' raises EV one for one."""
        assert expected_value(0.5, 1.9, 0.1) - expected_value(0.5, 1.9, 0.0) == pytest.approx(0.1, abs=1e-14)


class TestEdge:
    @pytest.mark.parametrize(
        "p_win, decimal, p_push, expected",
        [(0.55, 2.0, 0.0, 0.05), (0.5, 1.9, 0.1, 0.5 - 0.9 / 1.9), (0.3, 4.0, 0.0, 0.05), (0.5, 2.0, 0.0, 0.0)],
    )
    def test_hand_worked(self, p_win: float, decimal: float, p_push: float, expected: float) -> None:
        assert edge(p_win, decimal, p_push) == pytest.approx(expected, abs=1e-14)

    @pytest.mark.parametrize("p_win, decimal, p_push", BETS)
    def test_ev_equals_price_times_edge(self, p_win: float, decimal: float, p_push: float) -> None:
        assert expected_value(p_win, decimal, p_push) == pytest.approx(
            decimal * edge(p_win, decimal, p_push), abs=1e-13
        )

    @pytest.mark.parametrize("p_win, decimal, p_push", BETS)
    def test_sign_agrees_with_ev(self, p_win: float, decimal: float, p_push: float) -> None:
        ev, e = expected_value(p_win, decimal, p_push), edge(p_win, decimal, p_push)
        if abs(ev) > 1e-12:
            assert (ev > 0) == (e > 0)


# --------------------------------------------------------------------------
# Kelly
# --------------------------------------------------------------------------
class TestKelly:
    @pytest.mark.parametrize(
        "p_win, decimal, expected",
        [(0.55, 2.0, 0.10), (0.6, 2.0, 0.20), (0.4, 3.0, 0.10), (0.25, 5.0, 0.0625), (0.9, 1.5, 0.7)],
    )
    def test_classic_formula_without_push(self, p_win: float, decimal: float, expected: float) -> None:
        """f* = (p b - q) / b."""
        b = decimal - 1.0
        assert kelly_fraction(p_win, decimal) == pytest.approx((p_win * b - (1 - p_win)) / b, abs=1e-14)
        assert kelly_fraction(p_win, decimal) == pytest.approx(expected, abs=1e-14)

    def test_hand_worked_with_push(self) -> None:
        # p=0.5, push=0.1, lose=0.4, b=1.1: (0.55 - 0.4) / (1.1 * 0.9) = 5/33
        assert kelly_fraction(0.5, 2.1, 0.1) == pytest.approx(5 / 33, abs=1e-14)

    @pytest.mark.parametrize("p_win, decimal, p_push", BETS)
    def test_equals_ev_over_net_odds_and_decided_probability(
        self, p_win: float, decimal: float, p_push: float
    ) -> None:
        ev = expected_value(p_win, decimal, p_push)
        decided = 1.0 - p_push
        if decided <= 0.0:
            assert kelly_fraction(p_win, decimal, p_push) == 0.0
            return
        expected = min(max(ev / ((decimal - 1.0) * decided), 0.0), 1.0)
        assert kelly_fraction(p_win, decimal, p_push) == pytest.approx(expected, abs=1e-13)

    @pytest.mark.parametrize("p_win, decimal, p_push", [b for b in BETS if b[2] < 1.0])
    def test_push_is_equivalent_to_conditioning_on_a_decision(
        self, p_win: float, decimal: float, p_push: float
    ) -> None:
        """Pushes do not change the bankroll, so Kelly only sees P(win | no push)."""
        conditional = p_win / (1.0 - p_push)
        assert kelly_fraction(p_win, decimal, p_push) == pytest.approx(
            kelly_fraction(conditional, decimal), abs=1e-13
        )

    @pytest.mark.parametrize(
        "p_win, decimal, p_push",
        [
            (0.55, 2.0, 0.0), (0.6, 21 / 11, 0.0), (0.3, 4.2, 0.0), (0.08, 15.0, 0.0), (0.92, 1.12, 0.0),
            (0.5, 2.1, 0.1), (0.48, 2.05, 0.06), (0.2, 6.5, 0.15), (0.7, 1.3, 0.25), (0.4, 2.0, 0.3),
            (0.02, 80.0, 0.5), (0.5, 3.0, 0.45),
        ],
    )
    def test_matches_numerical_maximum_of_log_growth(self, p_win: float, decimal: float, p_push: float) -> None:
        result = minimize_scalar(
            lambda f: -_log_growth(f, p_win, decimal, p_push),
            bounds=(0.0, 1.0 - 1e-9),
            method="bounded",
            options={"xatol": 1e-12},
        )
        kelly = kelly_fraction(p_win, decimal, p_push)
        assert kelly > 0.0
        assert kelly == pytest.approx(result.x, abs=2e-6)
        assert _log_growth(kelly, p_win, decimal, p_push) >= -result.fun - 1e-15

    @pytest.mark.parametrize("p_win, decimal, p_push", [(0.55, 2.0, 0.0), (0.5, 2.1, 0.1), (0.2, 6.5, 0.15)])
    def test_is_a_stationary_point_and_a_maximum(self, p_win: float, decimal: float, p_push: float) -> None:
        kelly = kelly_fraction(p_win, decimal, p_push)
        h = 1e-5
        up = _log_growth(kelly + h, p_win, decimal, p_push)
        down = _log_growth(kelly - h, p_win, decimal, p_push)
        here = _log_growth(kelly, p_win, decimal, p_push)
        assert (up - down) / (2 * h) == pytest.approx(0.0, abs=1e-8)
        assert here > up and here > down

    def test_is_zero_for_negative_or_zero_ev(self) -> None:
        assert kelly_fraction(0.5, 21 / 11) == 0.0
        assert kelly_fraction(0.5, 2.0) == 0.0
        assert kelly_fraction(0.0, 5.0) == 0.0
        assert kelly_fraction(0.4, 2.0, 0.1) == 0.0
        assert kelly_fraction(0.45, 2.0, 0.1) == 0.0  # exactly break-even with the push
        for p_win, decimal, p_push in BETS:
            if expected_value(p_win, decimal, p_push) <= 0.0:
                assert kelly_fraction(p_win, decimal, p_push) == 0.0

    def test_is_positive_exactly_when_ev_is_positive(self) -> None:
        for p_win, decimal, p_push in BETS:
            ev = expected_value(p_win, decimal, p_push)
            if ev > 1e-12:
                assert 0.0 < kelly_fraction(p_win, decimal, p_push) <= 1.0

    def test_sure_things_stake_everything(self) -> None:
        assert kelly_fraction(1.0, 2.0) == 1.0
        assert kelly_fraction(0.7, 2.0, 0.3) == 1.0  # can only win or push

    def test_certain_push_stakes_nothing(self) -> None:
        assert kelly_fraction(0.0, 2.0, 1.0) == 0.0

    def test_grows_with_edge_and_is_bounded_by_edge_over_net_odds(self) -> None:
        values = [kelly_fraction(p, 2.0) for p in (0.5, 0.52, 0.55, 0.6, 0.75, 0.99)]
        assert values == sorted(values)
        assert all(0.0 <= v <= 1.0 for v in values)


class TestExpectedLogGrowth:
    @pytest.mark.parametrize("p_win, decimal, p_push", BETS)
    @pytest.mark.parametrize("stake", [0.0, 0.01, 0.2, 0.75])
    def test_matches_direct_formula(self, stake: float, p_win: float, decimal: float, p_push: float) -> None:
        assert expected_log_growth(stake, p_win, decimal, p_push) == pytest.approx(
            _log_growth(stake, p_win, decimal, p_push), abs=1e-14
        )

    def test_zero_stake_has_zero_growth(self) -> None:
        for p_win, decimal, p_push in BETS:
            assert expected_log_growth(0.0, p_win, decimal, p_push) == 0.0

    def test_hand_worked(self) -> None:
        expected = 0.55 * math.log(1.1) + 0.45 * math.log(0.9)
        assert expected_log_growth(0.1, 0.55, 2.0) == pytest.approx(expected, abs=1e-15)

    def test_kelly_beats_every_other_stake(self) -> None:
        for p_win, decimal, p_push in ((0.55, 2.0, 0.0), (0.5, 2.1, 0.1), (0.2, 6.5, 0.15)):
            kelly = kelly_fraction(p_win, decimal, p_push)
            best = expected_log_growth(kelly, p_win, decimal, p_push)
            assert best > 0.0
            for stake in np.linspace(0.0, 0.99, 100):
                assert expected_log_growth(float(stake), p_win, decimal, p_push) <= best + 1e-15

    def test_overbetting_destroys_growth(self) -> None:
        """Growth is concave: twice Kelly is roughly break-even, three times loses."""
        kelly = kelly_fraction(0.55, 2.0)
        assert expected_log_growth(kelly / 2, 0.55, 2.0) > 0.0
        assert abs(expected_log_growth(2 * kelly, 0.55, 2.0)) < 2e-4
        assert expected_log_growth(3 * kelly, 0.55, 2.0) < 0.0
        assert expected_log_growth(0.05, 0.5, 21 / 11) < 0.0  # any stake on a -EV bet

    @pytest.mark.parametrize("stake", [1.0, 1.5, -0.01, NAN])
    def test_invalid_stake_raises(self, stake: float) -> None:
        """Rejected by the argument check itself, not by a math domain error further down."""
        with pytest.raises(ValueError, match="stake_fraction"):
            expected_log_growth(stake, 0.55, 2.0)


class TestFractionalKelly:
    def test_default_is_a_quarter(self) -> None:
        assert fractional_kelly(0.55, 2.0) == pytest.approx(0.025, abs=1e-15)
        assert fractional_kelly(0.5, 2.1, 0.1) == pytest.approx(0.25 * 5 / 33, abs=1e-15)

    @pytest.mark.parametrize("fraction", [0.0, 0.1, 0.25, 0.5, 1.0, 2.0])
    @pytest.mark.parametrize("p_win, decimal, p_push", BETS)
    def test_scales_full_kelly(self, fraction: float, p_win: float, decimal: float, p_push: float) -> None:
        assert fractional_kelly(p_win, decimal, p_push, fraction=fraction) == pytest.approx(
            fraction * kelly_fraction(p_win, decimal, p_push), abs=1e-15
        )

    def test_cap_limits_the_stake(self) -> None:
        assert fractional_kelly(0.9, 2.0, fraction=1.0) == pytest.approx(0.8, abs=1e-15)
        assert fractional_kelly(0.9, 2.0, fraction=1.0, cap=0.05) == 0.05
        assert fractional_kelly(0.9, 2.0, fraction=0.25, cap=0.05) == 0.05
        assert fractional_kelly(0.9, 2.0, fraction=0.25, cap=None) == pytest.approx(0.2, abs=1e-15)

    def test_cap_does_not_raise_small_stakes(self) -> None:
        assert fractional_kelly(0.55, 2.0, cap=0.05) == pytest.approx(0.025, abs=1e-15)
        assert fractional_kelly(0.5, 21 / 11, cap=0.05) == 0.0

    def test_zero_or_negative_cap_means_no_bet(self) -> None:
        assert fractional_kelly(0.9, 2.0, fraction=1.0, cap=0.0) == 0.0
        assert fractional_kelly(0.9, 2.0, fraction=1.0, cap=-0.05) == 0.0

    def test_never_exceeds_the_cap(self) -> None:
        for p_win, decimal, p_push in BETS:
            for cap in (0.0, 0.01, 0.03, 0.5):
                stake = fractional_kelly(p_win, decimal, p_push, fraction=1.0, cap=cap)
                assert 0.0 <= stake <= cap

    def test_negative_fraction_raises(self) -> None:
        with pytest.raises(ValueError):
            fractional_kelly(0.55, 2.0, fraction=-0.25)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "BUG (minor): fractional_kelly only rejects fraction < 0, so a NaN or infinite "
            "fraction passes validation and the recommended stake comes back as nan / inf"
        ),
    )
    @pytest.mark.parametrize("fraction", [NAN, INF])
    def test_non_finite_fraction_raises(self, fraction: float) -> None:
        with pytest.raises(ValueError):
            fractional_kelly(0.55, 2.0, fraction=fraction)

    def test_validates_the_bet(self) -> None:
        with pytest.raises(ValueError):
            fractional_kelly(1.2, 2.0)
        with pytest.raises(ValueError):
            fractional_kelly(0.5, 1.0)


# --------------------------------------------------------------------------
# shrink_prob
# --------------------------------------------------------------------------
class TestShrink:
    def test_endpoints_and_midpoint(self) -> None:
        assert shrink_prob(0.6, 0.5, 1.0) == 0.6
        assert shrink_prob(0.6, 0.5, 0.0) == 0.5
        assert shrink_prob(0.6, 0.5, 0.5) == pytest.approx(0.55, abs=1e-15)
        assert shrink_prob(0.4, 0.5, 0.25) == pytest.approx(0.475, abs=1e-15)

    def test_result_lies_between_model_and_breakeven(self) -> None:
        for confidence in np.linspace(0.0, 1.0, 11):
            value = shrink_prob(0.62, 0.5238, float(confidence))
            assert 0.5238 - 1e-15 <= value <= 0.62 + 1e-15

    def test_is_monotone_in_confidence(self) -> None:
        values = [shrink_prob(0.62, 0.5238, float(c)) for c in np.linspace(0.0, 1.0, 11)]
        assert all(a < b for a, b in zip(values, values[1:]))

    def test_shrinking_scales_ev_linearly(self) -> None:
        """EV is affine in p_win and zero at break-even, so EV(shrunk) = c * EV(model)."""
        for p_model, decimal, p_push in ((0.58, 21 / 11, 0.0), (0.5, 2.1, 0.1), (0.2, 6.5, 0.15)):
            breakeven = (1.0 - p_push) / decimal
            full = expected_value(p_model, decimal, p_push)
            for confidence in (0.0, 0.3, 0.7, 1.0):
                shrunk = shrink_prob(p_model, breakeven, confidence)
                assert expected_value(shrunk, decimal, p_push) == pytest.approx(confidence * full, abs=1e-13)

    def test_no_confidence_means_no_bet(self) -> None:
        breakeven = 11 / 21
        assert kelly_fraction(shrink_prob(0.6, breakeven, 0.0), 21 / 11) == pytest.approx(0.0, abs=1e-15)
        assert kelly_fraction(shrink_prob(0.6, breakeven, 0.5), 21 / 11) < kelly_fraction(0.6, 21 / 11)

    @pytest.mark.parametrize("confidence", [-0.01, 1.01, 2.0, NAN])
    def test_invalid_confidence_raises(self, confidence: float) -> None:
        with pytest.raises(ValueError):
            shrink_prob(0.6, 0.5, confidence)


# --------------------------------------------------------------------------
# bet_variance
# --------------------------------------------------------------------------
class TestVariance:
    @pytest.mark.parametrize("p_win, decimal, p_push", BETS)
    def test_matches_outcome_enumeration(self, p_win: float, decimal: float, p_push: float) -> None:
        _, var = _moments(p_win, decimal, p_push)
        assert bet_variance(p_win, decimal, p_push) == pytest.approx(var, abs=1e-12)
        assert bet_variance(p_win, decimal, p_push) >= -1e-15

    def test_hand_worked(self) -> None:
        assert bet_variance(0.5, 2.0) == pytest.approx(1.0, abs=1e-15)  # +-1 coin flip
        # no push: outcomes b and -1 are a scaled Bernoulli: p (1-p) (b+1)^2
        assert bet_variance(0.3, 4.2) == pytest.approx(0.3 * 0.7 * 4.2**2, abs=1e-13)

    def test_certain_outcomes_have_no_variance(self) -> None:
        assert bet_variance(1.0, 3.0) == pytest.approx(0.0, abs=1e-15)
        assert bet_variance(0.0, 3.0) == pytest.approx(0.0, abs=1e-15)
        assert bet_variance(0.0, 3.0, 1.0) == pytest.approx(0.0, abs=1e-15)

    def test_pushes_reduce_variance(self) -> None:
        assert bet_variance(0.45, 2.0, 0.1) < bet_variance(0.5, 2.0, 0.0)

    @pytest.mark.parametrize(
        "p_win, decimal, p_push",
        [(0.55, 2.0, 0.0), (0.45, 1.91, 0.1), (0.2, 6.5, 0.15), (0.08, 15.0, 0.0), (0.7, 1.3, 0.25)],
    )
    def test_matches_monte_carlo(self, p_win: float, decimal: float, p_push: float, rng: np.random.Generator) -> None:
        """Simulate n = 400,000 bets.

        The sample mean has standard error sqrt(var / n) and the sample
        variance sqrt((m4 - var^2) / n) with m4 the fourth central moment,
        both computed exactly from the three-point distribution; tolerances
        are five standard errors.
        """
        n = 400_000
        b = decimal - 1.0
        u = rng.random(n)
        profit = np.where(u < p_win, b, np.where(u < p_win + p_push, 0.0, -1.0))
        mean, var = _moments(p_win, decimal, p_push)
        m4 = (
            p_win * (b - mean) ** 4
            + p_push * (0.0 - mean) ** 4
            + (1.0 - p_win - p_push) * (-1.0 - mean) ** 4
        )
        assert float(profit.mean()) == pytest.approx(
            expected_value(p_win, decimal, p_push), abs=5.0 * math.sqrt(var / n)
        )
        assert float(profit.var()) == pytest.approx(
            bet_variance(p_win, decimal, p_push), abs=5.0 * math.sqrt((m4 - var * var) / n)
        )


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------
BAD_BETS = [
    (0.5, 1.0, 0.0),      # decimal must exceed 1
    (0.5, 0.5, 0.0),
    (0.5, 0.0, 0.0),
    (0.5, -2.0, 0.0),
    (0.5, NAN, 0.0),
    (0.5, INF, 0.0),      # an infinite price would give an infinite EV and a NaN Kelly stake
    (0.5, -INF, 0.0),
    (-0.01, 2.0, 0.0),
    (1.01, 2.0, 0.0),
    (NAN, 2.0, 0.0),
    (0.5, 2.0, -0.01),
    (0.5, 2.0, 1.01),
    (0.5, 2.0, NAN),
    (0.6, 2.0, 0.5),      # p_win + p_push > 1
    (0.7, 2.0, 0.300001),
]


@pytest.mark.parametrize("func", [expected_value, edge, kelly_fraction, bet_variance, fractional_kelly])
@pytest.mark.parametrize("p_win, decimal, p_push", BAD_BETS)
def test_invalid_bets_raise(func, p_win: float, decimal: float, p_push: float) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        func(p_win, decimal, p_push)


@pytest.mark.parametrize("p_win, decimal, p_push", BAD_BETS)
def test_expected_log_growth_validates_the_bet(p_win: float, decimal: float, p_push: float) -> None:
    with pytest.raises(ValueError):
        expected_log_growth(0.1, p_win, decimal, p_push)


def test_probabilities_overshooting_one_by_float_noise_are_accepted() -> None:
    """Probabilities that sum to 1 + 1e-15 are rounding noise, not invalid input."""
    p_win, p_push = 0.3, 0.7 + 1e-15
    assert 1.0 - p_win - p_push < 0.0  # guard: a naive p_lose >= 0 check would reject this
    assert expected_value(p_win, 2.0, p_push) == pytest.approx(0.3, abs=1e-12)
    assert edge(p_win, 2.0, p_push) == pytest.approx(0.15, abs=1e-12)
    assert kelly_fraction(p_win, 2.0, p_push) == 1.0
    assert bet_variance(p_win, 2.0, p_push) == pytest.approx(0.3 * 0.7, abs=1e-12)
    assert expected_value(0.6, 2.0, 0.4 + 5e-10) == pytest.approx(0.6, abs=1e-9)


def test_results_are_finite_for_extreme_but_valid_prices() -> None:
    for decimal in (1.0 + 1e-9, 1.0001, 1e6, 1e12):
        for p_win in (0.0, 1e-9, 0.5, 1.0):
            assert math.isfinite(expected_value(p_win, decimal))
            assert math.isfinite(bet_variance(p_win, decimal))
            assert 0.0 <= kelly_fraction(p_win, decimal) <= 1.0
