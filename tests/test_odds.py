"""Independent verification of :mod:`evmodel.odds`.

Expected values are derived without the implementation: exact rational
arithmetic (:class:`fractions.Fraction`) for the conversion formulas and
hand-worked constants for the well-known prices (-110, +150, ...).
"""

from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction
from typing import List, Union

import numpy as np
import pytest

from evmodel.odds import (
    american_to_decimal,
    american_to_prob,
    breakeven_prob,
    decimal_to_american,
    decimal_to_prob,
    format_american,
    parse_odds,
    prob_to_american,
    prob_to_decimal,
)

NAN = float("nan")
INF = float("inf")


def _exact_decimal(american: int) -> Fraction:
    """Decimal odds of an integer American price, as an exact fraction."""
    if american > 0:
        return 1 + Fraction(american, 100)
    return 1 + Fraction(100, -american)


def _american_grid() -> List[int]:
    """Every price from 100 to 1000 plus sparse long shots, both signs."""
    magnitudes = list(range(100, 1001)) + [1250, 2000, 5000, 10000, 25000, 100000, 1000000]
    return [sign * m for m in magnitudes for sign in (1, -1)]


# --------------------------------------------------------------------------
# American <-> decimal <-> probability
# --------------------------------------------------------------------------
class TestConversions:
    @pytest.mark.parametrize(
        "american, decimal",
        [
            (150, 2.5),
            (-110, 21 / 11),
            (100, 2.0),
            (-100, 2.0),
            (-200, 1.5),
            (300, 4.0),
            (-400, 1.25),
            (-2000, 1.05),
            (1200, 13.0),
            (-105, 1 + 100 / 105),
            (10000, 101.0),
        ],
    )
    def test_american_to_decimal_hand_worked(self, american: int, decimal: float) -> None:
        assert american_to_decimal(american) == pytest.approx(decimal, rel=1e-15)

    def test_american_to_decimal_matches_exact_rationals(self) -> None:
        for american in _american_grid():
            expected = float(_exact_decimal(american))
            assert american_to_decimal(american) == pytest.approx(expected, rel=1e-15)

    def test_american_to_prob_matches_exact_rationals(self) -> None:
        """Implied probability is |a|/(|a|+100) for favourites, 100/(a+100) for dogs."""
        for american in _american_grid():
            if american < 0:
                expected = Fraction(-american, -american + 100)
            else:
                expected = Fraction(100, american + 100)
            assert american_to_prob(american) == pytest.approx(float(expected), rel=1e-15)

    @pytest.mark.parametrize(
        "american, prob",
        [(-110, 110 / 210), (150, 0.4), (100, 0.5), (-100, 0.5), (-300, 0.75), (900, 0.1)],
    )
    def test_american_to_prob_hand_worked(self, american: int, prob: float) -> None:
        assert american_to_prob(american) == pytest.approx(prob, rel=1e-15)

    def test_even_money_is_continuous(self) -> None:
        """+100 and -100 are the same price and the scale has no jump there."""
        assert american_to_decimal(100) == american_to_decimal(-100) == 2.0
        assert american_to_decimal(-100.0001) == pytest.approx(2.0, abs=1e-5)
        assert american_to_decimal(100.0001) == pytest.approx(2.0, abs=1e-5)

    def test_decimal_two_maps_to_plus_100(self) -> None:
        assert decimal_to_american(2.0) == 100.0
        assert decimal_to_american(2) == 100.0

    def test_american_round_trip_over_grid(self) -> None:
        for american in _american_grid():
            back = decimal_to_american(american_to_decimal(american))
            expected = 100 if american == -100 else american  # -100 normalises to +100
            assert back == pytest.approx(expected, rel=1e-12)

    def test_american_round_trip_non_integer(self) -> None:
        for american in (-100.5, -137.25, -999.9, 100.5, 104.75, 1234.56):
            assert decimal_to_american(american_to_decimal(american)) == pytest.approx(
                american, rel=1e-12
            )

    def test_decimal_round_trip_over_grid(self) -> None:
        for decimal in np.geomspace(1.0005, 5000.0, 600):
            back = american_to_decimal(decimal_to_american(float(decimal)))
            assert back == pytest.approx(float(decimal), rel=1e-12)

    def test_probability_round_trips_over_grid(self) -> None:
        for prob in np.linspace(0.0005, 0.9995, 1999):
            p = float(prob)
            assert decimal_to_prob(prob_to_decimal(p)) == pytest.approx(p, rel=1e-13)
            assert american_to_prob(prob_to_american(p)) == pytest.approx(p, rel=1e-12)
            assert prob_to_decimal(p) == pytest.approx(1.0 / p, rel=1e-15)

    def test_prob_to_american_sign_convention(self) -> None:
        """Favourites (p > 1/2) are negative, underdogs positive, 1/2 is +100."""
        assert prob_to_american(0.5) == 100.0
        for prob in np.linspace(0.01, 0.99, 99):
            american = prob_to_american(float(prob))
            assert abs(american) >= 100.0 - 1e-9
            if prob > 0.5 + 1e-12:
                assert american < 0
            elif prob < 0.5 - 1e-12:
                assert american > 0

    @pytest.mark.parametrize(
        "prob, american",
        [(0.75, -300.0), (0.25, 300.0), (0.6, -150.0), (0.4, 150.0), (0.8, -400.0), (0.1, 900.0)],
    )
    def test_prob_to_american_hand_worked(self, prob: float, american: float) -> None:
        assert prob_to_american(prob) == pytest.approx(american, rel=1e-12)

    def test_decimal_is_monotone_in_probability(self) -> None:
        decimals = [prob_to_decimal(float(p)) for p in np.linspace(0.01, 0.99, 99)]
        assert all(a > b for a, b in zip(decimals, decimals[1:]))

    def test_american_scale_orders_prices_within_each_sign(self) -> None:
        dogs = [american_to_decimal(a) for a in range(100, 2001, 25)]
        favourites = [american_to_decimal(a) for a in range(-100, -2001, -25)]
        assert all(a < b for a, b in zip(dogs, dogs[1:]))
        assert all(a > b for a, b in zip(favourites, favourites[1:]))

    def test_numeric_input_types(self) -> None:
        assert american_to_decimal(np.int64(-110)) == pytest.approx(21 / 11)
        assert american_to_decimal(np.float64(150.0)) == 2.5
        assert decimal_to_prob(np.float32(2.0)) == 0.5
        assert decimal_to_american(Fraction(5, 2)) == 150.0
        assert isinstance(american_to_decimal(150), float)
        assert isinstance(decimal_to_american(2.5), float)


class TestConversionValidation:
    @pytest.mark.parametrize("american", [0, 1, -1, 50, -50, 99, -99, 99.999, -99.999, NAN, INF, -INF])
    def test_american_inside_dead_zone_or_non_finite_raises(self, american: float) -> None:
        with pytest.raises(ValueError):
            american_to_decimal(american)
        with pytest.raises(ValueError):
            american_to_prob(american)

    @pytest.mark.parametrize("decimal", [1.0, 0.999, 0.0, -2.0, NAN, INF, -INF])
    def test_decimal_to_american_rejects_invalid(self, decimal: float) -> None:
        with pytest.raises(ValueError):
            decimal_to_american(decimal)

    @pytest.mark.parametrize("decimal", [1.0, 0.999, 0.0, -2.0, NAN, INF, -INF])
    def test_decimal_to_prob_rejects_invalid(self, decimal: float) -> None:
        """Infinite odds would give an implied probability of 0, which no de-vig accepts."""
        with pytest.raises(ValueError):
            decimal_to_prob(decimal)

    @pytest.mark.parametrize("prob", [0.0, 1.0, -0.1, 1.1, NAN, INF])
    def test_prob_conversions_reject_outside_open_interval(self, prob: float) -> None:
        with pytest.raises(ValueError):
            prob_to_decimal(prob)
        with pytest.raises(ValueError):
            prob_to_american(prob)


# --------------------------------------------------------------------------
# breakeven_prob
# --------------------------------------------------------------------------
class TestBreakeven:
    def test_without_push_is_implied_probability(self) -> None:
        for decimal in (1.01, 1.5, 21 / 11, 2.0, 3.5, 101.0):
            assert breakeven_prob(decimal) == pytest.approx(1.0 / decimal, rel=1e-15)
            assert breakeven_prob(decimal) == pytest.approx(decimal_to_prob(decimal), rel=1e-15)

    @pytest.mark.parametrize(
        "decimal, push, expected",
        [(2.0, 0.1, 0.45), (2.0, 0.5, 0.25), (1.25, 0.2, 0.64), (4.0, 0.08, 0.23)],
    )
    def test_with_push_hand_worked(self, decimal: float, push: float, expected: float) -> None:
        assert breakeven_prob(decimal, push) == pytest.approx(expected, rel=1e-14)

    def test_expected_value_is_zero_at_breakeven(self) -> None:
        """EV = p*(d-1) - (1-p-push) computed directly must vanish."""
        for decimal in (1.2, 1.91, 2.0, 2.6, 7.5):
            for push in (0.0, 0.03, 0.12, 0.4, 0.9):
                p = breakeven_prob(decimal, push)
                ev = p * (decimal - 1.0) - (1.0 - p - push)
                assert ev == pytest.approx(0.0, abs=1e-14)
                assert 0.0 < p <= 1.0 - push

    def test_push_lowers_the_breakeven(self) -> None:
        values = [breakeven_prob(1.91, push) for push in np.linspace(0.0, 0.9, 10)]
        assert all(a > b for a, b in zip(values, values[1:]))

    @pytest.mark.parametrize(
        "decimal, push",
        [(1.0, 0.0), (0.5, 0.0), (NAN, 0.0), (INF, 0.0), (2.0, 1.0), (2.0, -0.01), (2.0, 1.5), (2.0, NAN)],
    )
    def test_invalid_inputs_raise(self, decimal: float, push: float) -> None:
        with pytest.raises(ValueError):
            breakeven_prob(decimal, push)


# --------------------------------------------------------------------------
# format_american
# --------------------------------------------------------------------------
class TestFormatAmerican:
    @pytest.mark.parametrize(
        "american, text",
        [
            (105, "+105"),
            (-110, "-110"),
            (100, "+100"),
            (-100, "-100"),
            (150.0, "+150"),
            (-2000, "-2000"),
            (1200, "+1200"),
            (1000000, "+1000000"),
            (np.float64(-137.0), "-137"),
        ],
    )
    def test_integers_render_with_explicit_sign(self, american: float, text: str) -> None:
        assert format_american(american) == text

    @pytest.mark.parametrize(
        "american, text",
        [
            (104.4, "+104"),
            (104.6, "+105"),
            (-110.4, "-110"),
            (-110.6, "-111"),
            (149.99999999, "+150"),
            (-149.99999999, "-150"),
            (100.49, "+100"),
            (-100.49, "-100"),
            (100.51, "+101"),
            (-100.51, "-101"),
        ],
    )
    def test_rounds_to_nearest_integer(self, american: float, text: str) -> None:
        assert format_american(american) == text

    @pytest.mark.parametrize("american", [104.5, 105.5, -104.5, -105.5, 100.5, -100.5, 250.5])
    def test_exact_ties_round_to_a_neighbour(self, american: float) -> None:
        """Tie-breaking is unspecified (the implementation uses banker's rounding).

        ``104.5 -> "+104"`` but ``105.5 -> "+106"``; either neighbour is an
        acceptable rendering, anything else is not.
        """
        neighbours = {math.floor(american), math.ceil(american)}
        assert int(format_american(american)) in neighbours

    @pytest.mark.parametrize(
        "american, text",
        [(99.6, "+100"), (99.5, "+100"), (-99.6, "-100"), (-99.5, "-100")],
    )
    def test_values_just_inside_the_dead_zone_snap_to_even_money(
        self, american: float, text: str
    ) -> None:
        assert format_american(american) == text

    def test_never_renders_a_magnitude_below_100(self) -> None:
        for prob in np.linspace(0.001, 0.999, 999):
            text = format_american(prob_to_american(float(prob)))
            assert text[0] in "+-"
            assert abs(int(text)) >= 100

    def test_round_trips_through_parse_odds(self) -> None:
        for american in _american_grid():
            text = format_american(american)
            assert parse_odds(text) == pytest.approx(float(_exact_decimal(american)), rel=1e-15)

    def test_fair_price_rendering_of_common_probabilities(self) -> None:
        assert format_american(prob_to_american(0.5)) == "+100"
        assert format_american(prob_to_american(0.6)) == "-150"
        assert format_american(prob_to_american(0.4)) == "+150"
        assert format_american(prob_to_american(110 / 210)) == "-110"

    @pytest.mark.parametrize("american", [NAN, INF, -INF])
    def test_non_finite_raises(self, american: float) -> None:
        with pytest.raises(ValueError):
            format_american(american)

    @pytest.mark.parametrize("american", [0, 50, -50, 2.5, 1.91, -1.5, 99.4, -99.4, 99.49])
    def test_values_that_are_not_american_odds_raise(self, american: float) -> None:
        """A decimal price passed by mistake must not be rendered as '+100'."""
        with pytest.raises(ValueError):
            format_american(american)


# --------------------------------------------------------------------------
# parse_odds
# --------------------------------------------------------------------------
class TestParseOddsAuto:
    @pytest.mark.parametrize(
        "text, decimal",
        [
            ("+150", 2.5),
            ("-110", 21 / 11),
            ("+100", 2.0),
            ("-100", 2.0),
            ("150", 2.5),
            ("100", 2.0),
            ("1000", 11.0),
            ("-2000", 1.05),
            ("+0150", 2.5),
            ("-110.0", 21 / 11),
            ("150.0", 2.5),
        ],
    )
    def test_american_strings(self, text: str, decimal: float) -> None:
        assert parse_odds(text) == pytest.approx(decimal, rel=1e-15)

    @pytest.mark.parametrize(
        "text, decimal",
        [("1.91", 1.91), ("2.5", 2.5), ("2", 2.0), ("1.0001", 1.0001), ("99.99", 99.99), ("15", 15.0)],
    )
    def test_decimal_strings(self, text: str, decimal: float) -> None:
        assert parse_odds(text) == pytest.approx(decimal, rel=1e-15)

    @pytest.mark.parametrize(
        "text, decimal",
        [
            ("5/2", 3.5),
            ("1/1", 2.0),
            ("10/11", 21 / 11),
            ("1/2", 1.5),
            ("100/1", 101.0),
            ("5 / 2", 3.5),
            (" 11/10 ", 2.1),
        ],
    )
    def test_fractional_strings(self, text: str, decimal: float) -> None:
        assert parse_odds(text) == pytest.approx(decimal, rel=1e-15)

    @pytest.mark.parametrize("text", ["EVEN", "even", "Even", "evens", "EVENS", "ev", "EV", "pk", "PK", "pick", " Even "])
    def test_even_money_words(self, text: str) -> None:
        assert parse_odds(text) == 2.0

    @pytest.mark.parametrize(
        "text, decimal",
        [
            ("−110", 21 / 11),  # U+2212 MINUS SIGN, as copied from a web page
            (" −250 ", 1.4),
            ("  +150  ", 2.5),
            ("\t-110\n", 21 / 11),
            (" +200 ", 3.0),  # non-breaking spaces
        ],
    )
    def test_unicode_minus_and_whitespace(self, text: str, decimal: float) -> None:
        assert parse_odds(text) == pytest.approx(decimal, rel=1e-15)

    @pytest.mark.parametrize("text", ["–110", "—110", "‒110"])
    def test_other_dashes_never_yield_a_wrong_price(self, text: str) -> None:
        """En/em/figure dashes are not documented: parse as -110 or refuse."""
        try:
            value = parse_odds(text)
        except ValueError:
            return
        assert value == pytest.approx(21 / 11, rel=1e-15)

    @pytest.mark.parametrize(
        "value, decimal",
        [
            (150, 2.5),
            (-110, 21 / 11),
            (100, 2.0),
            (-100, 2.0),
            (1.91, 1.91),
            (2, 2.0),
            (2.5, 2.5),
            (99.99, 99.99),
            (np.int64(-110), 21 / 11),
            (np.float64(1.91), 1.91),
            (np.float32(2.5), 2.5),
            (np.int32(250), 3.5),
            (Decimal("1.91"), 1.91),
            (Decimal("-110"), 21 / 11),
            (Fraction(5, 2), 2.5),
        ],
    )
    def test_numeric_types(self, value: Union[int, float], decimal: float) -> None:
        result = parse_odds(value)
        assert result == pytest.approx(decimal, rel=1e-15)
        assert isinstance(result, float)

    def test_magnitude_100_is_the_american_boundary(self) -> None:
        """Documented rule: |x| >= 100 is American, (1, 100) is decimal."""
        assert parse_odds("99.99") == 99.99
        assert parse_odds(99.99) == 99.99
        assert parse_odds("100") == 2.0
        assert parse_odds(100.0) == 2.0
        assert parse_odds(101.0) == pytest.approx(2.01)

    def test_decimal_that_looks_american_needs_an_explicit_format(self) -> None:
        """A 149/1 long shot quoted as decimal 150.0 is read as +150 in auto mode."""
        assert parse_odds("150.0") == 2.5
        assert parse_odds(150.0) == 2.5
        assert parse_odds("150.0", fmt="decimal") == 150.0
        assert parse_odds(150.0, fmt="decimal") == 150.0

    def test_leading_sign_forces_american(self) -> None:
        """A signed number is American even when its magnitude looks decimal."""
        with pytest.raises(ValueError):
            parse_odds("+2.5")
        with pytest.raises(ValueError):
            parse_odds("-1.91")
        with pytest.raises(ValueError):
            parse_odds(-1.91)

    @pytest.mark.parametrize(
        "value",
        [
            "",
            "   ",
            "abc",
            "even money",
            "+EVEN",
            "+ 150",
            "1,000",
            "+1,000",
            "-5/2",
            "5/-2",
            "5/2/1",
            "1.5/2",
            "0/1",
            "5/0",
            "0/0",
            "1",
            "1.0",
            "0.5",
            "0",
            "-0",
            "+50",
            "-99",
            "nan",
            "inf",
            "-inf",
            "--110",
            "110-",
        ],
    )
    def test_unparseable_strings_raise(self, value: str) -> None:
        with pytest.raises(ValueError):
            parse_odds(value)

    @pytest.mark.parametrize("value", [1, 1.0, 0, 0.5, -0.0, -1, -50, -99.5, NAN, INF, -INF, True, False])
    def test_invalid_numbers_raise(self, value: float) -> None:
        with pytest.raises(ValueError):
            parse_odds(value)

    @pytest.mark.parametrize("value", [None, [150], {"price": 150}])
    def test_non_numeric_objects_raise(self, value: object) -> None:
        with pytest.raises((TypeError, ValueError)):
            parse_odds(value)  # type: ignore[arg-type]

    def test_agrees_with_conversions_over_american_grid(self) -> None:
        for american in _american_grid():
            expected = float(_exact_decimal(american))
            assert parse_odds(american) == pytest.approx(expected, rel=1e-15)
            assert parse_odds(str(american)) == pytest.approx(expected, rel=1e-15)
            assert parse_odds(f"{american:+d}") == pytest.approx(expected, rel=1e-15)


class TestParseOddsExplicitFormat:
    @pytest.mark.parametrize(
        "value, fmt, decimal",
        [
            ("150", "american", 2.5),
            ("+150", "american", 2.5),
            (-110, "american", 21 / 11),
            ("150", "decimal", 150.0),
            ("1.5", "decimal", 1.5),
            (2.5, "decimal", 2.5),
            ("+2.5", "decimal", 2.5),
            ("5/2", "fractional", 3.5),
            ("2.5", "fractional", 3.5),
            (2.5, "fractional", 3.5),
            (0.5, "fractional", 1.5),
            ("even", "american", 2.0),
            ("EVEN", "decimal", 2.0),
            ("evens", "fractional", 2.0),
        ],
    )
    def test_valid(self, value: Union[str, float], fmt: str, decimal: float) -> None:
        assert parse_odds(value, fmt=fmt) == pytest.approx(decimal, rel=1e-15)

    @pytest.mark.parametrize(
        "value, fmt",
        [
            ("50", "american"),
            ("2.0", "american"),
            (1.91, "american"),
            ("5/2", "american"),
            ("-110", "decimal"),
            ("1", "decimal"),
            ("1.0", "decimal"),
            (0.5, "decimal"),
            ("5/2", "decimal"),
            ("nan", "decimal"),
            (NAN, "decimal"),
            ("0", "fractional"),
            ("-1", "fractional"),
            ("0/5", "fractional"),
            ("nan", "fractional"),
            ("2.5", "bogus"),
            (150, "bogus"),
            ("5/2", "bogus"),
            ("even", "bogus"),   # the format is validated before even-money words
            ("EVEN", ""),
            ("2.5", ""),
            ("2.5", "AUTO"),
            ("inf", "decimal"),
            (INF, "decimal"),
            ("-inf", "decimal"),
            ("inf", "fractional"),
            (INF, "fractional"),
            ("inf", "american"),
        ],
    )
    def test_invalid(self, value: Union[str, float], fmt: str) -> None:
        with pytest.raises(ValueError):
            parse_odds(value, fmt=fmt)

    @pytest.mark.parametrize("fmt", ["auto", "american", "decimal", "fractional"])
    def test_result_is_always_a_finite_price_above_one(self, fmt: str) -> None:
        for value in ("+150", "-110", "150", "1.91", "2.5", "5/2", "even", 250, 3.5, "1e3", "101"):
            try:
                price = parse_odds(value, fmt=fmt)
            except ValueError:
                continue
            assert math.isfinite(price) and price > 1.0
