"""Odds conversions.

Three representations are used throughout the package:

* **American** odds (``-110``, ``+150``): what US sportsbooks display.
* **Decimal** odds (``1.909``, ``2.50``): total return per 1 unit staked,
  stake included.  All internal maths uses decimal odds.
* **Implied probability** (``1 / decimal``): the break-even win probability
  of a price.  For a real sportsbook price this still contains the book's
  margin ("vig"); see :mod:`evmodel.devig` for removing it.
"""

from __future__ import annotations

import math
import re
from typing import Union

Number = Union[int, float]

__all__ = [
    "american_to_decimal",
    "decimal_to_american",
    "american_to_prob",
    "decimal_to_prob",
    "prob_to_decimal",
    "prob_to_american",
    "breakeven_prob",
    "format_american",
    "parse_odds",
]


def american_to_decimal(american: Number) -> float:
    """Convert American odds to decimal odds.

    ``+150 -> 2.5``, ``-110 -> 1.9090...``.  ``+100`` and ``-100`` are both
    even money (2.0).  Values strictly between -100 and +100 are not valid
    American odds and raise ``ValueError``.
    """
    a = float(american)
    if math.isnan(a) or math.isinf(a):
        raise ValueError(f"invalid American odds: {american!r}")
    if -100.0 < a < 100.0:
        raise ValueError(
            f"invalid American odds {american!r}: magnitude must be at least 100"
        )
    if a > 0:
        return 1.0 + a / 100.0
    return 1.0 + 100.0 / -a


def decimal_to_american(decimal: Number) -> float:
    """Convert decimal odds to American odds (unrounded float).

    Decimal odds of exactly 2.0 map to ``+100``.
    """
    d = float(decimal)
    if not d > 1.0 or math.isinf(d) or math.isnan(d):
        raise ValueError(f"decimal odds must be finite and > 1, got {decimal!r}")
    if d >= 2.0:
        return (d - 1.0) * 100.0
    return -100.0 / (d - 1.0)


def decimal_to_prob(decimal: Number) -> float:
    """Implied (break-even) probability of decimal odds: ``1 / decimal``."""
    d = float(decimal)
    if not d > 1.0 or not math.isfinite(d):
        raise ValueError(f"decimal odds must be finite and > 1, got {decimal!r}")
    return 1.0 / d


def american_to_prob(american: Number) -> float:
    """Implied (break-even) probability of American odds."""
    return decimal_to_prob(american_to_decimal(american))


def prob_to_decimal(prob: Number) -> float:
    """Fair decimal odds for a win probability in (0, 1)."""
    p = float(prob)
    if not 0.0 < p < 1.0:
        raise ValueError(f"probability must be in (0, 1), got {prob!r}")
    return 1.0 / p


def prob_to_american(prob: Number) -> float:
    """Fair American odds for a win probability in (0, 1) (unrounded)."""
    return decimal_to_american(prob_to_decimal(prob))


def breakeven_prob(decimal: Number, push_prob: float = 0.0) -> float:
    """Win probability at which a bet at ``decimal`` odds has zero EV.

    With a push probability ``push_prob`` (stake returned), EV per unit is
    ``p_win * (d - 1) - (1 - p_win - push_prob)``; solving EV = 0 gives
    ``p_win = (1 - push_prob) / d``.
    """
    d = float(decimal)
    if not d > 1.0 or not math.isfinite(d):
        raise ValueError(f"decimal odds must be finite and > 1, got {decimal!r}")
    if not 0.0 <= push_prob < 1.0:
        raise ValueError(f"push_prob must be in [0, 1), got {push_prob!r}")
    return (1.0 - push_prob) / d


def format_american(american: Number) -> str:
    """Render American odds the way sportsbooks do: ``+105``, ``-110``."""
    a = float(american)
    if not math.isfinite(a) or abs(a) < 99.5:
        raise ValueError(f"invalid American odds: {american!r}")
    rounded = int(round(a))
    if -100 < rounded < 100:
        # Only reachable by rounding from just inside the valid range.
        rounded = 100 if a > 0 else -100
    return f"+{rounded}" if rounded > 0 else f"{rounded}"


_EVEN = {"even", "evens", "ev", "pk", "pick"}
_FRACTION = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")


def parse_odds(value: Union[str, Number], fmt: str = "auto") -> float:
    """Parse a price in American, decimal or fractional form to decimal odds.

    ``fmt`` may be ``"american"``, ``"decimal"``, ``"fractional"`` or
    ``"auto"``.  In auto mode a leading ``+``/``-`` or a magnitude >= 100
    means American, ``a/b`` means fractional, anything else in (1, 100) is
    decimal.  ``"EVEN"`` is 2.0.
    """
    if fmt not in ("auto", "american", "decimal", "fractional"):
        raise ValueError(f"unknown odds format: {fmt!r}")
    if isinstance(value, str):
        text = value.strip().lower().replace("−", "-")  # unicode minus
        if text in _EVEN:
            return 2.0
        m = _FRACTION.match(text)
        if m and fmt in ("auto", "fractional"):
            num, den = int(m.group(1)), int(m.group(2))
            if den == 0 or num == 0:
                raise ValueError(f"invalid fractional odds: {value!r}")
            return 1.0 + num / den
        signed = text.startswith(("+", "-"))
        try:
            number = float(text)
        except ValueError as exc:
            raise ValueError(f"cannot parse odds: {value!r}") from exc
    else:
        number = float(value)
        signed = number < 0

    if not math.isfinite(number):
        raise ValueError(f"odds must be finite, got {value!r}")
    if fmt == "american":
        return american_to_decimal(number)
    if fmt == "decimal":
        if not number > 1.0:
            raise ValueError(f"decimal odds must be > 1, got {value!r}")
        return number
    if fmt == "fractional":
        if not number > 0:
            raise ValueError(f"fractional odds must be > 0, got {value!r}")
        return 1.0 + number

    if signed or abs(number) >= 100.0:
        return american_to_decimal(number)
    if number > 1.0:
        return number
    raise ValueError(f"cannot interpret {value!r} as odds")
