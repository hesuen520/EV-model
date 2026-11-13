"""Reading a stored result row: missing values, and the result after
regulation time.

A row that comes out of a table holding several leagues carries ``NaN`` where
a league has no value (a soccer match has no overtime flag; an NHL game
does).  ``NaN`` is truthy in Python, so every reader goes through
:func:`clean_row` and :func:`flag` instead of ``bool(...)`` or ``... or ...``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple

__all__ = ["clean_row", "flag", "regulation_result", "OVERTIME_SPORTS"]

#: Sports whose final score can include overtime.  For these a three-way
#: (regulation) result cannot be read off the final score unless the row says
#: whether overtime was played.
OVERTIME_SPORTS = frozenset({"hockey", "basketball", "football", "baseball"})


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    try:
        differs = value != value          # true for NaN-like scalars (numpy NaN, NaT)
    except (TypeError, ValueError):
        return False
    try:
        return bool(differs)
    except (TypeError, ValueError):
        # pandas.NA compares to NA, whose truth value is undefined; arrays are not scalars.
        return type(value).__name__ == "NAType"


def clean_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """A plain dict with every missing value (``NaN``, ``NaT``, ``pd.NA``) as ``None``."""
    return {key: (None if _missing(value) else value) for key, value in dict(row).items()}


def flag(value: Any) -> Optional[bool]:
    """A stored 0/1 flag: ``True``, ``False`` or ``None`` when it is not on file."""
    if _missing(value):
        return None
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return None


def regulation_result(ev: Mapping[str, Any]) -> Tuple[Optional[str], str]:
    """Who led after regulation time: the home ID, the away ID or ``"draw"``.

    Returns ``(None, reason)`` when the row cannot decide it.  Order of
    evidence: a regulation score on file; a flag saying the game went to
    overtime or a shootout (then it was level); the final score, but only
    where that is known to be the regulation score - either the flags say no
    overtime was played, or the sport has none recorded in the final score.
    """
    ev = clean_row(ev)
    rh, ra = ev.get("reg_home_score"), ev.get("reg_away_score")
    if rh is not None and ra is not None:
        return ("draw" if rh == ra else (ev["home_id"] if rh > ra else ev["away_id"])), "regulation score on file"
    overtime, shootout = flag(ev.get("went_overtime")), flag(ev.get("went_shootout"))
    if overtime or shootout:
        return "draw", "the game went beyond regulation, so it was level after it"
    hs, as_ = ev.get("home_score"), ev.get("away_score")
    if hs is None or as_ is None:
        return None, "no score on file"
    if overtime is None and shootout is None and ev.get("sport") in OVERTIME_SPORTS:
        return None, "the final score may include overtime and the row does not say whether it was played"
    return ("draw" if hs == as_ else (ev["home_id"] if hs > as_ else ev["away_id"])), "final score"
