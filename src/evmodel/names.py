"""Player-name normalisation and matching.

Odds feeds and stats feeds spell the same player differently ("Nikola Jokić"
vs "Nikola Jokic", "Michael Porter Jr." vs "Michael Porter", "C.J. Stroud" vs
"CJ Stroud", "A. St. Brown" vs "Amon-Ra St. Brown").  Everything in the
package joins on :func:`player_key`, a conservative canonical form, and
:class:`NameMatcher` resolves the leftovers.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import pandas as pd

__all__ = ["player_key", "NameMatcher"]

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
# After a comma a lone "V" is far more often a first initial ("Williams, V")
# than a generational suffix, so it is not treated as one there.
_COMMA_SUFFIXES = _SUFFIXES - {"v"}
# Hyphens and apostrophes (ASCII and typographic) join the letters around them.
_JOINERS = "-'`" + "".join(map(chr, (0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015, 0x2018, 0x2019, 0x02BC, 0x00B4, 0x2032)))
_PUNCT = re.compile(r"[^a-z0-9\s]")
_SPACE = re.compile(r"\s+")

# Characters that NFKD does not decompose into ASCII + combining mark.
_TRANSLATE = str.maketrans(
    {
        "ø": "o", "Ø": "O", "đ": "d", "Đ": "D", "ł": "l", "Ł": "L",
        "ß": "ss", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE",
        "þ": "th", "Þ": "Th", "ð": "d", "Ð": "D", "ı": "i",
    }
)


def _ascii(text: str) -> str:
    text = text.translate(_TRANSLATE)
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def player_key(name: object) -> str:
    """Canonical join key for a player name.

    Lower-cases, strips accents, drops punctuation (so ``C.J.`` -> ``cj`` and
    ``St. Brown`` -> ``st brown``; hyphens and apostrophes, ASCII or
    typographic, are removed without leaving a space so ``Amon-Ra`` ->
    ``amonra`` and ``O'Neale`` -> ``oneale``), reorders ``Last, First`` and
    removes generational suffixes (Jr., Sr., II, III, IV, V).  Missing values
    (``None``, NaN, ``pd.NA``) give the empty string.

    >>> player_key("Nikola Jokić")
    'nikola jokic'
    >>> player_key("Michael Porter Jr.")
    'michael porter'
    >>> player_key("C.J. Stroud") == player_key("CJ Stroud")
    True
    """
    if name is None:
        return ""
    try:
        if pd.isna(name):
            return ""
    except (TypeError, ValueError):
        pass  # not a scalar: fall through to its text form
    text = str(name)
    for ch in _JOINERS:  # before accent stripping, which would split U+00B4
        text = text.replace(ch, "")
    text = _ascii(text).lower().strip()
    if not text or text == "nan":
        return ""
    parts = [part.strip() for part in text.split(",")]
    # "Porter, Michael, Jr." / "Michael Porter, Jr.": drop a trailing suffix part.
    if len(parts) > 1 and parts[-1].replace(".", "") in _COMMA_SUFFIXES:
        parts = parts[:-1]
    if len(parts) == 2 and parts[0] and parts[1]:
        # "Last, First" -> "First Last"; "Porter, Michael Jr." loses its suffix.
        last, first = parts
        first_tokens = first.replace(".", "").split()
        while len(first_tokens) > 1 and first_tokens[-1] in _SUFFIXES:
            first_tokens.pop()
        text = f"{' '.join(first_tokens)} {last}"
    else:
        text = " ".join(parts)
    text = text.replace(".", "")
    text = _PUNCT.sub(" ", text)
    tokens = [t for t in _SPACE.split(text) if t]
    while len(tokens) > 1 and tokens[-1] in _SUFFIXES:
        tokens.pop()
    # "C. J. Stroud" -> "cj stroud": merge a leading run of single initials.
    run = 0
    while run < len(tokens) - 1 and len(tokens[run]) == 1 and tokens[run].isalpha():
        run += 1
    if run > 1:
        tokens = ["".join(tokens[:run])] + tokens[run:]
    return " ".join(tokens)


def _split(key: str) -> tuple:
    tokens = key.split()
    if not tokens:
        return "", ""
    return tokens[0], " ".join(tokens[1:])


def _team_code(team: object) -> str:
    """Upper-cased team hint, or ``""`` when it is missing (None, NaN, blank)."""
    if team is None:
        return ""
    try:
        if pd.isna(team):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(team).strip().upper()
    return "" if text in {"", "NAN", "NONE", "<NA>"} else text


class NameMatcher:
    """Resolve names from one feed against a roster of known players.

    Matching is attempted in order of decreasing strictness:

    1. exact :func:`player_key` match;
    2. explicit alias supplied by the caller;
    3. same surname and compatible first name (initial or prefix, e.g.
       ``"A. St. Brown"`` or ``"Nic Claxton"`` vs ``"Nicolas Claxton"``) when
       that identifies exactly one candidate.

    Nothing fuzzier is attempted: names that merely look alike are different
    people far too often (Jalen and Jaden McDaniels, Avery Williams and Avery
    Williamson).  Supply ``aliases`` for genuine spelling differences.

    A team hint, when given, vetoes candidates known to play elsewhere.
    Ambiguous or unknown names return ``None`` rather than a guess: a wrong
    match silently corrupts an edge calculation, a missing one only skips a
    prop.
    """

    def __init__(
        self,
        names: Iterable[str],
        teams: Optional[Iterable[Optional[str]]] = None,
        aliases: Optional[Dict[str, str]] = None,
    ) -> None:
        self._keys: Dict[str, str] = {}
        self._teams: Dict[str, set] = defaultdict(set)
        self._by_last: Dict[str, List[str]] = defaultdict(list)
        names = list(names)
        team_list = list(teams) if teams is not None else [None] * len(names)
        if len(team_list) != len(names):
            raise ValueError("names and teams must have the same length")
        for name, team in zip(names, team_list):
            key = player_key(name)
            if not key:
                continue
            if key not in self._keys:
                self._keys[key] = str(name)
                self._by_last[_split(key)[1]].append(key)
            code = _team_code(team)
            if code:
                self._teams[key].add(code)
        self._aliases = {player_key(k): player_key(v) for k, v in (aliases or {}).items()}

    def __contains__(self, name: str) -> bool:
        return player_key(name) in self._keys

    def display_name(self, key: str) -> Optional[str]:
        """Original spelling registered for a canonical key."""
        return self._keys.get(key)

    def _team_ok(self, key: str, team: object) -> bool:
        """False only when a team hint is given and contradicts the roster."""
        code = _team_code(team)
        if not code:
            return True
        known = self._teams.get(key)
        return not known or code in known

    def match(self, name: str, team: Optional[str] = None) -> Optional[str]:
        """Return the canonical key of the roster player matching ``name``.

        ``team`` is an optional hint (missing values are ignored); a
        candidate whose known team(s) do not include it is never returned.
        """
        key = player_key(name)
        if not key:
            return None
        if key in self._keys:
            return key if self._team_ok(key, team) else None
        alias = self._aliases.get(key)
        if alias and alias in self._keys:
            return alias if self._team_ok(alias, team) else None

        first, last = _split(key)
        if not last:
            return None
        same_last = [c for c in self._by_last.get(last, []) if self._team_ok(c, team)]
        compatible = [
            c for c in same_last if _split(c)[0].startswith(first) or first.startswith(_split(c)[0])
        ]
        if len(compatible) == 1:
            return compatible[0]
        return None  # unknown, or ambiguous (two "J. Williams" on one team)
