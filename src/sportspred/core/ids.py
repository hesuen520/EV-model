"""Stable identifiers and text normalisation for entity matching."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any

__all__ = ["norm_text", "slug", "stable_id", "person_key", "loose_person_key", "surname_key"]

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


_GENERATION = {"jr", "sr", "ii", "iii", "iv", "v"}


def person_key(value: Any) -> str:
    """Comparison key for a person's name: :func:`norm_text` without a trailing
    generational suffix, so ``"Jimmy Butler"`` and ``"Jimmy Butler III"`` (two
    sources' spellings of one player) compare equal.  Only ever used to match a
    name *within one team*, where two different people sharing it is not a
    practical risk."""
    words = norm_text(value).split()
    while len(words) > 2 and words[-1] in _GENERATION:
        words.pop()
    return " ".join(words)


def loose_person_key(value: Any) -> str:
    """:func:`person_key` with runs of single letters joined, so ``"O.G.
    Anunoby"`` and ``"OG Anunoby"`` compare equal.  A second attempt after the
    exact key has failed - never the first."""
    out: list = []
    run = False
    for word in person_key(value).split():
        single = len(word) == 1
        if single and run:
            out[-1] += word
        else:
            out.append(word)
        run = single
    return " ".join(out)


def surname_key(value: Any) -> str:
    """Last word of :func:`person_key` - enough to notice that a statement may
    be about a known player, never enough to say that it is."""
    words = person_key(value).split()
    return words[-1] if len(words) > 1 else ""


def norm_text(value: Any) -> str:
    """Accent-free, lower-case, single-spaced alphanumeric form of ``value``.

    ``"Luka Dončić"`` and ``"LUKA  DONCIC"`` both become ``"luka doncic"``.
    Used as the comparison key for names; never shown to a person.
    """
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    # Characters NFKD does not decompose but which commonly appear in names.
    text = (
        text.replace("ø", "o").replace("Ø", "O").replace("đ", "d").replace("Đ", "D")
        .replace("ł", "l").replace("Ł", "L").replace("ß", "ss").replace("æ", "ae")
        .replace("Æ", "AE").replace("œ", "oe").replace("Œ", "OE").replace("&", " and ")
    )
    return _NON_ALNUM.sub(" ", text.lower()).strip()


def slug(value: Any) -> str:
    """URL/ID-safe form: ``norm_text`` joined with hyphens."""
    return norm_text(value).replace(" ", "-")


def stable_id(prefix: str, *parts: Any) -> str:
    """Deterministic ID from normalised parts: same inputs, same ID, forever.

    The ID is ``<prefix>_<16 hex chars>`` of a SHA-1 over the normalised
    parts, so it survives re-ingestion, restarts and different machines.
    """
    if not prefix or not parts:
        raise ValueError("stable_id needs a prefix and at least one part")
    key = "|".join(norm_text(p) for p in parts)
    if not key.replace("|", ""):
        raise ValueError("stable_id parts are all empty")
    return f"{prefix}_{hashlib.sha1(key.encode('utf-8')).hexdigest()[:16]}"
