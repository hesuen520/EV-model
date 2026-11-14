"""Market keys.

A model is validated and trained for a *model market*: ``winner``,
``total``, ``player_prop:points``.  A prediction about one player is recorded
under a key that also names him - ``player_prop:points:<player id>`` - so
that every (event, market) pair in the ledger is one selection.  The helpers
here go back from the ledger key to the model market wherever results are
grouped by model (calibration bands, monitoring, display).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

__all__ = ["model_market", "market_family", "market_subject", "market_text", "SUBJECT_FAMILIES"]


def market_family(market: str) -> str:
    return str(market).split(":", 1)[0]


#: Families whose ledger keys name a subject: ``<family>:<detail>:<subject id>``.
SUBJECT_FAMILIES = ("player_prop", "finish")


def model_market(market: str) -> str:
    """``player_prop:points:1629029`` -> ``player_prop:points`` and
    ``finish:win:p_ab12`` -> ``finish:win``; any other key unchanged."""
    parts = str(market).split(":")
    return ":".join(parts[:2]) if parts[0] in SUBJECT_FAMILIES and len(parts) > 2 else str(market)


def market_subject(market: str) -> Optional[str]:
    """The subject (player, driver) of a per-subject ledger key, else ``None``."""
    parts = str(market).split(":")
    return parts[2] if parts[0] in SUBJECT_FAMILIES and len(parts) > 2 else None


def market_text(prediction: Mapping[str, Any]) -> str:
    """How a prediction's market is shown: the model market, and the line where
    the selection label does not already state it."""
    market = str(prediction["market"])
    line = prediction.get("line")
    if market_subject(market) is not None or line is None or line != line:
        return model_market(market)
    return f"{market} {line:+g}"
