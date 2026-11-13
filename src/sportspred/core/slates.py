"""Which Pacific/Honolulu slate a recorded prediction belongs to.

A slate is one HST calendar day.  An event whose source gives only a date is
stored at 00:00Z of that date, which is the *previous* afternoon in Honolulu;
such an event belongs to the slate of its own calendar date, not to the day
before.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any, Mapping

import pandas as pd

from .timeutil import hst_date

__all__ = ["predicted_date_only", "slate_date_of"]


def _loads(value: Any, default: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value) if isinstance(value, str) and value else default
    except ValueError:
        return default


def predicted_date_only(pred: Mapping[str, Any]) -> bool:
    """Was the event's start only a calendar date when this prediction was made?

    Recorded as ``display.date_only``; older rows are recognised by the
    ``start_time_published`` check the engine attaches to date-only events.
    """
    display = _loads(pred.get("display_json", pred.get("display")), {})
    if display.get("date_only"):
        return True
    checks = _loads(pred.get("checks_json", pred.get("checks")), [])
    return any(isinstance(c, dict) and c.get("key") == "start_time_published" for c in checks)


def slate_date_of(pred: Mapping[str, Any]) -> date:
    """HST slate of a prediction: the event's own date when only a date was
    published, otherwise the Honolulu calendar day of its start."""
    start = pd.Timestamp(pred["event_start_utc"])
    if predicted_date_only(pred):
        return start.date()
    return hst_date(start.to_pydatetime() if start.tzinfo else start.tz_localize("UTC").to_pydatetime())
