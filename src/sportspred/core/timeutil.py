"""Time handling: store UTC, display Pacific/Honolulu (HST, UTC-10, no DST).

Rules enforced here and relied on everywhere else:

* Every stored timestamp is an ISO-8601 string in UTC with a ``Z`` suffix and
  second precision (``2026-11-12T03:30:00Z``).  Strings in this one format
  sort lexicographically in time order, which SQLite comparisons rely on.
* Naive datetimes are rejected unless the caller states the zone they are in.
  Silently assuming a zone is how events end up on the wrong day.
* Display and "which slate is this game on" always use Pacific/Honolulu.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone, tzinfo
from typing import Optional, Tuple, Union
from zoneinfo import ZoneInfo

__all__ = [
    "UTC",
    "HST",
    "DISPLAY_TZ_NAME",
    "NaiveDatetimeError",
    "utcnow",
    "parse_ts",
    "iso_z",
    "to_hst",
    "fmt_hst",
    "hst_date",
    "hst_day_bounds_utc",
    "age_hours",
]

UTC = timezone.utc
DISPLAY_TZ_NAME = "Pacific/Honolulu"
HST = ZoneInfo(DISPLAY_TZ_NAME)

TimeLike = Union[str, datetime, date, "pandas.Timestamp"]  # noqa: F821 - doc only


class NaiveDatetimeError(ValueError):
    """A timestamp without a time zone was given and no zone was stated."""


def utcnow() -> datetime:
    """Current time, timezone-aware UTC, second precision."""
    return datetime.now(UTC).replace(microsecond=0)


def _zone(tz: Union[str, tzinfo]) -> tzinfo:
    return ZoneInfo(tz) if isinstance(tz, str) else tz


def parse_ts(value: TimeLike, *, assume_tz: Optional[Union[str, tzinfo]] = None) -> datetime:
    """Parse ``value`` into a timezone-aware UTC ``datetime``.

    Accepts aware datetimes, pandas Timestamps, ISO-8601 strings (``Z`` or a
    numeric offset) and, when ``assume_tz`` is given, naive datetimes, date
    objects and date-only strings (interpreted as midnight in that zone).

    Raises :class:`NaiveDatetimeError` for a zone-less value without
    ``assume_tz`` and ``ValueError`` for anything unparseable or missing.
    """
    if value is None:
        raise ValueError("timestamp is missing")
    # pandas.Timestamp is a datetime subclass; NaT is not usable.
    if hasattr(value, "to_pydatetime"):
        if value != value:  # NaT
            raise ValueError("timestamp is missing (NaT)")
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime.combine(value, time(0, 0))
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("timestamp is an empty string")
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"unparseable timestamp {value!r}") from exc
    else:
        raise ValueError(f"unsupported timestamp type {type(value).__name__}")

    if dt.tzinfo is None or dt.utcoffset() is None:
        if assume_tz is None:
            raise NaiveDatetimeError(
                f"timestamp {value!r} has no time zone; pass assume_tz=... to state one"
            )
        dt = dt.replace(tzinfo=_zone(assume_tz))
    return dt.astimezone(UTC)


def iso_z(value: TimeLike, *, assume_tz: Optional[Union[str, tzinfo]] = None) -> str:
    """Canonical storage form: ``YYYY-MM-DDTHH:MM:SSZ`` in UTC."""
    dt = parse_ts(value, assume_tz=assume_tz).replace(microsecond=0)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def to_hst(value: TimeLike) -> datetime:
    """Convert an aware timestamp to Pacific/Honolulu."""
    return parse_ts(value).astimezone(HST)


def fmt_hst(value: Optional[TimeLike], *, date_only: bool = False) -> str:
    """Human display in HST, e.g. ``Wed Nov 11, 2026 5:30 PM HST``.

    ``date_only=True`` is for events whose source gives a date but no start
    time: the calendar date is shown without a fabricated clock time.
    """
    if value is None or value != value:
        return "time unknown"
    if date_only:
        # A date-only event is stored at 00:00Z of its calendar date; showing
        # the HST conversion would move it to the previous day, so keep the
        # source's own calendar date and say that the time is not known.
        return parse_ts(value).strftime("%a %b %-d, %Y") + " (start time not published)"
    local = to_hst(value)
    return local.strftime("%a %b %-d, %Y %-I:%M %p") + " HST"


def hst_date(value: TimeLike) -> date:
    """The Pacific/Honolulu calendar date an instant falls on."""
    return to_hst(value).date()


def hst_day_bounds_utc(day: date) -> Tuple[datetime, datetime]:
    """UTC ``[start, end)`` of a Pacific/Honolulu calendar day."""
    start_local = datetime.combine(day, time(0, 0), tzinfo=HST)
    return start_local.astimezone(UTC), (start_local + timedelta(days=1)).astimezone(UTC)


def age_hours(then: TimeLike, now: Optional[TimeLike] = None) -> float:
    """Hours elapsed from ``then`` to ``now`` (default: the current time)."""
    end = utcnow() if now is None else parse_ts(now)
    return (end - parse_ts(then)).total_seconds() / 3600.0
