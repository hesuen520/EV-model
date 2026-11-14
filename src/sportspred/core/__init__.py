"""Shared primitives: time handling, stable IDs, enumerations."""

from .enums import (
    Availability,
    CheckState,
    CoverageStatus,
    EventStatus,
    OutcomeType,
    RecStatus,
    TimePrecision,
)
from .ids import norm_text, slug, stable_id
from .timeutil import (
    HST,
    UTC,
    NaiveDatetimeError,
    age_hours,
    fmt_hst,
    hst_date,
    hst_day_bounds_utc,
    iso_z,
    parse_ts,
    to_hst,
    utcnow,
)

__all__ = [
    "Availability",
    "CheckState",
    "CoverageStatus",
    "EventStatus",
    "OutcomeType",
    "RecStatus",
    "TimePrecision",
    "norm_text",
    "slug",
    "stable_id",
    "HST",
    "UTC",
    "NaiveDatetimeError",
    "age_hours",
    "fmt_hst",
    "hst_date",
    "hst_day_bounds_utc",
    "iso_z",
    "parse_ts",
    "to_hst",
    "utcnow",
]
