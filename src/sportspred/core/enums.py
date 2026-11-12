"""Enumerations shared by every module.  Values are the strings stored on disk."""

from __future__ import annotations

from enum import Enum

__all__ = [
    "EventStatus",
    "TimePrecision",
    "OutcomeType",
    "RecStatus",
    "CoverageStatus",
    "Availability",
    "CheckState",
    "COVERAGE_DEFINITIONS",
    "REC_STATUS_DEFINITIONS",
]


class _Str(str, Enum):
    def __str__(self) -> str:  # so f-strings and CSV writers show the value
        return self.value


class EventStatus(_Str):
    """Life-cycle state of an event.  New selections need ``SCHEDULED``."""

    SCHEDULED = "scheduled"   # upcoming, start time in the future
    LIVE = "live"             # in progress
    FINAL = "final"           # completed with an official result
    POSTPONED = "postponed"   # not played at the scheduled time; may be rescheduled
    CANCELED = "canceled"     # will not be played
    UNKNOWN = "unknown"       # source did not say; never treated as upcoming


class TimePrecision(_Str):
    """How precisely the source states an event's start."""

    DATETIME = "datetime"  # an actual start instant
    DATE = "date"          # calendar date only; stored at 00:00Z of that date
    PERIOD = "period"      # only a tournament/round window is known (e.g. tennis week)


class OutcomeType(_Str):
    BINARY = "binary"        # two competitors, exactly one wins (draws impossible or resolved)
    THREE_WAY = "three_way"  # home / draw / away
    MULTI = "multi"          # field of competitors with a finishing order
    OVER_UNDER = "over_under"  # a quantity versus a line; pushes possible on whole numbers


class RecStatus(_Str):
    """Recommendation status of a pregame prediction."""

    FINAL = "FINAL"
    PROVISIONAL = "PROVISIONAL"
    CONDITIONAL = "CONDITIONAL"
    PASS = "PASS"


REC_STATUS_DEFINITIONS = {
    RecStatus.FINAL: (
        "Every configured pregame check for this league and market is complete and "
        "fresh. The result itself remains uncertain."
    ),
    RecStatus.PROVISIONAL: (
        "A probability exists, but at least one required pregame check is missing or "
        "stale. Re-run after the named check is available."
    ),
    RecStatus.CONDITIONAL: (
        "The selection holds only if a stated condition is true (for example a named "
        "player's availability). It is not a finalized selection."
    ),
    RecStatus.PASS: (
        "No selection: the evidence is inadequate or an explicit selection rule failed."
    ),
}


class CoverageStatus(_Str):
    """How far a (sport, league, market) is actually operational."""

    VALIDATED = "validated"
    EXPERIMENTAL = "experimental"
    DEMO_ONLY = "demo-only"
    ADAPTER_ONLY = "adapter-only"
    UNAVAILABLE = "unavailable"


COVERAGE_DEFINITIONS = {
    CoverageStatus.VALIDATED: (
        "Real data ingested; chronological walk-forward validation and an untouched "
        "final test period were run; the selected model passed every configured gate "
        "(sample size, skill over the naive baseline with its uncertainty interval, "
        "calibration, point-in-time features)."
    ),
    CoverageStatus.EXPERIMENTAL: (
        "Real data ingested and the model runs end to end with a recorded validation "
        "run, but at least one gate is not met (small sample, weak or uncertain skill, "
        "calibration, retrospective inputs, or no realistic line history)."
    ),
    CoverageStatus.DEMO_ONLY: (
        "Runs only on synthetic demo data, which is isolated and labelled. No claim "
        "about real events."
    ),
    CoverageStatus.ADAPTER_ONLY: (
        "An adapter and/or documented CSV import and a model specification exist, but "
        "no real data has been ingested and validated in this installation."
    ),
    CoverageStatus.UNAVAILABLE: "Not implemented.",
}


class Availability(_Str):
    """Availability of a player or competitor as stated by a source.

    Three situations must never be confused:

    * confirmed absence  -> ``OUT``;
    * uncertain availability -> ``DOUBTFUL`` / ``QUESTIONABLE`` / ``PROBABLE`` /
      ``UNCERTAIN``;
    * missing information -> no report row exists at all (not a status here).
    """

    OUT = "out"                  # confirmed absent
    DOUBTFUL = "doubtful"        # unlikely to play
    QUESTIONABLE = "questionable"
    PROBABLE = "probable"
    UNCERTAIN = "uncertain"      # source says status is undetermined
    AVAILABLE = "available"      # confirmed available / confirmed starter

    @property
    def is_confirmed_absence(self) -> bool:
        return self is Availability.OUT

    @property
    def is_uncertain(self) -> bool:
        return self in (
            Availability.DOUBTFUL,
            Availability.QUESTIONABLE,
            Availability.PROBABLE,
            Availability.UNCERTAIN,
        )


class CheckState(_Str):
    """State of one configured pregame check."""

    COMPLETE = "complete"   # information on file, fresh, and re-checked close enough to the start
    RECHECK = "recheck_needed"  # on file and fresh, but must be re-checked nearer the start
    STALE = "stale"         # on file but older than the freshness window
    MISSING = "missing"     # nothing on file
    NOT_REQUIRED = "not_required"
