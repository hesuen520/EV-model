"""Evaluation of the configured pregame checks for one event.

A check is COMPLETE only when information is on file, is no older than its
freshness window, and (where configured) dates from close enough to the start.
Age is measured from the source's publication time when it states one,
otherwise from the time the check was made.  "The source was checked and listed nothing" completes a check; "the
source has not been checked" does not.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from ..config import Settings
from ..core.enums import CheckState
from ..core.timeutil import fmt_hst, iso_z
from ..sports.base import CheckSpec

__all__ = ["evaluate_checks", "checks_complete"]

_DONE = ("reported", "nothing_listed")


def _latest(log: pd.DataFrame, key: str, *, event_id: Optional[str], team_id: Optional[str]) -> Optional[pd.Series]:
    if log.empty:
        return None
    part = log[(log["check_key"] == key) & log["outcome"].isin(_DONE)]
    mask = pd.Series(False, index=part.index)
    if team_id is not None:
        # A team-level check applies if it is for this event or not tied to any event.
        mask |= (part["team_id"] == team_id) & (part["event_id"].isna() | (part["event_id"] == event_id))
    else:
        mask |= part["event_id"] == event_id
    part = part[mask]
    if part.empty:
        return None
    # The newest information wins: by publication time when the source states
    # one, otherwise by the time the check was made.
    info_time = part["published_at_utc"].where(part["published_at_utc"].notna(), part["checked_at_utc"])
    return part.assign(_info=info_time).sort_values(["_info", "checked_at_utc"]).iloc[-1]


def evaluate_checks(
    specs: Sequence[CheckSpec],
    row: Mapping[str, Any],
    check_log: pd.DataFrame,
    *,
    now: pd.Timestamp,
    settings: Settings,
    names: Optional[Mapping[str, str]] = None,
) -> List[Dict[str, Any]]:
    """State of every required check for the event in ``row``."""
    names = names or {}
    start = pd.Timestamp(row["start_time_utc"])
    out: List[Dict[str, Any]] = []
    for spec in specs:
        if not spec.required:
            continue
        max_age = float(settings.selection.max_age_hours.get(spec.key, spec.max_age_hours))
        recheck = settings.selection.recheck_within_hours.get(spec.key, spec.recheck_within_hours)
        if spec.key == "prior_games_complete":
            pending = int(row.get("n_pending_prior", 0) or 0)
            out.append({
                "key": spec.key, "label": spec.label, "subject": "event",
                "state": (CheckState.COMPLETE if pending == 0 else CheckState.MISSING).value,
                "detail": ("every earlier game of both sides is final" if pending == 0 else
                           f"{pending} earlier scheduled game(s) of these sides are not final yet; "
                           "re-run after they finish"),
                "checked_at_utc": iso_z(now), "published_at_utc": None, "source_id": None,
            })
            continue
        subjects = ([("event", None)] if spec.scope == "event" else
                    [(row["side_a_id"], row["side_a_id"]), (row["side_b_id"], row["side_b_id"])])
        for subject, team_id in subjects:
            label = "event" if team_id is None else names.get(team_id, team_id)
            hit = _latest(check_log, spec.key, event_id=row["event_id"], team_id=team_id)
            window = f"no older than {max_age:g} hours"
            if recheck is not None:
                window += f" and re-checked within {float(recheck):g} hours of the start ({fmt_hst(start)})"
            entry: Dict[str, Any] = {
                "key": spec.key, "label": spec.label, "subject": label, "checked_at_utc": None,
                "published_at_utc": None, "source_id": None,
            }
            if hit is None:
                entry.update(state=CheckState.MISSING.value,
                             detail=f"nothing on file for {label}. Needed: {window}.")
            else:
                # The information is as old as its publication, not as recent
                # as our download of it: when the source states a publication
                # time, freshness is measured from that time.
                has_pub = isinstance(hit["published_at_utc"], str) and hit["published_at_utc"]
                checked = pd.Timestamp(hit["published_at_utc"] if has_pub else hit["checked_at_utc"])
                age = (now - checked).total_seconds() / 3600.0
                entry.update(checked_at_utc=hit["checked_at_utc"], published_at_utc=hit["published_at_utc"],
                             source_id=hit["source_id"])
                found = "nothing listed" if hit["outcome"] == "nothing_listed" else "report on file"
                when = (f"{found} for {label}, {'published' if has_pub else 'checked'} {fmt_hst(checked)} "
                        f"({age:.1f} h ago)")
                if age > max_age:
                    entry.update(state=CheckState.STALE.value, detail=f"{when}: older than {max_age:g} hours.")
                elif recheck is not None and (start - checked).total_seconds() / 3600.0 > float(recheck):
                    entry.update(state=CheckState.RECHECK.value,
                                 detail=f"{when}: must be re-checked within {float(recheck):g} hours of the start.")
                else:
                    entry.update(state=CheckState.COMPLETE.value, detail=when + ".")
            out.append(entry)
    return out


def checks_complete(checks: Sequence[Mapping[str, Any]]) -> bool:
    return all(c["state"] == CheckState.COMPLETE.value for c in checks)
