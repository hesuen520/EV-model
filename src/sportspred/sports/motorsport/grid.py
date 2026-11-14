"""Importing a starting grid.

The model starts from the official starting grid, which no reachable feed
publishes before a race.  The user reads it from the organiser's document and
imports it; the reading's time is recorded, and a prediction made at ``now``
uses the newest grid read at or before ``now``.

CSV columns: ``driver`` (the name he races under), ``grid`` (his slot, 1 =
pole; ``PL`` for a pit-lane start).  Optional: ``constructor`` (needed for a
driver with no race on file, or one who has changed team since his last
race).  The whole file is checked before anything is stored.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from ...adapters.csv_import import CsvImportError, _cell, _read, _source, _time
from ...core.timeutil import iso_z, utcnow
from .adapter import CONSTRUCTOR_SCOPE, LEAGUE
from .features import IMPORTED_GRID, F1Config, build_finish_frame

__all__ = ["import_grid", "GRID_TEMPLATE"]

GRID_TEMPLATE = (
    "# The official starting grid of the next race. grid: the slot (1 = pole) or PL for a pit-lane start.\n"
    "# constructor is only needed for a driver new to the series or to his team.\n"
    "driver,grid,constructor\n"
    "Example Driver,1,\n"
    "Another Driver,2,\n"
    "Third Driver,PL,Example Team\n")
_MIN_FIELD = 10


def import_grid(store: Any, league: str, path: Any, *, tz: Optional[str] = None, collected_at: Any = None,
                event_id: Optional[str] = None, horizon_days: float = 7.0) -> Dict[str, int]:
    """Store one reading of a race's starting grid and log the ``starting_grid``
    check for that race.  ``event_id`` names the race (the id it was ingested
    with); by default it is the next scheduled race within ``horizon_days``."""
    if league != LEAGUE:
        raise CsvImportError(path, [f"{league} has no starting-grid import"])
    df = _read(path, ["driver", "grid"])
    errors: List[str] = []
    if collected_at is None:
        collected = iso_z(utcnow())
    else:
        collected = _time(str(collected_at), tz, 0, "collection time", errors)
        if collected is None:
            raise CsvImportError(path, [e.replace("line 0: ", "") for e in errors])
    tolerance = getattr(store, "wall_clock_tolerance_seconds", None)
    if tolerance is not None and collected > iso_z(utcnow() + pd.Timedelta(seconds=float(tolerance))):
        raise CsvImportError(path, [f"collection time {collected} is in the future; a grid cannot have been read yet"])

    read_at = pd.Timestamp(collected)
    events = store.events(league, with_results=False)
    ahead = events[(events["status"] == "scheduled") & (events["start_time_utc"] > read_at)].sort_values("start_time_utc")
    if event_id is not None:
        race = events[(events["provider_event_id"].astype(str) == str(event_id)) | (events["event_id"] == event_id)]
        if race.empty:
            raise CsvImportError(path, [f"no race {event_id!r} on file for {league}"])
        if race.iloc[0]["event_id"] not in set(ahead["event_id"]):
            raise CsvImportError(path, [f"race {event_id!r} is not a scheduled race that starts after {collected}"])
        race = race.iloc[0]
    else:
        soon = ahead[ahead["start_time_utc"] <= read_at + pd.Timedelta(days=horizon_days)]
        if soon.empty:
            raise CsvImportError(path, [f"no scheduled race within {horizon_days:g} days of {collected}; give an "
                                        "event id"])
        race = soon.iloc[0]

    names = store.participants(scope=league)
    by_name = {str(n).casefold(): pid for n, pid in zip(names["name"], names["participant_id"])}
    by_short = {str(n).casefold(): pid for n, pid in zip(names["short_name"], names["participant_id"])
                if isinstance(n, str) and n}
    teams = store.participants(scope=CONSTRUCTOR_SCOPE)
    team_by_name = {str(n).casefold(): pid for n, pid in zip(teams["name"], teams["participant_id"])}
    last_team = build_finish_frame(store, league, "win", now=collected, cfg=F1Config()).extra.get("last_team", {})

    rows, seen_drivers, seen_slots = [], {}, {}
    for i, row in df.iterrows():
        line = i + 2
        said = _cell(row, "driver")
        if not said:
            errors.append(f"line {line}: driver is required")
            continue
        driver = by_name.get(said.casefold()) or by_short.get(said.casefold())
        if driver is None:
            errors.append(f"line {line}: driver {said!r} is not on file; use the name he races under (ingest first "
                          "if he is new to the series)")
            continue
        if driver in seen_drivers:
            errors.append(f"line {line}: {said!r} is already on line {seen_drivers[driver]}")
            continue
        seen_drivers[driver] = line
        slot_text = (_cell(row, "grid") or "").upper()
        pit_lane, slot = slot_text == "PL", None
        if not pit_lane:
            if not slot_text.isdigit() or int(slot_text) < 1:
                errors.append(f"line {line}: grid {row['grid']!r} must be a whole number from 1, or PL")
                continue
            slot = int(slot_text)
            if slot in seen_slots:
                errors.append(f"line {line}: grid slot {slot} is already taken on line {seen_slots[slot]}")
                continue
            seen_slots[slot] = line
        team_said = _cell(row, "constructor")
        team = team_by_name.get(team_said.casefold()) if team_said else last_team.get(driver)
        if team is None:
            errors.append(f"line {line}: " + (f"constructor {team_said!r} is not on file" if team_said else
                                              f"{said!r} has no race on file; give his constructor"))
            continue
        rows.append({"event_id": race["event_id"], "driver_id": driver, "constructor_id": team,
                     "grid": float(slot) if slot is not None else float("nan"), "pit_lane": pit_lane,
                     "quali_gap_pct": float("nan"), "collected_at_utc": collected})
    if not errors and len(rows) < _MIN_FIELD:
        errors.append(f"only {len(rows)} drivers; a grid of fewer than {_MIN_FIELD} is not a starting grid")
    if not errors and seen_slots and 1 not in seen_slots:
        errors.append("nobody is on pole (slot 1); this is not a complete starting grid")
    if errors:
        raise CsvImportError(path, errors)

    source = _source(store, path, "starting grid", league)
    new = pd.DataFrame(rows).assign(source_id=source)
    if store.has_table(league, IMPORTED_GRID):
        old = store.read_table(league, IMPORTED_GRID)
        same = (old["event_id"] == race["event_id"]) & (old["collected_at_utc"] == collected)
        new = pd.concat([old[~same], new], ignore_index=True)       # the same reading twice is stored once
    store.write_table(league, IMPORTED_GRID, new, source_ids=[source], mode="replace")
    store.log_check(league, "starting_grid", "reported", event_id=race["event_id"], checked_at=collected,
                    source_id=source, detail=f"{len(rows)} drivers")
    return {"drivers": len(rows), "pit_lane_starts": int(sum(r["pit_lane"] for r in rows))}
