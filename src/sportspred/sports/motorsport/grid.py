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

import json
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
_EMPTY_SLOTS = 4                 # slots left empty by cars that withdrew after qualifying
_RECENT_START_HOURS = 12.0       # a race that started this recently is the one a reading is about


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
    # A grid read while a race is on (or just after) is that race's grid, not the next one's.
    under_way = events[~events["status"].isin(["canceled", "postponed"]) & (events["start_time_utc"] <= read_at)
                       & (events["start_time_utc"] > read_at - pd.Timedelta(hours=_RECENT_START_HOURS))]
    if event_id is None and len(under_way):
        raise CsvImportError(path, [f"a race started at {iso_z(under_way['start_time_utc'].max())}, before this grid "
                                    f"was read ({collected}); a grid for a later race needs its event id"])
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
    # The grid exists only once qualifying has been held.
    fmt = json.loads(race["format_json"]) if isinstance(race.get("format_json"), str) else {}
    qualifying = fmt.get("qualifying_utc")
    if isinstance(qualifying, str) and collected < qualifying:
        raise CsvImportError(path, [f"qualifying for this race is scheduled for {qualifying}; a starting grid read "
                                    f"at {collected} cannot be its grid"])

    names = store.participants(scope=league)
    by_name: Dict[str, List[str]] = {}
    for n, pid in zip(names["name"], names["participant_id"]):
        by_name.setdefault(str(n).casefold(), []).append(pid)
    teams = store.participants(scope=CONSTRUCTOR_SCOPE)
    team_by_name = {str(n).casefold(): pid for n, pid in zip(teams["name"], teams["participant_id"])}
    last_team = build_finish_frame(store, league, "win", now=collected, cfg=F1Config()).extra.get("last_team", {})
    # Who is "current": a driver with a start this season or last.  Three-letter codes are
    # reused over the years (ALB was Albers before it was Albon), so a code is matched among
    # current drivers only, and a driver who is not current must be named with his team.
    entries = store.read_table(league, "race_entries") if store.has_table(league, "race_entries") else None
    current: set = set()
    if entries is not None:
        seasons = events.set_index("event_id")["season"].astype(int)
        latest = int(seasons.loc[seasons.index.intersection(entries["event_id"].unique())].max())
        recent = set(seasons[seasons >= latest - 1].index)
        current = set(entries.loc[entries["event_id"].isin(recent) & entries["started"].astype(bool), "driver_id"])
    by_short: Dict[str, List[str]] = {}
    for n, pid in zip(names["short_name"], names["participant_id"]):
        if isinstance(n, str) and n and pid in current:
            by_short.setdefault(n.casefold(), []).append(pid)

    rows, seen_drivers, seen_slots = [], {}, {}
    for i, row in df.iterrows():
        line = i + 2
        said = _cell(row, "driver")
        if not said:
            errors.append(f"line {line}: driver is required")
            continue
        found = by_name.get(said.casefold()) or by_short.get(said.casefold()) or []
        if len(found) > 1:
            errors.append(f"line {line}: {said!r} fits more than one driver on file; use his full name")
            continue
        if not found:
            errors.append(f"line {line}: driver {said!r} is not on file; use the name he races under (ingest first "
                          "if he is new to the series)")
            continue
        driver = found[0]
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
            if slot > len(df) + _EMPTY_SLOTS:
                errors.append(f"line {line}: grid slot {slot} is beyond a field of {len(df)} drivers")
                continue
            if slot in seen_slots:
                errors.append(f"line {line}: grid slot {slot} is already taken on line {seen_slots[slot]}")
                continue
            seen_slots[slot] = line
        team_said = _cell(row, "constructor")
        team = team_by_name.get(team_said.casefold()) if team_said else last_team.get(driver)
        if not team_said and team is not None and driver not in current:
            errors.append(f"line {line}: {said!r} has not started a race this season or last; give his constructor")
            continue
        if team is None:
            errors.append(f"line {line}: " + (f"constructor {team_said!r} is not on file" if team_said else
                                              f"{said!r} has no race on file; give his constructor"))
            continue
        rows.append({"event_id": race["event_id"], "driver_id": driver, "constructor_id": team,
                     "grid": float(slot) if slot is not None else float("nan"), "pit_lane": pit_lane,
                     "quali_gap_pct": float("nan"), "collected_at_utc": collected})
    if not errors and len(rows) < _MIN_FIELD:
        errors.append(f"only {len(rows)} drivers; a grid of fewer than {_MIN_FIELD} is not a starting grid")
    if not errors and 1 not in seen_slots:
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
