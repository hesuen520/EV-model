"""Documented CSV import: the way in for any league whose feed is unreachable.

Four files are understood (templates: ``sportspred templates <dir>``; column
reference: ``docs/CSV_IMPORT.md``):

* **events** - schedule rows, optionally with results in the same file;
* **results** - results for events already on file;
* **availability** - injury / line-up statements as published;
* **checks** - a record that a pregame check was done and what it found.

Rules enforced on every file:

* a row that is wrong is an error with its line number - never silently
  skipped or "repaired"; nothing is written unless the whole file is valid;
* timestamps must carry a zone (``2026-11-13T00:30:00Z``,
  ``2026-11-12 19:30-05:00``) or the caller must state one with ``tz``;
  a zone is never guessed;
* blank cells stay missing - nothing is filled in;
* unknown participants are an error unless ``create_participants`` is set,
  so a typo cannot create a phantom team;
* every import is recorded as a source (file path, SHA-256, collection time)
  and re-importing the same file changes nothing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from ..core.enums import Availability, EventStatus
from ..core.ids import norm_text
from ..core.timeutil import NaiveDatetimeError, iso_z, parse_ts, utcnow
from ..sports import get_league

__all__ = ["CsvImportError", "import_events", "import_results", "import_availability", "import_checks",
           "write_templates", "TEMPLATES"]

PathLike = Union[str, Path]


class CsvImportError(ValueError):
    """The file is invalid; ``errors`` lists every problem with its line."""

    def __init__(self, path: PathLike, errors: Sequence[str]) -> None:
        self.errors = list(errors)
        shown = "\n  ".join(self.errors[:25])
        more = "" if len(self.errors) <= 25 else f"\n  ... and {len(self.errors) - 25} more"
        super().__init__(f"{path}: {len(self.errors)} problem(s); nothing was imported\n  {shown}{more}")


def _read(path: PathLike, required: Sequence[str]) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, comment="#")
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise CsvImportError(path, [f"missing required column(s): {', '.join(missing)}"])
    return df.apply(lambda col: col.str.strip())


def _source(store: Any, path: PathLike, kind: str, league: str) -> str:
    data = Path(path).read_bytes()
    return store.add_source("csv", str(Path(path).resolve()), sha256=hashlib.sha256(data).hexdigest(),
                            n_bytes=len(data), license="user supplied", notes=f"{kind} import for {league}")


def _cell(row: pd.Series, name: str) -> Optional[str]:
    value = row.get(name, "")
    return None if value is None or value == "" else str(value)


def _num(row: pd.Series, name: str, line: int, errors: List[str]) -> Optional[float]:
    value = _cell(row, name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        errors.append(f"line {line}: {name} {value!r} is not a number")
        return None


def _time(value: Optional[str], tz: Optional[str], line: int, name: str, errors: List[str]) -> Optional[str]:
    if value is None:
        return None
    try:
        return iso_z(parse_ts(value, assume_tz=tz))
    except NaiveDatetimeError:
        errors.append(f"line {line}: {name} {value!r} has no time zone; add an offset (e.g. -05:00 or Z) "
                      "or pass --tz with the zone the file is written in")
    except ValueError:
        errors.append(f"line {line}: {name} {value!r} is not a valid timestamp")
    return None


def _participants(store: Any, league: str, names: Sequence[str], *, create: bool, kind: str,
                  errors: List[str]) -> Dict[str, str]:
    """name -> participant_id; unknown names are errors unless ``create``."""
    plugin, spec = get_league(league)
    unique = [n for n in dict.fromkeys(names) if n]
    frame = pd.DataFrame({"name": unique})
    found = store.resolve_participants(frame, sport=spec.sport, scope=league, kind=kind, create=False,
                                       is_synthetic=league.startswith("demo_"))
    unknown = [n for n, pid in zip(unique, found) if pid is None]
    if unknown and not create:
        errors.append("unknown participant(s): " + ", ".join(sorted(unknown)[:15])
                      + (" ..." if len(unknown) > 15 else "")
                      + " - fix the spelling, or pass --create-participants if they are new")
        return {n: pid for n, pid in zip(unique, found) if pid is not None}
    return dict(zip(unique, found))


def import_events(store: Any, league: str, path: PathLike, *, tz: Optional[str] = None,
                  create_participants: bool = False) -> Dict[str, int]:
    """Import schedule rows (and results, where the row has them).

    Required columns: ``season, start_time, status, home, away`` - for sports
    without a home side put the two competitors in ``home``/``away`` in any
    order and set ``neutral_site`` to 1.  Optional: ``event_id`` (your stable
    ID; keeps the event's identity through a reschedule), ``time_precision``
    (``datetime`` | ``date`` | ``period``), ``neutral_site`` (0/1), ``venue``,
    ``venue_city``, ``stage``, ``discipline``, ``best_of``, ``surface``,
    ``home_score``, ``away_score``, ``winner`` (name, for results without
    scores), ``draw`` (1 for a drawn event), ``method``.
    """
    plugin, spec = get_league(league)
    df = _read(path, ["season", "start_time", "status", "home", "away"])
    errors: List[str] = []
    valid = {s.value for s in EventStatus}
    names = list(df["home"]) + list(df["away"])
    rows, results, seen = [], [], set()
    for i, row in df.iterrows():
        line = i + 2
        status = (_cell(row, "status") or "").lower()
        if status not in valid:
            errors.append(f"line {line}: status {row['status']!r} is not one of {sorted(valid)}")
        precision = (_cell(row, "time_precision") or "datetime").lower()
        if precision not in ("datetime", "date", "period"):
            errors.append(f"line {line}: time_precision {precision!r} must be datetime, date or period")
        start = _time(_cell(row, "start_time"), tz if precision == "datetime" else (tz or "UTC"), line,
                      "start_time", errors)
        home, away = _cell(row, "home"), _cell(row, "away")
        if not home or not away:
            errors.append(f"line {line}: home and away are both required")
            continue
        if norm_text(home) == norm_text(away):
            errors.append(f"line {line}: home and away are the same ({home})")
        if not _cell(row, "season"):
            errors.append(f"line {line}: season is required")
        key = _cell(row, "event_id") or f"{start}|{norm_text(home)}|{norm_text(away)}"
        if key in seen:
            errors.append(f"line {line}: duplicate event ({key})")
        seen.add(key)
        neutral = _cell(row, "neutral_site")
        if neutral is not None and neutral not in ("0", "1"):
            errors.append(f"line {line}: neutral_site must be 0, 1 or blank")
        fmt: Dict[str, Any] = {}
        for extra in ("best_of", "surface", "game_number", "overs"):
            if _cell(row, extra) is not None:
                fmt[extra] = _cell(row, extra)
        hs, aw = _num(row, "home_score", line, errors), _num(row, "away_score", line, errors)
        winner, draw = _cell(row, "winner"), _cell(row, "draw") == "1"
        has_result = hs is not None or aw is not None or winner is not None or draw
        if has_result and status != "final":
            errors.append(f"line {line}: a result is given but status is {status!r}, not 'final'")
        if (hs is None) != (aw is None):
            errors.append(f"line {line}: give both scores or neither")
        if winner is not None and norm_text(winner) not in (norm_text(home), norm_text(away)):
            errors.append(f"line {line}: winner {winner!r} is neither home nor away")
        if status == "final" and not has_result:
            errors.append(f"line {line}: status is 'final' but no score, winner or draw is given")
        rows.append({
            "league": league, "season": _cell(row, "season"), "start_time_utc": start, "status": status,
            "time_precision": precision, "provider": "csv", "provider_event_id": _cell(row, "event_id"),
            "home_name": home, "away_name": away, "neutral_site": None if neutral is None else int(neutral),
            "venue_name": _cell(row, "venue"), "venue_city": _cell(row, "venue_city"),
            "stage": _cell(row, "stage") or "regular", "discipline": _cell(row, "discipline"), "format": fmt,
            "is_synthetic": league.startswith("demo_"),
        })
        results.append({"home_score": hs, "away_score": aw, "winner_name": winner, "is_draw": int(draw),
                        "method": _cell(row, "method"), "has": has_result})
    ids: Dict[str, str] = {}
    if not create_participants:
        ids = _participants(store, league, names, create=False, kind="team", errors=errors)
    if errors:
        raise CsvImportError(path, errors)
    source = _source(store, path, "events", league)
    if create_participants:
        frame = pd.DataFrame({"name": [n for n in dict.fromkeys(names) if n]})
        made = store.resolve_participants(frame, sport=spec.sport, scope=league, kind="team",
                                          is_synthetic=league.startswith("demo_"))
        ids = dict(zip(frame["name"], made))
    events = pd.DataFrame(rows)
    events["home_id"] = events["home_name"].map(ids)
    events["away_id"] = events["away_name"].map(ids)
    stored, report = store.upsert_events(events, sport=spec.sport, source_id=source)
    res = pd.DataFrame(results)
    res["event_id"] = stored["event_id"].to_numpy()
    res["winner_id"] = [
        None if w is None else (h if norm_text(w) == norm_text(hn) else a)
        for w, hn, h, a in zip(res["winner_name"], events["home_name"], events["home_id"], events["away_id"])
    ]
    res = res[res["has"]]
    n_results = 0
    if len(res):
        rr = store.upsert_results(res[["event_id", "home_score", "away_score", "winner_id", "is_draw", "method"]]
                                  .assign(is_synthetic=league.startswith("demo_")), source_id=source)
        n_results = rr.inserted + rr.updated
    return {"events_new": report.inserted, "events_updated": report.updated, "events_unchanged": report.unchanged,
            "results_written": n_results}


def import_results(store: Any, league: str, path: PathLike) -> Dict[str, int]:
    """Results for events already on file.  Required: ``event_id`` (the ID used
    when the event was imported, or the store's own event ID) and either
    ``home_score`` + ``away_score`` or ``winner`` / ``draw``."""
    get_league(league)
    df = _read(path, ["event_id"])
    errors: List[str] = []
    known = store.query("SELECT provider_event_id, event_id FROM events WHERE league=?", [league])
    by_native = dict(zip(known["provider_event_id"].dropna(), known["event_id"]))
    by_id = set(known["event_id"])
    names = store.participant_names()
    events = store.events(league, with_results=False, include_synthetic=True).set_index("event_id")
    rows = []
    for i, row in df.iterrows():
        line = i + 2
        key = _cell(row, "event_id")
        eid = by_native.get(key) or (key if key in by_id else None)
        if eid is None:
            errors.append(f"line {line}: no event {key!r} on file for {league}")
            continue
        hs, aw = _num(row, "home_score", line, errors), _num(row, "away_score", line, errors)
        winner, draw = _cell(row, "winner"), _cell(row, "draw") == "1"
        if hs is None and aw is None and winner is None and not draw:
            errors.append(f"line {line}: no score, winner or draw given")
        winner_id = None
        if winner is not None:
            ev = events.loc[eid]
            match = [pid for pid in (ev["home_id"], ev["away_id"]) if norm_text(names.get(pid, "")) == norm_text(winner)]
            if not match:
                errors.append(f"line {line}: winner {winner!r} is not a participant of event {key}")
            else:
                winner_id = match[0]
        rows.append({"event_id": eid, "home_score": hs, "away_score": aw, "winner_id": winner_id,
                     "is_draw": int(draw), "method": _cell(row, "method"),
                     "is_synthetic": league.startswith("demo_")})
    if errors:
        raise CsvImportError(path, errors)
    rr = store.upsert_results(pd.DataFrame(rows), source_id=_source(store, path, "results", league))
    return {"results_new": rr.inserted, "results_corrected": rr.updated, "results_unchanged": rr.unchanged}


def import_availability(store: Any, league: str, path: PathLike, *, tz: Optional[str] = None,
                        league_wide: bool = False, check_key: str = "injury_report",
                        horizon_hours: float = 48.0, collected_at: Any = None,
                        report_published_at: Any = None) -> Dict[str, int]:
    """Availability statements as published.

    Required columns: ``team, player, status`` (``out`` | ``doubtful`` |
    ``questionable`` | ``probable`` | ``uncertain`` | ``available``).
    Optional: ``role`` (e.g. ``starting_goalie``, ``probable_pitcher``),
    ``detail``, ``published_at`` (when the source published it),
    ``collected_at`` (when you read it; default: now).

    How old the report is: ``report_published_at`` says when the report this
    file transcribes was published, and the pregame check is then aged from
    that time.  Without it the check is as old as its collection time (the
    list is taken to be the source's current one when it was read).  A row's
    own ``published_at`` dates that single statement - a long-term injury is
    old news on a current list - so it does not age the check.

    The matching pregame check is logged as ``reported`` for every team in the
    file.  With ``league_wide`` (the file is a complete league report) teams
    with an event inside ``horizon_hours`` that are *not* in the file are
    logged as ``nothing_listed`` - "checked, nobody listed" is recorded as
    such and is not the same as "never checked".
    """
    get_league(league)
    df = _read(path, ["team", "player", "status"])
    errors: List[str] = []
    valid = {a.value for a in Availability}
    default_collected = iso_z(collected_at or utcnow())
    report_published = None
    if report_published_at is not None:
        report_published = _time(str(report_published_at), tz, 0, "report publication time", errors)
        if report_published and report_published > default_collected:
            errors.append("the report's publication time is later than its collection time")
    ids = _participants(store, league, list(df["team"]), create=False, kind="team", errors=errors)
    rows = []
    for i, row in df.iterrows():
        line = i + 2
        status = (_cell(row, "status") or "").lower()
        if status not in valid:
            errors.append(f"line {line}: status {row['status']!r} is not one of {sorted(valid)}")
        if not _cell(row, "player"):
            errors.append(f"line {line}: player is required")
        collected = _time(_cell(row, "collected_at"), tz, line, "collected_at", errors) or default_collected
        published = _time(_cell(row, "published_at"), tz, line, "published_at", errors)
        if published and published > collected:
            errors.append(f"line {line}: published_at is later than collected_at")
        rows.append({"league": league, "team_id": ids.get(_cell(row, "team")), "player_name": _cell(row, "player"),
                     "status": status, "role": _cell(row, "role"), "detail": _cell(row, "detail"),
                     "published_at_utc": published, "collected_at_utc": collected,
                     "is_synthetic": league.startswith("demo_")})
    if errors:
        raise CsvImportError(path, errors)
    source = _source(store, path, "availability", league)
    frame = pd.DataFrame(rows)
    n = store.add_availability(frame, source_id=source) if len(frame) else 0
    listed = {t: c for t, c in zip(frame.get("team_id", []), frame.get("collected_at_utc", []))}
    for team, when in listed.items():
        store.log_check(league, check_key, "reported", checked_at=when, team_id=team, source_id=source,
                        published_at=report_published, is_synthetic=league.startswith("demo_"))
    nothing = 0
    if league_wide:
        upcoming = store.upcoming(league, now=default_collected, horizon_hours=horizon_hours,
                                  include_synthetic=league.startswith("demo_"))
        for team in (set(upcoming["home_id"].dropna()) | set(upcoming["away_id"].dropna())) - set(listed):
            store.log_check(league, check_key, "nothing_listed", checked_at=default_collected, team_id=team,
                            source_id=source, published_at=report_published,
                            detail="league-wide report; team not listed",
                            is_synthetic=league.startswith("demo_"))
            nothing += 1
    return {"statements": n, "teams_reported": len(listed), "teams_nothing_listed": nothing}


def import_checks(store: Any, league: str, path: PathLike, *, tz: Optional[str] = None) -> Dict[str, int]:
    """Pregame check records.  Required: ``check_key, outcome`` (``reported`` |
    ``nothing_listed`` | ``not_published``), ``checked_at`` and one of
    ``team`` or ``event_id``.  Optional: ``published_at, detail``."""
    get_league(league)
    df = _read(path, ["check_key", "outcome", "checked_at"])
    errors: List[str] = []
    teams = [t for t in df.get("team", pd.Series(dtype=str)) if t]
    ids = _participants(store, league, teams, create=False, kind="team", errors=errors)
    known = store.query("SELECT provider_event_id, event_id FROM events WHERE league=?", [league])
    by_native = dict(zip(known["provider_event_id"].dropna(), known["event_id"]))
    rows = []
    for i, row in df.iterrows():
        line = i + 2
        outcome = (_cell(row, "outcome") or "").lower()
        if outcome not in ("reported", "nothing_listed", "not_published"):
            errors.append(f"line {line}: outcome {row['outcome']!r} must be reported, nothing_listed or not_published")
        checked = _time(_cell(row, "checked_at"), tz, line, "checked_at", errors)
        team, event = _cell(row, "team"), _cell(row, "event_id")
        if not team and not event:
            errors.append(f"line {line}: give a team or an event_id")
        event_id = None if event is None else by_native.get(event, event)
        if event is not None and event_id not in set(known["event_id"]):
            errors.append(f"line {line}: no event {event!r} on file for {league}")
        rows.append({"check_key": _cell(row, "check_key"), "outcome": outcome, "checked_at": checked,
                     "team_id": ids.get(team) if team else None, "event_id": event_id,
                     "published_at": _time(_cell(row, "published_at"), tz, line, "published_at", errors),
                     "detail": _cell(row, "detail")})
    if errors:
        raise CsvImportError(path, errors)
    source = _source(store, path, "checks", league)
    for r in rows:
        store.log_check(league, r["check_key"], r["outcome"], checked_at=r["checked_at"], team_id=r["team_id"],
                        event_id=r["event_id"], published_at=r["published_at"], source_id=source, detail=r["detail"],
                        is_synthetic=league.startswith("demo_"))
    return {"checks": len(rows)}


TEMPLATES: Dict[str, str] = {
    "events_template.csv": (
        "# One row per event. Timestamps need a zone (Z or an offset) or use --tz. Blank cells stay missing.\n"
        "event_id,season,start_time,status,home,away,neutral_site,venue,venue_city,stage,best_of,"
        "home_score,away_score,winner,draw,method\n"
        "2026-ipl-014,2026,2026-04-02T14:00:00Z,final,Example Kings,Example Royals,0,Example Oval,Example City,"
        "regular,,182,176,,,\n"
        "2026-ipl-015,2026,2026-04-03T14:00:00Z,scheduled,Example Royals,Example Kings,0,,,regular,,,,,,\n"),
    "results_template.csv": (
        "# Results for events already imported; event_id is the ID used in the events file.\n"
        "event_id,home_score,away_score,winner,draw,method\n"
        "2026-ipl-015,150,151,,,\n"),
    "availability_template.csv": (
        "# status: out | doubtful | questionable | probable | uncertain | available\n"
        "team,player,status,role,detail,published_at,collected_at\n"
        "Example Kings,Example Player,out,,hamstring,2026-04-03T08:00:00Z,2026-04-03T09:15:00Z\n"
        "Example Royals,Example Keeper,available,starting_goalie,confirmed starter,,2026-04-03T12:30:00Z\n"),
    "checks_template.csv": (
        "# outcome: reported | nothing_listed | not_published. Give a team or an event_id.\n"
        "check_key,team,event_id,outcome,checked_at,published_at,detail\n"
        "injury_report,Example Kings,,nothing_listed,2026-04-03T12:00:00Z,,official report checked\n"
        "weather,,2026-ipl-015,reported,2026-04-03T12:00:00Z,,forecast checked\n"),
}


def write_templates(directory: PathLike) -> List[Path]:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, text in TEMPLATES.items():
        (out / name).write_text(text, encoding="utf-8")
        paths.append(out / name)
    return paths
