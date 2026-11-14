"""Durable store: SQLite for entities, events, provenance and the ledger;
parquet for bulk statistics tables.

Guarantees the rest of the platform relies on:

* **Stable IDs and de-duplication.**  Re-ingesting the same source is a no-op;
  the same game arriving from two providers becomes one event with two
  cross-references.
* **Provenance.**  Every row can carry a ``source_id`` pointing at the provider,
  URL, collection time and (when known) publication time.
* **Append-only ledger.**  Predictions and combos cannot be updated or deleted
  (SQL triggers), must be created before the event starts (SQL CHECK), and are
  hash-chained so out-of-band edits are detectable.
* **Synthetic isolation.**  Synthetic rows are only accepted for leagues whose
  key starts with ``demo_``, and those leagues accept nothing else.  Queries
  exclude synthetic rows unless asked.
"""

from __future__ import annotations

import hashlib
import json
import math
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..core.enums import Availability, EventStatus, RecStatus, TimePrecision
from ..core.ids import norm_text, stable_id
from ..core.timeutil import iso_z, parse_ts, utcnow

__all__ = ["Store", "StoreError", "SyntheticIsolationError", "LedgerError", "UpsertReport", "DEMO_PREFIX"]

log = logging.getLogger(__name__)

#: A date-only event is stored at 00:00Z of its calendar date.  That date begins
#: up to this many hours earlier (UTC+14), so nothing is recorded for the event
#: later than this long before the stored instant.
DATE_ONLY_LEAD_HOURS = 14.0

DEMO_PREFIX = "demo_"

PathLike = Union[str, Path]


class StoreError(ValueError):
    """Invalid data was offered to the store."""


class SyntheticIsolationError(StoreError):
    """Synthetic and real data were about to be mixed."""


class LedgerError(StoreError):
    """A ledger rule was violated (late prediction, mutation, bad probabilities)."""


@dataclass
class UpsertReport:
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    merged: int = 0          # matched to an existing event from another provider
    duplicates: int = 0      # same-provider duplicates folded into an existing event
    issues: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"inserted={self.inserted} updated={self.updated} unchanged={self.unchanged} "
            f"merged={self.merged} duplicates={self.duplicates} issues={len(self.issues)}"
        )


def _no_nan(value: Any) -> Any:
    """``NaN`` / ``inf`` are not JSON.  A missing value read back from a
    frame arrives as ``NaN``; it is stored as ``null``."""
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _no_nan(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_no_nan(v) for v in value]
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def _json(value: Any) -> str:
    return json.dumps(_no_nan(value), sort_keys=True, default=_json_default, separators=(",", ":"),
                      allow_nan=False)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (datetime, pd.Timestamp)):
        return iso_z(obj)
    if hasattr(obj, "value"):  # enums
        return obj.value
    return str(obj)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _opt(value: Any) -> Any:
    """None for missing scalars; numpy scalars to Python."""
    if isinstance(value, (list, dict, tuple)):
        return value
    if _is_missing(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _opt_int(value: Any) -> Optional[int]:
    value = _opt(value)
    return None if value is None else int(value)


def is_demo_league(league: str) -> bool:
    return str(league).startswith(DEMO_PREFIX)


def _check_synthetic(league: str, is_synthetic: bool, what: str) -> None:
    if is_synthetic and not is_demo_league(league):
        raise SyntheticIsolationError(
            f"refusing to write synthetic {what} into real league {league!r}; "
            f"synthetic data must use a league key starting with {DEMO_PREFIX!r}"
        )
    if not is_synthetic and is_demo_league(league):
        raise SyntheticIsolationError(
            f"league {league!r} is reserved for synthetic demo data; {what} must be flagged is_synthetic"
        )


class Store:
    """Open (and create if needed) the store rooted at ``root``."""

    def __init__(self, root: PathLike = "data/store") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "sportspred.sqlite"
        self.tables_dir = self.root / "tables"
        self.models_dir = self.root / "models"
        self.tables_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------ connection
    @contextmanager
    def conn(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.db_path, timeout=60.0)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("PRAGMA busy_timeout = 60000")
        # With this on, INSERT OR REPLACE fires the DELETE triggers, so it cannot
        # be used to rewrite an append-only row behind the triggers' back.
        con.execute("PRAGMA recursive_triggers = ON")
        try:
            yield con
            con.commit()
        except BaseException:
            con.rollback()
            raise
        finally:
            con.close()

    def _init_schema(self) -> None:
        ddl = resources.files("sportspred.storage").joinpath("schema.sql").read_text(encoding="utf-8")
        with self.conn() as con:
            con.execute("PRAGMA journal_mode = WAL")
            con.executescript(ddl)
            con.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', '1')"
            )
            # Stores created before the column existed: add it (earlier rows keep NULL
            # and their hashes, which never covered it, still verify).
            for table in ("predictions", "combos"):
                cols = {r["name"] for r in con.execute(f"PRAGMA table_info({table})")}
                if "recorded_at_utc" not in cols:
                    con.execute(f"ALTER TABLE {table} ADD COLUMN recorded_at_utc TEXT")

    def query(self, sql: str, params: Sequence[Any] = ()) -> pd.DataFrame:
        with self.conn() as con:
            return pd.read_sql_query(sql, con, params=list(params))

    # ------------------------------------------------------------ provenance
    def add_source(
        self,
        provider: str,
        url: Optional[str] = None,
        *,
        external_id: Optional[str] = None,
        collected_at: Optional[Any] = None,
        published_at: Optional[Any] = None,
        sha256: Optional[str] = None,
        n_bytes: Optional[int] = None,
        license: Optional[str] = None,
        notes: Optional[str] = None,
        is_synthetic: bool = False,
    ) -> str:
        """Record where data came from and when; returns the ``source_id``.

        The ID covers provider, URL, external ID and content hash, so
        re-collecting unchanged content returns the same ID (its first
        ``collected_at`` is kept) while changed content gets a new one.
        """
        collected = iso_z(collected_at) if collected_at is not None else iso_z(utcnow())
        published = iso_z(published_at) if published_at is not None else None
        sid = stable_id("src", provider, url or "", external_id or "", sha256 or collected)
        with self.conn() as con:
            con.execute(
                """INSERT OR IGNORE INTO sources
                   (source_id, provider, url, external_id, collected_at_utc, published_at_utc,
                    sha256, n_bytes, license, notes, is_synthetic)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (sid, provider, url, external_id, collected, published, sha256, n_bytes,
                 license, notes, int(bool(is_synthetic))),
            )
        return sid

    def sources(self, source_ids: Optional[Iterable[str]] = None) -> pd.DataFrame:
        if source_ids is None:
            return self.query("SELECT * FROM sources ORDER BY collected_at_utc")
        ids = [s for s in dict.fromkeys(source_ids) if s]
        if not ids:
            return self.query("SELECT * FROM sources WHERE 0")
        marks = ",".join("?" * len(ids))
        return self.query(f"SELECT * FROM sources WHERE source_id IN ({marks})", ids)

    # ---------------------------------------------------------- participants
    def resolve_participants(
        self,
        frame: pd.DataFrame,
        *,
        sport: str,
        scope: str,
        kind: str = "team",
        name_col: str = "name",
        provider: Optional[str] = None,
        provider_id_col: Optional[str] = None,
        short_name_col: Optional[str] = None,
        country_col: Optional[str] = None,
        is_synthetic: bool = False,
        create: bool = True,
    ) -> pd.Series:
        """Map each row of ``frame`` to a canonical ``participant_id``.

        Matching order: (provider, provider id) cross-reference, then the
        normalised name within ``scope``.  Unknown entities are created when
        ``create`` is true, otherwise they map to ``None`` (never to a guess).
        A known provider ID whose name changed (relocation, re-brand) keeps its
        participant and gains the new name as an alias.
        """
        if name_col not in frame.columns:
            raise StoreError(f"participants frame lacks the name column {name_col!r}")
        now = iso_z(utcnow())
        with self.conn() as con:
            xref: Dict[str, str] = {}
            if provider is not None:
                xref = {
                    r["provider_id"]: r["participant_id"]
                    for r in con.execute(
                        "SELECT provider_id, participant_id FROM participant_xref WHERE provider=? AND scope=?",
                        (provider, scope),
                    )
                }
            alias = {
                r["alias_norm"]: r["participant_id"]
                for r in con.execute(
                    "SELECT alias_norm, participant_id FROM participant_alias WHERE scope=?", (scope,)
                )
            }
            ids_of: Dict[str, set] = {}
            for known_id, participant in xref.items():
                ids_of.setdefault(participant, set()).add(known_id)
            out: List[Optional[str]] = []
            for row in frame.itertuples(index=False):
                rec = row._asdict()
                name = _opt(rec.get(name_col))
                pid_raw = _opt(rec.get(provider_id_col)) if provider_id_col else None
                pid_key = None if pid_raw is None else str(pid_raw)
                if isinstance(pid_raw, float) and pid_raw.is_integer():
                    pid_key = str(int(pid_raw))
                key = norm_text(name)
                found: Optional[str] = None
                if pid_key is not None and pid_key in xref:
                    found = xref[pid_key]
                elif key and key in alias:
                    found = alias[key]
                if found is None:
                    if not create or not key:
                        out.append(None)
                        continue
                    found = stable_id("p", sport, scope, kind, key)
                    con.execute(
                        """INSERT OR IGNORE INTO participants
                           (participant_id, sport, scope, kind, name, short_name, country, meta_json,
                            is_synthetic, created_at_utc)
                           VALUES (?,?,?,?,?,?,?,?,?,?)""",
                        (
                            found, sport, scope, kind, str(name),
                            _opt(rec.get(short_name_col)) if short_name_col else None,
                            _opt(rec.get(country_col)) if country_col else None,
                            None, int(bool(is_synthetic)), now,
                        ),
                    )
                if key and key not in alias:
                    con.execute(
                        "INSERT OR IGNORE INTO participant_alias (scope, alias_norm, participant_id, source) VALUES (?,?,?,?)",
                        (scope, key, found, provider),
                    )
                    alias[key] = found
                if provider is not None and pid_key is not None and pid_key not in xref:
                    others = ids_of.get(found, set())
                    if others:
                        # One provider, two of its own IDs, one name: they are treated as one
                        # participant (names are all there is to go on) - but that may be two
                        # people or clubs with the same name, so it is put on record.
                        self._issue(con, scope, None, "name_shared_by_provider_ids",
                                    f"{provider} ids {sorted(others)} and {pid_key} share the name {name!r}; "
                                    "treated as one participant - check that they are")
                    con.execute(
                        "INSERT OR IGNORE INTO participant_xref (provider, scope, provider_id, participant_id) VALUES (?,?,?,?)",
                        (provider, scope, pid_key, found),
                    )
                    xref[pid_key] = found
                    ids_of.setdefault(found, set()).add(pid_key)
                out.append(found)
        return pd.Series(out, index=frame.index, dtype="object")

    def resolve_participant(self, sport: str, scope: str, name: str, **kwargs: Any) -> Optional[str]:
        provider_id = kwargs.pop("provider_id", None)
        frame = pd.DataFrame({"name": [name], "pid": [provider_id]})
        return self.resolve_participants(
            frame, sport=sport, scope=scope, provider_id_col="pid" if provider_id is not None else None, **kwargs
        ).iloc[0]

    def add_alias(self, scope: str, alias: str, participant_id: str, source: str = "manual") -> None:
        key = norm_text(alias)
        if not key:
            raise StoreError("alias is empty after normalisation")
        with self.conn() as con:
            row = con.execute(
                "SELECT participant_id FROM participant_alias WHERE scope=? AND alias_norm=?", (scope, key)
            ).fetchone()
            if row is not None and row["participant_id"] != participant_id:
                raise StoreError(
                    f"alias {alias!r} in scope {scope!r} already points at {row['participant_id']}"
                )
            con.execute(
                "INSERT OR IGNORE INTO participant_alias (scope, alias_norm, participant_id, source) VALUES (?,?,?,?)",
                (scope, key, participant_id, source),
            )

    def participants(self, scope: Optional[str] = None, sport: Optional[str] = None) -> pd.DataFrame:
        sql, params = "SELECT * FROM participants WHERE 1=1", []
        if scope is not None:
            sql += " AND scope=?"
            params.append(scope)
        if sport is not None:
            sql += " AND sport=?"
            params.append(sport)
        return self.query(sql, params)

    def participant_names(self, ids: Optional[Iterable[str]] = None) -> Dict[str, str]:
        df = self.query("SELECT participant_id, name FROM participants")
        names = dict(zip(df["participant_id"], df["name"]))
        if ids is None:
            return names
        return {i: names.get(i, i) for i in ids}

    # ---------------------------------------------------------------- events
    _TRACKED = ("start_time_utc", "status", "home_id", "away_id", "venue_name", "neutral_site")

    def upsert_events(
        self,
        frame: pd.DataFrame,
        *,
        sport: str,
        source_id: Optional[str] = None,
        match_tolerance_hours: float = 6.0,
        duplicate_tolerance_hours: float = 2.0,
    ) -> Tuple[pd.DataFrame, UpsertReport]:
        """Insert or update events; returns the frame with ``event_id`` and a report.

        Required columns: ``league, season, start_time_utc, status, provider``
        plus either ``home_id``/``away_id`` (two-sided) or ``participants``
        (a list of participant IDs).  ``start_time_utc`` must be timezone-aware
        (or an ISO string with zone); naive values are rejected.
        Optional: ``provider_event_id, stage, discipline, time_precision,
        neutral_site, venue_name, venue_city, venue_id, format`` (dict),
        ``source_id, is_synthetic``.
        """
        report = UpsertReport()
        required = {"league", "season", "start_time_utc", "status", "provider"}
        missing = required - set(frame.columns)
        if missing:
            raise StoreError(f"events frame lacks required columns: {sorted(missing)}")
        if frame.empty:
            return frame.assign(event_id=pd.Series(dtype="object")), report
        df = frame.copy()
        two_sided = {"home_id", "away_id"} <= set(df.columns)
        if not two_sided and "participants" not in df.columns:
            raise StoreError("events need home_id/away_id or a participants list")

        valid_status = {s.value for s in EventStatus}
        valid_precision = {p.value for p in TimePrecision}
        now = iso_z(utcnow())
        event_ids: List[str] = []
        with self.conn() as con:
            leagues = [str(x) for x in df["league"].unique()]
            xref: Dict[Tuple[str, str, str], str] = {}
            by_sides: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
            existing: Dict[str, Dict[str, Any]] = {}
            has_result: set = set()
            for league in leagues:
                for r in con.execute(
                    "SELECT provider, provider_event_id, event_id FROM event_xref WHERE league=?", (league,)
                ):
                    xref[(league, r["provider"], r["provider_event_id"])] = r["event_id"]
                for r in con.execute("SELECT * FROM events WHERE league=?", (league,)):
                    rec = dict(r)
                    existing[rec["event_id"]] = rec
                    if rec["home_id"] and rec["away_id"]:
                        by_sides.setdefault((league, rec["home_id"], rec["away_id"]), []).append(rec)
                for r in con.execute(
                    "SELECT r.event_id FROM results r JOIN events e ON e.event_id = r.event_id WHERE e.league=?",
                    (league,),
                ):
                    has_result.add(r["event_id"])

            for row in df.to_dict("records"):
                league = str(row["league"])
                synthetic = bool(_opt(row.get("is_synthetic")) or False)
                _check_synthetic(league, synthetic, "events")
                status = str(_opt(row.get("status")) or "unknown")
                if status not in valid_status:
                    raise StoreError(f"invalid event status {status!r}")
                precision = str(_opt(row.get("time_precision")) or "datetime")
                if precision not in valid_precision:
                    raise StoreError(f"invalid time_precision {precision!r}")
                start = iso_z(row["start_time_utc"])  # raises on naive / missing
                provider = str(row["provider"])
                prov_eid = _opt(row.get("provider_event_id"))
                prov_eid = None if prov_eid is None else str(prov_eid)
                home = _opt(row.get("home_id")) if two_sided else None
                away = _opt(row.get("away_id")) if two_sided else None
                participants = _opt(row.get("participants")) if "participants" in row else None
                if two_sided and participants is None:
                    if home is None or away is None:
                        raise StoreError(
                            f"event {prov_eid or start} in {league} is missing a participant id"
                        )
                    if home == away:
                        raise StoreError(f"event {prov_eid or start} in {league} has identical sides")
                fmt = _opt(row.get("format")) or {}
                if not isinstance(fmt, Mapping):
                    raise StoreError("event 'format' must be a mapping")
                game_no = fmt.get("game_number")
                sid = _opt(row.get("source_id")) or source_id
                rec = {
                    "sport": sport,
                    "league": league,
                    "season": str(row["season"]),
                    "stage": str(_opt(row.get("stage")) or "regular"),
                    "discipline": _opt(row.get("discipline")),
                    "start_time_utc": start,
                    "time_precision": precision,
                    "status": status,
                    "home_id": home,
                    "away_id": away,
                    "neutral_site": _opt_int(row.get("neutral_site")),
                    "venue_name": _opt(row.get("venue_name")),
                    "venue_city": _opt(row.get("venue_city")),
                    "venue_id": None if _opt(row.get("venue_id")) is None else str(_opt(row.get("venue_id"))),
                    "participants_json": None if participants is None else _json(list(participants)),
                    "format_json": _json(dict(fmt)) if fmt else None,
                    "provider": provider,
                    "provider_event_id": prov_eid,
                    "source_id": sid,
                    "is_synthetic": int(synthetic),
                }

                eid = xref.get((league, provider, prov_eid)) if prov_eid else None
                matched_other = False
                if eid is None and home and away:
                    cands = self._match_candidates(
                        by_sides.get((league, home, away), []), start, precision, game_no,
                        provider, match_tolerance_hours, duplicate_tolerance_hours,
                        has_native_id=prov_eid is not None,
                    )
                    if len(cands) == 1 and cands[0]["provider"] == provider and (
                            prov_eid is None or not cands[0].get("provider_event_id")):
                        # The same provider's own record of this game, addressed without a native
                        # ID (a CSV row with no event_id): this is an update of it, not a duplicate.
                        eid = cands[0]["event_id"]
                    elif len(cands) == 1:
                        eid = cands[0]["event_id"]
                        matched_other = True
                        if cands[0]["provider"] == provider:
                            report.duplicates += 1
                            self._issue(con, league, eid, "duplicate_event",
                                        f"{provider}:{prov_eid} duplicates {cands[0]['provider_event_id']}")
                        else:
                            report.merged += 1
                    elif len(cands) > 1:
                        msg = f"{provider}:{prov_eid or start} matches {len(cands)} existing events; kept separate"
                        report.issues.append(msg)
                        self._issue(con, league, None, "ambiguous_event_match", msg)
                if eid is None:
                    if prov_eid:
                        eid = stable_id("evt", league, provider, prov_eid)
                    else:
                        who = [home, away] if home and away else list(participants or [])
                        eid = stable_id("evt", league, rec["season"], start[:10], *who, game_no or 1)
                    if eid in existing:
                        matched_other = False  # same natural key re-imported
                old = existing.get(eid)
                if old is None:
                    con.execute(
                        """INSERT INTO events (event_id, sport, league, season, stage, discipline, start_time_utc,
                               time_precision, status, home_id, away_id, neutral_site, venue_name, venue_city,
                               venue_id, participants_json, format_json, provider, provider_event_id, source_id,
                               first_seen_utc, updated_at_utc, is_synthetic)
                           VALUES (:event_id,:sport,:league,:season,:stage,:discipline,:start_time_utc,
                               :time_precision,:status,:home_id,:away_id,:neutral_site,:venue_name,:venue_city,
                               :venue_id,:participants_json,:format_json,:provider,:provider_event_id,:source_id,
                               :now,:now,:is_synthetic)""",
                        {**rec, "event_id": eid, "now": now},
                    )
                    stored = {**rec, "event_id": eid}
                    existing[eid] = stored
                    if home and away:
                        by_sides.setdefault((league, home, away), []).append(stored)
                    report.inserted += 1
                else:
                    changes = self._event_changes(old, rec, primary=not matched_other)
                    if (changes.get("status") in ("scheduled", "unknown", "live") and old.get("status") == "final"
                            and eid in has_result):
                        # A stale schedule row must not reopen a game whose result is on file,
                        # nor move it: the result belongs to the game as it was played.
                        del changes["status"]
                        changes.pop("start_time_utc", None)
                        changes.pop("time_precision", None)
                        msg = (f"{provider}:{prov_eid or start} lists a finished game as {rec['status']}; "
                               "the result on file was kept")
                        report.issues.append(msg)
                        self._issue(con, league, eid, "stale_schedule_row", msg)
                    if changes:
                        sets = ", ".join(f"{k}=?" for k in changes)
                        con.execute(
                            f"UPDATE events SET {sets}, updated_at_utc=? WHERE event_id=?",
                            [*changes.values(), now, eid],
                        )
                        for fld, new in changes.items():
                            if fld in self._TRACKED:
                                con.execute(
                                    "INSERT INTO event_changes (event_id, changed_at_utc, field, old_value, new_value, source_id) VALUES (?,?,?,?,?,?)",
                                    (eid, now, fld, None if old.get(fld) is None else str(old.get(fld)),
                                     None if new is None else str(new), sid),
                                )
                        old.update(changes)
                        report.updated += 1
                    else:
                        report.unchanged += 1
                if prov_eid and (league, provider, prov_eid) not in xref:
                    con.execute(
                        "INSERT OR IGNORE INTO event_xref (provider, league, provider_event_id, event_id) VALUES (?,?,?,?)",
                        (provider, league, prov_eid, eid),
                    )
                    xref[(league, provider, prov_eid)] = eid
                event_ids.append(eid)
        df["event_id"] = event_ids
        return df, report

    @staticmethod
    def _match_candidates(
        pool: List[Dict[str, Any]], start: str, precision: str, game_no: Any, provider: str,
        tol_hours: float, dup_hours: float, has_native_id: bool = False,
    ) -> List[Dict[str, Any]]:
        if not pool:
            return []
        t = parse_ts(start)
        out = []
        for ev in pool:
            other = parse_ts(ev["start_time_utc"])
            both_exact = precision == "datetime" and ev["time_precision"] == "datetime"
            same_provider = ev["provider"] == provider
            if both_exact:
                limit = dup_hours if same_provider else tol_hours
                close = abs((t - other).total_seconds()) <= limit * 3600.0
            else:
                # A calendar date from one source may be the UTC date or the
                # local date of the same game: allow one day either way.
                close = abs((t.date() - other.date()).days) <= (0 if same_provider else 1)
            if not close:
                continue
            other_no = json.loads(ev["format_json"]).get("game_number") if ev.get("format_json") else None
            if game_no is not None and other_no is not None and str(game_no) != str(other_no):
                continue
            if same_provider and ev.get("provider_event_id") and has_native_id:
                # One provider, two different native IDs: the provider itself
                # says these are distinct events (a doubleheader, an All-Star
                # mini-tournament).  Never fold them together.
                continue
            out.append(ev)
        return out

    @staticmethod
    def _event_changes(old: Mapping[str, Any], new: Mapping[str, Any], *, primary: bool) -> Dict[str, Any]:
        """Fields to update.  A cross-provider match only fills gaps; it never
        overwrites the first provider's values (the first provider stays the
        event's source of record)."""
        changes: Dict[str, Any] = {}
        fields = (
            "season", "stage", "discipline", "start_time_utc", "time_precision", "status", "home_id",
            "away_id", "neutral_site", "venue_name", "venue_city", "venue_id", "participants_json",
            "format_json", "source_id",
        )
        for fld in fields:
            nv, ov = new.get(fld), old.get(fld)
            if nv is None or nv == ov:
                continue
            if primary:
                if fld == "time_precision" and ov == "datetime" and nv != "datetime":
                    continue
                if fld == "start_time_utc" and old.get("time_precision") == "datetime" and new.get("time_precision") != "datetime":
                    continue
                changes[fld] = nv
            elif ov is None:
                changes[fld] = nv
            elif fld == "start_time_utc" and old.get("time_precision") != "datetime" and new.get("time_precision") == "datetime":
                changes[fld] = nv
                changes["time_precision"] = "datetime"
        return changes

    @staticmethod
    def _issue(con: sqlite3.Connection, league: Optional[str], event_id: Optional[str], kind: str, detail: str) -> None:
        con.execute(
            "INSERT INTO data_issues (created_at_utc, league, event_id, kind, detail) VALUES (?,?,?,?,?)",
            (iso_z(utcnow()), league, event_id, kind, detail),
        )

    def log_issue(self, kind: str, detail: str, *, league: Optional[str] = None, event_id: Optional[str] = None) -> None:
        with self.conn() as con:
            self._issue(con, league, event_id, kind, detail)

    def issues(self, league: Optional[str] = None) -> pd.DataFrame:
        if league is None:
            return self.query("SELECT * FROM data_issues ORDER BY issue_id")
        return self.query("SELECT * FROM data_issues WHERE league=? ORDER BY issue_id", [league])

    # --------------------------------------------------------------- results
    def upsert_results(
        self,
        frame: pd.DataFrame,
        *,
        source_id: Optional[str] = None,
        strict_scores: bool = True,
    ) -> UpsertReport:
        """Record verified results and mark their events final.

        Required: ``event_id``.  Optional: ``home_score, away_score, winner_id,
        is_draw, reg_home_score, reg_away_score, went_overtime, went_shootout,
        method, ranking`` (list), ``detail`` (dict), ``completed_at_utc``.
        With ``strict_scores`` a two-sided result whose winner contradicts the
        scores is rejected; pass ``False`` for sports where the higher number
        does not always win (e.g. rain-adjusted cricket).
        A changed result is kept in ``result_history`` before being replaced.
        """
        report = UpsertReport()
        if frame.empty:
            return report
        if "event_id" not in frame.columns:
            raise StoreError("results frame lacks event_id")
        now = iso_z(utcnow())
        with self.conn() as con:
            ids = [str(x) for x in frame["event_id"].unique()]
            events: Dict[str, Dict[str, Any]] = {}
            for i in range(0, len(ids), 500):
                chunk = ids[i:i + 500]
                marks = ",".join("?" * len(chunk))
                for r in con.execute(f"SELECT * FROM events WHERE event_id IN ({marks})", chunk):
                    events[r["event_id"]] = dict(r)
            prior: Dict[str, Dict[str, Any]] = {}
            for i in range(0, len(ids), 500):
                chunk = ids[i:i + 500]
                marks = ",".join("?" * len(chunk))
                for r in con.execute(f"SELECT * FROM results WHERE event_id IN ({marks})", chunk):
                    prior[r["event_id"]] = dict(r)

            for row in frame.to_dict("records"):
                eid = str(row["event_id"])
                ev = events.get(eid)
                if ev is None:
                    raise StoreError(f"result refers to unknown event {eid}")
                synthetic = bool(_opt(row.get("is_synthetic")) or ev["is_synthetic"])
                _check_synthetic(ev["league"], synthetic, "results")
                hs, as_ = _opt(row.get("home_score")), _opt(row.get("away_score"))
                winner = _opt(row.get("winner_id"))
                is_draw = bool(_opt(row.get("is_draw")) or False)
                ranking = _opt(row.get("ranking"))
                for label, val in (("home_score", hs), ("away_score", as_)):
                    if val is not None and float(val) < 0:
                        raise StoreError(f"negative {label} for event {eid}")
                if ev["home_id"] and ev["away_id"]:
                    if winner is None and not is_draw and hs is not None and as_ is not None:
                        if float(hs) == float(as_):
                            is_draw = True
                        else:
                            winner = ev["home_id"] if float(hs) > float(as_) else ev["away_id"]
                    if winner is not None and winner not in (ev["home_id"], ev["away_id"]):
                        raise StoreError(f"winner {winner} is not a participant of event {eid}")
                    if winner is not None and is_draw:
                        raise StoreError(f"event {eid} has both a winner and a draw")
                    if winner is None and not is_draw:
                        raise StoreError(f"result for event {eid} has neither winner, draw nor scores")
                    if (
                        winner is not None and hs is not None and as_ is not None
                        and float(hs) != float(as_)
                        and (winner == ev["home_id"]) != (float(hs) > float(as_))
                    ):
                        msg = f"winner contradicts scores {hs}-{as_} for event {eid}"
                        if strict_scores:
                            raise StoreError(msg)
                        self._issue(con, ev["league"], eid, "winner_score_mismatch", msg)
                elif ranking is None and winner is None:
                    raise StoreError(f"multi-competitor result for {eid} needs a ranking or winner")
                detail = _opt(row.get("detail"))
                rec = {
                    "event_id": eid,
                    "home_score": None if hs is None else float(hs),
                    "away_score": None if as_ is None else float(as_),
                    "winner_id": winner,
                    "is_draw": int(is_draw),
                    "reg_home_score": _opt(row.get("reg_home_score")),
                    "reg_away_score": _opt(row.get("reg_away_score")),
                    "went_overtime": _opt_int(row.get("went_overtime")),
                    "went_shootout": _opt_int(row.get("went_shootout")),
                    "method": _opt(row.get("method")),
                    "ranking_json": None if ranking is None else _json(list(ranking)),
                    "detail_json": None if detail is None else _json(detail),
                    "completed_at_utc": None if _opt(row.get("completed_at_utc")) is None else iso_z(row["completed_at_utc"]),
                    "source_id": _opt(row.get("source_id")) or source_id,
                    "verified_at_utc": now,
                    "is_synthetic": int(synthetic),
                }
                old = prior.get(eid)
                compare = ("home_score", "away_score", "winner_id", "is_draw", "reg_home_score",
                           "reg_away_score", "went_overtime", "went_shootout", "method", "ranking_json")
                if old is not None and all(old.get(k) == rec.get(k) for k in compare):
                    report.unchanged += 1
                else:
                    if old is not None:
                        con.execute(
                            "INSERT INTO result_history (event_id, replaced_at_utc, previous_json) VALUES (?,?,?)",
                            (eid, now, _json(old)),
                        )
                        self._issue(con, ev["league"], eid, "result_corrected",
                                    f"result changed from {old.get('home_score')}-{old.get('away_score')} "
                                    f"to {rec['home_score']}-{rec['away_score']}")
                        report.updated += 1
                    else:
                        report.inserted += 1
                    con.execute(
                        """INSERT OR REPLACE INTO results
                           (event_id, home_score, away_score, winner_id, is_draw, reg_home_score, reg_away_score,
                            went_overtime, went_shootout, method, ranking_json, detail_json, completed_at_utc,
                            source_id, verified_at_utc, is_synthetic)
                           VALUES (:event_id,:home_score,:away_score,:winner_id,:is_draw,:reg_home_score,
                            :reg_away_score,:went_overtime,:went_shootout,:method,:ranking_json,:detail_json,
                            :completed_at_utc,:source_id,:verified_at_utc,:is_synthetic)""",
                        rec,
                    )
                    prior[eid] = rec
                if ev["status"] != EventStatus.FINAL.value:
                    con.execute(
                        "UPDATE events SET status='final', updated_at_utc=? WHERE event_id=?", (now, eid)
                    )
                    con.execute(
                        "INSERT INTO event_changes (event_id, changed_at_utc, field, old_value, new_value, source_id) VALUES (?,?,?,?,?,?)",
                        (eid, now, "status", ev["status"], "final", rec["source_id"]),
                    )
                    ev["status"] = "final"
        return report

    def events(
        self,
        league: Optional[Union[str, Sequence[str]]] = None,
        *,
        status: Optional[Union[str, Sequence[str]]] = None,
        start_from: Optional[Any] = None,
        start_before: Optional[Any] = None,
        with_results: bool = True,
        include_synthetic: bool = False,
        event_ids: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Events (optionally joined with results), ordered by start time.

        ``start_time_utc`` is returned as a timezone-aware pandas datetime.
        """
        cols = "e.*"
        join = ""
        if with_results:
            cols += (", r.home_score, r.away_score, r.winner_id, r.is_draw, r.reg_home_score, r.reg_away_score,"
                     " r.went_overtime, r.went_shootout, r.method, r.ranking_json, r.detail_json,"
                     " r.completed_at_utc, r.source_id AS result_source_id, r.verified_at_utc")
            join = " LEFT JOIN results r ON r.event_id = e.event_id"
        sql = f"SELECT {cols} FROM events e{join} WHERE 1=1"
        params: List[Any] = []
        if league is not None:
            leagues = [league] if isinstance(league, str) else list(league)
            sql += f" AND e.league IN ({','.join('?' * len(leagues))})"
            params += leagues
        if status is not None:
            statuses = [status] if isinstance(status, str) else list(status)
            sql += f" AND e.status IN ({','.join('?' * len(statuses))})"
            params += [str(s) for s in statuses]
        if start_from is not None:
            sql += " AND e.start_time_utc >= ?"
            params.append(iso_z(start_from))
        if start_before is not None:
            sql += " AND e.start_time_utc < ?"
            params.append(iso_z(start_before))
        if event_ids is not None:
            ids = list(event_ids)
            if not ids:
                sql += " AND 0"
            else:
                sql += f" AND e.event_id IN ({','.join('?' * len(ids))})"
                params += ids
        if not include_synthetic:
            sql += " AND e.is_synthetic = 0"
        sql += " ORDER BY e.start_time_utc, e.event_id"
        df = self.query(sql, params)
        df["start_time_utc"] = pd.to_datetime(df["start_time_utc"], utc=True, format="%Y-%m-%dT%H:%M:%SZ")
        return df

    def upcoming(
        self,
        league: Union[str, Sequence[str]],
        *,
        now: Optional[Any] = None,
        horizon_hours: Optional[float] = None,
        include_synthetic: bool = False,
    ) -> pd.DataFrame:
        """Scheduled events that have not started as of ``now``.

        An event only counts as upcoming when its status is ``scheduled`` AND
        its start time is still in the future.  For date-only events the stored
        00:00Z of the calendar date is the cutoff, which is deliberately
        conservative: without a start time we cannot know the event has not
        begun.
        """
        t0 = utcnow() if now is None else parse_ts(now)
        t1 = None if horizon_hours is None else t0 + timedelta(hours=float(horizon_hours))
        df = self.events(league, status="scheduled", start_from=t0, start_before=t1,
                         with_results=False, include_synthetic=include_synthetic)
        return df[df["start_time_utc"] > pd.Timestamp(t0)].reset_index(drop=True)

    def cancel_events(self, event_ids: Sequence[str], reason: str, *, source_id: Optional[str] = None) -> int:
        """Mark events as cancelled (for example records the source has
        withdrawn or duplicated).  Nothing is deleted: the change is logged
        and the reason recorded as a data issue."""
        now = iso_z(utcnow())
        n = 0
        with self.conn() as con:
            for eid in event_ids:
                row = con.execute("SELECT league, status FROM events WHERE event_id=?", (eid,)).fetchone()
                if row is None or row["status"] == "canceled":
                    continue
                con.execute("UPDATE events SET status='canceled', updated_at_utc=? WHERE event_id=?", (now, eid))
                con.execute(
                    "INSERT INTO event_changes (event_id, changed_at_utc, field, old_value, new_value, source_id) "
                    "VALUES (?,?,?,?,?,?)", (eid, now, "status", row["status"], "canceled", source_id))
                self._issue(con, row["league"], eid, "event_withdrawn", reason)
                n += 1
        return n

    def event_changes(self, event_id: str) -> pd.DataFrame:
        return self.query("SELECT * FROM event_changes WHERE event_id=? ORDER BY change_id", [event_id])

    # ---------------------------------------------------------------- tables
    def _table_path(self, league: str, name: str) -> Path:
        safe = lambda s: "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)  # noqa: E731
        return self.tables_dir / safe(league) / f"{safe(name)}.parquet"

    def write_table(
        self,
        league: str,
        name: str,
        frame: pd.DataFrame,
        *,
        source_ids: Optional[Sequence[str]] = None,
        is_synthetic: bool = False,
        mode: str = "merge",
    ) -> Path:
        """Write the bulk table ``name`` of ``league``.

        ``mode="merge"`` (default): when the table already exists and both it
        and ``frame`` carry an ``event_id`` column, rows of events that are
        **not** in ``frame`` are kept, so ingesting a subset of seasons can
        never silently discard the history of the others.  Rows of events
        that are in ``frame`` are replaced.  ``mode="replace"`` writes
        ``frame`` as the whole table.
        """
        if mode not in ("merge", "replace"):
            raise StoreError(f"unknown table write mode {mode!r}")
        _check_synthetic(league, is_synthetic, f"table {name!r}")
        path = self._table_path(league, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if mode == "merge" and path.exists() and "event_id" in frame.columns:
            old = pd.read_parquet(path)
            if "event_id" in old.columns:
                kept = old[~old["event_id"].isin(set(frame["event_id"]))]
                if len(kept):
                    frame = pd.concat([kept, frame], ignore_index=True)
        tmp = path.with_suffix(".parquet.part")
        frame.to_parquet(tmp, index=False)
        tmp.replace(path)
        with self.conn() as con:
            con.execute(
                """INSERT OR REPLACE INTO tables_manifest
                   (league, name, n_rows, updated_at_utc, source_ids_json, columns_json, is_synthetic)
                   VALUES (?,?,?,?,?,?,?)""",
                (league, name, int(len(frame)), iso_z(utcnow()), _json(list(source_ids or [])),
                 _json([str(c) for c in frame.columns]), int(bool(is_synthetic))),
            )
        return path

    def has_table(self, league: str, name: str) -> bool:
        return self._table_path(league, name).exists()

    def read_table(self, league: str, name: str, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
        path = self._table_path(league, name)
        if not path.exists():
            raise FileNotFoundError(
                f"no table {name!r} for league {league!r}; run ingestion or import a CSV first"
            )
        return pd.read_parquet(path, columns=list(columns) if columns else None)

    def table_manifest(self, league: Optional[str] = None) -> pd.DataFrame:
        if league is None:
            return self.query("SELECT * FROM tables_manifest ORDER BY league, name")
        return self.query("SELECT * FROM tables_manifest WHERE league=? ORDER BY name", [league])

    # ----------------------------------------------- availability and checks
    def add_availability(self, frame: pd.DataFrame, *, source_id: Optional[str] = None) -> int:
        """Append availability statements (never updates existing rows).

        Required: ``league, player_name, status, collected_at_utc``.
        Optional: ``event_id, team_id, player_id, role, detail,
        published_at_utc, source_id, is_synthetic``.
        """
        required = {"league", "player_name", "status", "collected_at_utc"}
        missing = required - set(frame.columns)
        if missing:
            raise StoreError(f"availability frame lacks required columns: {sorted(missing)}")
        valid = {a.value for a in Availability}
        n = 0
        with self.conn() as con:
            for row in frame.to_dict("records"):
                league = str(row["league"])
                synthetic = bool(_opt(row.get("is_synthetic")) or False)
                _check_synthetic(league, synthetic, "availability")
                status = str(row["status"]).strip().lower()
                if status not in valid:
                    raise StoreError(
                        f"invalid availability status {row['status']!r}; use one of {sorted(valid)}"
                    )
                collected = iso_z(row["collected_at_utc"])
                published = None if _opt(row.get("published_at_utc")) is None else iso_z(row["published_at_utc"])
                if published is not None and published > collected:
                    raise StoreError("availability published_at_utc is later than collected_at_utc")
                sid = _opt(row.get("source_id")) or source_id
                rid = stable_id(
                    "avl", league, _opt(row.get("event_id")) or "", _opt(row.get("team_id")) or "",
                    row["player_name"], status, _opt(row.get("role")) or "", published or "", collected,
                )
                cur = con.execute(
                    """INSERT OR IGNORE INTO availability
                       (report_id, league, event_id, team_id, player_id, player_name, status, role, detail,
                        published_at_utc, collected_at_utc, source_id, is_synthetic)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (rid, league, _opt(row.get("event_id")), _opt(row.get("team_id")), _opt(row.get("player_id")),
                     str(row["player_name"]), status, _opt(row.get("role")), _opt(row.get("detail")),
                     published, collected, sid, int(synthetic)),
                )
                n += cur.rowcount
        return n

    def availability_as_of(
        self,
        league: str,
        as_of: Any,
        *,
        team_ids: Optional[Sequence[str]] = None,
        event_id: Optional[str] = None,
        include_synthetic: bool = False,
    ) -> pd.DataFrame:
        """Statements in force at ``as_of``: the latest per (team, player, role),
        from each team's newest report of its kind.

        Only rows *collected* at or before ``as_of`` are visible, so a
        historical prediction can never see later news.

        A report lists who is doubtful *now*; a player it no longer lists has
        been cleared.  So when a newer report of the same kind has been
        checked for a team (the check log records every report read, including
        "nothing listed"), statements that team carried from an older report
        are dropped.  Statements from a source with no logged check are kept
        until a newer statement about the same player and role replaces them.
        """
        cutoff = iso_z(as_of)
        sql = "SELECT * FROM availability WHERE league=? AND collected_at_utc <= ?"
        params: List[Any] = [league, cutoff]
        if not include_synthetic:
            sql += " AND is_synthetic = 0"
        df = self.query(sql, params)
        if team_ids is not None:
            df = df[df["team_id"].isin(list(team_ids))]
        if event_id is not None:
            df = df[df["event_id"].isna() | (df["event_id"] == event_id)]
        if df.empty:
            return df
        df = self._drop_superseded_reports(df, league, cutoff, include_synthetic)
        if df.empty:
            return df
        df = df.sort_values(["collected_at_utc", "published_at_utc"], na_position="first")
        key = df["team_id"].fillna("") + "|" + df["player_name"].map(norm_text) + "|" + df["role"].fillna("")
        return df.loc[~key.duplicated(keep="last")].reset_index(drop=True)

    def _drop_superseded_reports(self, statements: pd.DataFrame, league: str, cutoff: str,
                                 include_synthetic: bool) -> pd.DataFrame:
        """Remove statements whose team has since had a newer report of the same kind.

        A *report* is one import: a source read at one time.  Its statements
        carry that source and collection time, and the check it logged carries
        the same pair, which is how a statement finds its report and the
        report's kind (``check_key``).  For each team and kind the report in
        force is the one with the newest *information* - by publication time
        where the source states one, otherwise by the time it was read, the
        same order the engine uses to pick the check it shows.  Statements of
        any other report of that kind are dropped.
        """
        checks = self.checks_as_of(league, cutoff, include_synthetic=include_synthetic)
        checks = checks[checks["outcome"].isin(["reported", "nothing_listed"]) & checks["team_id"].notna()]
        if checks.empty:
            return statements

        def report_key(source: Any, read_at: Any) -> Tuple[str, str]:
            return ("" if _is_missing(source) else str(source), str(read_at))

        kinds: Dict[Tuple[str, str], set] = {}
        for src, read_at, key in zip(checks["source_id"], checks["checked_at_utc"], checks["check_key"]):
            kinds.setdefault(report_key(src, read_at), set()).add(key)
        info = checks["published_at_utc"].where(checks["published_at_utc"].notna(), checks["checked_at_utc"])
        ranked = checks.assign(_info=info).sort_values(["_info", "checked_at_utc"])
        newest = ranked.drop_duplicates(["team_id", "check_key"], keep="last").set_index(["team_id", "check_key"])
        keep = []
        for src, team, collected in zip(statements["source_id"], statements["team_id"],
                                        statements["collected_at_utc"]):
            mine = report_key(src, collected)
            of_kinds = kinds.get(mine)
            if not of_kinds or _is_missing(team):
                keep.append(True)                       # no report of a known kind behind this statement
                continue
            current = False
            for kind in of_kinds:
                if (team, kind) not in newest.index:
                    current = True
                    break
                latest = newest.loc[(team, kind)]
                if report_key(latest["source_id"], latest["checked_at_utc"]) == mine:
                    current = True
                    break
            keep.append(current)
        return statements.loc[keep]

    def teams_with_statements(self, league: str, check_key: str, as_of: Any, *,
                              include_synthetic: bool = False) -> set:
        """Teams that still carry a statement from a report of this kind.

        An importer of a league-wide report calls this to find the teams the
        new report no longer mentions: it must log "nothing listed" for them,
        or their old statements would stay in force for ever."""
        cutoff = iso_z(as_of)
        current = self.availability_as_of(league, cutoff, include_synthetic=include_synthetic)
        if current.empty:
            return set()
        checks = self.checks_as_of(league, cutoff, include_synthetic=include_synthetic)
        checks = checks[checks["check_key"] == check_key]
        reports = {("" if _is_missing(s) else str(s), str(t)) for s, t in zip(checks["source_id"], checks["checked_at_utc"])}
        mine = [("" if _is_missing(s) else str(s), str(t)) in reports
                for s, t in zip(current["source_id"], current["collected_at_utc"])]
        return set(current.loc[mine, "team_id"].dropna())

    def log_check(
        self,
        league: str,
        check_key: str,
        outcome: str,
        *,
        checked_at: Any,
        event_id: Optional[str] = None,
        team_id: Optional[str] = None,
        published_at: Optional[Any] = None,
        source_id: Optional[str] = None,
        detail: Optional[str] = None,
        is_synthetic: bool = False,
    ) -> str:
        """Record that a pregame check ran and what it found.

        ``outcome``: ``reported`` (source listed something), ``nothing_listed``
        (source was published and listed nothing for this team/event) or
        ``not_published`` (source had no report yet).
        """
        _check_synthetic(league, is_synthetic, "check log")
        if event_id is None and team_id is None:
            raise StoreError("a check must refer to an event or a team")
        checked = iso_z(checked_at)
        published = None if published_at is None else iso_z(published_at)
        if published is not None and published > checked:
            raise StoreError("a check's published_at is later than its checked_at: a report cannot be read before "
                             "it is published")
        cid = stable_id("chk", league, check_key, event_id or "", team_id or "", outcome, checked, published or "")
        with self.conn() as con:
            con.execute(
                """INSERT OR IGNORE INTO check_log
                   (check_id, league, check_key, event_id, team_id, outcome, published_at_utc,
                    checked_at_utc, source_id, detail, is_synthetic)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (cid, league, check_key, event_id, team_id, outcome, published, checked, source_id,
                 detail, int(bool(is_synthetic))),
            )
        return cid

    # ---------------------------------------------------------- prop thresholds
    def add_prop_lines(self, rows: Sequence[Mapping[str, Any]], *, source_id: Optional[str] = None) -> int:
        """Record thresholds of player-statistic markets.  Each row needs
        ``league, event_id, player_id, player_name, stat, line,
        collected_at_utc``; ``team_id`` and ``book`` are optional."""
        payload = []
        for row in rows:
            line = float(row["line"])
            if not math.isfinite(line) or line < 0.5:
                raise StoreError(f"prop threshold {row['line']!r} is not a number of one half or more")
            if (2.0 * line) != round(2.0 * line):
                raise StoreError(f"prop threshold {line!r} is not a whole or half number")
            synthetic = is_demo_league(row["league"])
            payload.append((row["league"], row["event_id"], _opt(row.get("team_id")), str(row["player_id"]),
                            row["player_name"], row["stat"], line, _opt(row.get("book")),
                            iso_z(row["collected_at_utc"]), source_id, int(synthetic)))
        with self.conn() as con:
            before = con.total_changes
            # The same reading imported twice is stored once.
            con.executemany(
                """INSERT OR IGNORE INTO prop_lines (league, event_id, team_id, player_id, player_name, stat, line,
                       book, collected_at_utc, source_id, is_synthetic) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                payload)
            return int(con.total_changes - before)

    def prop_lines_as_of(self, league: str, as_of: Any, *, stat: Optional[str] = None) -> pd.DataFrame:
        """The threshold in force for each (event, player, statistic): the
        newest one collected at or before ``as_of``."""
        sql = "SELECT * FROM prop_lines WHERE league=? AND collected_at_utc <= ?"
        params: List[Any] = [league, iso_z(as_of)]
        if stat is not None:
            sql += " AND stat=?"
            params.append(stat)
        df = self.query(sql + " ORDER BY collected_at_utc, line_id", params)
        if df.empty:
            return df
        return df.drop_duplicates(["event_id", "player_id", "stat"], keep="last").reset_index(drop=True)

    def checks_as_of(self, league: str, as_of: Any, *, include_synthetic: bool = False) -> pd.DataFrame:
        sql = "SELECT * FROM check_log WHERE league=? AND checked_at_utc <= ?"
        if not include_synthetic:
            sql += " AND is_synthetic = 0"
        return self.query(sql + " ORDER BY checked_at_utc", [league, iso_z(as_of)])

    # ---------------------------------------------------------------- ledger
    _PRED_JSON = ("probs", "reasons", "checks", "confidence", "factors", "features", "data_quality",
                  "sources", "display")
    _PRED_HASHED = (
        "prediction_id", "created_at_utc", "event_id", "event_start_utc", "sport", "league", "market",
        "line", "selection", "selection_label", "prob", "prob_low", "prob_high", "probs_json",
        "model_name", "model_version", "validation_status", "rec_status", "reasons_json", "checks_json",
        "confidence_json", "factors_json", "features_json", "data_quality_json", "sources_json",
        "display_json", "supersedes_id", "is_synthetic",
    )

    #: A real event's prediction must carry a creation time within this many
    #: seconds of the machine's own clock, so a "pregame" record cannot be
    #: written afterwards with an earlier time.  Synthetic demo leagues, which
    #: replay invented dates, are exempt.  ``None`` switches the test off (the
    #: test-suite does, because it replays fixed dates in temporary stores).
    wall_clock_tolerance_seconds: Optional[float] = 300.0

    @classmethod
    def _hashed(cls, row: Mapping[str, Any]) -> Dict[str, Any]:
        """The fields a prediction's hash covers."""
        out = {k: row[k] for k in cls._PRED_HASHED}
        if row.get("recorded_at_utc") is not None:
            out["recorded_at_utc"] = row["recorded_at_utc"]
        return out

    @staticmethod
    def _hash(payload: Mapping[str, Any], prev: Optional[str]) -> str:
        body = json.dumps({k: payload[k] for k in sorted(payload)}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(((prev or "") + body).encode("utf-8")).hexdigest()

    def insert_prediction(self, record: Mapping[str, Any]) -> str:
        """Append one immutable pregame prediction; returns its ``prediction_id``.

        ``record`` keys: ``created_at_utc, event_id, market, selection,
        selection_label, probs`` (mapping outcome -> probability, summing to
        1), ``model_name, model_version, validation_status, rec_status`` and
        optional ``line, prob, prob_low, prob_high, reasons, checks,
        confidence, factors, features, data_quality, sources, display,
        supersedes_id, is_synthetic``.

        Rejected (``LedgerError``) when the event is unknown, is not scheduled,
        has already started at ``created_at_utc``, when the probabilities are
        invalid, or when ``supersedes_id`` is not an earlier prediction of the
        same event and market.
        """
        rec = dict(record)
        for key in ("created_at_utc", "event_id", "market", "selection", "selection_label", "probs",
                    "model_name", "model_version", "validation_status", "rec_status"):
            if rec.get(key) is None:
                raise LedgerError(f"prediction lacks {key!r}")
        created = iso_z(rec["created_at_utc"])
        status = str(rec["rec_status"])
        if status not in {s.value for s in RecStatus}:
            raise LedgerError(f"invalid rec_status {status!r}")
        probs = {str(k): float(v) for k, v in dict(rec["probs"]).items()}
        if not probs and status != RecStatus.PASS.value:
            raise LedgerError("a non-PASS prediction needs an outcome distribution")
        if probs:
            if any((not np.isfinite(v)) or v < 0.0 or v > 1.0 for v in probs.values()):
                raise LedgerError(f"outcome probabilities outside [0, 1]: {probs}")
            if abs(sum(probs.values()) - 1.0) > 1e-6:
                raise LedgerError(f"outcome probabilities sum to {sum(probs.values()):.6f}, not 1")
        prob = _opt(rec.get("prob"))
        if status != RecStatus.PASS.value and prob is None:
            raise LedgerError("a non-PASS prediction needs the probability of its selection")
        for key in ("prob", "prob_low", "prob_high"):
            val = _opt(rec.get(key))
            if val is not None and not (0.0 <= float(val) <= 1.0):
                raise LedgerError(f"{key}={val} is outside [0, 1]")

        with self.conn() as con:
            # The row's hash chains to the newest row.  Take the write lock before reading
            # it, or two writers at once would both chain to the same row and break the chain.
            con.execute("BEGIN IMMEDIATE")
            ev = con.execute("SELECT * FROM events WHERE event_id=?", (rec["event_id"],)).fetchone()
            if ev is None:
                raise LedgerError(f"prediction refers to unknown event {rec['event_id']}")
            synthetic = bool(ev["is_synthetic"])
            _check_synthetic(ev["league"], bool(_opt(rec.get("is_synthetic")) or synthetic), "predictions")
            if ev["status"] != EventStatus.SCHEDULED.value:
                raise LedgerError(
                    f"event {ev['event_id']} is {ev['status']}; selections are only generated for scheduled events"
                )
            start = ev["start_time_utc"]
            if created >= start:
                raise LedgerError(
                    f"prediction time {created} is not before the event start {start}; "
                    "selections are only generated pregame"
                )
            if ev["time_precision"] != TimePrecision.DATETIME.value:
                earliest = iso_z(parse_ts(start) - timedelta(hours=DATE_ONLY_LEAD_HOURS))
                if created >= earliest:
                    raise LedgerError(
                        f"event {ev['event_id']} has only a date ({start[:10]}); that date begins at {earliest} in "
                        "the easternmost time zone, so the event may already have started"
                    )
            recorded = iso_z(utcnow())
            tolerance = self.wall_clock_tolerance_seconds
            if not synthetic and tolerance is not None:
                drift = abs((parse_ts(created) - parse_ts(recorded)).total_seconds())
                if drift > tolerance or recorded >= start:
                    raise LedgerError(
                        f"prediction time {created} does not match this machine's clock ({recorded}); a record "
                        "for a real event cannot be written with another time, before or after the fact"
                    )
            sup = _opt(rec.get("supersedes_id"))
            if sup is not None:
                prev_row = con.execute(
                    "SELECT event_id, market, created_at_utc FROM predictions WHERE prediction_id=?", (sup,)
                ).fetchone()
                if prev_row is None:
                    raise LedgerError(f"supersedes_id {sup} does not exist")
                if prev_row["event_id"] != rec["event_id"] or prev_row["market"] != rec["market"]:
                    raise LedgerError("a prediction can only supersede one for the same event and market")
                if prev_row["created_at_utc"] > created:
                    raise LedgerError("a prediction cannot supersede a later one")
            line = _opt(rec.get("line"))
            pid = _opt(rec.get("prediction_id")) or stable_id(
                "prd", rec["event_id"], rec["market"], "" if line is None else repr(float(line)),
                rec["selection"], rec["model_version"], created, status, start,
                str(rec["validation_status"]), _json(rec.get("reasons") or []),
                "" if prob is None else repr(round(float(prob), 6)), str(rec["model_name"]),
            )
            # Whether the event had only a date is part of the record: grading needs it
            # later to tell a published start time from a reschedule.
            if ev["time_precision"] != TimePrecision.DATETIME.value:
                rec["display"] = {**dict(rec.get("display") or {}), "date_only": True}
            payload: Dict[str, Any] = {
                "prediction_id": pid,
                "created_at_utc": created,
                "event_id": rec["event_id"],
                "event_start_utc": start,
                "sport": ev["sport"],
                "league": ev["league"],
                "market": str(rec["market"]),
                "line": None if line is None else float(line),
                "selection": str(rec["selection"]),
                "selection_label": str(rec["selection_label"]),
                "prob": None if prob is None else float(prob),
                "prob_low": None if _opt(rec.get("prob_low")) is None else float(rec["prob_low"]),
                "prob_high": None if _opt(rec.get("prob_high")) is None else float(rec["prob_high"]),
                "model_name": str(rec["model_name"]),
                "model_version": str(rec["model_version"]),
                "validation_status": str(rec["validation_status"]),
                "rec_status": status,
                "supersedes_id": sup,
                "is_synthetic": int(synthetic),
            }
            payload["probs_json"] = _json(probs)
            defaults = {"reasons": [], "checks": [], "confidence": {}, "factors": {}, "features": {},
                        "data_quality": {}, "sources": [], "display": {}}
            for key in self._PRED_JSON[1:]:
                payload[f"{key}_json"] = _json(rec.get(key) if rec.get(key) is not None else defaults[key])
            last = con.execute("SELECT content_hash FROM predictions ORDER BY seq DESC LIMIT 1").fetchone()
            prev_hash = None if last is None else last["content_hash"]
            payload["recorded_at_utc"] = recorded
            payload["prev_hash"] = prev_hash
            payload["content_hash"] = self._hash(self._hashed(payload), prev_hash)
            cols = list(payload)
            try:
                con.execute(
                    f"INSERT INTO predictions ({','.join(cols)}) VALUES ({','.join(':' + c for c in cols)})",
                    payload,
                )
            except sqlite3.IntegrityError as exc:
                raise LedgerError(f"prediction rejected by the ledger: {exc}") from exc
        return pid

    def predictions(
        self,
        *,
        league: Optional[Union[str, Sequence[str]]] = None,
        market: Optional[str] = None,
        event_ids: Optional[Sequence[str]] = None,
        start_from: Optional[Any] = None,
        start_before: Optional[Any] = None,
        latest_only: bool = False,
        include_synthetic: bool = False,
        slate: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Ledger rows.

        ``latest_only`` keeps only the newest prediction of each (event,
        market), judged over the *whole* ledger: a row is dropped as soon as a
        later one exists for the same event and market, even when the later
        one has another line, another status, or a start time outside the
        requested window (the event moved).  Earlier rows stay stored.

        ``slate`` (a ``date``) selects one Pacific/Honolulu slate, placing an
        event whose start is only a date on its own calendar date.
        """
        if slate is not None:
            from datetime import timedelta

            from ..core.slates import slate_date_of
            from ..core.timeutil import hst_day_bounds_utc

            lo, hi = hst_day_bounds_utc(slate)
            wide = self.predictions(league=league, market=market, event_ids=event_ids,
                                    start_from=lo - timedelta(hours=24), start_before=hi + timedelta(hours=24),
                                    latest_only=latest_only, include_synthetic=include_synthetic)
            if wide.empty:
                return wide
            keep = [slate_date_of(row) == slate for row in wide.to_dict("records")]
            return wide.loc[keep].reset_index(drop=True)
        sql, params = "SELECT * FROM predictions WHERE 1=1", []
        if league is not None:
            leagues = [league] if isinstance(league, str) else list(league)
            sql += f" AND league IN ({','.join('?' * len(leagues))})"
            params += leagues
        if market is not None:
            sql += " AND market=?"
            params.append(market)
        if start_from is not None:
            sql += " AND event_start_utc >= ?"
            params.append(iso_z(start_from))
        if start_before is not None:
            sql += " AND event_start_utc < ?"
            params.append(iso_z(start_before))
        if event_ids is not None:
            ids = list(event_ids)
            if not ids:
                sql += " AND 0"
            else:
                sql += f" AND event_id IN ({','.join('?' * len(ids))})"
                params += ids
        if not include_synthetic:
            sql += " AND is_synthetic = 0"
        df = self.query(sql + " ORDER BY seq", params)
        if latest_only and not df.empty:
            current = self.query("SELECT MAX(seq) AS seq FROM predictions GROUP BY event_id, market")["seq"]
            df = df.loc[df["seq"].isin(set(current.tolist()))].reset_index(drop=True)
        return df

    _COMBO_HASHED = ("combo_id", "created_at_utc", "slate_date_hst", "n_legs", "role", "leg_ids_json", "joint_prob",
                     "joint_prob_low", "joint_method", "dependency_json", "rec_status", "message", "rules_json",
                     "is_synthetic")
    _LEDGER_TRIGGERS = ("predictions_no_update", "predictions_no_delete", "predictions_no_replace",
                        "combos_no_update", "combos_no_delete", "combos_no_replace")

    def verify_ledger(self) -> Tuple[bool, List[str]]:
        """Audit the ledger; returns ``(ok, problems)``.

        Predictions: the hash chain (an edited, removed or reordered row), the
        immutability triggers, and the clock evidence - a real event's row
        whose recorded clock time is at or after the event's start, or far
        from its stated creation time, was not written pregame.  Combos: each
        row's content hash and the triggers.

        Not detectable from inside the database: removal of the newest rows,
        or a wrong machine clock.  Keep the head hash and the count elsewhere.
        """
        problems: List[str] = []
        with self.conn() as con:
            rows = [dict(r) for r in con.execute("SELECT * FROM predictions ORDER BY seq")]
            combos = [dict(r) for r in con.execute("SELECT * FROM combos ORDER BY seq")]
            triggers = {
                r["name"] for r in con.execute("SELECT name FROM sqlite_master WHERE type='trigger'")
            }
        for name in self._LEDGER_TRIGGERS:
            if name not in triggers:
                problems.append(f"immutability trigger {name} is missing")
        prev: Optional[str] = None
        clock_seen = False
        # The clock evidence is audited whenever the clock rule is in force (it always is,
        # outside the test-suite, which replays fixed dates in temporary stores).
        slack = self.wall_clock_tolerance_seconds
        for row in rows:
            if row["prev_hash"] != prev:
                problems.append(f"{row['prediction_id']}: chain broken (a row was removed or reordered)")
            expect = self._hash(self._hashed(row), row["prev_hash"])
            if expect != row["content_hash"]:
                problems.append(f"{row['prediction_id']}: content does not match its hash (row was edited)")
            prev = row["content_hash"]
            recorded = row.get("recorded_at_utc")
            if recorded is None:
                if clock_seen:
                    problems.append(f"{row['prediction_id']}: no recorded clock time, although earlier rows have one")
                continue
            clock_seen = True
            if not row["is_synthetic"] and slack is not None:
                if recorded >= row["event_start_utc"]:
                    problems.append(f"{row['prediction_id']}: written at {recorded}, not before the event start "
                                    f"{row['event_start_utc']}")
                elif abs((parse_ts(recorded) - parse_ts(row["created_at_utc"])).total_seconds()) > slack:
                    problems.append(f"{row['prediction_id']}: stated time {row['created_at_utc']} differs from the "
                                    f"clock time {recorded} at which it was written")
        for row in combos:
            body = {k: row[k] for k in self._COMBO_HASHED}
            if row.get("recorded_at_utc") is not None:
                body["recorded_at_utc"] = row["recorded_at_utc"]
            if self._hash(body, None) != row["content_hash"]:
                problems.append(f"{row['combo_id']}: combo content does not match its hash (row was edited)")
        return (not problems), problems

    def insert_combo(self, record: Mapping[str, Any]) -> str:
        rec = dict(record)
        created = iso_z(rec["created_at_utc"])
        legs = [str(x) for x in rec.get("leg_ids", [])]
        cid = _opt(rec.get("combo_id")) or stable_id(
            "cmb", created, rec["role"], rec.get("slate_date_hst"), *(legs or ["none"]),
            *(["synthetic"] if rec.get("is_synthetic") else []),      # a demo run never shares an id with a real one
        )
        payload = {
            "combo_id": cid,
            "created_at_utc": created,
            "slate_date_hst": str(rec["slate_date_hst"]),
            "n_legs": len(legs),
            "role": str(rec["role"]),
            "leg_ids_json": _json(legs),
            "joint_prob": _opt(rec.get("joint_prob")),
            "joint_prob_low": _opt(rec.get("joint_prob_low")),
            "joint_method": str(rec.get("joint_method") or "none"),
            "dependency_json": _json(rec.get("dependency") or {}),
            "rec_status": str(rec["rec_status"]),
            "message": _opt(rec.get("message")),
            "rules_json": _json(rec.get("rules") or {}),
            "is_synthetic": int(bool(rec.get("is_synthetic"))),
        }
        recorded = iso_z(utcnow())
        with self.conn() as con:
            rows = {
                r["prediction_id"]: dict(r)
                for r in con.execute(
                    "SELECT prediction_id, created_at_utc, event_start_utc, rec_status, is_synthetic FROM predictions "
                    f"WHERE prediction_id IN ({','.join('?' * len(legs)) or 'NULL'})",
                    legs,
                )
            }
            unknown = [x for x in legs if x not in rows]
            if unknown:
                raise LedgerError(f"combo refers to predictions not in the ledger: {unknown}")
            # A combo is a pregame record like its legs: made after every leg was predicted,
            # before any leg's event starts, and - for real events - at the time the clock says.
            synthetic = bool(payload["is_synthetic"])
            if rows:
                if any(bool(r["is_synthetic"]) != synthetic for r in rows.values()):
                    raise LedgerError("a combo cannot mix synthetic and real legs, nor be labelled as the other kind")
                if created < max(r["created_at_utc"] for r in rows.values()):
                    raise LedgerError("a combo cannot be dated before one of its own legs was predicted")
                if created >= min(r["event_start_utc"] for r in rows.values()):
                    raise LedgerError("a combo cannot be recorded once one of its events has started")
                if payload["rec_status"] == RecStatus.FINAL.value and any(
                        r["rec_status"] != RecStatus.FINAL.value for r in rows.values()):
                    raise LedgerError("a combo is only as final as its least final leg")
            tolerance = self.wall_clock_tolerance_seconds
            if not synthetic and tolerance is not None and abs(
                    (parse_ts(created) - parse_ts(recorded)).total_seconds()) > tolerance:
                raise LedgerError(
                    f"combo time {created} does not match this machine's clock ({recorded}); a record of real "
                    "events cannot be written with another time")
            payload["recorded_at_utc"] = recorded
            payload["content_hash"] = self._hash({k: v for k, v in payload.items()}, None)
            cols = list(payload)
            con.execute(
                f"INSERT OR IGNORE INTO combos ({','.join(cols)}) VALUES ({','.join(':' + c for c in cols)})",
                payload,
            )
        return cid

    def combos(self, *, slate_date_hst: Optional[str] = None, include_synthetic: bool = False) -> pd.DataFrame:
        sql, params = "SELECT * FROM combos WHERE 1=1", []
        if slate_date_hst is not None:
            sql += " AND slate_date_hst=?"
            params.append(str(slate_date_hst))
        if not include_synthetic:
            sql += " AND is_synthetic = 0"
        return self.query(sql + " ORDER BY seq", params)

    # ---------------------------------------------------- grading and review
    def add_grade(self, prediction_id: str, outcome: str, result: Mapping[str, Any], *,
                  graded_at: Optional[Any] = None, result_source_id: Optional[str] = None,
                  supersedes_grade_id: Optional[int] = None) -> int:
        with self.conn() as con:
            cur = con.execute(
                """INSERT INTO grades (prediction_id, graded_at_utc, outcome, result_json, result_source_id,
                                       supersedes_grade_id) VALUES (?,?,?,?,?,?)""",
                (prediction_id, iso_z(graded_at or utcnow()), outcome, _json(dict(result)), result_source_id,
                 supersedes_grade_id),
            )
            return int(cur.lastrowid)

    def grades(self, latest_only: bool = True) -> pd.DataFrame:
        df = self.query("SELECT * FROM grades ORDER BY grade_id")
        if latest_only and not df.empty:
            df = df.loc[~df["prediction_id"].duplicated(keep="last")].reset_index(drop=True)
        return df

    def add_review(self, prediction_id: str, category: str, evidence: Mapping[str, Any],
                   notes: Optional[str] = None, *, created_at: Optional[Any] = None) -> int:
        with self.conn() as con:
            cur = con.execute(
                "INSERT INTO reviews (prediction_id, created_at_utc, category, evidence_json, notes) VALUES (?,?,?,?,?)",
                (prediction_id, iso_z(created_at or utcnow()), category, _json(dict(evidence)), notes),
            )
            return int(cur.lastrowid)

    def reviews(self) -> pd.DataFrame:
        return self.query("SELECT * FROM reviews ORDER BY review_id")

    # ------------------------------------------------- validation and models
    def save_validation_run(self, report: Mapping[str, Any]) -> str:
        """Persist a validation report (see ``validation.backtest``)."""
        league = str(report["league"])
        synthetic = bool(report.get("is_synthetic"))
        _check_synthetic(league, synthetic, "validation runs")
        created = iso_z(report.get("created_at_utc") or utcnow())
        run_id = report.get("run_id") or stable_id("val", league, report["market"], created)
        body = dict(report)
        body["run_id"] = run_id
        body["created_at_utc"] = created
        with self.conn() as con:
            con.execute(
                """INSERT OR REPLACE INTO validation_runs
                   (run_id, created_at_utc, sport, league, market, selected_model, gates_passed,
                    is_synthetic, point_in_time, report_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (run_id, created, str(report["sport"]), league, str(report["market"]),
                 report.get("selected_model"), int(bool(report.get("gates_passed"))), int(synthetic),
                 int(bool(report.get("point_in_time"))), _json(body)),
            )
        return run_id

    def validation_runs(self, league: Optional[str] = None, market: Optional[str] = None,
                        include_synthetic: bool = True) -> pd.DataFrame:
        sql, params = "SELECT * FROM validation_runs WHERE 1=1", []
        if league is not None:
            sql += " AND league=?"
            params.append(league)
        if market is not None:
            sql += " AND market=?"
            params.append(market)
        if not include_synthetic:
            sql += " AND is_synthetic = 0"
        return self.query(sql + " ORDER BY created_at_utc", params)

    def latest_validation(self, league: str, market: str) -> Optional[Dict[str, Any]]:
        df = self.validation_runs(league, market)
        if df.empty:
            return None
        return json.loads(df.iloc[-1]["report_json"])

    def register_model(
        self,
        *,
        model_version: str,
        sport: str,
        league: str,
        market: str,
        model_name: str,
        artifact_path: PathLike,
        trained_at: Optional[Any] = None,
        train_start: Optional[Any] = None,
        train_end: Optional[Any] = None,
        n_train: Optional[int] = None,
        validation_run_id: Optional[str] = None,
        promote: bool = False,
        is_synthetic: bool = False,
        info: Optional[Mapping[str, Any]] = None,
    ) -> None:
        _check_synthetic(league, is_synthetic, "models")
        with self.conn() as con:
            if promote:
                con.execute("UPDATE models SET promoted=0 WHERE league=? AND market=?", (league, market))
            con.execute(
                """INSERT OR REPLACE INTO models
                   (model_version, sport, league, market, model_name, trained_at_utc, train_start_utc,
                    train_end_utc, n_train, artifact_path, validation_run_id, promoted, is_synthetic, info_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (model_version, sport, league, market, model_name, iso_z(trained_at or utcnow()),
                 None if train_start is None else iso_z(train_start),
                 None if train_end is None else iso_z(train_end),
                 n_train, str(artifact_path), validation_run_id, int(bool(promote)), int(bool(is_synthetic)),
                 _json(dict(info or {}))),
            )

    def promoted_model(self, league: str, market: str) -> Optional[Dict[str, Any]]:
        df = self.query(
            "SELECT * FROM models WHERE league=? AND market=? AND promoted=1 ORDER BY trained_at_utc DESC LIMIT 1",
            [league, market],
        )
        return None if df.empty else df.iloc[0].to_dict()

    def models(self) -> pd.DataFrame:
        return self.query("SELECT * FROM models ORDER BY league, market, trained_at_utc")
