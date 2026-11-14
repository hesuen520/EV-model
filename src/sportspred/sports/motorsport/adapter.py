"""Formula 1 results from F1DB, a public database published on GitHub.

One zip of CSV files holds every race since 1950 with its classification,
qualifying, starting grid and session times.  Licence: CC BY 4.0 (F1DB).

What is stored:

* one **event** per Grand Prix (the race itself, not the weekend), with the
  drivers as participants, the race start in UTC where the source gives a
  time (date only before the source records times), and the circuit;
* the **result**: winner and the classified order;
* a bulk table ``race_entries`` with one row per driver and race: constructor,
  starting grid position (pit-lane starts flagged), qualifying position and
  gap to pole, finishing position, whether he started and whether he was
  classified, and the stated reason for a retirement.

Nothing is invented: a race with no classification on file stays scheduled.
"""

from __future__ import annotations

import io
import zipfile
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ...adapters.http import Fetcher, NotFound
from .. import DataUnavailable
from ..base import IngestReport

__all__ = ["F1DB_URL", "FIRST_SEASON", "ingest_f1", "normalize_f1db", "NOT_STARTED", "LEAGUE"]

LEAGUE = "f1"
PROVIDER = "f1db"
F1DB_URL = "https://github.com/f1db/f1db/releases/latest/download/f1db-csv.zip"
_LICENCE = "CC BY 4.0 (F1DB, github.com/f1db/f1db)"
FIRST_SEASON = 2000
#: Classification texts of a driver who did not take the start.
NOT_STARTED = frozenset({"DNS", "DNQ", "DNPQ", "DNP", "EX", "WD"})
CONSTRUCTOR_SCOPE = "f1_constructors"


def _read(z: zipfile.ZipFile, name: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(z.read(name)), low_memory=False)


def normalize_f1db(z: zipfile.ZipFile, first_season: int = FIRST_SEASON) -> Dict[str, pd.DataFrame]:
    """``races`` (one row per Grand Prix) and ``entries`` (one row per driver
    and race) from the F1DB CSV files, in the source's own identifiers."""
    races = _read(z, "f1db-races.csv")
    races = races[races["year"] >= first_season].copy()
    circuits = _read(z, "f1db-circuits.csv").set_index("id")
    gps = _read(z, "f1db-grands-prix.csv").set_index("id")
    has_time = races["time"].notna()
    stamp = races["date"].astype(str) + "T" + races["time"].fillna("00:00").astype(str) + ":00Z"
    races["start_time_utc"] = pd.to_datetime(stamp, utc=True)
    races["time_precision"] = np.where(has_time, "datetime", "date")
    q_stamp = races["qualifyingDate"].astype(str) + "T" + races["qualifyingTime"].fillna("00:00").astype(str) + ":00Z"
    races["qualifying_utc"] = pd.to_datetime(q_stamp.where(races["qualifyingDate"].notna()), utc=True, errors="coerce")
    races["circuit_name"] = races["circuitId"].map(circuits["fullName"])
    races["circuit_place"] = races["circuitId"].map(circuits["placeName"])
    races["gp_name"] = races["grandPrixId"].map(gps["fullName"])
    races["has_sprint"] = races["sprintRaceDate"].notna()

    res = _read(z, "f1db-races-race-results.csv")
    res = res[res["year"] >= first_season].copy()
    qual = _read(z, "f1db-races-qualifying-results.csv")
    qual = qual[qual["year"] >= first_season]
    best = qual[["timeMillis", "q1Millis", "q2Millis", "q3Millis"]].min(axis=1)      # his best lap of the session
    qual = qual.assign(best_ms=best)[["raceId", "driverId", "best_ms"]].drop_duplicates(["raceId", "driverId"])
    pole = qual.groupby("raceId")["best_ms"].min().rename("pole_ms")
    entries = res.merge(qual, on=["raceId", "driverId"], how="left").merge(pole, on="raceId", how="left")
    text = entries["positionText"].astype(str)
    entries["started"] = ~text.isin(NOT_STARTED)
    entries["classified"] = entries["positionNumber"].notna()
    entries["pit_lane"] = entries["gridPositionText"].astype(str).eq("PL")
    out = pd.DataFrame({
        "race": entries["raceId"].astype(int).astype(str), "driver": entries["driverId"].astype(str),
        "constructor": entries["constructorId"].astype(str),
        "order": entries["positionDisplayOrder"].astype(float), "position": entries["positionNumber"].astype(float),
        "position_text": text, "started": entries["started"].to_numpy(), "classified": entries["classified"].to_numpy(),
        "grid": entries["gridPositionNumber"].astype(float), "pit_lane": entries["pit_lane"].to_numpy(),
        "quali_pos": entries["qualificationPositionNumber"].astype(float),
        "quali_gap_pct": 100.0 * (entries["best_ms"] - entries["pole_ms"]) / entries["pole_ms"],
        "laps": entries["laps"].astype(float), "reason": entries["reasonRetired"],
    })
    out = out.drop_duplicates(["race", "driver"], keep="first")          # a shared drive counts once
    drivers = _read(z, "f1db-drivers.csv")[["id", "name", "fullName", "abbreviation"]]
    constructors = _read(z, "f1db-constructors.csv")[["id", "name"]]
    return {"races": races, "entries": out, "drivers": drivers, "constructors": constructors}


def ingest_f1(store: Any, *, seasons: Optional[Sequence[int]] = None, refresh: Optional[bool] = None,
              fetcher: Optional[Fetcher] = None) -> IngestReport:
    fetcher = fetcher or Fetcher()
    report = IngestReport(league=LEAGUE)
    try:
        got = fetcher.fetch(F1DB_URL, provider=PROVIDER, relpath="f1db/f1db-csv.zip", max_age_hours=6.0,
                            refresh=refresh, license=_LICENCE)
    except NotFound as exc:
        raise DataUnavailable(f"the F1DB release file is not reachable ({exc}); import results with "
                              "`sportspred import events --league f1`") from exc
    if got.stale:
        report.warnings.append(f"refresh failed; using the copy collected {got.collected_at_utc}")
    source = got.register(store, notes="F1DB release (CSV)")
    report.sources.append(source)
    with zipfile.ZipFile(got.path) as z:
        data = normalize_f1db(z, min(seasons) if seasons else FIRST_SEASON)
    races, entries = data["races"], data["entries"]

    # ---- participants: drivers and constructors, by the source's stable ids
    used = data["drivers"][data["drivers"]["id"].isin(set(entries["driver"]))]
    # ``name`` is the name he races under (Lance Stroll), not the full legal one.
    driver_ids = store.resolve_participants(used, sport="motorsport", scope=LEAGUE, kind="person", name_col="name",
                                            provider=PROVIDER, provider_id_col="id", short_name_col="abbreviation")
    driver_of = dict(zip(used["id"], driver_ids))
    teams = data["constructors"][data["constructors"]["id"].isin(set(entries["constructor"]))]
    team_ids = store.resolve_participants(teams, sport="motorsport", scope=CONSTRUCTOR_SCOPE, kind="team",
                                          name_col="name", provider=PROVIDER, provider_id_col="id")
    team_of = dict(zip(teams["id"], team_ids))
    entries = entries.assign(driver_id=entries["driver"].map(driver_of), constructor_id=entries["constructor"].map(team_of))

    # ---- events: a race with a classification is final; one without is scheduled
    by_race = {r: part for r, part in entries.groupby("race", sort=False)}
    latest_field: List[str] = []
    rows = []
    for race in races.sort_values("start_time_utc").itertuples():
        rid = str(int(race.id))
        part = by_race.get(rid)
        if part is not None:
            latest_field = part.loc[part["started"], "driver_id"].tolist()
        rows.append({
            "league": LEAGUE, "season": str(int(race.year)), "start_time_utc": race.start_time_utc,
            "time_precision": race.time_precision, "status": "final" if part is not None else "scheduled",
            "provider": PROVIDER, "provider_event_id": rid, "stage": "regular",
            "venue_name": race.circuit_name, "venue_city": race.circuit_place, "neutral_site": 1,
            # Who races next is only known from the entry list; until a grid is imported the
            # field of the latest race stands in (it is replaced by the classification later).
            "participants": part["driver_id"].tolist() if part is not None else list(latest_field),
            "format": {"name": race.officialName if isinstance(race.officialName, str) else race.gp_name,
                       "grand_prix": race.grandPrixId, "round": int(race.round), "circuit": race.circuitId,
                       "circuit_type": race.circuitType, "sprint_weekend": bool(race.has_sprint),
                       "qualifying_utc": None if pd.isna(race.qualifying_utc) else race.qualifying_utc.strftime("%Y-%m-%dT%H:%M:%SZ")},
            "source_id": source,
        })
    events, up = store.upsert_events(pd.DataFrame(rows), sport="motorsport")
    report.warnings += up.issues
    eid_of = dict(zip(events["provider_event_id"].astype(str), events["event_id"]))
    entries = entries.assign(event_id=entries["race"].map(eid_of))
    entries = entries[entries["event_id"].notna()]

    results = []
    for event_id, part in entries.groupby("event_id", sort=False):
        ranked = part[part["classified"]].sort_values("position")
        if ranked.empty:
            store.log_issue("race_without_classification", str(event_id), league=LEAGUE, event_id=event_id)
            continue
        results.append({"event_id": event_id, "winner_id": ranked["driver_id"].iloc[0],
                        "ranking": ranked["driver_id"].tolist(), "source_id": source})
    store.upsert_results(pd.DataFrame(results))

    meta = races.assign(event_id=races["id"].astype(int).astype(str).map(eid_of))[
        ["event_id", "year", "round", "grandPrixId", "circuitId", "circuitType", "has_sprint", "qualifying_utc"]]
    meta = meta.rename(columns={"grandPrixId": "grand_prix", "circuitId": "circuit", "circuitType": "circuit_type"})
    meta["qualifying_utc"] = meta["qualifying_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    store.write_table(LEAGUE, "races", meta, source_ids=[source], mode="replace")
    keep = ["event_id", "driver_id", "constructor_id", "driver", "constructor", "order", "position", "position_text",
            "started", "classified", "grid", "pit_lane", "quali_pos", "quali_gap_pct", "laps", "reason"]
    store.write_table(LEAGUE, "race_entries", entries[keep], source_ids=[source], mode="replace")
    report.tables["races"], report.tables["race_entries"] = len(meta), len(entries)

    stored = store.events(LEAGUE, with_results=True)
    report.n_events = len(stored)
    report.n_results = int(stored["winner_id"].notna().sum())
    report.n_upcoming = len(store.upcoming(LEAGUE))
    if len(stored):
        report.first_event = stored["start_time_utc"].min().strftime("%Y-%m-%d")
        report.last_event = stored["start_time_utc"].max().strftime("%Y-%m-%d")
    return report
