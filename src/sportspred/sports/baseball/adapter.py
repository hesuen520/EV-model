"""MLB ingestion from the Chadwick Bureau's GitHub mirrors of Retrosheet.

Sources (all under ``raw.githubusercontent.com/chadwickbureau``):

* ``retrosheet/master/seasons/<year>/GL<year>.TXT`` - Retrosheet game logs, one
  161-field row per regular-season game; ``retrosheet/master/gamelog/GLWC.TXT``,
  ``GLDV.TXT``, ``GLLC.TXT`` and ``GLWS.TXT`` - the same rows for every
  postseason game;
* ``retrosheet/master/seasons/<year>/TEAM<year>`` (club names of that season),
  ``reference/ballparks.csv`` (park names and cities) and
  ``reference/biofile.csv`` (throwing hand);
* ``retrosplits/master/daybyday/playing-<year>.csv`` - one row per player per
  game, from which each pitcher's line (start or relief, outs, batters faced,
  runs, home runs, walks, hit batters, strikeouts) is taken;
* ``retrosheet/master/seasons/<year>/<year>schedule.csv`` - the original
  schedule of a season, read only for seasons that have no game log yet.

What the files do and do not contain:

* the calendar date of each game, its doubleheader game number and a day/night
  flag, but **no start time** - events are stored with date precision;
* the park, the final score, team batting totals and the pitchers and lineups
  that **actually started**.  Who was announced as the probable starter before
  the game is not recorded anywhere, so starter features built from these
  files are a reconstruction, not pregame information;
* **no** betting odds, weather, injuries or announced lineups;
* complete seasons only: a season appears months after it ends, and the files
  carry no games in progress.

Retrosheet's notice applies to everything read here: "The information used
here was obtained free of charge from and is copyrighted by Retrosheet.
Interested parties may contact Retrosheet at www.retrosheet.org."
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ...adapters.http import Fetched, Fetcher, NotFound
from ...core.timeutil import utcnow
from ...sports import DataUnavailable
from ..base import IngestReport

__all__ = [
    "RETROSHEET_NOTICE", "PROVIDER", "FIRST_SEASON", "ADAPTER_LEAGUES", "N_GAMELOG_FIELDS",
    "normalize_gamelog", "normalize_schedule", "normalize_pitching", "neutral_sites", "ingest_retrosheet",
]

log = logging.getLogger(__name__)

RETROSHEET_NOTICE = (
    "The information used here was obtained free of charge from and is copyrighted by Retrosheet. "
    "Interested parties may contact Retrosheet at www.retrosheet.org."
)
_SPLITS_LICENSE = RETROSHEET_NOTICE + " Day-by-day files: Open Database License (ODbL) 1.0."
_RS = "https://raw.githubusercontent.com/chadwickbureau/retrosheet/master"
_SPLITS = "https://raw.githubusercontent.com/chadwickbureau/retrosplits/master/daybyday"
PROVIDER = "retrosheet"
ADAPTER_LEAGUES = ("mlb",)
#: Club codes are linked across relocations from this season on (see ``_FRANCHISE``).
FIRST_SEASON = 2000
N_GAMELOG_FIELDS = 161
#: The mirror's file-name case differs between seasons.
_GAMELOG_NAMES = ("GL{y}.TXT", "gl{y}.txt")
_POSTSEASON_LOGS = ("GLWC.TXT", "GLDV.TXT", "GLLC.TXT", "GLWS.TXT")
#: Files that change upstream (latest seasons, all-time postseason logs,
#: reference tables) are re-downloaded when the cached copy is older than this.
_REFRESH_HOURS = 24.0
#: A park hosting fewer regular-season home games of a club than this in a
#: season is a special venue (London, Tokyo, a 2020 postseason bubble ...).
HOME_PARK_MIN_GAMES = 10

#: Retrosheet code of a relocated or renamed club -> its current code.
_FRANCHISE = {"FLO": "MIA", "MON": "WAS", "OAK": "ATH"}
#: Retrosheet labels a club by the city of its code; these are the clubs' own names.
_DISPLAY = {"ANA": "Los Angeles Angels", "ATH": "Athletics"}

#: Team batting totals in game-log order; the visitors' block starts at field
#: 22 and the home block at field 50 (1-based, as in Retrosheet's field list).
_BATTING = ("ab", "h", "d", "t", "hr", "rbi", "sh", "sf", "hbp", "bb")
_BATTING_KEPT = ("ab", "h", "d", "t", "hr", "hbp", "bb")
_PITCHING = {"P_GS": "is_start", "P_OUT": "outs", "P_TBF": "bf", "P_R": "runs", "P_HR": "hr",
             "P_BB": "bb", "P_HP": "hbp", "P_SO": "so"}
#: Where a game has records from several kinds of source file, prefer the
#: event file, then the deduced event file, then the box score.
_SOURCE_RANK = {"evt": 0, "ded": 1, "box": 2}


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _iso_date(yyyymmdd: pd.Series) -> pd.Series:
    """``20230514`` -> ``2023-05-14``; anything else -> None."""
    ok = yyyymmdd.str.fullmatch(r"\d{8}")
    text = yyyymmdd.str[:4] + "-" + yyyymmdd.str[4:6] + "-" + yyyymmdd.str[6:8]
    return text.where(ok, None)


def normalize_gamelog(raw: pd.DataFrame, league: str, stage: str = "regular") -> pd.DataFrame:
    """Retrosheet game-log rows -> one canonical row per game (club codes still Retrosheet's).

    ``raw`` is the header-less file read as strings.  The provider event id is
    Retrosheet's game id (home code + date + game number), so the two games of
    a doubleheader are distinct events.
    """
    if raw.shape[1] != N_GAMELOG_FIELDS:
        raise ValueError(f"a Retrosheet game log has {N_GAMELOG_FIELDS} fields, this file has {raw.shape[1]}")
    number = _num(raw[1]).fillna(0).astype(int)        # 0 single game, 1 / 2 games of a doubleheader
    out = pd.DataFrame({
        "provider_event_id": raw[6] + raw[0] + raw[1],
        "league": league,
        "season": raw[0].str[:4],
        "stage": stage,
        "start_time_utc": pd.to_datetime(raw[0], format="%Y%m%d", utc=True),
        "time_precision": "date",
        "status": "final",
        "provider": PROVIDER,
        "home_code": raw[6],
        "away_code": raw[3],
        "home_score": _num(raw[10]),
        "away_score": _num(raw[9]),
        "venue_id": raw[16].where(raw[16] != "", None),
        "game_number": number.clip(lower=1),
        "doubleheader": (number > 0).astype(int),
        "day_night": raw[12].map({"D": "day", "N": "night"}),
        # A suspended game is listed on the date it began; its result only
        # existed once it was completed on this later date.
        "resumed_on": _iso_date(raw[13].str.split(",").str[0]),
        "forfeit": raw[14] != "",
        "home_sp": raw[103].where(raw[103] != "", None),
        "away_sp": raw[101].where(raw[101] != "", None),
        "home_sp_name": raw[104],
        "away_sp_name": raw[102],
    })
    for side, first in (("away", 21), ("home", 49)):
        for k, stat in enumerate(_BATTING):
            if stat in _BATTING_KEPT:
                out[f"{side}_{stat}"] = _num(raw[first + k])
    return out.drop_duplicates("provider_event_id", keep="last").reset_index(drop=True)


def normalize_schedule(raw: pd.DataFrame, league: str) -> pd.DataFrame:
    """Retrosheet original-schedule rows -> scheduled events (date precision).

    Postponed games are dropped: their make-up date is known, but not the game
    number they will carry, so they cannot be keyed reliably in advance.
    """
    raw = raw[raw["Postponed"].fillna("") == ""]
    number = _num(raw["Num"]).fillna(0).astype(int)
    out = pd.DataFrame({
        "provider_event_id": raw["Home"] + raw["Date"] + raw["Num"],
        "league": league,
        "season": raw["Date"].str[:4],
        "stage": "regular",
        "start_time_utc": pd.to_datetime(raw["Date"], format="%Y%m%d", utc=True),
        "time_precision": "date",
        "status": "scheduled",
        "provider": PROVIDER,
        "home_code": raw["Home"],
        "away_code": raw["Visitor"],
        "venue_id": raw["Location"].where(raw["Location"] != "", None) if "Location" in raw else None,
        "game_number": number.clip(lower=1),
        "doubleheader": (number > 0).astype(int),
        "day_night": raw["Day/Night"].str.upper().map({"D": "day", "N": "night"}),
    })
    return out.drop_duplicates("provider_event_id", keep="last").reset_index(drop=True)


def normalize_pitching(raw: pd.DataFrame) -> pd.DataFrame:
    """Day-by-day player rows -> one pitching line per (game, club, pitcher)."""
    pit = raw[raw["P_G"] > 0]
    rank = pit["game.source"].map(_SOURCE_RANK).fillna(len(_SOURCE_RANK))
    pit = pit[rank == rank.groupby(pit["game.key"]).transform("min")]
    out = (pit.groupby(["game.key", "team.key", "person.key"], sort=False)[list(_PITCHING)]
           .sum(min_count=1).reset_index().rename(columns=_PITCHING))
    out["is_start"] = (out["is_start"] > 0).astype(int)
    return out.rename(columns={"game.key": "provider_event_id", "team.key": "team_code",
                               "person.key": "pitcher_id"})


def neutral_sites(games: pd.DataFrame) -> np.ndarray:
    """1 where the park is not one of the home club's regular parks that season.

    Uses only the season's list of games (who hosts whom, where), which is
    public before any of them is played.
    """
    regular = games[games["stage"] == "regular"]
    counts = regular.groupby(["season", "home_code", "venue_id"]).size()
    key = pd.MultiIndex.from_frame(games[["season", "home_code", "venue_id"]])
    n = counts.reindex(key).fillna(0).to_numpy()
    return (n < HOME_PARK_MIN_GAMES).astype(int)


class _Session:
    """One ingestion run: fetches files, registers them and collects warnings."""

    def __init__(self, store, fetcher: Fetcher, refresh: Optional[bool], report: IngestReport) -> None:
        self.store, self.fetcher, self.refresh, self.report = store, fetcher, refresh, report

    def get(self, url: str, relpath: str, *, mutable: bool, notes: str,
            license: str = RETROSHEET_NOTICE) -> Tuple[Fetched, str]:
        got = self.fetcher.fetch(url, provider=PROVIDER, relpath=relpath, refresh=self.refresh,
                                 max_age_hours=_REFRESH_HOURS if mutable else None, license=license)
        if got.stale:
            self.report.warnings.append(
                f"{relpath}: refresh failed; using the copy collected {got.collected_at_utc}")
        sid = got.register(self.store, notes=notes)
        self.report.sources.append(sid)
        return got, sid

    def gamelog(self, season: int, mutable: bool) -> Tuple[Fetched, str]:
        for name in _GAMELOG_NAMES:
            rel = f"seasons/{season}/{name.format(y=season)}"
            try:
                return self.get(f"{_RS}/{rel}", rel, mutable=mutable, notes=f"game log {season}")
            except NotFound:
                continue
        raise NotFound(f"no Retrosheet game log for {season} in the mirror")


def _read_gamelog(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, header=None, dtype=str, keep_default_na=False)


def _team_names(session: _Session, seasons: Sequence[int], current: int) -> pd.DataFrame:
    """Club names per season (``code, name, season``) from the ``TEAM<year>`` files."""
    parts: List[pd.DataFrame] = []
    for season in seasons:
        rel = f"seasons/{season}/TEAM{season}"
        try:
            got, _ = session.get(f"{_RS}/{rel}", rel, mutable=season >= current - 1,
                                 notes=f"club names {season}")
        except NotFound:
            session.report.warnings.append(f"season {season}: no club-name file; clubs keep their codes as names")
            continue
        teams = pd.read_csv(got.path, header=None, names=["code", "lg", "city", "nick"], dtype=str,
                            keep_default_na=False)
        parts.append(pd.DataFrame({"code": teams["code"], "name": teams["city"] + " " + teams["nick"],
                                   "season": season}))
    if not parts:
        return pd.DataFrame({"code": pd.Series(dtype=str), "name": pd.Series(dtype=str),
                             "season": pd.Series(dtype=int)})
    return pd.concat(parts, ignore_index=True)


def _resolve_teams(store, league: str, games: pd.DataFrame, names: pd.DataFrame) -> Dict[str, str]:
    """Retrosheet club code -> participant id, one participant per franchise."""
    used = pd.concat([
        games[["home_code", "season"]].set_axis(["code", "season"], axis=1),
        games[["away_code", "season"]].set_axis(["code", "season"], axis=1),
    ]).drop_duplicates()
    used["season"] = used["season"].astype(int)
    known = used.merge(names, on=["code", "season"], how="left")
    known["name"] = known["name"].fillna(known["code"])
    known["franchise"] = known["code"].replace(_FRANCHISE)
    known = known.sort_values(["season", "code"])
    # Register every historical name (oldest first) so each becomes an alias.
    first = known.drop_duplicates(["franchise", "name"], keep="first")
    first = first.assign(pid=store.resolve_participants(
        first, sport="baseball", scope=league, kind="team", name_col="name", provider=PROVIDER,
        provider_id_col="franchise", short_name_col="code").to_numpy())
    pid_of = dict(zip(first["franchise"], first["pid"]))
    latest = known.drop_duplicates("franchise", keep="last")
    with store.conn() as con:   # show the current name
        for franchise, name, code in zip(latest["franchise"], latest["name"], latest["code"]):
            con.execute("UPDATE participants SET name=?, short_name=? WHERE participant_id=?",
                        (_DISPLAY.get(franchise, name), code, pid_of[franchise]))
    for franchise, display in _DISPLAY.items():
        if franchise in pid_of:
            store.add_alias(league, display, pid_of[franchise], source=PROVIDER)
    return {code: pid_of[franchise] for code, franchise in zip(known["code"], known["franchise"])}


def _merge_table(store, league: str, name: str, new: pd.DataFrame, key: str, source_ids: List[str]) -> int:
    """Write ``new`` rows, keeping stored rows whose ``key`` is not being replaced
    (an ingest of a few seasons must not drop the statistics of the others)."""
    if store.has_table(league, name):
        old = store.read_table(league, name)
        new = pd.concat([old[~old[key].isin(set(new[key]))], new], ignore_index=True)
    store.write_table(league, name, new, source_ids=source_ids)
    return len(new)


def _pitching_lines(session: _Session, seasons: Sequence[int], current: int) -> Tuple[pd.DataFrame, List[str]]:
    parts: List[pd.DataFrame] = []
    sources: List[str] = []
    cols = ["game.key", "game.source", "team.key", "person.key", "P_G", *_PITCHING]
    for season in seasons:
        rel = f"daybyday/playing-{season}.csv"
        try:
            got, sid = session.get(f"{_SPLITS}/playing-{season}.csv", rel, mutable=season >= current - 1,
                                   notes=f"player day-by-day {season}", license=_SPLITS_LICENSE)
        except NotFound:
            session.report.warnings.append(
                f"season {season}: no player day-by-day file; starter and bullpen features will be missing")
            continue
        sources.append(sid)
        parts.append(normalize_pitching(pd.read_csv(got.path, usecols=cols, low_memory=False)))
    lines = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return lines, sources


def _future_schedules(session: _Session, league: str, after: int, current: int) -> pd.DataFrame:
    """Scheduled games of seasons newer than the newest game log, if the
    mirror has published their schedule."""
    parts: List[pd.DataFrame] = []
    today = pd.Timestamp(utcnow()).normalize()
    for season in range(after + 1, current + 2):
        rel = f"seasons/{season}/{season}schedule.csv"
        try:
            got, sid = session.get(f"{_RS}/{rel}", rel, mutable=True, notes=f"schedule {season}")
        except NotFound:
            session.report.warnings.append(f"season {season}: no schedule file in the mirror")
            continue
        sched = normalize_schedule(pd.read_csv(got.path, dtype=str, keep_default_na=False), league)
        past = sched["start_time_utc"] <= today
        if past.any():
            session.report.warnings.append(
                f"season {season}: {int(past.sum())} scheduled games are in the past and have no result in "
                "the mirror; they are not stored and ratings do not include them")
        sched = sched[~past].assign(source_id=sid)
        if len(sched):
            parts.append(sched)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def ingest_retrosheet(store, league: str, *, seasons: Optional[Sequence[int]] = None,
                      refresh: Optional[bool] = None, fetcher: Optional[Fetcher] = None) -> IngestReport:
    """Download, normalise and store Retrosheet seasons for ``league``."""
    if league not in ADAPTER_LEAGUES:
        raise DataUnavailable(
            f"no public dataset adapter for baseball league {league!r}; import results with "
            "`sportspred import events`")
    report = IngestReport(league=league)
    session = _Session(store, fetcher or Fetcher(), refresh, report)
    current = utcnow().year
    wanted = sorted(set(seasons)) if seasons else list(range(FIRST_SEASON, current + 1))
    if any(s < FIRST_SEASON for s in wanted):
        report.warnings.append(f"seasons before {FIRST_SEASON} are not supported and were skipped")
        wanted = [s for s in wanted if s >= FIRST_SEASON]

    parts: List[pd.DataFrame] = []
    have: List[int] = []
    for season in wanted:
        try:
            got, sid = session.gamelog(season, mutable=season >= current - 1)
        except NotFound:
            report.warnings.append(
                f"season {season}: no game log in the mirror" + (" yet" if season >= current - 1 else ""))
            continue
        parts.append(normalize_gamelog(_read_gamelog(got.path), league).assign(source_id=sid))
        have.append(season)
    if not parts:
        raise DataUnavailable(f"no Retrosheet game logs could be obtained for {league}")
    for name in _POSTSEASON_LOGS:
        try:
            got, sid = session.get(f"{_RS}/gamelog/{name}", f"gamelog/{name}", mutable=True,
                                   notes=f"postseason game log {name}")
        except NotFound:
            report.warnings.append(f"{name}: postseason game log missing upstream")
            continue
        post = normalize_gamelog(_read_gamelog(got.path), league, stage="playoffs").assign(source_id=sid)
        parts.append(post[post["season"].astype(int).isin(have)])
    games = pd.concat(parts, ignore_index=True)
    upcoming = _future_schedules(session, league, max(have), current) if max(wanted) >= current else pd.DataFrame()
    if len(upcoming):
        games = pd.concat([games, upcoming], ignore_index=True)
    games = games.drop_duplicates("provider_event_id", keep="first").reset_index(drop=True)

    # ---- venues and participants
    got, _ = session.get(f"{_RS}/reference/ballparks.csv", "reference/ballparks.csv", mutable=True,
                         notes="ballparks")
    parks = pd.read_csv(got.path, dtype=str, keep_default_na=False).set_index("PARKID")
    games["venue_name"] = games["venue_id"].map(parks["NAME"].str.split(";").str[0])
    games["venue_city"] = games["venue_id"].map(parks["CITY"])
    games["neutral_site"] = neutral_sites(games)
    seasons_seen = sorted(games["season"].astype(int).unique())
    pid_of = _resolve_teams(store, league, games, _team_names(session, seasons_seen, current))
    games["home_id"] = games["home_code"].map(pid_of)
    games["away_id"] = games["away_code"].map(pid_of)
    games["format"] = [
        {"game_number": int(n), "doubleheader": bool(dh), **({"day_night": dn} if isinstance(dn, str) else {})}
        for n, dh, dn in zip(games["game_number"], games["doubleheader"], games["day_night"])
    ]

    events, up = store.upsert_events(games, sport="baseball")
    report.warnings += up.issues
    eid_of = dict(zip(events["provider_event_id"], events["event_id"]))

    # ---- results
    final = events[events["status"] == "final"].copy()
    scored = final["home_score"].notna() & final["away_score"].notna() & ~final["forfeit"].astype(bool)
    for _, row in final[~scored].iterrows():
        store.log_issue("final_without_valid_score", f"{row['provider_event_id']}: forfeit or missing score",
                        league=league, event_id=row["event_id"])
    final = final[scored]
    final["detail"] = [None if r is None or r != r else {"resumed_on": r} for r in final["resumed_on"]]
    store.upsert_results(final[["event_id", "home_score", "away_score", "detail", "source_id"]])

    # ---- bulk tables
    log_sources = sorted(set(final["source_id"]))
    stat_cols = ["event_id", "resumed_on", "home_sp", "away_sp", "home_sp_name", "away_sp_name",
                 *[f"{side}_{stat}" for side in ("home", "away") for stat in _BATTING_KEPT]]
    report.tables["game_stats"] = _merge_table(store, league, "game_stats", final[stat_cols], "event_id",
                                               log_sources)
    lines, line_sources = _pitching_lines(session, have, current)
    if len(lines):
        lines["event_id"] = lines["provider_event_id"].map(eid_of)
        lines["team_id"] = lines["team_code"].map(pid_of)
        lines = lines[lines["event_id"].notna() & lines["team_id"].notna()]
        lines = lines.drop(columns=["provider_event_id", "team_code"])
        report.tables["pitching"] = _merge_table(store, league, "pitching", lines, "event_id", line_sources)
        got, sid = session.get(f"{_RS}/reference/biofile.csv", "reference/biofile.csv", mutable=True,
                               notes="player register (throwing hand)")
        bio = pd.read_csv(got.path, dtype=str, usecols=["PLAYERID", "LAST", "NICKNAME", "THROWS"])
        bio = bio[bio["PLAYERID"].isin(set(lines["pitcher_id"]))]
        pitchers = pd.DataFrame({
            "pitcher_id": bio["PLAYERID"],
            "name": (bio["NICKNAME"].fillna("") + " " + bio["LAST"].fillna("")).str.strip(),
            "throws": bio["THROWS"].where(bio["THROWS"].isin(["L", "R"]), None),
        })
        report.tables["pitchers"] = _merge_table(store, league, "pitchers", pitchers, "pitcher_id", [sid])
    else:
        report.warnings.append("no pitching lines available")

    stored = store.events(league, with_results=True)
    report.n_events = len(stored)
    report.n_results = int(stored["winner_id"].notna().sum())
    report.n_upcoming = len(store.upcoming(league))
    if len(stored):
        report.first_event = stored["start_time_utc"].min().strftime("%Y-%m-%d")
        report.last_event = stored["start_time_utc"].max().strftime("%Y-%m-%d")
    if not report.n_upcoming:
        report.warnings.append(
            "no upcoming games: the mirror publishes completed seasons only and no schedule of a "
            "future season is available; import a schedule to get predictions")
    return report
