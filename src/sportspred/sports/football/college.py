"""College football (FBS) ingestion from the sportsdataverse public release files.

Source: https://github.com/sportsdataverse/sportsdataverse-data, release tag
``cfb_schedules``, one parquet per season from 2001 (cfbfastR exports of the
ESPN / CollegeFootballData schedule).

What the files do and do not contain:

* schedules with UTC kickoff times (a flag marks times that are still to be
  announced), final scores, neutral-site flag, each team's division, and the
  remaining schedule of the current season;
* **no** betting lines, quarterbacks, injuries, box scores or weather.

Only games with at least one FBS team are stored.  Games between two FBS teams
are the league's events proper; a game against a lower-division opponent is
kept as context (it is part of the FBS team's record and rest) and every
lower-division opponent is rated as one pooled entity per division, because a
single game a year cannot rate such a team on its own.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ...adapters.http import Fetcher, NotFound
from ...core.timeutil import utcnow
from ...sports import DataUnavailable
from ..base import IngestReport

__all__ = ["ingest_cfb", "normalize_cfb", "CFB_URL", "CFB_LICENSE", "CFB_FIRST_SEASON"]

log = logging.getLogger(__name__)

CFB_URL = ("https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
           "cfb_schedules/cfb_schedules_{season}.parquet")
CFB_LICENSE = "sportsdataverse-data public release (ESPN / CollegeFootballData-derived); no licence file published"
CFB_FIRST_SEASON = 2001
PROVIDER = "espn"                  # game and team IDs in these files are ESPN's
_LOCAL_TZ = "America/New_York"
_STATUS = {"STATUS_CANCELED": "canceled", "STATUS_CANCELLED": "canceled", "STATUS_POSTPONED": "postponed",
           "STATUS_FORFEIT": "unknown"}


def current_cfb_season(today: Optional[pd.Timestamp] = None) -> int:
    """A season is labelled by the year it starts in; the next one is published around August."""
    today = pd.Timestamp(utcnow()) if today is None else pd.Timestamp(today)
    return today.year if today.month >= 8 else today.year - 1


def normalize_cfb(raw: pd.DataFrame, league: str, season: int) -> pd.DataFrame:
    """Schedule rows with an FBS participant -> canonical event rows (IDs still ESPN's)."""
    div = lambda c: raw[c].astype("string").str.lower().fillna("other")  # noqa: E731
    keep = raw["season_type"].isin(["regular", "postseason"]) & ((div("home_division") == "fbs")
                                                                 | (div("away_division") == "fbs"))
    raw = raw[keep]
    start = pd.to_datetime(raw["start_date"], utc=True, format="ISO8601", errors="coerce")
    local = start.dt.tz_convert(_LOCAL_TZ)
    # Older files have no flag: there, local midnight is the placeholder for
    # an unannounced kickoff.  Where the flag exists it alone decides, so a
    # genuine late kickoff (Hawaii) is not mistaken for a placeholder.
    flag = raw["start_time_tbd"]
    midnight = (local.dt.hour == 0) & (local.dt.minute == 0)
    tbd = flag.fillna(False).astype(bool) | (flag.isna() & midnight)
    # A kickoff that is still "to be announced" is a placeholder at local
    # midnight: keep the calendar date and say that the time is not known.
    date_only = pd.to_datetime(local.dt.strftime("%Y-%m-%d"), format="%Y-%m-%d").dt.tz_localize("UTC")
    hp = pd.to_numeric(raw["home_points"], errors="coerce")
    ap = pd.to_numeric(raw["away_points"], errors="coerce")
    played = raw["completed"].fillna(False).astype(bool)
    scored = played & hp.notna() & ap.notna() & (hp != ap) & ((hp + ap) > 0)
    status = pd.Series(np.where(scored, "final", np.where(played, "unknown", "scheduled")), index=raw.index)
    if "status" in raw.columns:
        mapped = raw["status"].map(_STATUS)
        status = status.where(mapped.isna() | scored, mapped)
    df = pd.DataFrame({
        "provider_event_id": raw["game_id"].astype("int64").astype(str),
        "league": league,
        "season": str(season),
        "start_time_utc": start.where(~tbd, date_only),
        "time_precision": np.where(tbd, "date", "datetime"),
        "local_date": local.dt.strftime("%Y-%m-%d"),
        "status": status,
        "provider": PROVIDER,
        "home_espn": raw["home_id"].astype("int64"),
        "away_espn": raw["away_id"].astype("int64"),
        "home_name": raw["home_team"].astype(str),
        "away_name": raw["away_team"].astype(str),
        "home_division": div("home_division"),
        "away_division": div("away_division"),
        "home_score": hp.where(scored),
        "away_score": ap.where(scored),
        "neutral_site": raw["neutral_site"].fillna(False).astype(bool).astype(int),
        "venue_name": raw["venue"],
        "stage": np.where(raw["season_type"] == "postseason", "postseason", "regular"),
        "week": pd.to_numeric(raw["week"], errors="coerce"),
        "conference_game": raw["conference_game"].fillna(False).astype(bool).astype(int),
    })
    df = df[df["start_time_utc"].notna() & (df["home_espn"] != df["away_espn"])]
    df = df.drop_duplicates("provider_event_id", keep="last")
    # The source occasionally carries one game twice under two IDs (a
    # date-only copy and a timed copy).  Keep one record per pairing and local
    # date, preferring the timed one, so a result is never applied twice.
    df = df.sort_values("time_precision")          # "date" < "datetime": the timed copy sorts last
    df = df.drop_duplicates(["home_espn", "away_espn", "local_date"], keep="last")
    return df.sort_values(["start_time_utc", "provider_event_id"]).reset_index(drop=True)


def ingest_cfb(store, league: str, *, seasons: Optional[Sequence[int]] = None,
               refresh: Optional[bool] = None, fetcher: Optional[Fetcher] = None) -> IngestReport:
    """Download, normalise and store FBS schedules and results."""
    fetcher = fetcher or Fetcher()
    current = current_cfb_season()
    wanted = list(seasons) if seasons else list(range(CFB_FIRST_SEASON, current + 1))
    report = IngestReport(league=league)
    parts: List[pd.DataFrame] = []
    for season in wanted:
        url = CFB_URL.format(season=season)
        try:
            got = fetcher.fetch(url, provider="sportsdataverse", relpath=f"cfb_schedules/cfb_schedules_{season}.parquet",
                                max_age_hours=6.0 if season >= current else None, refresh=refresh,
                                license=CFB_LICENSE)
        except NotFound:
            report.warnings.append(f"season {season}: no schedule file published")
            continue
        if got.stale:
            report.warnings.append(f"season {season}: refresh failed; using the copy collected {got.collected_at_utc}")
        sid = got.register(store, notes=f"{league} schedule {season}")
        report.sources.append(sid)
        part = normalize_cfb(pd.read_parquet(got.path), league, season)
        part["source_id"] = sid
        parts.append(part)
    if not parts:
        raise DataUnavailable(f"no schedule files could be obtained for {league}")
    sched = pd.concat(parts, ignore_index=True).drop_duplicates("provider_event_id", keep="last")
    sched = sched.reset_index(drop=True)

    teams = pd.concat([
        sched[["home_espn", "home_name", "start_time_utc"]].set_axis(["espn", "name", "t"], axis=1),
        sched[["away_espn", "away_name", "start_time_utc"]].set_axis(["espn", "name", "t"], axis=1),
    ]).sort_values("t")
    names = teams.drop_duplicates(["espn", "name"], keep="first")
    names = names.assign(pid=store.resolve_participants(
        names, sport="football", scope=league, kind="team", name_col="name", provider=PROVIDER,
        provider_id_col="espn").to_numpy())
    pid_of: Dict[int, str] = dict(zip(names["espn"].astype(int), names["pid"]))
    latest = teams.drop_duplicates("espn", keep="last")
    with store.conn() as con:   # show the current name
        for espn, name in zip(latest["espn"].astype(int), latest["name"]):
            con.execute("UPDATE participants SET name=? WHERE participant_id=?", (name, pid_of[espn]))
    sched["home_id"] = sched["home_espn"].astype(int).map(pid_of)
    sched["away_id"] = sched["away_espn"].astype(int).map(pid_of)

    events, up = store.upsert_events(sched, sport="football")
    report.warnings += up.issues
    # Records of the ingested seasons that the source no longer carries, or
    # that were duplicates of a game kept above, are retired (logged, never
    # deleted) so their results cannot be applied a second time.
    on_file = store.query(
        "SELECT event_id, season FROM events WHERE league=? AND provider=? AND status != 'canceled'",
        [league, PROVIDER])
    gone = on_file[on_file["season"].isin(set(sched["season"])) & ~on_file["event_id"].isin(set(events["event_id"]))]
    if len(gone):
        n = store.cancel_events(gone["event_id"].tolist(),
                                "record withdrawn by the source or a duplicate of another record of the same game")
        report.warnings.append(f"{n} duplicate or withdrawn game records were retired")
    final = events[events["status"] == "final"]
    store.upsert_results(final[["event_id", "home_score", "away_score", "source_id"]])
    unscored = int((events["status"] == "unknown").sum())
    if unscored:
        report.warnings.append(f"{unscored} games are marked completed without a usable score and are ignored")

    games = events[["event_id", "local_date", "week", "conference_game", "home_division", "away_division"]].copy()
    for side in ("home", "away"):
        fbs = games[f"{side}_division"] == "fbs"
        games[f"{side}_entity"] = events[f"{side}_id"].where(fbs, "pool:" + games[f"{side}_division"].astype(str))
    games["target"] = ((games["home_division"] == "fbs") & (games["away_division"] == "fbs")).astype(int)
    store.write_table(league, "games", games, source_ids=report.sources)
    report.tables["games"] = len(games)

    stored = store.events(league, with_results=True)
    report.n_events = len(stored)
    report.n_results = int(stored["home_score"].notna().sum())
    report.n_upcoming = len(store.upcoming(league))
    report.first_event = stored["start_time_utc"].min().strftime("%Y-%m-%d")
    report.last_event = stored["start_time_utc"].max().strftime("%Y-%m-%d")
    return report
