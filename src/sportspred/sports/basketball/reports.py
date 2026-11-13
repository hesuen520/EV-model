"""The official NBA injury report as it stood before each day's games.

The archive (see :mod:`sportspred.adapters.nba_injury_report`) holds every
published version of the league's report since 2018-12-17.  That makes an
honest backtest of availability possible: for each game we can know what the
league had published *before* it, rather than who turned out to play.

**Which version counts.**  One instant per slate: the newest version
published at least ``lead_minutes`` before the first tip-off of the game's
Pacific/Honolulu day (for the NBA, the same games as the US calendar day).
Every game of the slate uses the report as it stood then.  Later versions -
including the late-afternoon updates before a night game - are deliberately
not used: features of a slate are built from one information time, the same
rule every other feature follows, and what a live user has is then never
*less* than what the backtest had.

A team with nobody listed does not appear in a version at all, so "nobody
listed" and "team not covered" look the same in a version; the table records
per game whether any version covering that game existed (``on_file``).
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import pandas as pd

from ...adapters.http import Fetcher, NotFound
from ...adapters.nba_injury_report import injury_report_url, normalize_injury_report
from ...core.ids import person_key
from ...core.timeutil import DISPLAY_TZ_NAME

__all__ = ["FIRST_REPORT_SEASON", "LEAD_MINUTES", "TABLE", "pregame_reports", "load_archive"]

FIRST_REPORT_SEASON = 2019          # the archive starts on 2018-12-17, in the 2018-19 season
LEAD_MINUTES = 60.0
TABLE = "pregame_report"
_LICENCE = "sportsdataverse-data public release (official NBA injury report archive)"


def load_archive(fetcher: Fetcher, seasons: Sequence[int], current: int, refresh: Optional[bool] = None):
    """Normalised listings of every archived season in ``seasons`` and the
    fetched files (for provenance).  A season with no file is skipped."""
    parts, fetched = [], []
    for season in seasons:
        if season < FIRST_REPORT_SEASON:
            continue
        try:
            got = fetcher.fetch(injury_report_url(season), provider="sportsdataverse",
                                relpath=f"nba_stats_injury_report/injury_report_{season}.parquet",
                                max_age_hours=6.0 if season >= current else None, refresh=refresh, license=_LICENCE)
        except NotFound:
            continue
        fetched.append(got)
        raw = pd.read_parquet(got.path)
        if len(raw):
            parts.append(normalize_injury_report(raw))
    listings = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return listings, fetched


def pregame_reports(listings: pd.DataFrame, events: pd.DataFrame, team_ids: Mapping[str, str], *,
                    lead_minutes: float = LEAD_MINUTES) -> pd.DataFrame:
    """One row per listed player in the version in force for each game.

    ``listings``: output of ``normalize_injury_report`` (all versions).
    ``events``: ``event_id, start_time_utc`` (tz-aware), ``home_id, away_id``.
    ``team_ids``: report team name -> participant id.

    Returns two frames.  The first has one row per listed player: ``event_id,
    team_id, player_name, player_key, status, published_at_utc, cutoff_utc``.
    The second has one row per game that a version covered at all
    (``event_id, published_at_utc``) - a team with nobody listed has no rows
    in the first frame, so coverage has to be recorded separately.
    """
    empty = pd.DataFrame(columns=["event_id", "team_id", "player_name", "player_key", "status",
                                  "published_at_utc", "cutoff_utc"])
    covered = pd.DataFrame(columns=["event_id", "published_at_utc"])
    if listings is None or listings.empty or events.empty:
        return empty, covered
    rep = listings.assign(team_id=listings["team_name"].map(team_ids))
    rep = rep[rep["team_id"].notna()]
    ev = events[["event_id", "start_time_utc", "home_id", "away_id"]].copy()
    slate = ev["start_time_utc"].dt.tz_convert(DISPLAY_TZ_NAME).dt.strftime("%Y-%m-%d")
    ev["cutoff"] = ev.groupby(slate)["start_time_utc"].transform("min") - pd.Timedelta(minutes=lead_minutes)
    ev["game_date"] = ev["start_time_utc"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    long = pd.concat([ev.rename(columns={"home_id": "team_id"})[["event_id", "team_id", "game_date", "cutoff"]],
                      ev.rename(columns={"away_id": "team_id"})[["event_id", "team_id", "game_date", "cutoff"]]])
    merged = rep.merge(long, on=["team_id", "game_date"], how="inner")
    merged = merged[merged["published_at"] <= merged["cutoff"]]          # nothing published after the slate's cut-off
    if merged.empty:
        return empty, covered
    # The version in force for a game: the newest one, before the cut-off, that lists that game at all.
    newest = merged.groupby("event_id")["published_at"].transform("max")
    current = merged[merged["published_at"] == newest]
    covered = (current.drop_duplicates("event_id")[["event_id", "published_at"]]
               .assign(published_at_utc=lambda d: d["published_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
               [["event_id", "published_at_utc"]].reset_index(drop=True))
    listed = current[current["status"].notna() & current["player_name"].notna()]
    out = pd.DataFrame({
        "event_id": listed["event_id"].to_numpy(),
        "team_id": listed["team_id"].to_numpy(),
        "player_name": listed["player_name"].to_numpy(),
        "player_key": [person_key(n) for n in listed["player_name"]],
        "status": listed["status"].to_numpy(),
        "published_at_utc": listed["published_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy(),
        "cutoff_utc": listed["cutoff"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy(),
    })
    out = out.drop_duplicates(["event_id", "team_id", "player_key"], keep="last").reset_index(drop=True)
    return out, covered
