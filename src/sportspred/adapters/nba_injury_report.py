"""Official NBA injury report, every published version, from the
sportsdataverse public archive.

Source: ``nba_stats_injury_report`` release of sportsdataverse-data
(https://github.com/sportsdataverse/sportsdataverse-data), described there as
"The NBA's official injury report - every published version, one row per
listing", 2018-12-17 to the present, one file per season, no key.

What matters for honest use:

* each row belongs to a report *version* (``report_datetime``, US Eastern
  clock time).  The archive is refreshed periodically, so its newest version
  can be many hours older than the newest report the league has published.
  The version time is stored as ``published_at``; the freshness rules of the
  pregame checks are applied to that time, not to the time of download;
* a team that has not filed yet appears as ``NOT YET SUBMITTED``: that is
  recorded as a check with outcome ``not_published`` and completes nothing;
* statuses are the league's own: Out, Doubtful, Questionable, Probable,
  Available.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from ..core.timeutil import iso_z, parse_ts, utcnow
from .http import Fetcher

__all__ = ["injury_report_url", "normalize_injury_report", "import_latest_report"]

_SDV = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/nba_stats_injury_report"
_STATUS = {"out": "out", "doubtful": "doubtful", "questionable": "questionable", "probable": "probable",
           "available": "available"}


def injury_report_url(season: int) -> str:
    """``season`` is the year the season ends in (2027 for 2026-27)."""
    return f"{_SDV}/injury_report_{season}.parquet"


def _name(text: Any) -> Optional[str]:
    """``"Miller, Brandon"`` -> ``"Brandon Miller"`` (suffixes stay with the surname)."""
    if not isinstance(text, str) or not text.strip():
        return None
    if "," not in text:
        return text.strip()
    last, first = [p.strip() for p in text.split(",", 1)]
    return f"{first} {last}".strip()


def normalize_injury_report(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per listing with UTC version time and platform statuses.
    ``status`` is ``None`` for a team that had not submitted."""
    version = pd.to_datetime(raw["report_datetime"]).dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    out = pd.DataFrame({
        "published_at": version,
        "game_date": raw["game_date"].astype(str),
        "matchup": raw["report_matchup"],
        "team_name": raw["team"],
        "player_name": raw["report_player_name"].map(_name),
        "status": raw["current_status"].astype("string").str.lower().map(_STATUS),
        "detail": raw["reason"],
        "not_submitted": raw["reason"].astype("string").str.upper().eq("NOT YET SUBMITTED").fillna(False),
    })
    return out[out["published_at"].notna() & out["team_name"].notna()].reset_index(drop=True)


def import_latest_report(store: Any, raw: pd.DataFrame, *, now: Any = None, source_id: Optional[str] = None,
                         league: str = "nba") -> Dict[str, Any]:
    """Store the newest report version published at or before ``now`` for
    games that have not started, and log each team's ``injury_report`` check.

    Returns counts plus the version time, so the caller can show how old the
    information is.
    """
    now_dt = utcnow() if now is None else parse_ts(now)
    rep = normalize_injury_report(raw)
    rep = rep[rep["published_at"] <= pd.Timestamp(now_dt)]
    if rep.empty:
        return {"version_utc": None, "statements": 0, "teams_reported": 0, "teams_nothing_listed": 0,
                "teams_not_submitted": 0, "unknown_teams": 0}
    version = rep["published_at"].max()
    rep = rep[rep["published_at"] == version]
    ids = store.resolve_participants(rep.assign(name=rep["team_name"]), sport="basketball", scope=league,
                                     kind="team", create=False)
    rep = rep.assign(team_id=ids.to_numpy())
    unknown = int(rep["team_id"].isna().sum())
    rep = rep[rep["team_id"].notna()]
    upcoming = store.upcoming(league, now=now_dt, horizon_hours=72.0)
    playing = set(upcoming["home_id"]) | set(upcoming["away_id"])
    rep = rep[rep["team_id"].isin(playing)]
    published, collected = iso_z(version), iso_z(now_dt)
    listed = rep[rep["status"].notna() & rep["player_name"].notna()]
    n = 0
    if len(listed):
        n = store.add_availability(pd.DataFrame({
            "league": league, "team_id": listed["team_id"], "player_name": listed["player_name"],
            "status": listed["status"], "detail": listed["detail"], "published_at_utc": published,
            "collected_at_utc": collected, "source_id": source_id,
        }))
    reported = set(listed["team_id"])
    waiting = set(rep.loc[rep["not_submitted"], "team_id"]) - reported
    # A team in a listed matchup with no rows of its own filed a clean report.
    names = store.participant_names()
    in_report: set = set()
    for _, ev in upcoming.iterrows():
        pair = {ev["home_id"], ev["away_id"]}
        if pair & (reported | waiting):
            in_report |= pair
    clean = in_report - reported - waiting
    # A team that is about to play, carried statements from an earlier version and is not
    # in this one at all: the newest version lists nobody for it.
    clean |= (store.teams_with_statements(league, "injury_report", collected) & playing) - reported - waiting
    for team, outcome in ([(t, "reported") for t in reported] + [(t, "not_published") for t in waiting]
                          + [(t, "nothing_listed") for t in clean]):
        store.log_check(league, "injury_report", outcome, checked_at=collected, published_at=published,
                        team_id=team, source_id=source_id,
                        detail=f"official NBA injury report, version published {published}")
    return {"version_utc": published, "statements": n, "teams_reported": len(reported),
            "teams_nothing_listed": len(clean), "teams_not_submitted": len(waiting), "unknown_teams": unknown,
            "not_submitted": sorted(names.get(t, t) for t in waiting)}


def fetch_and_import(store: Any, fetcher: Fetcher, season: int, *, now: Any = None) -> Dict[str, Any]:
    got = fetcher.fetch(injury_report_url(season), provider="sportsdataverse",
                        relpath=f"nba_stats_injury_report/injury_report_{season}.parquet", max_age_hours=0.25,
                        license="sportsdataverse-data public release (official NBA injury report archive)")
    sid = got.register(store, notes=f"NBA official injury report archive {season}")
    out = import_latest_report(store, pd.read_parquet(got.path), now=now, source_id=sid)
    out["archive_collected_utc"] = got.collected_at_utc
    out["archive_stale"] = got.stale
    return out
