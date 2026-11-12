"""ESPN site API adapter: league-wide injury list and daily scoreboard.

**Status: adapter-only.**  The ESPN host is blocked in the environment this
was written in, so these parsers follow the response shape of ESPN's public,
undocumented site API as commonly observed (``site.api.espn.com/apis/site/v2``)
and are tested against hand-written fixtures of that shape - they have **not**
been run against the live service here.  ESPN publishes no terms that grant
API access; treat the endpoints as unofficial, low-volume and liable to
change, and check ESPN's terms of use before relying on them.

What it gives the platform when it does work:

* ``injuries``  -> availability statements per team (as published, with the
  statement's own date as ``published_at``) and an ``injury_report`` check for
  every team with an upcoming event: ``reported`` when the team is listed,
  ``nothing_listed`` when the league-wide list was fetched and the team is
  absent.  That is what lets a prediction move from PROVISIONAL to FINAL.
* ``scoreboard`` -> event status, start-time changes and final scores for one
  day, so results can be verified without waiting for the bulk dataset.

The native IDs are ESPN's, the same IDs the sportsdataverse files carry, so
teams and events line up with ingested history without name matching.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd

from ..core.timeutil import iso_z, parse_ts, utcnow
from ..sports import DataUnavailable, get_league
from .http import Fetcher

__all__ = ["ESPN_PATHS", "injuries_url", "scoreboard_url", "parse_injuries", "parse_scoreboard",
           "import_injuries", "import_scoreboard", "fetch_json"]

log = logging.getLogger(__name__)

_BASE = "https://site.api.espn.com/apis/site/v2/sports"
PROVIDER = "espn"

ESPN_PATHS: Dict[str, str] = {
    "nba": "basketball/nba", "wnba": "basketball/wnba",
    "ncaam": "basketball/mens-college-basketball", "ncaaw": "basketball/womens-college-basketball",
    "nfl": "football/nfl", "ncaaf": "football/college-football", "nhl": "hockey/nhl", "mlb": "baseball/mlb",
}

#: ESPN wording -> platform availability status.  Anything unrecognised maps
#: to "uncertain": an unknown label is never read as "available".
_STATUS: Dict[str, str] = {
    "out": "out", "injured reserve": "out", "ir": "out", "suspension": "out", "suspended": "out",
    "out for season": "out", "physically unable to perform": "out", "non-football injury": "out",
    "doubtful": "doubtful", "questionable": "questionable", "game-time decision": "questionable",
    "gtd": "questionable", "day-to-day": "questionable", "dtd": "questionable",
    "probable": "probable", "active": "available", "available": "available",
}
_SCORE_STATUS = {
    "STATUS_FINAL": "final", "STATUS_FULL_TIME": "final", "STATUS_SCHEDULED": "scheduled",
    "STATUS_POSTPONED": "postponed", "STATUS_CANCELED": "canceled", "STATUS_SUSPENDED": "postponed",
    "STATUS_IN_PROGRESS": "live", "STATUS_HALFTIME": "live", "STATUS_END_PERIOD": "live",
    "STATUS_OVERTIME": "live", "STATUS_DELAYED": "scheduled", "STATUS_RAIN_DELAY": "live",
}


def _path(league: str) -> str:
    if league not in ESPN_PATHS:
        raise DataUnavailable(f"no ESPN path is configured for league {league!r}")
    return ESPN_PATHS[league]


def injuries_url(league: str) -> str:
    return f"{_BASE}/{_path(league)}/injuries"


def scoreboard_url(league: str, day: Optional[str] = None) -> str:
    """``day`` is the US schedule date ``YYYYMMDD`` (ESPN groups games by it)."""
    url = f"{_BASE}/{_path(league)}/scoreboard"
    return f"{url}?dates={day}" if day else url


def fetch_json(url: str, fetcher: Fetcher, *, relpath: str) -> Tuple[Any, Any]:
    """Download a JSON document (always fresh); returns ``(payload, fetched)``."""
    got = fetcher.fetch(url, provider="espn_api", relpath=relpath, refresh=True,
                        license="ESPN site API (unofficial; subject to ESPN terms of use)")
    if got.stale:
        raise DataUnavailable(f"{url} could not be refreshed; the cached copy is not used for live checks")
    return json.loads(got.path.read_text(encoding="utf-8")), got


def _availability(status_text: Optional[str], fantasy_text: Optional[str]) -> str:
    for text in (fantasy_text, status_text):
        key = (text or "").strip().lower()
        if key in _STATUS:
            return _STATUS[key]
    return "uncertain"


def parse_injuries(payload: Mapping[str, Any]) -> pd.DataFrame:
    """One row per listed player: ``team_espn, team_name, player_espn,
    player_name, status, detail, published_at_utc`` (missing dates stay missing)."""
    rows: List[Dict[str, Any]] = []
    for team in payload.get("injuries") or []:
        team_id, team_name = team.get("id"), team.get("displayName")
        for item in team.get("injuries") or []:
            athlete = item.get("athlete") or {}
            name = athlete.get("displayName") or athlete.get("fullName")
            if not name:
                continue
            details = item.get("details") or {}
            fantasy = (details.get("fantasyStatus") or {})
            published = None
            if item.get("date"):
                try:
                    published = iso_z(parse_ts(item["date"]))
                except ValueError:
                    published = None
            parts = [details.get("type"), details.get("detail"), details.get("side"), item.get("shortComment")]
            rows.append({
                "team_espn": None if team_id is None else str(team_id), "team_name": team_name,
                "player_espn": None if athlete.get("id") is None else str(athlete["id"]),
                "player_name": str(name),
                "status": _availability(item.get("status"), fantasy.get("description") or fantasy.get("abbreviation")),
                "status_text": item.get("status"),
                "detail": "; ".join(str(p) for p in parts if p) or None,
                "published_at_utc": published,
            })
    return pd.DataFrame(rows, columns=["team_espn", "team_name", "player_espn", "player_name", "status",
                                       "status_text", "detail", "published_at_utc"])


def import_injuries(store: Any, league: str, payload: Mapping[str, Any], *, collected_at: Any = None,
                    source_id: Optional[str] = None, horizon_hours: float = 48.0) -> Dict[str, int]:
    """Store the injury list and log the ``injury_report`` check for every
    team with an event starting within ``horizon_hours``.

    Teams ESPN lists that this store does not know are skipped and counted
    (``unknown_teams``); they are never matched by guesswork.
    """
    plugin, spec = get_league(league)
    collected = iso_z(collected_at or utcnow())
    reports = parse_injuries(payload)
    known = store.resolve_participants(
        reports.assign(name=reports["team_name"]), sport=spec.sport, scope=league, kind="team",
        provider=PROVIDER, provider_id_col="team_espn", create=False) if len(reports) else pd.Series(dtype=object)
    reports = reports.assign(team_id=known.to_numpy() if len(reports) else [])
    unknown = int(reports["team_id"].isna().sum())
    reports = reports[reports["team_id"].notna()]
    n = 0
    if len(reports):
        frame = pd.DataFrame({
            "league": league, "team_id": reports["team_id"], "player_id": reports["player_espn"],
            "player_name": reports["player_name"], "status": reports["status"], "detail": reports["detail"],
            # A statement dated later than our collection time is a clock
            # disagreement; keep the row but do not claim it was published in the future.
            "published_at_utc": reports["published_at_utc"].where(reports["published_at_utc"] <= collected),
            "collected_at_utc": collected, "source_id": source_id,
        })
        n = store.add_availability(frame)
    upcoming = store.upcoming(league, now=collected, horizon_hours=horizon_hours)
    teams = set(upcoming["home_id"].dropna()) | set(upcoming["away_id"].dropna())
    listed = set(reports["team_id"])
    for team in teams:
        store.log_check(league, "injury_report", "reported" if team in listed else "nothing_listed",
                        checked_at=collected, team_id=team, source_id=source_id,
                        detail="ESPN league-wide injury list")
    return {"reports": n, "teams_reported": len(teams & listed), "teams_nothing_listed": len(teams - listed),
            "unknown_teams": unknown}


def parse_scoreboard(payload: Mapping[str, Any], league: str) -> pd.DataFrame:
    """One row per game: native IDs, UTC start, status, sides and scores."""
    rows: List[Dict[str, Any]] = []
    for event in payload.get("events") or []:
        comp = (event.get("competitions") or [{}])[0]
        sides = {c.get("homeAway"): c for c in comp.get("competitors") or []}
        home, away = sides.get("home"), sides.get("away")
        if not home or not away or not event.get("date"):
            continue
        status_name = (((event.get("status") or {}).get("type") or {}).get("name")) or ""
        venue = comp.get("venue") or {}

        def score(side: Mapping[str, Any]) -> Optional[float]:
            try:
                return float(side.get("score"))
            except (TypeError, ValueError):
                return None

        rows.append({
            "provider_event_id": str(event.get("id")), "league": league,
            "season": str(((event.get("season") or {}).get("year")) or ""),
            "start_time_utc": parse_ts(event["date"]), "status": _SCORE_STATUS.get(status_name, "unknown"),
            "provider": PROVIDER,
            "home_espn": str((home.get("team") or {}).get("id")), "away_espn": str((away.get("team") or {}).get("id")),
            "home_name": (home.get("team") or {}).get("displayName"),
            "away_name": (away.get("team") or {}).get("displayName"),
            "home_score": score(home), "away_score": score(away),
            "neutral_site": None if comp.get("neutralSite") is None else int(bool(comp["neutralSite"])),
            "venue_name": venue.get("fullName"), "venue_city": (venue.get("address") or {}).get("city"),
            "venue_id": None if venue.get("id") is None else str(venue["id"]),
        })
    return pd.DataFrame(rows)


def import_scoreboard(store: Any, league: str, payload: Mapping[str, Any], *,
                      source_id: Optional[str] = None) -> Dict[str, int]:
    """Update event status/start times and record final scores.

    Only games whose two teams are already known to the store are imported;
    this adapter refreshes events, it does not create leagues from scratch.
    """
    plugin, spec = get_league(league)
    games = parse_scoreboard(payload, league)
    if games.empty:
        return {"events": 0, "results": 0, "skipped": 0}
    for side in ("home", "away"):
        games[f"{side}_id"] = store.resolve_participants(
            games.assign(name=games[f"{side}_name"]), sport=spec.sport, scope=league, kind="team",
            provider=PROVIDER, provider_id_col=f"{side}_espn", create=False).to_numpy()
    usable = games[games["home_id"].notna() & games["away_id"].notna() & (games["season"] != "")]
    skipped = len(games) - len(usable)
    if usable.empty:
        return {"events": 0, "results": 0, "skipped": skipped}
    events, report = store.upsert_events(usable, sport=spec.sport, source_id=source_id)
    final = events[(events["status"] == "final") & events["home_score"].notna() & events["away_score"].notna()
                   & (events["home_score"] != events["away_score"])]
    res = store.upsert_results(final[["event_id", "home_score", "away_score"]], source_id=source_id)
    return {"events": report.inserted + report.updated + report.unchanged,
            "results": res.inserted + res.updated, "skipped": skipped}
