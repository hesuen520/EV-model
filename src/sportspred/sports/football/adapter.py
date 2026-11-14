"""NFL ingestion from the nflverse-data release files.

Source: https://github.com/nflverse/nflverse-data (CC-BY-4.0), three assets:

* ``schedules/games.csv`` - every game since 1999 including the unplayed part
  of the current season: date, kickoff (US Eastern clock time; 1999 has dates
  only), teams, final score, overtime flag, neutral-site flag, roof, surface,
  division-game flag, the starting quarterbacks, a point spread, a total and
  (2006 onwards, complete from 2010) moneylines and spread/total prices;
* ``stats_team/stats_team_week_<season>.csv`` - team box score per game with
  nflfastR expected points added (EPA);
* ``stats_player/stats_player_week_<season>.parquet`` - player box score per
  game; only quarterback rows are kept.

What the files do **not** contain, and what that means here:

* no timestamp or bookmaker for any betting number.  The nflverse
  documentation describes ``spread_line`` as "the spread line for the game"
  and says nothing about opening or closing, so the lines are stored and
  labelled as *undated*;
* the quarterback columns are the **actual** starters of played games (known
  only at kickoff) and a projection for the next week or two.  No
  as-announced history exists;
* ``temp`` and ``wind`` are observed at the game, and a retractable roof's
  ``open``/``closed`` state is decided on the day (it is blank for unplayed
  games);
* no injury reports are ingested: the nflverse injury files for 2025 onwards
  carry no modification timestamps, so their pregame timing cannot be verified.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ...adapters.http import Fetcher, NotFound
from ...core.timeutil import iso_z, utcnow
from ...sports import DataUnavailable
from ..base import IngestReport
from .teams import franchise, team_name

__all__ = ["ingest_nflverse", "normalize_games", "american_implied", "GAMES_URL", "LICENSE", "PROVIDER",
           "KICKOFF_TZ"]

log = logging.getLogger(__name__)

_BASE = "https://github.com/nflverse/nflverse-data/releases/download"
GAMES_URL = f"{_BASE}/schedules/games.csv"
LICENSE = "nflverse-data release files, CC-BY-4.0"
PROVIDER = "nflverse"
KICKOFF_TZ = "America/New_York"    # games.csv states every kickoff in US Eastern clock time
FIRST_STATS_SEASON = 1999

_STAGE = {"REG": "regular", "WC": "playoffs", "DIV": "playoffs", "CON": "playoffs", "SB": "playoffs"}
_GAME_COLS = [
    "week", "game_type", "weekday", "div_game", "roof", "surface", "temp", "wind", "spread_line", "total_line",
    "home_moneyline", "away_moneyline", "home_spread_odds", "away_spread_odds", "under_odds", "over_odds",
    "home_qb_id", "home_qb_name", "away_qb_id", "away_qb_name", "stadium_id",
]


def american_implied(odds: pd.Series) -> pd.Series:
    """Raw implied probability of an American price (still containing the bookmaker's margin)."""
    o = pd.to_numeric(odds, errors="coerce").astype(float)
    return pd.Series(np.where(o < 0, -o / (-o + 100.0), 100.0 / (o + 100.0)), index=odds.index).where(o.notna())


def kickoff_utc(gameday: pd.Series, gametime: pd.Series) -> pd.DataFrame:
    """``start_time_utc`` and ``time_precision`` from an Eastern date and clock time.

    A game without a kickoff time is stored at 00:00Z of its calendar date
    with ``time_precision='date'``; no clock time is invented.
    """
    day = pd.to_datetime(gameday, format="%Y-%m-%d")
    has_time = gametime.notna() & gametime.astype(str).str.match(r"^\d{1,2}:\d{2}$")
    local = pd.to_datetime(gameday.astype(str) + " " + gametime.where(has_time, "00:00").astype(str),
                           format="%Y-%m-%d %H:%M")
    timed = local.dt.tz_localize(KICKOFF_TZ, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    has_time &= timed.notna()
    start = timed.where(has_time, day.dt.tz_localize("UTC"))
    return pd.DataFrame({"start_time_utc": start, "time_precision": np.where(has_time, "datetime", "date")})


def normalize_games(raw: pd.DataFrame, league: str) -> pd.DataFrame:
    """games.csv rows -> canonical event rows (teams still as abbreviations)."""
    when = kickoff_utc(raw["gameday"], raw["gametime"])
    scored = raw["home_score"].notna() & raw["away_score"].notna()
    df = pd.DataFrame({
        "provider_event_id": raw["game_id"].astype(str),
        "league": league,
        "season": raw["season"].astype(int).astype(str),
        "start_time_utc": when["start_time_utc"],
        "time_precision": when["time_precision"],
        "local_date": raw["gameday"].astype(str),
        "status": np.where(scored, "final", "scheduled"),
        "provider": PROVIDER,
        "home_abbr": raw["home_team"].astype(str),
        "away_abbr": raw["away_team"].astype(str),
        "home_score": pd.to_numeric(raw["home_score"], errors="coerce"),
        "away_score": pd.to_numeric(raw["away_score"], errors="coerce"),
        "went_overtime": pd.to_numeric(raw["overtime"], errors="coerce"),
        "neutral_site": raw["location"].astype(str).str.lower().eq("neutral").astype(int),
        "venue_name": raw["stadium"],
        "venue_id": raw["stadium_id"],
        "stage": raw["game_type"].map(_STAGE).fillna("regular"),
    })
    for col in _GAME_COLS:
        df[col] = raw[col] if col in raw.columns else np.nan
    df["surface"] = df["surface"].astype("string").str.strip().str.lower()
    df["roof"] = df["roof"].astype("string").str.strip().str.lower()
    df["format"] = [{"week": int(w), "game_type": str(t)} for w, t in zip(raw["week"], raw["game_type"])]
    return df[df["start_time_utc"].notna()].drop_duplicates("provider_event_id", keep="last").reset_index(drop=True)


def _team_stats(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per team and game: plays, EPA, passing and rushing splits, turnovers."""
    num = lambda c: pd.to_numeric(raw[c], errors="coerce").fillna(0.0) if c in raw.columns else 0.0  # noqa: E731
    dropbacks = num("attempts") + num("sacks_suffered")
    out = pd.DataFrame({
        "game_id": raw["game_id"].astype(str),
        "team": raw["team"],
        "plays": dropbacks + num("carries"),
        "dropbacks": dropbacks,
        "carries": num("carries"),
        "pass_epa": num("passing_epa"),
        "rush_epa": num("rushing_epa"),
        "turnovers": num("passing_interceptions") + num("sack_fumbles_lost") + num("rushing_fumbles_lost")
        + num("receiving_fumbles_lost"),
    })
    out["epa"] = out["pass_epa"] + out["rush_epa"]
    return out[out["team"].notna() & (out["plays"] > 0)]


def _qb_stats(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per quarterback and game: his dropbacks plus carries and the EPA on them."""
    qb = raw[raw["position"].astype(str) == "QB"]
    num = lambda c: pd.to_numeric(qb[c], errors="coerce").fillna(0.0)  # noqa: E731
    out = pd.DataFrame({
        "game_id": qb["game_id"].astype(str),
        "team": qb["team"],
        "player_id": qb["player_id"].astype(str),
        "player_name": qb["player_display_name"].astype(str),
        "plays": num("attempts") + num("sacks_suffered") + num("carries"),
        "epa": num("passing_epa") + num("rushing_epa"),
    })
    return out[out["team"].notna() & qb["player_id"].notna() & (out["plays"] > 0)]


def _resolve_teams(store, league: str, sched: pd.DataFrame) -> Dict[str, str]:
    """Franchise abbreviation -> participant ID.  Earlier names of a franchise
    (Oakland Raiders, ...) become aliases of the same participant."""
    abbrs = pd.concat([
        sched[["home_abbr", "start_time_utc"]].set_axis(["abbr", "t"], axis=1),
        sched[["away_abbr", "start_time_utc"]].set_axis(["abbr", "t"], axis=1),
    ]).sort_values("t").drop_duplicates("abbr", keep="first")
    teams = pd.DataFrame({
        "franchise": [franchise(a) for a in abbrs["abbr"]],
        "name": [team_name(a) for a in abbrs["abbr"]],
        "abbr": abbrs["abbr"].to_numpy(),
    })
    teams["pid"] = store.resolve_participants(
        teams, sport="football", scope=league, kind="team", name_col="name", provider=PROVIDER,
        provider_id_col="franchise", short_name_col="abbr",
    ).to_numpy()
    pid_of = dict(zip(teams["franchise"], teams["pid"]))
    with store.conn() as con:   # show the franchise's current name
        for fr, pid in pid_of.items():
            con.execute("UPDATE participants SET name=?, short_name=? WHERE participant_id=?",
                        (team_name(fr), fr, pid))
    return pid_of


def _record_projected_starters(store, league: str, events: pd.DataFrame, pid_of: Dict[str, str],
                               source_id: str, collected_at: str) -> int:
    """The schedule file lists a projected starting quarterback for the next
    week or two.  That is a third-party projection, not a team announcement,
    so it is recorded as a *probable* QB1 statement stamped with the time this
    installation collected it."""
    now = pd.Timestamp(utcnow())
    up = events[(events["status"] == "scheduled") & (events["start_time_utc"] > now)]
    rows: List[dict] = []
    for side in ("home", "away"):
        part = up[up[f"{side}_qb_id"].notna() & up[f"{side}_qb_name"].notna()]
        for _, r in part.iterrows():
            rows.append({
                "league": league, "event_id": r["event_id"], "team_id": pid_of[franchise(r[f"{side}_abbr"])],
                "player_id": str(r[f"{side}_qb_id"]), "player_name": str(r[f"{side}_qb_name"]),
                "status": "probable", "role": "QB1",
                "detail": "projected starter listed in the nflverse schedule file (not a team announcement)",
                "collected_at_utc": collected_at, "source_id": source_id,
            })
    return store.add_availability(pd.DataFrame(rows)) if rows else 0


def ingest_nflverse(store, league: str, *, seasons: Optional[Sequence[int]] = None,
                    refresh: Optional[bool] = None, fetcher: Optional[Fetcher] = None) -> IngestReport:
    """Download, normalise and store the NFL schedule, results and statistics."""
    fetcher = fetcher or Fetcher()
    report = IngestReport(league=league)
    got = fetcher.fetch(GAMES_URL, provider=PROVIDER, relpath="schedules/games.csv", max_age_hours=6.0,
                        refresh=refresh, license=LICENSE)
    if got.stale:
        report.warnings.append(f"schedule refresh failed; using the copy collected {got.collected_at_utc}")
    sid = got.register(store, notes=f"{league} schedule, results, lines and starting quarterbacks")
    report.sources.append(sid)
    raw = pd.read_csv(got.path)
    if seasons:
        raw = raw[raw["season"].isin(list(seasons))]
    if raw.empty:
        raise DataUnavailable(f"the nflverse schedule file has no rows for seasons {list(seasons or [])}")
    sched = normalize_games(raw, league)
    sched["source_id"] = sid
    pid_of = _resolve_teams(store, league, sched)
    sched["home_id"] = [pid_of[franchise(a)] for a in sched["home_abbr"]]
    sched["away_id"] = [pid_of[franchise(a)] for a in sched["away_abbr"]]

    events, up = store.upsert_events(sched, sport="football")
    report.warnings += up.issues
    eid_of = dict(zip(events["provider_event_id"], events["event_id"]))
    final = events[events["status"] == "final"]
    store.upsert_results(final[["event_id", "home_score", "away_score", "went_overtime", "source_id"]])
    games = events[["event_id", "provider_event_id", "season", "local_date", "home_abbr", "away_abbr", *_GAME_COLS]]
    store.write_table(league, "games", games.rename(columns={"provider_event_id": "game_id"}), source_ids=[sid])
    report.tables["games"] = len(games)

    current = int(raw["season"].max())
    wanted = sorted(int(s) for s in raw["season"].unique() if int(s) >= FIRST_STATS_SEASON)
    for name, pattern, reader, norm in (
        ("team_stats", "stats_team/stats_team_week_{s}.csv", pd.read_csv, _team_stats),
        ("qb_stats", "stats_player/stats_player_week_{s}.parquet", pd.read_parquet, _qb_stats),
    ):
        parts: List[pd.DataFrame] = []
        sids: List[str] = []
        for season in wanted:
            rel = pattern.format(s=season)
            try:
                f = fetcher.fetch(f"{_BASE}/{rel}", provider=PROVIDER, relpath=rel, refresh=refresh,
                                  max_age_hours=6.0 if season >= current else None, license=LICENSE)
            except NotFound:
                report.warnings.append(f"season {season}: {rel} is not published")
                continue
            if f.stale:
                report.warnings.append(f"season {season}: {name} refresh failed; using the cached copy")
            sids.append(f.register(store, notes=f"{league} {name} {season}"))
            parts.append(norm(reader(f.path)))
        if not parts:
            report.warnings.append(f"no {name} files available: the features built from them will be missing")
            continue
        table = pd.concat(parts, ignore_index=True)
        table["event_id"] = table["game_id"].map(eid_of)
        table["team_id"] = table["team"].map(lambda a: pid_of.get(franchise(a)))
        table = table[table["event_id"].notna() & table["team_id"].notna()].drop(columns=["game_id", "team"])
        store.write_table(league, name, table.reset_index(drop=True), source_ids=sids)
        report.sources += sids
        report.tables[name] = len(table)

    n_proj = _record_projected_starters(store, league, events, pid_of, sid, got.collected_at_utc)
    if n_proj:
        report.tables["projected_qb_statements (new)"] = n_proj
    stored = store.events(league, with_results=True)
    report.n_events = len(stored)
    report.n_results = int(stored["home_score"].notna().sum())
    report.n_upcoming = len(store.upcoming(league))
    report.first_event = stored["start_time_utc"].min().strftime("%Y-%m-%d")
    report.last_event = stored["start_time_utc"].max().strftime("%Y-%m-%d")
    late = stored[(stored["status"] == "scheduled") & (stored["start_time_utc"] <= pd.Timestamp(utcnow()))]
    if len(late):
        report.warnings.append(f"{len(late)} game(s) have started but have no result in the source yet "
                               f"(checked {iso_z(utcnow())})")
    return report
