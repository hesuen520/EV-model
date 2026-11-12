"""Basketball ingestion from the sportsdataverse public release files.

Source: https://github.com/sportsdataverse/sportsdataverse-data (hoopR / wehoop
exports of ESPN's schedule and box-score feeds), one parquet per season,
labelled by the year the season **ends** in.  The files are refreshed daily
upstream; the current season is re-downloaded when the cached copy is older
than a few hours.

What the files do and do not contain:

* schedules, final scores, venues, neutral-site flag, team and player box
  scores for completed games, and the remaining schedule of the current season;
* **no** betting odds, and **no** pregame injury reports or lineups.  The
  player box records who actually played, which is only known afterwards.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ...adapters.http import Fetcher, NotFound
from ...core.timeutil import utcnow
from ...sports import DataUnavailable
from ..base import IngestReport

__all__ = ["SdvLeague", "SDV_LEAGUES", "ingest_sdv", "current_season_label"]

log = logging.getLogger(__name__)

_SDV = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
_LICENSE = "sportsdataverse-data public release (ESPN-derived); see docs/DATA_SOURCES.md"
PROVIDER = "espn"   # the native IDs in these files are ESPN's


@dataclass(frozen=True)
class SdvLeague:
    league: str
    schedule_tag: str
    schedule_stem: str
    team_tag: str
    player_tag: str
    first_season: int
    season_start_month: int        # month in which a new season label begins


SDV_LEAGUES: Dict[str, SdvLeague] = {
    "nba": SdvLeague("nba", "espn_nba_schedules", "nba_schedule", "espn_nba_team_boxscores",
                     "espn_nba_player_boxscores", 2003, 10),
    "wnba": SdvLeague("wnba", "espn_wnba_schedules", "wnba_schedule", "espn_wnba_team_boxscores",
                      "espn_wnba_player_boxscores", 2003, 4),
    "ncaam": SdvLeague("ncaam", "espn_mens_college_basketball_schedules", "mbb_schedule",
                       "espn_mens_college_basketball_team_boxscores",
                       "espn_mens_college_basketball_player_boxscores", 2006, 10),
    "ncaaw": SdvLeague("ncaaw", "espn_womens_college_basketball_schedules", "wbb_schedule",
                       "espn_womens_college_basketball_team_boxscores",
                       "espn_womens_college_basketball_player_boxscores", 2006, 10),
}

_STATUS = {
    "STATUS_FINAL": "final", "STATUS_FULL_TIME": "final", "STATUS_SCHEDULED": "scheduled",
    "STATUS_POSTPONED": "postponed", "STATUS_CANCELED": "canceled", "STATUS_CANCELLED": "canceled",
    "STATUS_SUSPENDED": "postponed", "STATUS_FORFEIT": "final",
    "STATUS_IN_PROGRESS": "live", "STATUS_HALFTIME": "live", "STATUS_END_PERIOD": "live",
    "STATUS_END_OF_PERIOD": "live", "STATUS_OVERTIME": "live", "STATUS_DELAYED": "scheduled",
    "STATUS_TBD": "scheduled",
}
_STAGE = {1: "preseason", 2: "regular", 3: "playoffs", 4: "offseason", 5: "play_in", 6: "playoffs"}

_TEAM_STATS = {
    "team_score": "pts", "field_goals_made": "fgm", "field_goals_attempted": "fga",
    "three_point_field_goals_made": "fg3m", "three_point_field_goals_attempted": "fg3a",
    "free_throws_made": "ftm", "free_throws_attempted": "fta", "offensive_rebounds": "oreb",
    "defensive_rebounds": "dreb", "total_rebounds": "reb", "assists": "ast", "steals": "stl",
    "blocks": "blk", "fouls": "pf",
}
_PLAYER_STATS = {
    "minutes": "minutes", "points": "pts", "field_goals_made": "fgm", "field_goals_attempted": "fga",
    "three_point_field_goals_made": "fg3m", "three_point_field_goals_attempted": "fg3a",
    "free_throws_made": "ftm", "free_throws_attempted": "fta", "offensive_rebounds": "oreb",
    "defensive_rebounds": "dreb", "rebounds": "reb", "assists": "ast", "steals": "stl", "blocks": "blk",
    "turnovers": "tov", "fouls": "pf",
}


def current_season_label(cfg: SdvLeague, today: Optional[pd.Timestamp] = None) -> int:
    """Season-end-year label of the season in progress (or about to start)."""
    today = pd.Timestamp(utcnow()) if today is None else pd.Timestamp(today)
    if cfg.season_start_month >= 7:   # seasons that span two calendar years
        return today.year + 1 if today.month >= cfg.season_start_month - 1 else today.year
    return today.year


def _fetch(fetcher: Fetcher, tag: str, stem: str, season: int, current: int, refresh: Optional[bool]):
    url = f"{_SDV}/{tag}/{stem}_{season}.parquet"
    return fetcher.fetch(
        url, provider="sportsdataverse", relpath=f"{tag}/{stem}_{season}.parquet",
        max_age_hours=6.0 if season >= current else None, refresh=refresh, license=_LICENSE,
    )


def _col(df: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    return df[name] if name in df.columns else pd.Series(default, index=df.index)


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalize_schedule(raw: pd.DataFrame, league: str, season: int) -> pd.DataFrame:
    """ESPN schedule rows -> canonical event rows (IDs still ESPN's)."""
    df = pd.DataFrame({
        "provider_event_id": raw["game_id"].astype("int64").astype(str),
        "league": league,
        "season": str(season),
        "start_time_utc": pd.to_datetime(raw["date"], utc=True, errors="coerce"),
        "status": raw["status_type_name"].map(_STATUS).fillna("unknown"),
        "provider": PROVIDER,
        "home_espn": _num(raw["home_id"]).astype("Int64"),
        "away_espn": _num(raw["away_id"]).astype("Int64"),
        "home_name": raw["home_display_name"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip(),
        "away_name": raw["away_display_name"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip(),
        "home_abbr": _col(raw, "home_abbreviation", None),
        "away_abbr": _col(raw, "away_abbreviation", None),
        "home_score": _num(_col(raw, "home_score")),
        "away_score": _num(_col(raw, "away_score")),
        "neutral_site": _col(raw, "neutral_site", False).fillna(False).astype(bool).astype(int),
        "venue_name": _col(raw, "venue_full_name", None),
        "venue_city": _col(raw, "venue_address_city", None),
        "venue_id": _num(_col(raw, "venue_id")).astype("Int64").astype(str).replace("<NA>", None),
        "season_type": _num(_col(raw, "season_type", 2)).fillna(2).astype(int),
        "type_abbr": _col(raw, "type_abbreviation", "STD").fillna("STD").astype(str),
        "note": _col(raw, "notes_headline", "").fillna("").astype(str),
        "periods": _num(_col(raw, "status_period")),
        "time_valid": _col(raw, "time_valid", True).fillna(True).astype(bool),
    })
    df["stage"] = df["season_type"].map(_STAGE).fillna("regular")
    df.loc[df["type_abbr"].str.upper().eq("ALLSTAR"), "stage"] = "allstar"
    df["time_precision"] = np.where(df["time_valid"], "datetime", "date")
    df = df[df["start_time_utc"].notna()]
    # Placeholder rows for games whose teams are not determined yet.
    placeholder = (
        df["home_name"].str.upper().isin({"TBD", "TBA", "NAN", ""})
        | df["away_name"].str.upper().isin({"TBD", "TBA", "NAN", ""})
        | df["home_espn"].isna() | df["away_espn"].isna() | (df["home_espn"] == df["away_espn"])
        | (df["home_espn"] < 0) | (df["away_espn"] < 0)
    )
    df = df[~placeholder]
    return df.drop_duplicates("provider_event_id", keep="last").reset_index(drop=True)


def normalize_team_box(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "provider_event_id": raw["game_id"].astype("int64").astype(str),
        "team_espn": _num(raw["team_id"]).astype("Int64"),
        "is_home": raw["team_home_away"].astype(str).str.lower().eq("home").astype(int),
    })
    for src, dst in _TEAM_STATS.items():
        out[dst] = _num(_col(raw, src))
    tov = _num(_col(raw, "total_turnovers"))
    out["tov"] = tov.where(tov.notna(), _num(_col(raw, "turnovers")))
    return out.dropna(subset=["team_espn"]).drop_duplicates(["provider_event_id", "team_espn"], keep="last")


def normalize_player_box(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "provider_event_id": raw["game_id"].astype("int64").astype(str),
        "team_espn": _num(raw["team_id"]).astype("Int64"),
        "player_id": _num(raw["athlete_id"]).astype("Int64").astype(str),
        "player_name": raw["athlete_display_name"].astype(str),
        "position": _col(raw, "athlete_position_abbreviation", None),
        "starter": _col(raw, "starter", False).fillna(False).astype(bool).astype(int),
    })
    for src, dst in _PLAYER_STATS.items():
        out[dst] = _num(_col(raw, src))
    out["minutes"] = out["minutes"].fillna(0.0)
    out = out[out["player_id"].ne("<NA>") & out["team_espn"].notna()]
    return out.drop_duplicates(["provider_event_id", "team_espn", "player_id"], keep="last")


def ingest_sdv(store, league: str, *, seasons: Optional[Sequence[int]] = None,
               refresh: Optional[bool] = None, fetcher: Optional[Fetcher] = None) -> IngestReport:
    """Download, normalise and store one basketball league."""
    if league not in SDV_LEAGUES:
        raise DataUnavailable(
            f"no public dataset adapter for basketball league {league!r}; import results with "
            "`sportspred import events`"
        )
    cfg = SDV_LEAGUES[league]
    fetcher = fetcher or Fetcher()
    current = current_season_label(cfg)
    wanted = list(seasons) if seasons else list(range(cfg.first_season, current + 1))
    report = IngestReport(league=league)
    sched_parts: List[pd.DataFrame] = []
    team_parts: List[pd.DataFrame] = []
    player_parts: List[pd.DataFrame] = []
    source_by_season: Dict[int, str] = {}
    box_sources: List[str] = []
    for season in wanted:
        try:
            got = _fetch(fetcher, cfg.schedule_tag, cfg.schedule_stem, season, current, refresh)
        except NotFound:
            if season >= current:
                report.warnings.append(f"season {season}: no schedule file published yet")
                continue
            report.warnings.append(f"season {season}: schedule file missing upstream")
            continue
        if got.stale:
            report.warnings.append(
                f"season {season}: refresh failed; using the copy collected {got.collected_at_utc}")
        sid = got.register(store, notes=f"{league} schedule {season}")
        source_by_season[season] = sid
        report.sources.append(sid)
        sched = normalize_schedule(pd.read_parquet(got.path), league, season)
        sched["source_id"] = sid
        sched_parts.append(sched)
        for tag, stem, parts, norm in (
            (cfg.team_tag, "team_box", team_parts, normalize_team_box),
            (cfg.player_tag, "player_box", player_parts, normalize_player_box),
        ):
            try:
                box = _fetch(fetcher, tag, stem, season, current, refresh)
            except NotFound:
                report.warnings.append(f"season {season}: {stem} file missing upstream")
                continue
            box_sources.append(box.register(store, notes=f"{league} {stem} {season}"))
            raw = pd.read_parquet(box.path)
            if len(raw):
                parts.append(norm(raw))
    if not sched_parts:
        raise DataUnavailable(f"no schedule files could be obtained for {league}")
    sched = pd.concat(sched_parts, ignore_index=True)
    sched = sched.drop_duplicates("provider_event_id", keep="last").reset_index(drop=True)

    # ---- participants: ESPN team IDs are stable across re-brands and moves.
    teams = pd.concat([
        sched[["home_espn", "home_name", "home_abbr", "start_time_utc"]].set_axis(
            ["espn", "name", "abbr", "t"], axis=1),
        sched[["away_espn", "away_name", "away_abbr", "start_time_utc"]].set_axis(
            ["espn", "name", "abbr", "t"], axis=1),
    ]).sort_values("t")
    # Register every historical name (oldest first) so each becomes an alias,
    # then the participant's display name is its first one; fine for matching.
    names = teams.drop_duplicates(["espn", "name"], keep="first")
    names = names.assign(pid=store.resolve_participants(
        names, sport="basketball", scope=league, kind="team", name_col="name", provider=PROVIDER,
        provider_id_col="espn", short_name_col="abbr",
    ).to_numpy())
    pid_of = dict(zip(names["espn"].astype(int), names["pid"]))
    latest = teams.drop_duplicates("espn", keep="last")
    with store.conn() as con:   # show the current name
        for espn, name, abbr in zip(latest["espn"].astype(int), latest["name"], latest["abbr"]):
            con.execute("UPDATE participants SET name=?, short_name=? WHERE participant_id=?",
                        (name, abbr, pid_of[espn]))
    sched["home_id"] = sched["home_espn"].astype(int).map(pid_of)
    sched["away_id"] = sched["away_espn"].astype(int).map(pid_of)
    sched["format"] = [{"note": n} if n else {} for n in sched["note"]]

    events, up = store.upsert_events(sched, sport="basketball")
    report.warnings += up.issues
    eid_of = dict(zip(events["provider_event_id"], events["event_id"]))

    final = events[(events["status"] == "final")].copy()
    scored = final["home_score"].notna() & final["away_score"].notna() & (
        (final["home_score"] + final["away_score"]) > 0) & (final["home_score"] != final["away_score"])
    for _, row in final[~scored].iterrows():
        store.log_issue("final_without_valid_score",
                        f"{row['provider_event_id']} {row['home_name']} v {row['away_name']}",
                        league=league, event_id=row["event_id"])
    final = final[scored]
    final["went_overtime"] = np.where(final["periods"].notna(), (final["periods"] > 4).astype(float), np.nan)
    final["reg_home_score"] = np.nan
    final["reg_away_score"] = np.nan
    store.upsert_results(final[["event_id", "home_score", "away_score", "went_overtime", "source_id"]])

    def finish(parts: List[pd.DataFrame], name: str) -> None:
        if not parts:
            report.warnings.append(f"no {name} rows available")
            return
        box = pd.concat(parts, ignore_index=True)
        box["event_id"] = box["provider_event_id"].map(eid_of)
        box = box[box["event_id"].notna() & box["team_espn"].astype("Int64").astype(int).isin(pid_of)]
        box["team_id"] = box["team_espn"].astype(int).map(pid_of)
        box = box.drop(columns=["provider_event_id", "team_espn"])
        store.write_table(league, name, box, source_ids=box_sources)
        report.tables[name] = len(box)

    finish(team_parts, "team_box")
    finish(player_parts, "player_box")

    stored = store.events(league, with_results=True)
    report.n_events = len(stored)
    report.n_results = int(stored["winner_id"].notna().sum())
    report.n_upcoming = len(store.upcoming(league))
    if len(stored):
        report.first_event = stored["start_time_utc"].min().strftime("%Y-%m-%d")
        report.last_event = stored["start_time_utc"].max().strftime("%Y-%m-%d")
    return report
