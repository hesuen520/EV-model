"""NHL ingestion from the sportsdataverse public release files.

Source: https://github.com/sportsdataverse/sportsdataverse-data (the
``fastRhockey`` exports of the NHL's own web API), one parquet per dataset and
season, labelled by the year the season **ends** in.  Four datasets are read:

* ``nhl_schedules`` - every regular-season and playoff game with its start
  instant in UTC, venue, state and final score, including the unplayed rest of
  the current season;
* ``nhl_shots_by_period`` - one row per period played, which says whether a
  game needed overtime or a shootout;
* ``nhl_goalie_boxscores`` - each dressed goalie's shots faced, saves and
  whether he started;
* ``nhl_pbp_lite`` - play-by-play, from which every unblocked shot attempt is
  kept with its distance, angle, shot type and the skaters on the ice.

What the files do **not** contain: betting odds, pregame injury reports,
confirmed starting goalies or line-ups.  The goalie box score records who
actually started, which is only known at puck drop.  The ``nhl_team_boxscores``
files carry no game identifier before the 2023-24 season and are not used.
The twelve round-robin seeding games of August 2020 are absent from the
schedule files and are therefore not ingested.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ...adapters.http import Fetched, Fetcher, NotFound
from ...core.timeutil import utcnow
from ...sports import DataUnavailable
from ..base import IngestReport
from .teams import canonical_code, team_names

__all__ = [
    "FIRST_SEASON", "current_season_label", "normalize_schedule", "flag_neutral_sites", "game_end_types",
    "regulation_results", "normalize_goalies", "normalize_shots", "ingest_nhl",
]

log = logging.getLogger(__name__)

_SDV = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
_LICENSE = ("sportsdataverse-data release files (repository licence MIT, data badge CC BY 4.0); "
            "NHL web API data collected by fastRhockey")
PROVIDER = "nhl"                 # game IDs and tri-codes are the NHL's own
FIRST_SEASON = 2010              # 2009-10: the first season published

#: dataset -> (release tag, file stem)
_DATASETS: Dict[str, Tuple[str, str]] = {
    "schedule": ("nhl_schedules", "nhl_schedule"),
    "periods": ("nhl_shots_by_period", "shots_by_period"),
    "goalies": ("nhl_goalie_boxscores", "goalie_box"),
    "pbp": ("nhl_pbp_lite", "play_by_play_lite"),
}
_STATUS = {
    "OFF": "final", "FINAL": "final", "OVER": "final", "FUT": "scheduled", "PRE": "scheduled",
    "LIVE": "live", "CRIT": "live", "PPD": "postponed", "SUSP": "postponed", "CNCL": "canceled",
}
_STAGE = {"R": "regular", "P": "playoffs"}
_PLACEHOLDER = {"", "TBD", "TBA", "NAN", "NONE"}
_PBP_COLUMNS = ["game_id", "event_type", "event_team_type", "period_type", "secondary_type",
                "shot_distance", "shot_angle", "home_skaters", "away_skaters", "home_goalie_in",
                "away_goalie_in"]
_SHOT_EVENTS = ("SHOT", "MISSED_SHOT", "GOAL")     # unblocked attempts; a blocked shot is located at the block
_END_RANK = {"REG": 0, "REGULAR": 0, "OT": 1, "OVERTIME": 1, "SO": 2, "SHOOTOUT": 2}
_END_NAME = np.array(["REG", "OT", "SO"])


def current_season_label(today: Optional[pd.Timestamp] = None) -> int:
    """Season-end-year label of the season in progress or about to start."""
    today = pd.Timestamp(utcnow()) if today is None else pd.Timestamp(today)
    return today.year + 1 if today.month >= 8 else today.year


def _fetch(fetcher: Fetcher, dataset: str, season: int, current: int, refresh: Optional[bool]) -> Fetched:
    tag, stem = _DATASETS[dataset]
    return fetcher.fetch(
        f"{_SDV}/{tag}/{stem}_{season}.parquet", provider="sportsdataverse",
        relpath=f"{tag}/{stem}_{season}.parquet", max_age_hours=6.0 if season >= current else None,
        refresh=refresh, license=_LICENSE,
    )


def _game_key(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype(str)


def normalize_schedule(raw: pd.DataFrame, league: str, season: int) -> pd.DataFrame:
    """NHL schedule rows -> canonical event rows (teams still as tri-codes).

    ``game_time`` is the start instant in UTC.  Only regular-season and
    playoff games are kept; rows whose teams are not determined yet are
    dropped."""
    raw = raw[raw["game_type"].isin(_STAGE)]
    df = pd.DataFrame({
        "provider_event_id": _game_key(raw["game_id"]),
        "league": league,
        "season": str(season),
        "start_time_utc": pd.to_datetime(raw["game_time"], utc=True, errors="coerce"),
        "time_precision": "datetime",
        "status": raw["game_state"].astype(str).str.upper().map(_STATUS).fillna("unknown"),
        "stage": raw["game_type"].map(_STAGE),
        "provider": PROVIDER,
        "home_abbr": raw["home_team_abbr"].astype(str).str.strip().str.upper(),
        "away_abbr": raw["away_team_abbr"].astype(str).str.strip().str.upper(),
        "home_name": raw["home_team_name"].astype(str).str.strip(),
        "away_name": raw["away_team_name"].astype(str).str.strip(),
        "home_score": pd.to_numeric(raw["home_score"], errors="coerce"),
        "away_score": pd.to_numeric(raw["away_score"], errors="coerce"),
        "venue_name": raw["venue"].astype("string").str.strip(),
    })
    undetermined = (df["home_abbr"].isin(_PLACEHOLDER) | df["away_abbr"].isin(_PLACEHOLDER)
                    | (df["home_abbr"] == df["away_abbr"]))
    df = df[df["start_time_utc"].notna() & ~undetermined]
    return df.drop_duplicates("provider_event_id", keep="last").reset_index(drop=True)


def flag_neutral_sites(sched: pd.DataFrame, min_home_games: int = 5) -> pd.Series:
    """1 where a game is not played in the home club's regular arena, 0 where
    it is, missing where the venue is unknown.

    An arena counts as regular when it hosts at least ``min_home_games`` of
    the club's regular-season home games that season, so outdoor games,
    games abroad and the 2020 playoff bubble are flagged while a mid-season
    arena renaming is not.  Only the published schedule is used."""
    keyed = pd.DataFrame({"season": sched["season"], "home": sched["home_abbr"],
                          "venue": sched["venue_name"].astype("string").str.casefold()})
    counts = keyed[sched["stage"] == "regular"].value_counts()
    regular = set(counts[counts >= min_home_games].index)
    flags = [np.nan if key[2] is pd.NA else float(key not in regular)
             for key in keyed.itertuples(index=False, name=None)]
    return pd.Series(flags, index=sched.index)


def game_end_types(periods: pd.DataFrame, pbp: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """How each game ended: ``REG``, ``OT`` or ``SO``, indexed by game key.

    The per-period shot table is the primary source; play-by-play period
    types fill games it lacks.  ``sources_agree`` is false where both exist
    and differ (such a game's regulation result is not trusted)."""
    def latest(frame: Optional[pd.DataFrame]) -> pd.Series:
        if frame is None or frame.empty:
            return pd.Series(dtype="float64")
        rank = frame["period_type"].astype(str).str.upper().map(_END_RANK)
        return rank.groupby(_game_key(frame["game_id"])).max()

    both = pd.concat([latest(periods).rename("periods"), latest(pbp).rename("pbp")], axis=1)
    rank = both["periods"].fillna(both["pbp"])
    out = pd.DataFrame({
        "end_type": np.where(rank.notna(), _END_NAME[rank.fillna(0).astype(int)], None),
        "sources_agree": ~(both["periods"].notna() & both["pbp"].notna() & (both["periods"] != both["pbp"])),
    }, index=both.index)
    return out[rank.notna()]


def regulation_results(final: pd.DataFrame) -> pd.DataFrame:
    """Add the regulation result to final games that carry ``end_type``.

    Overtime and shootouts are sudden death, so a game that needed either was
    level after regulation and its final margin is exactly one goal (the
    shootout winner is credited one goal).  A game whose end type is unknown,
    disputed between sources, or inconsistent with that rule gets a missing
    regulation result rather than a guess."""
    out = final.copy()
    extra = out["end_type"].isin(["OT", "SO"])
    one_goal = (out["home_score"] - out["away_score"]).abs() == 1
    trusted = out["end_type"].notna() & out["sources_agree"].eq(True) & (~extra | one_goal)
    level = out[["home_score", "away_score"]].min(axis=1)
    out["went_overtime"] = np.where(trusted, extra.astype(float), np.nan)
    out["went_shootout"] = np.where(trusted, (out["end_type"] == "SO").astype(float), np.nan)
    out["reg_home_score"] = np.where(trusted, np.where(extra, level, out["home_score"]), np.nan)
    out["reg_away_score"] = np.where(trusted, np.where(extra, level, out["away_score"]), np.nan)
    return out


def normalize_goalies(raw: pd.DataFrame) -> pd.DataFrame:
    """Goalie box rows: who dressed, who started, shots faced and saves."""
    out = pd.DataFrame({
        "provider_event_id": _game_key(raw["game_id"]),
        "team_abbr": raw["team_abbrev"].astype(str).str.strip().str.upper(),
        "player_id": _game_key(raw["player_id"]),
        "player_name": raw["player_name"].astype(str).str.strip(),
        "starter": raw["starter"].fillna(False).astype(bool).astype(int),
        "shots_against": pd.to_numeric(raw["shots_against"], errors="coerce").fillna(0.0),
        "saves": pd.to_numeric(raw["saves"], errors="coerce").fillna(0.0),
    })
    out = out[out["player_id"].ne("<NA>")]
    return out.drop_duplicates(["provider_event_id", "team_abbr", "player_id"], keep="last").reset_index(drop=True)


def normalize_shots(pbp: pd.DataFrame) -> pd.DataFrame:
    """Unblocked shot attempts outside shootouts, one row per attempt.

    ``situation`` is the manpower as the shooting team saw it: ``5v5``;
    ``pp`` / ``sh`` (more / fewer skaters with both goalies in net); ``ev``
    (other even strength, e.g. 4-on-4); ``en`` when either net is empty or the
    manpower was not recorded."""
    x = pbp[pbp["event_type"].isin(_SHOT_EVENTS) & pbp["event_team_type"].isin(["home", "away"])
            & (pbp["period_type"].astype(str).str.upper() != "SHOOTOUT")]
    is_home = (x["event_team_type"] == "home").to_numpy()
    home_n = pd.to_numeric(x["home_skaters"], errors="coerce").to_numpy(dtype=float)
    away_n = pd.to_numeric(x["away_skaters"], errors="coerce").to_numpy(dtype=float)
    own, opp = np.where(is_home, home_n, away_n), np.where(is_home, away_n, home_n)
    goalies_in = ((pd.to_numeric(x["home_goalie_in"], errors="coerce") == 1)
                  & (pd.to_numeric(x["away_goalie_in"], errors="coerce") == 1)).to_numpy()
    situation = np.select(
        [~goalies_in | np.isnan(own) | np.isnan(opp), (own == 5) & (opp == 5), own > opp, own < opp],
        ["en", "5v5", "pp", "sh"], default="ev")
    return pd.DataFrame({
        "provider_event_id": _game_key(x["game_id"]).to_numpy(),
        "is_home": is_home.astype("int8"),
        "situation": situation,
        "on_goal": x["event_type"].isin(["SHOT", "GOAL"]).to_numpy().astype("int8"),
        "is_goal": (x["event_type"] == "GOAL").to_numpy().astype("int8"),
        "distance": pd.to_numeric(x["shot_distance"], errors="coerce").to_numpy(dtype="float32"),
        "angle": pd.to_numeric(x["shot_angle"], errors="coerce").abs().to_numpy(dtype="float32"),
        "shot_type": x["secondary_type"].astype("string").str.strip().str.lower().fillna("unknown").to_numpy(),
    })


def _register_teams(store, league: str, sched: pd.DataFrame) -> Dict[str, str]:
    """Tri-code -> participant ID, one participant per continuing team."""
    seen = pd.concat([
        sched[["home_abbr", "home_name", "start_time_utc"]].set_axis(["abbr", "name", "t"], axis=1),
        sched[["away_abbr", "away_name", "start_time_utc"]].set_axis(["abbr", "name", "t"], axis=1),
    ]).sort_values("t").drop_duplicates("abbr", keep="last")
    source_name = dict(zip(seen["abbr"], seen["name"]))
    canon = sorted({canonical_code(code) for code in seen["abbr"]})
    # Oldest name first, so every earlier name becomes an alias of one participant.
    rows = [{"code": code, "name": name}
            for code in canon for name in team_names(code, fallback=source_name.get(code))]
    names = pd.DataFrame(rows)
    names["pid"] = store.resolve_participants(
        names, sport="hockey", scope=league, kind="team", name_col="name", provider=PROVIDER,
        provider_id_col="code", short_name_col="code",
    ).to_numpy()
    pid_of = dict(zip(names["code"], names["pid"]))
    current = names.drop_duplicates("code", keep="last")
    with store.conn() as con:   # show the current name
        for code, name in zip(current["code"], current["name"]):
            con.execute("UPDATE participants SET name=?, short_name=? WHERE participant_id=?",
                        (name, code, pid_of[code]))
    return {code: pid_of[canonical_code(code)] for code in seen["abbr"]}


def ingest_nhl(store, league: str = "nhl", *, seasons: Optional[Sequence[int]] = None,
               refresh: Optional[bool] = None, fetcher: Optional[Fetcher] = None) -> IngestReport:
    """Download, normalise and store NHL schedules, results, goalie box
    scores and shot attempts."""
    fetcher = fetcher or Fetcher()
    current = current_season_label()
    wanted = list(seasons) if seasons else list(range(FIRST_SEASON, current + 1))
    report = IngestReport(league=league)
    parts: Dict[str, List[pd.DataFrame]] = {"schedule": [], "ends": [], "goalies": [], "shots": []}
    sources: Dict[str, List[str]] = {name: [] for name in _DATASETS}

    def load(dataset: str, season: int) -> Optional[pd.DataFrame]:
        try:
            got = _fetch(fetcher, dataset, season, current, refresh)
        except NotFound:
            late = " yet" if season >= current else ""
            report.warnings.append(f"season {season}: no {_DATASETS[dataset][0]} file published{late}")
            return None
        if got.stale:
            report.warnings.append(
                f"season {season} {dataset}: refresh failed; using the copy collected {got.collected_at_utc}")
        sources[dataset].append(got.register(store, notes=f"{league} {dataset} {season}"))
        return pd.read_parquet(got.path, columns=_PBP_COLUMNS if dataset == "pbp" else None)

    for season in wanted:
        raw = load("schedule", season)
        if raw is None:
            continue
        sched = normalize_schedule(raw, league, season)
        sched["source_id"] = sources["schedule"][-1]
        parts["schedule"].append(sched)
        periods, goalies, pbp = (load(name, season) for name in ("periods", "goalies", "pbp"))
        if periods is not None or pbp is not None:
            parts["ends"].append(game_end_types(periods if periods is not None else pd.DataFrame(), pbp))
        if goalies is not None and len(goalies):
            parts["goalies"].append(normalize_goalies(goalies))
        if pbp is not None and len(pbp):
            parts["shots"].append(normalize_shots(pbp))
    report.sources = [sid for ids in sources.values() for sid in ids]
    if not parts["schedule"]:
        raise DataUnavailable(f"no schedule files could be obtained for {league}")

    sched = pd.concat(parts["schedule"], ignore_index=True)
    sched = sched.drop_duplicates("provider_event_id", keep="last").reset_index(drop=True)
    sched["neutral_site"] = flag_neutral_sites(sched)
    pid_of = _register_teams(store, league, sched)
    sched["home_id"] = sched["home_abbr"].map(pid_of)
    sched["away_id"] = sched["away_abbr"].map(pid_of)
    events, up = store.upsert_events(sched, sport="hockey")
    report.warnings += up.issues
    eid_of = dict(zip(events["provider_event_id"], events["event_id"]))

    # ---- results, with the regulation result where it is reliably known
    final = events[events["status"] == "final"].copy()
    scored = final["home_score"].notna() & final["away_score"].notna() & (final["home_score"] != final["away_score"])
    for _, row in final[~scored].iterrows():
        store.log_issue("final_without_valid_score", f"{row['provider_event_id']} {row['home_abbr']} v {row['away_abbr']}",
                        league=league, event_id=row["event_id"])
    ends = pd.concat(parts["ends"]) if parts["ends"] else pd.DataFrame(columns=["end_type", "sources_agree"])
    final = regulation_results(final[scored].join(ends[~ends.index.duplicated(keep="last")], on="provider_event_id"))
    unknown = int(final["went_overtime"].isna().sum())
    if unknown:
        report.warnings.append(f"{unknown} final games have no trusted regulation result (excluded from winner_3way)")
        for _, row in final[final["went_overtime"].isna()].iterrows():
            store.log_issue("regulation_result_unknown", f"{row['provider_event_id']} end type {row['end_type']}",
                            league=league, event_id=row["event_id"])
    store.upsert_results(final[["event_id", "home_score", "away_score", "went_overtime", "went_shootout",
                                "reg_home_score", "reg_away_score", "source_id"]])

    # ---- bulk tables
    games = events[["event_id", "home_abbr", "away_abbr"]]
    store.write_table(league, "games", games, source_ids=sources["schedule"])
    report.tables["games"] = len(games)
    if parts["goalies"]:
        box = pd.concat(parts["goalies"], ignore_index=True)
        box["event_id"] = box["provider_event_id"].map(eid_of)
        box["team_id"] = box["team_abbr"].map(pid_of)
        box = box[box["event_id"].notna() & box["team_id"].notna()].drop(columns=["provider_event_id", "team_abbr"])
        store.write_table(league, "goalie_box", box, source_ids=sources["goalies"])
        report.tables["goalie_box"] = len(box)
    else:
        report.warnings.append("no goalie box scores available")
    if parts["shots"]:
        shots = pd.concat(parts["shots"], ignore_index=True)
        shots["event_id"] = shots["provider_event_id"].map(eid_of)
        shots = shots[shots["event_id"].notna()].drop(columns=["provider_event_id"])
        store.write_table(league, "shots", shots, source_ids=sources["pbp"])
        report.tables["shots"] = len(shots)
        without = int((~final["event_id"].isin(set(shots["event_id"]))).sum())
        if without:
            report.warnings.append(f"{without} final games have no play-by-play shots")
    else:
        report.warnings.append("no play-by-play available: shot and expected-goal features will be missing")

    stored = store.events(league, with_results=True)
    report.n_events = len(stored)
    report.n_results = int(stored["winner_id"].notna().sum())
    report.n_upcoming = len(store.upcoming(league))
    if len(stored):
        report.first_event = stored["start_time_utc"].min().strftime("%Y-%m-%d")
        report.last_event = stored["start_time_utc"].max().strftime("%Y-%m-%d")
    return report
