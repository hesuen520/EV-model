"""Club league ingestion: a results file plus a fixture feed.

**Results** - ``data/Matches.csv`` of
https://github.com/xgabora/Club-Football-Match-Data-2000-2025 (MIT licence;
a repackaging of Football-Data.co.uk).  One row per league match since
2000/01 for the five divisions used here: date, kick-off time in UK local
time (only from 2019/20; earlier rows are date-only), full-time score, shots
and shots on target (from about 2005/06 outside England), and Bet365
home/draw/away decimal odds.  It is refreshed upstream every few weeks, so it
normally lags the current season, and it has holes (La Liga 2006/07 is absent
and a few seasons before 2005 are incomplete).

**Fixtures** - ``<season>/<code>.json`` of
https://github.com/openfootball/football.json (public domain, CC0): the full
schedule of the current season with kick-off times in the league's local time
where published, and full-time scores of the matches entered so far.  It
supplies what the results file does not yet cover: later results and the
remaining fixtures.  The results file stays the source of record for every
date it covers; fixture rows in that window are only compared with it.

What neither file contains: line-ups, injuries, expected goals, cup or
European matches (so rest days only see league matches), or the time at which
an odds quote was taken.  The odds are stored in their own table and are read
only by the market benchmark, never by a model.

Not usable, and therefore not adapted: the results file's ``USA`` (MLS) and
other "extra league" rows, whose dates have day and month swapped whenever the
day is 12 or lower (matches appear in January).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ...adapters.http import Fetcher, NotFound
from ...core.timeutil import utcnow
from ...sports import DataUnavailable
from ..base import IngestReport
from .adapter import date_start, finish_report, resolve_teams, retire_vanished, write_matches
from .club_names import FIXTURE_NAMES, canonical_result_name

__all__ = ["ClubLeague", "CLUB_LEAGUES", "season_start_year", "kickoff_utc", "normalize_matches",
           "parse_fixtures", "ingest_club"]

RESULTS_PROVIDER = "xgabora"
FIXTURES_PROVIDER = "openfootball"
MATCHES_URL = "https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data-2000-2025/main/data/Matches.csv"
FIXTURES_URL = "https://raw.githubusercontent.com/openfootball/football.json/master/{label}/{code}.json"
_RESULTS_LICENCE = "MIT (xgabora/Club-Football-Match-Data-2000-2025; data from Football-Data.co.uk)"
_FIXTURES_LICENCE = "CC0 1.0 (openfootball/football.json, public domain)"
#: Kick-off times in the results file are UK local time for every league.
RESULTS_TZ = "Europe/London"
_MATCH_COLS = ["Division", "MatchDate", "MatchTime", "HomeTeam", "AwayTeam", "FTHome", "FTAway",
               "HomeTarget", "AwayTarget", "OddHome", "OddDraw", "OddAway"]


@dataclass(frozen=True)
class ClubLeague:
    league: str
    division: str          # Football-Data.co.uk division code in the results file
    fixtures_code: str     # file stem in the fixture feed
    fixtures_tz: str       # zone of the fixture feed's kick-off times


CLUB_LEAGUES: Dict[str, ClubLeague] = {
    "epl": ClubLeague("epl", "E0", "en.1", "Europe/London"),
    "laliga": ClubLeague("laliga", "SP1", "es.1", "Europe/Madrid"),
    "bundesliga": ClubLeague("bundesliga", "D1", "de.1", "Europe/Berlin"),
    "seriea": ClubLeague("seriea", "I1", "it.1", "Europe/Rome"),
    "ligue1": ClubLeague("ligue1", "F1", "fr.1", "Europe/Paris"),
}


def season_start_year(dates: pd.Series) -> pd.Series:
    """Year in which a match's season began.  Seasons turn over on 1 July,
    except in 2020 when the interrupted 2019/20 season ran into early August."""
    dates = pd.to_datetime(dates)
    turn = pd.to_datetime(dates.dt.year.astype(str) + np.where(dates.dt.year == 2020, "-08-10", "-07-01"))
    return (dates.dt.year - (dates < turn).astype(int)).astype(int)


def kickoff_utc(dates: pd.Series, times: pd.Series, tz: str) -> Tuple[pd.Series, np.ndarray]:
    """``(start_time_utc, time_precision)``: the local kick-off converted to
    UTC where a time is given, otherwise 00:00Z of the date (``date``)."""
    text = times.astype("string").str.strip()
    has_time = text.str.match(r"^\d{1,2}:\d{2}").fillna(False).to_numpy(dtype=bool)
    local = pd.to_datetime(dates.astype(str) + " " + text.where(has_time, "00:00"), errors="coerce")
    exact = local.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    ok = has_time & exact.notna().to_numpy()
    return date_start(dates).where(~ok, exact), np.where(ok, "datetime", "date")


def normalize_matches(raw: pd.DataFrame, cfg: ClubLeague) -> pd.DataFrame:
    """Results-file rows of one division -> canonical match rows (team names,
    not yet IDs) with the shots-on-target and odds columns alongside."""
    part = raw[raw["Division"] == cfg.division]
    part = part[part["FTHome"].notna() & part["FTAway"].notna()]
    home = part["HomeTeam"].astype(str).str.strip().map(canonical_result_name)
    away = part["AwayTeam"].astype(str).str.strip().map(canonical_result_name)
    start, precision = kickoff_utc(part["MatchDate"], part["MatchTime"], RESULTS_TZ)
    df = pd.DataFrame({
        "provider": RESULTS_PROVIDER,
        "provider_event_id": cfg.division + "|" + part["MatchDate"].astype(str) + "|" + home + "|" + away,
        "league": cfg.league,
        "date": pd.to_datetime(part["MatchDate"], errors="coerce"),
        "start_time_utc": start,
        "time_precision": precision,
        "status": "final",
        "stage": "regular",
        "home_name": home,
        "away_name": away,
        "home_score": part["FTHome"].astype(float),
        "away_score": part["FTAway"].astype(float),
        "neutral_site": 0,
        "sot_home": part["HomeTarget"].astype(float),
        "sot_away": part["AwayTarget"].astype(float),
        "odd_home": part["OddHome"].astype(float),
        "odd_draw": part["OddDraw"].astype(float),
        "odd_away": part["OddAway"].astype(float),
    })
    df = df[df["start_time_utc"].notna() & df["home_name"].ne(df["away_name"])].copy()
    df["season"] = season_start_year(df["date"]).astype(str)
    return df.drop_duplicates("provider_event_id", keep="last").reset_index(drop=True)


def _full_time(score: Any) -> Tuple[float, float]:
    """The feed writes a score as ``{"ft": [h, a], ...}`` or as ``[h, a]``."""
    ft = score.get("ft") if isinstance(score, dict) else score
    if isinstance(ft, (list, tuple)) and len(ft) == 2 and all(isinstance(g, (int, float)) for g in ft):
        return float(ft[0]), float(ft[1])
    return float("nan"), float("nan")


def parse_fixtures(payload: Dict[str, Any], cfg: ClubLeague) -> pd.DataFrame:
    """One fixture-feed file -> canonical match rows under the feed's own
    team names (``home_fixture_name`` / ``away_fixture_name``)."""
    recs = []
    for m in payload.get("matches", []):
        if not (m.get("date") and m.get("team1") and m.get("team2")):
            continue
        hs, as_ = _full_time(m.get("score"))
        note = str(m.get("status") or "").lower()
        status = ("final" if hs == hs else "canceled" if note.startswith("cancel")
                  else "postponed" if note == "postponed" else "scheduled")
        recs.append({"date": m["date"], "time": m.get("time"), "home_fixture_name": str(m["team1"]).strip(),
                     "away_fixture_name": str(m["team2"]).strip(), "home_score": hs, "away_score": as_,
                     "status": status})
    df = pd.DataFrame(recs, columns=["date", "time", "home_fixture_name", "away_fixture_name", "home_score",
                                     "away_score", "status"])
    start, precision = kickoff_utc(df["date"], df["time"], cfg.fixtures_tz)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["start_time_utc"], df["time_precision"] = start, precision
    df = df[df["date"].notna()].copy()
    df["season"] = season_start_year(df["date"]).astype(str)
    return df.reset_index(drop=True)


def _merge_table(store, league: str, name: str, new: pd.DataFrame, source_id: str) -> int:
    """Add or replace rows of a per-event table, keeping rows for events that
    are not part of this write (an ingest may cover only some seasons)."""
    if store.has_table(league, name):
        old = store.read_table(league, name)
        new = pd.concat([old[~old["event_id"].isin(new["event_id"])], new], ignore_index=True)
    store.write_table(league, name, new, source_ids=[source_id])
    return len(new)


def _fixture_rows(fx: pd.DataFrame, cfg: ClubLeague, results: pd.DataFrame, label: str,
                  report: IngestReport) -> pd.DataFrame:
    """Map a fixture file onto the canonical club names, compare it with the
    results file where the two overlap, and return the rows the results file
    does not cover yet."""
    table = FIXTURE_NAMES.get(cfg.league, {})
    known = set(results["home_name"]) | set(results["away_name"])
    for side in ("home", "away"):
        raw_name = fx[f"{side}_fixture_name"]
        fx[f"{side}_name"] = raw_name.map(table).where(raw_name.isin(table), raw_name.where(raw_name.isin(known)))
    unmapped = sorted(set(fx.loc[fx["home_name"].isna(), "home_fixture_name"])
                      | set(fx.loc[fx["away_name"].isna(), "away_fixture_name"]))
    if unmapped:
        report.warnings.append(
            f"{label}: fixtures of clubs without a name mapping were skipped ({', '.join(unmapped)}); "
            "add them to club_names.FIXTURE_NAMES")
    fx = fx[fx["home_name"].notna() & fx["away_name"].notna()]
    last = results["date"].max()
    overlap = fx[(fx["date"] <= last) & (fx["status"] == "final")].merge(
        results[["season", "home_name", "away_name", "date", "home_score", "away_score"]],
        on=["season", "home_name", "away_name"], how="left", suffixes=("", "_res"))
    agree = ((overlap["date"] == overlap["date_res"]) & (overlap["home_score"] == overlap["home_score_res"])
             & (overlap["away_score"] == overlap["away_score_res"]))
    if len(overlap) and not agree.all():
        report.warnings.append(
            f"{label}: {int((~agree).sum())} of {len(overlap)} fixture-feed results up to {last.date()} "
            "disagree with or are missing from the results file (results file kept)")
    new = fx[fx["date"] > last].copy()
    new["provider"] = FIXTURES_PROVIDER
    new["provider_event_id"] = cfg.fixtures_code + "|" + new["season"] + "|" + new["home_name"] + "|" + new["away_name"]
    new["league"], new["stage"], new["neutral_site"] = cfg.league, "regular", 0
    return new.drop_duplicates("provider_event_id", keep="last")


def ingest_club(store, league: str, *, seasons: Optional[Sequence[int]] = None,
                refresh: Optional[bool] = None, fetcher: Optional[Fetcher] = None) -> IngestReport:
    """Download, normalise and store one club league.  ``seasons`` restricts
    the write to seasons starting in those years."""
    cfg = CLUB_LEAGUES[league]
    fetcher = fetcher or Fetcher()
    report = IngestReport(league=league)
    got = fetcher.fetch(MATCHES_URL, provider=RESULTS_PROVIDER, relpath="Matches.csv", max_age_hours=24.0,
                        refresh=refresh, license=_RESULTS_LICENCE)
    if got.stale:
        report.warnings.append(f"results file: refresh failed; using the copy collected {got.collected_at_utc}")
    results_sid = got.register(store, notes="club results, shots on target and Bet365 odds (all divisions)")
    results = normalize_matches(pd.read_csv(got.path, usecols=_MATCH_COLS, low_memory=False), cfg)
    if results.empty:
        raise DataUnavailable(f"the results file has no rows for division {cfg.division} ({league})")
    wanted = None if not seasons else {str(int(s)) for s in seasons}
    rows = results if wanted is None else results[results["season"].isin(wanted)].copy()
    rows["source_id"] = results_sid
    names = pd.concat([rows["home_name"], rows["away_name"]])
    ids = resolve_teams(store, league, names, RESULTS_PROVIDER)
    rows["home_id"], rows["away_id"] = ids.iloc[: len(rows)].to_numpy(), ids.iloc[len(rows):].to_numpy()
    events = write_matches(store, rows, report)
    if wanted is None:
        retire_vanished(store, league, RESULTS_PROVIDER, rows["provider_event_id"], report)
    stats = events.loc[events["sot_home"].notna() & events["sot_away"].notna(), ["event_id", "sot_home", "sot_away"]]
    odds = events[["event_id", "odd_home", "odd_draw", "odd_away"]].dropna()
    report.tables["match_stats"] = _merge_table(store, league, "match_stats", stats, results_sid)
    report.tables["odds_1x2"] = _merge_table(store, league, "odds_1x2", odds, results_sid)

    sources = [results_sid]
    first = int(results["season"].astype(int).max())
    current = int(season_start_year(pd.Series([pd.Timestamp(utcnow()).tz_localize(None)])).iloc[0])
    for year in range(first, current + 1):
        if wanted is not None and str(year) not in wanted:
            continue
        label = f"{year}-{(year + 1) % 100:02d}"
        try:
            fixtures = fetcher.fetch(FIXTURES_URL.format(label=label, code=cfg.fixtures_code),
                                     provider=FIXTURES_PROVIDER, relpath=f"{label}/{cfg.fixtures_code}.json",
                                     max_age_hours=6.0, refresh=refresh, license=_FIXTURES_LICENCE)
        except NotFound:
            report.warnings.append(f"{label}: no fixture file published; no upcoming matches for that season")
            continue
        except DataUnavailable as exc:
            report.warnings.append(f"{label}: fixture feed unreachable ({exc}); results file only")
            continue
        fixtures_sid = fixtures.register(store, notes=f"{league} fixtures and scores {label}")
        sources.append(fixtures_sid)
        fx = parse_fixtures(json.loads(fixtures.path.read_text(encoding="utf-8")), cfg)
        new = _fixture_rows(fx, cfg, results, label, report)
        if new.empty:
            continue
        new["source_id"] = fixtures_sid
        fx_names = pd.concat([new["home_name"], new["away_name"]])
        fx_ids = resolve_teams(store, league, fx_names, RESULTS_PROVIDER)
        new["home_id"], new["away_id"] = fx_ids.iloc[: len(new)].to_numpy(), fx_ids.iloc[len(new):].to_numpy()
        write_matches(store, new, report)
    return finish_report(store, league, report, sources)
