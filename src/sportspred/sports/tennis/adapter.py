"""Tennis ingestion from a public mirror of Jeff Sackmann's match archive.

Source: https://github.com/Aneeshers/tennis-sackmann-archive (branch ``main``),
a mirror of the ``tennis_atp`` / ``tennis_wta`` repositories compiled by Jeff
Sackmann, which are no longer on GitHub.  **Licence: CC BY-NC-SA 4.0** -
attribution required, share-alike, and **non-commercial use only**.

What the files do and do not contain:

* one row per completed match with tournament, surface, round, best-of,
  score, and (for most tour-level, Challenger and ITF matches) serve
  statistics for both players;
* ``tourney_date`` is the tournament's start date (normally its Monday),
  **not** the date of the match.  Every match is therefore stored on the
  tournament start date with time precision ``period``;
* the winner is listed first.  Sides are stored in ID order, never in
  winner-first order, so nothing downstream can learn the result from the
  position of a name;
* no upcoming matches, no draws that have not been played, no injury or
  withdrawal information.  Walkovers are skipped (no match was played);
  retirements are kept and flagged ``RET``;
* the mirror is a snapshot: its newest matches are from mid-2026.

League keys keep levels apart: tour-level main draw, tour-level qualifying,
Challenger / WTA 125 main draw, and ITF.  Olympic singles count as tour main
draw.  Left out: Davis Cup and Billie Jean King Cup ties (team events);
qualifying below tour level; and a few hundred ITF-level and early-2000s
rows that sit in the tour-level file under level codes no league claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ...adapters.http import Fetcher, NotFound
from ...sports import DataUnavailable
from ..base import IngestReport

__all__ = ["TENNIS_LEAGUES", "TOUR_GROUP", "ingest_tennis", "store_matches", "normalize_matches", "MIRROR", "LICENCE"]

MIRROR = "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main"
LICENCE = "CC BY-NC-SA 4.0 (Jeff Sackmann / Tennis Abstract; non-commercial, attribution, share-alike)"
PROVIDER = "sackmann"
_ITF_W = {"10", "15", "25", "35", "40", "50", "60", "75", "80", "100", "50+H", "35+H", "75+H", "100+H", "25+H",
          "60+H", "80+H", "40+H"}


def _is_q(df: pd.DataFrame) -> pd.Series:
    """Qualifying rounds are Q1, Q2, Q3 ...  "QF" is the main-draw quarter-final."""
    return df["round"].astype(str).str.upper().str.match(r"^Q\d")


@dataclass(frozen=True)
class TennisLeague:
    key: str
    tour: str                       # men | women: the pool players are matched in
    folder: str
    stem: str
    first_year: int
    keep: Callable[[pd.DataFrame], pd.Series]


def _lvl(df: pd.DataFrame) -> pd.Series:
    return df["tourney_level"].astype(str).str.upper()


TENNIS_LEAGUES: Dict[str, TennisLeague] = {l.key: l for l in (
    TennisLeague("atp", "men", "atp", "atp_matches", 2000,
                 lambda d: _lvl(d).isin(["G", "M", "A", "F", "O"]) & ~_is_q(d)),          # O: Olympic singles
    TennisLeague("atp_qual", "men", "atp", "atp_matches_qual_chall", 2008,
                 lambda d: _lvl(d).isin(["G", "M", "A"]) & _is_q(d)),
    TennisLeague("atp_challenger", "men", "atp", "atp_matches_qual_chall", 2008,
                 lambda d: _lvl(d).eq("C") & ~_is_q(d)),
    TennisLeague("itf_men", "men", "atp", "atp_matches_futures", 2015, lambda d: ~_is_q(d)),
    TennisLeague("wta", "women", "wta", "wta_matches", 2000,
                 lambda d: _lvl(d).isin(["G", "PM", "P", "I", "F", "T1", "T2", "T3", "T4", "T5", "W", "O"]) & ~_is_q(d)),
    TennisLeague("wta_qual", "women", "wta", "wta_matches_qual_itf", 2008,
                 lambda d: _lvl(d).isin(["G", "PM", "P", "I", "T1", "T2", "T3", "T4", "T5", "W"]) & _is_q(d)),
    TennisLeague("wta_125", "women", "wta", "wta_matches_qual_itf", 2012, lambda d: _lvl(d).eq("C") & ~_is_q(d)),
    TennisLeague("itf_women", "women", "wta", "wta_matches_qual_itf", 2015,
                 lambda d: _lvl(d).isin(_ITF_W) & ~_is_q(d)),
)}
#: Leagues whose results rate the same pool of players.
TOUR_GROUP: Dict[str, List[str]] = {
    "men": [k for k, v in TENNIS_LEAGUES.items() if v.tour == "men"],
    "women": [k for k, v in TENNIS_LEAGUES.items() if v.tour == "women"],
}
_ROUND_ORDER = {"Q1": 0, "Q2": 1, "Q3": 2, "RR": 3, "R128": 3, "R64": 4, "R32": 5, "R16": 6, "QF": 7, "SF": 8,
                "BR": 9, "F": 9}
_STATS = ["svpt", "1stWon", "2ndWon", "bpFaced", "bpSaved", "ace", "df"]


def _numeric(df: pd.DataFrame, name: str) -> pd.Series:
    """A numeric column, all-missing when the file does not have it."""
    if name not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[name], errors="coerce")


def normalize_matches(raw: pd.DataFrame, league: str, year: int) -> pd.DataFrame:
    """Match rows -> canonical rows with sides in a result-independent order."""
    cfg = TENNIS_LEAGUES[league]
    df = raw[cfg.keep(raw)].copy()
    score = df["score"].astype(str).str.upper()
    df = df[~score.str.contains("W/O|WALKOVER", regex=True) & df["winner_id"].notna() & df["loser_id"].notna()]
    df = df[df["winner_id"] != df["loser_id"]]
    if df.empty:
        return pd.DataFrame()
    score = df["score"].astype(str).str.upper()
    out = pd.DataFrame({
        "provider_event_id": df["tourney_id"].astype(str) + "-" + df["match_num"].astype(str),
        "league": league, "season": str(year),
        "start_time_utc": pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce", utc=True),
        "time_precision": "period", "status": "final", "provider": PROVIDER, "neutral_site": 1,
        "venue_name": df["tourney_name"].astype(str), "stage": "main_draw",
        "discipline": "singles",
        "winner_pid": pd.to_numeric(df["winner_id"], errors="coerce").astype("Int64").astype(str),
        "loser_pid": pd.to_numeric(df["loser_id"], errors="coerce").astype("Int64").astype(str),
        "winner_name": df["winner_name"].astype(str), "loser_name": df["loser_name"].astype(str),
        "surface": df["surface"].astype(str).str.lower().replace({"nan": None}),
        "best_of": pd.to_numeric(df["best_of"], errors="coerce"),
        "round": df["round"].astype(str).str.upper(),
        "level": df["tourney_level"].astype(str),
        "method": np.where(score.str.contains("RET|DEF|ABN|ABD", regex=True), "RET", None),
        "minutes": _numeric(df, "minutes"),
    })
    for stat in _STATS:
        out[f"w_{stat}"] = _numeric(df, f"w_{stat}").to_numpy()
        out[f"l_{stat}"] = _numeric(df, f"l_{stat}").to_numpy()
    out["round_order"] = out["round"].map(_ROUND_ORDER).fillna(5).astype(int)
    out = out[out["start_time_utc"].notna()]
    return out.drop_duplicates("provider_event_id", keep="last").reset_index(drop=True)


def ingest_tennis(store, league: str, *, seasons: Optional[Sequence[int]] = None,
                  refresh: Optional[bool] = None, fetcher: Optional[Fetcher] = None) -> IngestReport:
    if league not in TENNIS_LEAGUES:
        raise DataUnavailable(f"no public dataset for tennis league {league!r}; use `sportspred import events`")
    cfg = TENNIS_LEAGUES[league]
    fetcher = fetcher or Fetcher()
    this_year = pd.Timestamp.now(tz="UTC").year
    years = list(seasons) if seasons else list(range(cfg.first_year, this_year + 1))
    report = IngestReport(league=league)
    parts: List[pd.DataFrame] = []
    for year in years:
        url = f"{MIRROR}/{cfg.folder}/{cfg.stem}_{year}.csv"
        try:
            got = fetcher.fetch(url, provider="tennis_sackmann_mirror", relpath=f"{cfg.folder}/{cfg.stem}_{year}.csv",
                                max_age_hours=24.0 if year >= this_year else None, refresh=refresh, license=LICENCE)
        except NotFound:
            report.warnings.append(f"{year}: no file in the mirror")
            continue
        sid = got.register(store, notes=f"{league} {year}")
        report.sources.append(sid)
        norm = normalize_matches(pd.read_csv(got.path, low_memory=False), league, year)
        if len(norm):
            parts.append(norm.assign(source_id=sid))
    if not parts:
        raise DataUnavailable(f"no match files could be obtained for {league}")
    store_matches(store, league, pd.concat(parts, ignore_index=True), report)
    report.warnings.append("the mirror is a snapshot with no upcoming matches; import fixtures to get predictions")
    return report


def store_matches(store, league: str, df: pd.DataFrame, report: Optional[IngestReport] = None, *,
                  is_synthetic: bool = False) -> IngestReport:
    """Write normalised match rows (see :func:`normalize_matches`) to the store."""
    cfg = TENNIS_LEAGUES[league]
    report = report or IngestReport(league=league)
    df = df.copy()
    df["is_synthetic"] = is_synthetic
    if "source_id" not in df.columns:
        df["source_id"] = None
    scope = f"tennis_{cfg.tour}"
    players = pd.concat([
        df[["winner_pid", "winner_name"]].set_axis(["pid", "name"], axis=1),
        df[["loser_pid", "loser_name"]].set_axis(["pid", "name"], axis=1),
    ]).drop_duplicates("pid")
    players["participant_id"] = store.resolve_participants(
        players, sport="tennis", scope=scope, kind="player", provider=PROVIDER, provider_id_col="pid",
        is_synthetic=is_synthetic).to_numpy()
    pid = dict(zip(players["pid"], players["participant_id"]))
    df["winner_id"] = df["winner_pid"].map(pid)
    df["loser_id"] = df["loser_pid"].map(pid)
    # Sides in ID order: the stored order says nothing about who won.
    first = df["winner_id"] < df["loser_id"]
    df["home_id"] = np.where(first, df["winner_id"], df["loser_id"])
    df["away_id"] = np.where(first, df["loser_id"], df["winner_id"])
    df["format"] = [{"surface": s, "best_of": None if b != b else int(b), "round": r, "level": lv}
                    for s, b, r, lv in zip(df["surface"], df["best_of"], df["round"], df["level"])]
    events, up = store.upsert_events(df, sport="tennis")
    report.warnings += up.issues
    store.upsert_results(events[["event_id", "winner_id", "method", "source_id", "is_synthetic"]])
    stats = pd.DataFrame({"event_id": events["event_id"], "surface": events["surface"], "best_of": events["best_of"],
                          "round_order": events["round_order"], "minutes": events["minutes"]})
    home_won = (events["winner_id"] == events["home_id"]).to_numpy()
    for stat in _STATS:
        w, l = events[f"w_{stat}"].to_numpy(dtype=float), events[f"l_{stat}"].to_numpy(dtype=float)
        stats[f"home_{stat}"] = np.where(home_won, w, l)
        stats[f"away_{stat}"] = np.where(home_won, l, w)
    store.write_table(league, "match_stats", stats, source_ids=report.sources, is_synthetic=is_synthetic)
    report.tables["match_stats"] = len(stats)
    stored = store.events(league, with_results=True, include_synthetic=is_synthetic)
    report.n_events, report.n_results, report.n_upcoming = len(stored), int(stored["winner_id"].notna().sum()), 0
    report.first_event = stored["start_time_utc"].min().strftime("%Y-%m-%d")
    report.last_event = stored["start_time_utc"].max().strftime("%Y-%m-%d")
    return report
