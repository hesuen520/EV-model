"""International football results from the martj42 public datasets.

Sources (one CSV pair per league, re-published in full upstream):

* men - https://github.com/martj42/international_results (CC0 1.0):
  full internationals since 1872, no Olympic, B-team or U-23 matches;
* women - https://github.com/martj42/womens-international-results (public
  domain, Unlicense): described upstream as work in progress - major
  tournaments are complete, many friendlies are missing.

What the files do and do not contain:

* ``results.csv``: date (no kick-off time), the two teams under their
  *current* names, the score, the tournament name, city, country and a
  neutral-venue flag.  The score is the one **after extra time where extra
  time was played, never including a penalty shoot-out**; the files do not
  give the 90-minute score separately and do not say which matches were
  knockout ties.  ``shootouts.csv`` lists the matches that went to penalties
  and who won them.
* Rows without a score are fixtures that have not been played; upstream only
  adds them occasionally (ahead of some tournaments).
* **No** kick-off times, line-ups, injuries, expected goals or betting odds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from ...adapters.http import Fetcher
from ..base import IngestReport
from .adapter import date_start, finish_report, resolve_teams, retire_vanished, write_matches, refuse_mass_disappearance

__all__ = ["IntlSource", "INTL_SOURCES", "competition_stage", "normalize_results", "ingest_international"]

PROVIDER = "martj42"
_RAW = "https://raw.githubusercontent.com/{repo}/master/{name}"
_REQUIRED = ("date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral")


@dataclass(frozen=True)
class IntlSource:
    league: str
    repo: str
    licence: str


INTL_SOURCES: Dict[str, IntlSource] = {
    "soccer_intl_men": IntlSource("soccer_intl_men", "martj42/international_results", "CC0 1.0 Universal"),
    "soccer_intl_women": IntlSource("soccer_intl_women", "martj42/womens-international-results",
                                    "public domain (Unlicense)"),
}


def competition_stage(tournament: pd.Series) -> pd.Series:
    """``friendly``, ``qualifying`` or ``regular`` (any other named competition:
    final tournaments, nations leagues, regional and invitational cups)."""
    name = tournament.fillna("").str.lower()
    stage = pd.Series("regular", index=tournament.index)
    stage[name.str.contains("qualif", regex=False)] = "qualifying"
    stage[name.eq("friendly")] = "friendly"
    return stage


def normalize_results(raw: pd.DataFrame, league: str, shootouts: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """``results.csv`` rows -> canonical match rows (team names, not yet IDs)."""
    missing = [c for c in _REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"international results file lacks columns {missing}")
    date = raw["date"].astype(str).str.strip()
    home = raw["home_team"].astype(str).str.strip()
    away = raw["away_team"].astype(str).str.strip()
    key = date + "|" + home + "|" + away
    # Two meetings of the same sides on one date are distinct matches.
    repeat = key.groupby(key).cumcount()
    df = pd.DataFrame({
        "provider": PROVIDER,
        "provider_event_id": key.where(repeat == 0, key + "|" + (repeat + 1).astype(str)),
        "league": league,
        "start_time_utc": date_start(date),
        "time_precision": "date",
        "home_name": home,
        "away_name": away,
        "home_score": pd.to_numeric(raw["home_score"], errors="coerce"),
        "away_score": pd.to_numeric(raw["away_score"], errors="coerce"),
        "neutral_site": raw["neutral"].astype(str).str.upper().eq("TRUE").astype(int),
        "venue_city": raw["city"] if "city" in raw.columns else None,
        "stage": competition_stage(raw["tournament"]),
        "tournament": raw["tournament"].astype(str),
        "country": raw["country"].astype(str) if "country" in raw.columns else "",
    })
    df = df[df["start_time_utc"].notna() & home.ne("") & away.ne("") & home.ne(away)].copy()
    df["season"] = df["start_time_utc"].dt.year.astype(str)
    played = df["home_score"].notna() & df["away_score"].notna()
    df["status"] = np.where(played, "final", "scheduled")
    df["format"] = [{"tournament": t, "country": c} for t, c in zip(df["tournament"], df["country"])]
    df["went_shootout"] = 0
    df["detail"] = None
    if shootouts is not None and len(shootouts):
        so = shootouts.assign(key=shootouts["date"].astype(str).str.strip() + "|"
                              + shootouts["home_team"].astype(str).str.strip() + "|"
                              + shootouts["away_team"].astype(str).str.strip())
        winner = dict(zip(so["key"], so["winner"].astype(str)))
        base_key = df["provider_event_id"].str.split("|").str[:3].str.join("|")
        hit = base_key.isin(winner) & played
        df.loc[hit, "went_shootout"] = 1
        df.loc[hit, "detail"] = [{"shootout_winner": winner[k]} for k in base_key[hit]]
    return df.drop(columns=["tournament", "country"]).reset_index(drop=True)


def ingest_international(store, league: str, *, seasons: Optional[Sequence[int]] = None,
                         refresh: Optional[bool] = None, fetcher: Optional[Fetcher] = None) -> IngestReport:
    """Download, normalise and store one international league.  ``seasons``
    restricts the write to those calendar years."""
    src = INTL_SOURCES[league]
    fetcher = fetcher or Fetcher()
    report = IngestReport(league=league)
    got = {}
    for name in ("results.csv", "shootouts.csv"):
        got[name] = fetcher.fetch(
            _RAW.format(repo=src.repo, name=name), provider=PROVIDER, relpath=f"{src.repo.split('/')[1]}/{name}",
            max_age_hours=6.0, refresh=refresh, license=src.licence)
        if got[name].stale:
            report.warnings.append(
                f"{name}: refresh failed; using the copy collected {got[name].collected_at_utc}")
    results_sid = got["results.csv"].register(store, notes=f"{league} results")
    shootouts_sid = got["shootouts.csv"].register(store, notes=f"{league} penalty shoot-outs")
    raw = pd.read_csv(got["results.csv"].path)
    rows = normalize_results(raw, league, pd.read_csv(got["shootouts.csv"].path))
    if len(rows) < len(raw):
        report.warnings.append(f"{len(raw) - len(rows)} rows without a usable date or with invalid teams were skipped")
    if seasons:
        rows = rows[rows["season"].astype(int).isin([int(s) for s in seasons])].copy()
    rows["source_id"] = results_sid
    names = pd.concat([rows["home_name"], rows["away_name"]])
    ids = resolve_teams(store, league, names, PROVIDER)
    rows["home_id"] = ids.iloc[: len(rows)].to_numpy()
    rows["away_id"] = ids.iloc[len(rows):].to_numpy()
    if not seasons:
        refuse_mass_disappearance(store, league, PROVIDER, rows["provider_event_id"])
    write_matches(store, rows, report)
    if not seasons:
        retire_vanished(store, league, PROVIDER, rows["provider_event_id"], report)
    return finish_report(store, league, report, [results_sid, shootouts_sid])
