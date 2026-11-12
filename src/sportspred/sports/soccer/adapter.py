"""Soccer ingestion: the steps shared by the international and club adapters.

Each source adapter (:mod:`.intl`, :mod:`.clubs`) turns downloaded files into
one frame of canonical match rows; this module writes those rows to the store
and keeps the store in step with files that are re-published in full.

Row columns: ``provider``, ``provider_event_id``, ``league``, ``season``,
``start_time_utc`` (tz-aware; 00:00Z of the calendar date when the source has
no kick-off time), ``time_precision``, ``status``, ``stage``, ``home_id``,
``away_id``, ``neutral_site``, ``home_score``, ``away_score`` (NaN until the
match has a result), ``source_id`` and optionally ``venue_city``, ``format``,
``went_shootout``, ``detail``.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from ..base import IngestReport

__all__ = ["SPORT", "date_start", "resolve_teams", "write_matches", "retire_vanished", "finish_report"]

SPORT = "soccer"

#: A re-published file that has lost more than this share of the matches
#: already stored is treated as damaged, not as a set of corrections.
_MAX_VANISHED_SHARE = 0.01

_RESULT_COLS = ("event_id", "home_score", "away_score", "went_shootout", "detail", "source_id")


def date_start(dates: pd.Series) -> pd.Series:
    """Stored start of a date-only match: 00:00Z of its calendar date."""
    return pd.to_datetime(dates, errors="coerce").dt.tz_localize("UTC")


def resolve_teams(store, league: str, names: pd.Series, provider: str, *, create: bool = True) -> pd.Series:
    """Participant ID per team name (``None`` for an unknown name when
    ``create`` is false).  Teams are matched within the league only."""
    unique = pd.DataFrame({"name": pd.unique(names.dropna())})
    ids = store.resolve_participants(unique, sport=SPORT, scope=league, kind="team", name_col="name",
                                     provider=provider, create=create)
    return names.map(dict(zip(unique["name"], ids)))


def write_matches(store, rows: pd.DataFrame, report: IngestReport) -> pd.DataFrame:
    """Upsert match rows and the results of the completed ones; returns the
    rows with their ``event_id``."""
    events, up = store.upsert_events(rows, sport=SPORT)
    report.warnings += up.issues
    final = events[events["status"] == "final"]
    store.upsert_results(final[[c for c in _RESULT_COLS if c in final.columns]])
    return events


def retire_vanished(store, league: str, provider: str, current_ids: pd.Series, report: IngestReport) -> None:
    """Cancel stored matches that the provider's file no longer lists.

    The sources are whole files that are corrected in place: when a date or a
    team name is fixed upstream the old row disappears and a new one arrives,
    and without this step the old match would stay behind as a duplicate.
    """
    stored = store.events(league, with_results=False)
    gone = stored[(stored["provider"] == provider) & (stored["status"] != "canceled")
                  & ~stored["provider_event_id"].isin(set(current_ids))]
    if gone.empty:
        return
    if len(gone) > _MAX_VANISHED_SHARE * len(stored):
        report.warnings.append(
            f"{len(gone)} stored matches are missing from the {provider} file; that is too many to be "
            "corrections, so nothing was cancelled (the download may be incomplete)")
        return
    keep = ["league", "season", "stage", "start_time_utc", "time_precision", "provider", "provider_event_id",
            "home_id", "away_id", "neutral_site"]
    store.upsert_events(gone[keep].assign(status="canceled"), sport=SPORT)
    for _, row in gone.iterrows():
        store.log_issue("vanished_upstream", f"{row['provider_event_id']} is no longer in the {provider} file",
                        league=league, event_id=row["event_id"])
    report.warnings.append(f"{len(gone)} stored matches no longer in the {provider} file were cancelled")


def finish_report(store, league: str, report: IngestReport, sources: Optional[List[str]] = None) -> IngestReport:
    report.sources += list(sources or [])
    stored = store.events(league, with_results=True)
    report.n_events = len(stored)
    report.n_results = int((stored["status"].eq("final") & stored["home_score"].notna()).sum())
    report.n_upcoming = len(store.upcoming(league))
    if len(stored):
        report.first_event = stored["start_time_utc"].min().strftime("%Y-%m-%d")
        report.last_event = stored["start_time_utc"].max().strftime("%Y-%m-%d")
    return report
