"""Live odds from The Odds API (https://the-odds-api.com), v4.

You need your own API key (free tier available).  Pass it explicitly or set
the ``ODDS_API_KEY`` environment variable.

Player props are only served by the *per-event* odds endpoint, so fetching a
slate means one request for the list of events plus one request per event.
Usage is metered in credits: an event-odds request costs
``markets x regions`` credits (see :func:`estimate_cost`); the remaining
allowance is reported in response headers and tracked on the client.

The parsing functions are separate from the HTTP client so they can be used
on saved snapshots and tested offline.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import pandas as pd
import requests

from ..markets import canonical_market_key, markets_for_sport
from ..odds import american_to_decimal
from ..schema import QUOTE_COLUMNS, empty_quotes, normalize_quotes

__all__ = [
    "SPORT_KEYS",
    "OddsApiError",
    "OddsApiClient",
    "parse_event_odds",
    "quotes_from_events",
    "estimate_cost",
    "save_snapshot",
    "load_snapshot",
    "default_markets",
]

BASE_URL = "https://api.the-odds-api.com/v4"

#: Package sport code -> The Odds API sport key.
SPORT_KEYS: Dict[str, str] = {
    "nba": "basketball_nba",
    "wnba": "basketball_wnba",
    "ncaab": "basketball_ncaab",
    "nfl": "americanfootball_nfl",
    "ncaaf": "americanfootball_ncaaf",
    "mlb": "baseball_mlb",
    "nhl": "icehockey_nhl",
}
_SPORT_CODES = {v: k for k, v in SPORT_KEYS.items()}

_OVER_NAMES = {"over", "yes"}
_UNDER_NAMES = {"under", "no"}


class OddsApiError(RuntimeError):
    """An error response from The Odds API."""

    def __init__(self, status: int, message: str, error_code: Optional[str] = None) -> None:
        super().__init__(f"The Odds API error {status}: {message}")
        self.status = status
        self.message = message
        self.error_code = error_code


def default_markets(sport: str) -> List[str]:
    """Registered (non yes/no) market keys for a sport."""
    return [m.key for m in markets_for_sport(sport) if not m.yes_no]


def estimate_cost(n_events: int, n_markets: int, n_regions: int = 1, historical: bool = False) -> int:
    """Credits needed to pull props for a slate.

    Each event-odds request costs ``markets x regions`` credits (10x for the
    historical endpoint); listing events is free.
    """
    per_event = n_markets * max(n_regions, 1) * (10 if historical else 1)
    return n_events * per_event


def _sport_code(sport_key: str) -> str:
    return _SPORT_CODES.get(sport_key, sport_key)


def _price_to_decimal(price: Any, odds_format: str) -> Optional[float]:
    try:
        value = float(price)
    except (TypeError, ValueError):
        return None
    if odds_format == "american":
        try:
            return american_to_decimal(value)
        except ValueError:
            return None
    return value if value > 1.0 else None


def parse_event_odds(
    payload: Mapping[str, Any], odds_format: str = "decimal", sport: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Flatten one event-odds response into quote rows.

    Over/Under (or Yes/No) outcomes for the same book, market, player and
    line are paired into a single row.  Yes/No markets without a ``point``
    are given the line ``0.5``.  Outcomes the parser cannot interpret are
    skipped rather than raising, so one odd market does not lose a slate.
    """
    if "data" in payload and "bookmakers" not in payload and isinstance(payload["data"], Mapping):
        payload = payload["data"]  # historical endpoint wraps the event
    sport_code = sport or _sport_code(str(payload.get("sport_key", "")))
    event = {
        "sport": sport_code,
        "event_id": str(payload.get("id", "")),
        "commence_time": payload.get("commence_time"),
        "home_team": payload.get("home_team") or "",
        "away_team": payload.get("away_team") or "",
    }
    rows: Dict[tuple, Dict[str, Any]] = {}
    for bookmaker in payload.get("bookmakers") or []:
        book = str(bookmaker.get("key", "")).lower()
        if not book:
            continue
        for market in bookmaker.get("markets") or []:
            raw_key = str(market.get("key", "")).lower()
            if not raw_key:
                continue
            _base, is_alternate = canonical_market_key(raw_key)
            last_update = market.get("last_update") or bookmaker.get("last_update")
            for outcome in market.get("outcomes") or []:
                name = str(outcome.get("name", "")).strip()
                description = str(outcome.get("description") or "").strip()
                side_name = name.lower()
                if side_name in _OVER_NAMES:
                    side, player = "over", description
                elif side_name in _UNDER_NAMES:
                    side, player = "under", description
                elif not description:
                    # Some one-sided markets put the player in `name`.
                    side, player = "over", name
                else:
                    continue
                if not player:
                    continue
                price = _price_to_decimal(outcome.get("price"), odds_format)
                if price is None:
                    continue
                point = outcome.get("point")
                try:
                    line = float(point) if point is not None else 0.5
                except (TypeError, ValueError):
                    continue
                key = (book, raw_key, player, line)
                row = rows.get(key)
                if row is None:
                    row = {
                        **event,
                        "book": book,
                        "market": raw_key,
                        "player": player,
                        "line": line,
                        "over_price": None,
                        "under_price": None,
                        "last_update": last_update,
                        "is_alternate": is_alternate,
                    }
                    rows[key] = row
                column = "over_price" if side == "over" else "under_price"
                # If a book lists the same side twice keep the better price.
                if row[column] is None or price > row[column]:
                    row[column] = price
    return list(rows.values())


def quotes_from_events(
    payloads: Iterable[Mapping[str, Any]], odds_format: str = "decimal", sport: Optional[str] = None
) -> pd.DataFrame:
    """Quotes table from an iterable of event-odds responses."""
    rows: List[Dict[str, Any]] = []
    for payload in payloads:
        rows.extend(parse_event_odds(payload, odds_format=odds_format, sport=sport))
    if not rows:
        return empty_quotes()
    return normalize_quotes(pd.DataFrame(rows))


def save_snapshot(payloads: Sequence[Mapping[str, Any]], path: Union[str, Path], meta: Optional[Mapping[str, Any]] = None) -> Path:
    """Write raw event-odds responses to a JSON file (for re-scanning and CLV)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"meta": dict(meta or {}), "events": list(payloads)}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(document), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_snapshot(path: Union[str, Path]) -> pd.DataFrame:
    """Load a snapshot written by :func:`save_snapshot` (or a bare list of events)."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(document, Mapping) and "events" in document:
        meta = document.get("meta") or {}
        events = document["events"]
    elif isinstance(document, list):
        meta, events = {}, document
    else:
        meta, events = {}, [document]
    return quotes_from_events(events, odds_format=meta.get("odds_format", "decimal"), sport=meta.get("sport"))


@dataclass
class Usage:
    """Credit usage reported by the API on the most recent response."""

    remaining: Optional[int] = None
    used: Optional[int] = None
    last: Optional[int] = None


class OddsApiClient:
    """Minimal client for the endpoints the scanner needs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        session: Optional[requests.Session] = None,
        timeout: float = 20.0,
        max_retries: int = 3,
        base_url: str = BASE_URL,
    ) -> None:
        self.api_key = api_key or os.environ.get("ODDS_API_KEY", "")
        if not self.api_key:
            raise OddsApiError(401, "no API key: pass api_key or set ODDS_API_KEY")
        self.session = session or requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_url = base_url.rstrip("/")
        self.usage = Usage()

    # -------------------------------------------------------------- plumbing
    def _get(self, path: str, params: Optional[Mapping[str, Any]] = None) -> Any:
        query = {k: v for k, v in (params or {}).items() if v not in (None, "")}
        query["apiKey"] = self.api_key
        url = f"{self.base_url}{path}"
        for attempt in range(self.max_retries + 1):
            response = self.session.get(url, params=query, timeout=self.timeout)
            self._record_usage(response.headers)
            if response.status_code == 429 and attempt < self.max_retries:
                time.sleep(min(2.0**attempt, 8.0))
                continue
            if response.status_code >= 400:
                message, code = response.text[:300], None
                try:
                    body = response.json()
                    message = body.get("message", message)
                    code = body.get("error_code")
                except ValueError:
                    pass
                raise OddsApiError(response.status_code, message, code)
            return response.json()
        raise OddsApiError(429, "rate limited")  # pragma: no cover - loop always returns/raises

    def _record_usage(self, headers: Mapping[str, str]) -> None:
        def as_int(name: str) -> Optional[int]:
            value = headers.get(name)
            try:
                return int(float(value)) if value is not None else None
            except (TypeError, ValueError):
                return None

        remaining = as_int("x-requests-remaining")
        if remaining is not None:
            self.usage = Usage(remaining, as_int("x-requests-used"), as_int("x-requests-last"))

    @staticmethod
    def _sport_key(sport: str) -> str:
        return SPORT_KEYS.get(sport.lower(), sport)

    # ------------------------------------------------------------- endpoints
    def sports(self) -> List[Dict[str, Any]]:
        """In-season sports (free)."""
        return self._get("/sports")

    def events(
        self,
        sport: str,
        commence_from: Optional[str] = None,
        commence_to: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Upcoming events for a sport (free).  Times are ISO-8601 ``...Z``."""
        return self._get(
            f"/sports/{self._sport_key(sport)}/events",
            {"dateFormat": "iso", "commenceTimeFrom": commence_from, "commenceTimeTo": commence_to},
        )

    def event_odds(
        self,
        sport: str,
        event_id: str,
        markets: Sequence[str],
        regions: Sequence[str] = ("us",),
        bookmakers: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Odds for one event, including player-prop markets (metered)."""
        params: Dict[str, Any] = {
            "markets": ",".join(markets),
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }
        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)
        else:
            params["regions"] = ",".join(regions)
        return self._get(f"/sports/{self._sport_key(sport)}/events/{event_id}/odds", params)

    def historical_event_odds(
        self,
        sport: str,
        event_id: str,
        date: str,
        markets: Sequence[str],
        regions: Sequence[str] = ("us",),
        bookmakers: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Snapshot of an event's odds at ``date`` (paid plans; 10x credits)."""
        params: Dict[str, Any] = {
            "date": date,
            "markets": ",".join(markets),
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }
        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)
        else:
            params["regions"] = ",".join(regions)
        return self._get(f"/historical/sports/{self._sport_key(sport)}/events/{event_id}/odds", params)

    # ------------------------------------------------------------ high level
    def fetch_props(
        self,
        sport: str,
        markets: Optional[Sequence[str]] = None,
        regions: Sequence[str] = ("us", "us2", "eu"),
        bookmakers: Optional[Sequence[str]] = None,
        max_events: Optional[int] = None,
        commence_from: Optional[str] = None,
        commence_to: Optional[str] = None,
        snapshot_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """Fetch player-prop quotes for every upcoming event of a sport.

        Returns a quotes table.  ``snapshot_path`` additionally saves the raw
        responses.  Check :func:`estimate_cost` first: a 10-game slate with 8
        markets and 3 regions costs 240 credits.
        """
        markets = list(markets) if markets else default_markets(sport)
        if not markets:
            raise ValueError(f"no markets given and none registered for sport {sport!r}")
        events = self.events(sport, commence_from, commence_to)
        if max_events is not None:
            events = events[:max_events]
        payloads = []
        for event in events:
            payloads.append(self.event_odds(sport, event["id"], markets, regions, bookmakers))
        if snapshot_path is not None:
            save_snapshot(
                payloads,
                snapshot_path,
                {
                    "sport": sport.lower(),
                    "odds_format": "decimal",
                    "markets": markets,
                    "fetched_at": pd.Timestamp.now(tz="UTC").isoformat(),
                },
            )
        return quotes_from_events(payloads, odds_format="decimal", sport=sport.lower())
