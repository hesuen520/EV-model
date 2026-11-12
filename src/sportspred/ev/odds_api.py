"""The Odds API v4 -> price rows.

**Written from the published documentation; not run against the live service
in the environment it was written in** (the host is blocked there).  The
parser follows the documented ``/v4/sports/{sport}/odds`` response: a list of
events, each with ``bookmakers[].markets[].outcomes[]`` where an outcome has
``name``, ``price`` and, for spreads and totals, ``point``.  Check the current
documentation (https://the-odds-api.com/liveapi/guides/v4/) before use; every
call spends credits (featured markets cost markets x regions per call).

The key is read from the ``ODDS_API_KEY`` environment variable only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import requests

from ..config import credential
from ..core.timeutil import iso_z, parse_ts
from ..sports import DataUnavailable

__all__ = ["SPORT_KEYS", "parse_odds_api", "fetch_odds_api"]

SPORT_KEYS: Dict[str, str] = {
    "nba": "basketball_nba", "wnba": "basketball_wnba", "ncaam": "basketball_ncaab", "nfl": "americanfootball_nfl",
    "ncaaf": "americanfootball_ncaaf", "nhl": "icehockey_nhl", "mlb": "baseball_mlb", "epl": "soccer_epl",
}
_MARKETS = {"h2h": "winner", "spreads": "spread", "totals": "total"}
_RULES = {"winner": "moneyline_incl_ot", "spread": "spread_incl_ot", "total": "total_incl_ot"}


def parse_odds_api(payload: Sequence[Dict[str, Any]], league: str, *, odds_format: str = "american") -> pd.DataFrame:
    """Rows in the price-import format (see ``PRICE_TEMPLATE``).

    A soccer ``h2h`` market has three outcomes (home / Draw / away) and is
    mapped to the regulation three-way market; everywhere else ``h2h`` is the
    winner including overtime.  The spread ``line`` is stored as the home
    side's handicap, the convention used across the platform.
    """
    rows: List[Dict[str, Any]] = []
    for event in payload or []:
        home, away, start = event.get("home_team"), event.get("away_team"), event.get("commence_time")
        if not home or not away or not start:
            continue
        date = iso_z(parse_ts(start))[:10]
        for book in event.get("bookmakers") or []:
            for market in book.get("markets") or []:
                kind = _MARKETS.get(market.get("key"))
                outcomes = market.get("outcomes") or []
                if kind is None or not outcomes:
                    continue
                three_way = kind == "winner" and any(str(o.get("name")).lower() == "draw" for o in outcomes)
                quoted = market.get("last_update") or book.get("last_update")
                if not quoted:
                    continue
                for o in outcomes:
                    name, price, point = str(o.get("name")), o.get("price"), o.get("point")
                    if price is None:
                        continue
                    line: Optional[float] = None
                    selection = "draw" if name.lower() == "draw" else name
                    if kind == "spread":
                        if point is None:
                            continue
                        line = float(point) if name == home else -float(point)
                    elif kind == "total":
                        if point is None:
                            continue
                        line, selection = float(point), name.lower()
                    text = f"{float(price):+g}" if odds_format == "american" else f"{float(price):g}"
                    rows.append({
                        "league": league, "event_id": "", "home": home, "away": away, "date": date,
                        "market": "winner_3way" if three_way else kind, "selection": selection,
                        "line": "" if line is None else line, "venue_kind": "sportsbook", "venue": book.get("key"),
                        "price": text, "ask": "", "bid": "", "quoted_at": iso_z(parse_ts(quoted)),
                        "rules_key": "three_way_regulation" if three_way else _RULES[kind],
                        "fee_key": "sportsbook", "is_executable": 1,
                    })
    return pd.DataFrame(rows)


def fetch_odds_api(league: str, *, regions: str = "us", markets: str = "h2h", timeout: float = 30.0) -> pd.DataFrame:
    """One ``/odds`` call for ``league`` (spends credits)."""
    key = credential("ODDS_API_KEY")
    if key is None:
        raise DataUnavailable("ODDS_API_KEY is not set; import prices from CSV instead (`sportspred ev template`)")
    if league not in SPORT_KEYS:
        raise DataUnavailable(f"no Odds API sport key is configured for {league!r}")
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEYS[league]}/odds"
    try:
        resp = requests.get(url, params={"apiKey": key, "regions": regions, "markets": markets,
                                         "oddsFormat": "american", "dateFormat": "iso"}, timeout=timeout)
    except requests.RequestException as exc:
        raise DataUnavailable(f"The Odds API is not reachable ({exc.__class__.__name__}); nothing was recorded") from exc
    if resp.status_code != 200:
        raise DataUnavailable(f"The Odds API answered HTTP {resp.status_code}; nothing was recorded")
    return parse_odds_api(resp.json(), league)
