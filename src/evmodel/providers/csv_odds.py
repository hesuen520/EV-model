"""Odds from a plain CSV file - no API needed.

Type (or export) the prices you see into a spreadsheet with one row per book
and prop::

    player,market,line,book,over,under
    Jalen Example,player_points,24.5,pinnacle,-125,+105
    Jalen Example,player_points,24.5,draftkings,-105,-115

``over`` / ``under`` accept American (``-110``, ``+105``), decimal (``1.91``)
or fractional (``10/11``) prices; leave a side blank if it is not offered.
Optional columns: ``sport``, ``event_id``, ``event`` (``"Away @ Home"``),
``home_team``, ``away_team``, ``commence_time``, ``last_update``,
``is_alternate``.  A template is produced by :func:`write_template`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from ..odds import parse_odds
from ..schema import normalize_quotes

__all__ = ["read_odds_csv", "write_template", "quotes_to_csv"]

_ALIASES = {
    "over_odds": "over",
    "over_price": "over",
    "under_odds": "under",
    "under_price": "under",
    "sportsbook": "book",
    "bookmaker": "book",
    "name": "player",
    "player_name": "player",
    "stat": "market",
    "prop": "market",
    "point": "line",
    "game": "event",
}

TEMPLATE = """player,market,line,book,over,under,sport,event
Jalen Example,player_points,24.5,pinnacle,-125,+105,nba,Away Aces @ Home Hawks
Jalen Example,player_points,24.5,fanduel,-122,+100,nba,Away Aces @ Home Hawks
Jalen Example,player_points,24.5,draftkings,-105,-115,nba,Away Aces @ Home Hawks
"""


def _price(value: object, odds_format: str) -> float:
    if value is None:
        return np.nan
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "-", "n/a", "na"}:
        return np.nan
    try:
        return parse_odds(text, odds_format)
    except ValueError:
        return np.nan


def read_odds_csv(
    path: Union[str, Path], sport: Optional[str] = None, odds_format: str = "auto"
) -> pd.DataFrame:
    """Read a CSV of prices into a quotes table.

    ``sport`` fills the sport column when the file has none.  Rows whose
    prices cannot be parsed on either side are dropped.
    """
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]
    raw = raw.rename(columns={c: _ALIASES[c] for c in raw.columns if c in _ALIASES})
    required = {"player", "market", "line", "book"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"odds CSV is missing columns: {sorted(missing)}")
    if "over" not in raw.columns and "under" not in raw.columns:
        raise ValueError("odds CSV needs an 'over' and/or 'under' price column")

    df = pd.DataFrame(
        {
            "player": raw["player"],
            "market": raw["market"].str.strip().str.lower(),
            "line": pd.to_numeric(raw["line"], errors="coerce"),
            "book": raw["book"].str.strip().str.lower(),
        }
    )
    df["over_price"] = raw["over"].map(lambda v: _price(v, odds_format)) if "over" in raw.columns else np.nan
    df["under_price"] = raw["under"].map(lambda v: _price(v, odds_format)) if "under" in raw.columns else np.nan
    df["sport"] = raw["sport"] if "sport" in raw.columns else (sport or "")
    if sport:
        df["sport"] = df["sport"].where(df["sport"].astype(str).str.strip() != "", sport)

    home = raw["home_team"] if "home_team" in raw.columns else pd.Series("", index=raw.index)
    away = raw["away_team"] if "away_team" in raw.columns else pd.Series("", index=raw.index)
    if "event" in raw.columns:
        parts = raw["event"].str.split(r"\s+(?:@|at|vs\.?|v)\s+", n=1, regex=True, expand=True)
        if parts.shape[1] == 2:
            away = away.where(away.str.strip() != "", parts[0].fillna("").str.strip())
            home = home.where(home.str.strip() != "", parts[1].fillna("").str.strip())
    df["home_team"], df["away_team"] = home, away
    if "event_id" in raw.columns:
        df["event_id"] = raw["event_id"]
    else:
        label = raw["event"] if "event" in raw.columns else (away + "@" + home)
        df["event_id"] = label.str.strip().str.lower().where(label.str.strip() != "", "event")
    for col in ("commence_time", "last_update"):
        df[col] = raw[col] if col in raw.columns else pd.NaT
    if "is_alternate" in raw.columns:
        df["is_alternate"] = raw["is_alternate"].str.strip().str.lower().isin({"1", "true", "yes", "y"})
    else:
        df["is_alternate"] = df["market"].str.endswith("_alternate")
    return normalize_quotes(df)


def write_template(path: Union[str, Path]) -> Path:
    """Write an example CSV showing the expected layout."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TEMPLATE, encoding="utf-8")
    return path


def quotes_to_csv(quotes: pd.DataFrame, path: Union[str, Path]) -> Path:
    """Save a quotes table in the layout :func:`read_odds_csv` reads (decimal prices)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = quotes.rename(columns={"over_price": "over", "under_price": "under"})
    out.to_csv(path, index=False)
    return path
