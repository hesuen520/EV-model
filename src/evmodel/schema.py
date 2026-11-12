"""Table contracts shared by every module.

The package passes pandas DataFrames between its stages.  This module is the
single place that says what columns those frames have.

**Quotes** - one row per (book, player, market, line): what a sportsbook is
offering right now.  Produced by the odds providers, consumed by the market
engine.

**Game logs** - one row per (player, game): what a player did.  Produced by
the stats providers, consumed by the projection engine.

**Opportunities** - one row per bettable side with its fair probability and
expected value.  Produced by the scanner.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from .names import player_key

__all__ = [
    "QUOTE_COLUMNS",
    "GAMELOG_COLUMNS",
    "OPPORTUNITY_COLUMNS",
    "normalize_quotes",
    "normalize_gamelogs",
    "empty_quotes",
    "prop_id",
]

#: Columns of a quotes table.
#:
#: sport          short sport code: ``nba``, ``nfl``, ``mlb``, ``nhl`` ...
#: event_id       provider's id for the game
#: commence_time  scheduled start (UTC, tz-aware) or NaT
#: home_team      full team name (may be empty)
#: away_team      full team name (may be empty)
#: book           bookmaker key, lower case (``draftkings``, ``pinnacle`` ...)
#: market         canonical market key (``player_points`` ...), see markets.py
#: player         player name as displayed by the feed
#: player_key     canonical join key from :func:`evmodel.names.player_key`
#: line           the prop line (``24.5``); ``0.5`` for yes/no markets
#: over_price     decimal odds for the over / "yes" (NaN if not offered)
#: under_price    decimal odds for the under / "no" (NaN if not offered)
#: last_update    when the book last changed this market (UTC) or NaT
#: is_alternate   True for alternate-line markets
QUOTE_COLUMNS: List[str] = [
    "sport",
    "event_id",
    "commence_time",
    "home_team",
    "away_team",
    "book",
    "market",
    "player",
    "player_key",
    "line",
    "over_price",
    "under_price",
    "last_update",
    "is_alternate",
]

#: Identity columns of a game-log table; stat columns (``points``,
#: ``rebounds``, ``receiving_yards`` ...) follow.
#:
#: sport, season   season is the starting year (2025 for NBA 2025-26)
#: game_id         provider's game id
#: game_date       date of the game (datetime64, no tz)
#: player_id       provider's player id (string)
#: player          display name
#: player_key      canonical join key
#: team, opponent  team abbreviations
#: is_home         bool
#: minutes         minutes played (NaN where the sport has no such notion)
GAMELOG_COLUMNS: List[str] = [
    "sport",
    "season",
    "game_id",
    "game_date",
    "player_id",
    "player",
    "player_key",
    "team",
    "opponent",
    "is_home",
    "minutes",
]

#: Columns of the scanner output, in display order.
OPPORTUNITY_COLUMNS: List[str] = [
    "sport",
    "event_id",
    "commence_time",
    "event",            # "Away @ Home"
    "player",
    "player_key",
    "market",
    "line",
    "side",             # "over" | "under"
    "book",
    "price",            # decimal odds offered
    "price_american",
    "fair_prob",        # P(win) used for EV (after blending)
    "push_prob",
    "fair_price_american",  # fair American odds of the side, pushes excluded
    "ev",               # expected profit per unit staked
    "ev_worst",         # EV under the de-vig method least favourable to the bet
    "edge",             # fair_prob - break-even probability
    "kelly",            # full-Kelly fraction of bankroll
    "stake_units",      # recommended stake in units
    "market_prob",      # P(win | no push) from the sharp-book consensus (NaN if none)
    "model_prob",       # P(win | no push) from the projection model (NaN if none)
    "source",           # "market" | "model" | "blend"
    "n_refs",           # reference books in the consensus
    "ref_books",        # comma separated, e.g. "pinnacle:24.5,fanduel:24.5"
    "ref_weight",       # total weight of the reference books
    "ref_hold",         # weighted average hold of the reference quotes
    "ref_disagreement", # weighted sd of reference probabilities (probability points)
    "line_gap",         # largest |ref line - offered line| among references
    "projection",       # model mean for the stat (NaN if none)
    "flags",            # comma separated cautions
    "last_update",
]


def prop_id(event_id: object, player_key_: object, market: object) -> str:
    """Stable identifier of one prop (all lines, all books)."""
    return f"{event_id}|{player_key_}|{market}"


def empty_quotes() -> pd.DataFrame:
    """An empty, correctly typed quotes table."""
    return normalize_quotes(pd.DataFrame(columns=QUOTE_COLUMNS))


_FALSE_TEXT = {"", "0", "false", "f", "no", "n", "nan", "none", "null", "<na>"}


def _text(series: pd.Series) -> pd.Series:
    """Strings with missing values as ``""`` and surrounding blanks removed."""
    return series.where(series.notna(), "").astype(str).str.strip()


def _id_text(series: pd.Series) -> pd.Series:
    """Identifiers as text; ``12345.0`` (an int column with gaps) becomes ``"12345"``."""
    def one(value: object) -> str:
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            return ""
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            return str(int(value))
        return str(value).strip()

    return series.map(one).astype(str)


def _bool(series: pd.Series) -> pd.Series:
    """Booleans from real booleans, numbers or text flags (``"false"`` is False)."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    def one(value: object) -> bool:
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            return False
        if isinstance(value, str):
            return value.strip().lower() not in _FALSE_TEXT
        return bool(value)

    return series.map(one).astype(bool)


def _epoch_to_utc(value: float) -> pd.Timestamp:
    """A plausible Unix epoch (seconds or milliseconds) as a UTC timestamp."""
    if not np.isfinite(value):
        return pd.NaT
    if 1e8 <= abs(value) < 1e11:  # seconds: 1973 .. 5138
        return pd.Timestamp(value, unit="s", tz="UTC")
    if 1e11 <= abs(value) < 1e14:  # milliseconds
        return pd.Timestamp(value, unit="ms", tz="UTC")
    return pd.NaT


def _to_utc(series: pd.Series) -> pd.Series:
    """Timestamps as tz-aware UTC; unparseable values become NaT.

    Accepts datetimes, any mix of ISO-like strings, and Unix epochs in
    seconds or milliseconds (bare numbers are never read as nanoseconds).
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, utc=True, errors="coerce")
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        stamps = [_epoch_to_utc(float(v)) for v in pd.to_numeric(series, errors="coerce")]
        return pd.to_datetime(pd.Series(stamps, index=series.index), utc=True, errors="coerce")

    def one(value: object) -> object:
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            return _epoch_to_utc(float(value))
        return value

    return pd.to_datetime(series.map(one), utc=True, errors="coerce", format="mixed")


def _looks_numeric(series: pd.Series) -> bool:
    """True for numeric columns and for text columns whose every value is a number."""
    if pd.api.types.is_bool_dtype(series):
        return False
    if pd.api.types.is_numeric_dtype(series):
        return True
    if pd.api.types.is_datetime64_any_dtype(series):
        return False
    present = series[series.notna()]
    present = present[present.astype(str).str.strip() != ""]
    if present.empty:
        return False
    return bool(pd.to_numeric(present, errors="coerce").notna().all())


def _to_local_dates(series: pd.Series) -> pd.Series:
    """Game dates as naive timestamps, keeping each value's own wall-clock time.

    Values may mix formats and UTC offsets (a season spans a daylight-saving
    change); an offset is dropped rather than converted so an evening game
    keeps its local calendar date.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        if getattr(series.dt, "tz", None) is not None:
            return series.dt.tz_localize(None)
        return series

    def one(value: object) -> pd.Timestamp:
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            return pd.NaT
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            # A bare number is a YYYYMMDD date or nothing (never nanoseconds).
            number = float(value)
            if not (number.is_integer() and 19000101 <= number <= 22001231):
                return pd.NaT
            value = str(int(number))
        try:
            stamp = pd.Timestamp(value)
        except (ValueError, TypeError):
            return pd.NaT
        if stamp is pd.NaT:
            return pd.NaT
        return stamp.tz_localize(None) if stamp.tzinfo is not None else stamp

    return pd.to_datetime(series.map(one), errors="coerce")


def normalize_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a quotes table to the canonical columns and dtypes.

    Missing optional columns are filled with defaults; rows that cannot be a
    valid quote (no player, no line, no price on either side, or a price that
    is not a finite number > 1) are dropped.  Returns a new frame with a
    fresh index.
    """
    df = df.copy()
    required = ["book", "market", "player", "line"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"quotes table is missing required columns: {missing}")

    defaults = {
        "sport": "",
        "event_id": "",
        "commence_time": pd.NaT,
        "home_team": "",
        "away_team": "",
        "over_price": np.nan,
        "under_price": np.nan,
        "last_update": pd.NaT,
        "is_alternate": False,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    for col in ("sport", "home_team", "away_team", "book", "market", "player"):
        df[col] = _text(df[col])
    df["event_id"] = _id_text(df["event_id"])
    df["sport"] = df["sport"].str.lower()
    df["book"] = df["book"].str.lower()
    df["market"] = df["market"].str.lower()
    derived = df["player"].map(player_key)
    if "player_key" not in df.columns:
        df["player_key"] = derived
    else:
        keys = _text(df["player_key"])
        df["player_key"] = keys.where(keys != "", derived)

    for col in ("line", "over_price", "under_price"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    df.loc[~np.isfinite(df["line"]), "line"] = np.nan
    for col in ("over_price", "under_price"):
        df.loc[~((df[col] > 1.0) & np.isfinite(df[col])), col] = np.nan
    df["commence_time"] = _to_utc(df["commence_time"])
    df["last_update"] = _to_utc(df["last_update"])
    df["is_alternate"] = _bool(df["is_alternate"])

    keep = (
        (df["player_key"] != "")
        & (df["book"] != "")
        & (df["market"] != "")
        & df["line"].notna()
        & (df["over_price"].notna() | df["under_price"].notna())
    )
    df = df.loc[keep, QUOTE_COLUMNS + [c for c in df.columns if c not in QUOTE_COLUMNS]]
    return df.reset_index(drop=True)


def normalize_gamelogs(df: pd.DataFrame, stat_columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Coerce a game-log table to the canonical identity columns and dtypes.

    ``stat_columns`` (default: every non-identity numeric-looking column) are
    converted to float.  Rows without a date or a player are dropped; the
    result is sorted by player and date.
    """
    df = df.copy()
    required = ["sport", "game_date", "player"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"game-log table is missing required columns: {missing}")

    defaults = {
        "season": np.nan,
        "game_id": "",
        "player_id": "",
        "team": "",
        "opponent": "",
        "is_home": False,
        "minutes": np.nan,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    df["sport"] = _text(df["sport"]).str.lower()
    df["game_date"] = _to_local_dates(df["game_date"])
    for col in ("player", "team", "opponent"):
        df[col] = _text(df[col])
    for col in ("game_id", "player_id"):
        df[col] = _id_text(df[col])
    derived = df["player"].map(player_key)
    if "player_key" not in df.columns:
        df["player_key"] = derived
    else:
        keys = _text(df["player_key"])
        df["player_key"] = keys.where(keys != "", derived)
    df["is_home"] = _bool(df["is_home"])
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").astype(float)
    df["season"] = pd.to_numeric(df["season"], errors="coerce")

    if stat_columns is not None:
        stats = [c for c in stat_columns if c in df.columns]
    else:
        stats = [c for c in df.columns if c not in GAMELOG_COLUMNS and _looks_numeric(df[c])]
    for col in stats:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    df = df[df["game_date"].notna() & (df["player_key"] != "")]
    ordered = GAMELOG_COLUMNS + [c for c in df.columns if c not in GAMELOG_COLUMNS]
    df = df[ordered].sort_values(["player_key", "game_date", "game_id"], kind="stable")
    return df.reset_index(drop=True)
