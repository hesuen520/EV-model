"""Bet tracking: a persistent ledger, grading, closing-line value and honest statistics.

Finding a +EV bet is a claim.  This module is how the claim gets checked:

* :func:`grade` / :func:`settle_pnl` turn a final stat value into a result and
  a profit.
* :func:`clv_price`, :func:`clv_ev` and :func:`clv_with_line_move` measure
  closing-line value (CLV): whether the market moved towards our bet after it
  was placed.  CLV converges far faster than profit, so it is the earliest
  evidence that an edge is real.
* :class:`BetLedger` stores placed bets in a plain CSV file.
* :func:`summarize`, :func:`breakdown` and :func:`calibration_table` report
  results next to what the model expected, with the sampling error spelled
  out, so that luck is not mistaken for skill (or the reverse).

Typical use::

    ledger = BetLedger("data/bets.csv")
    bet_id = ledger.add(opportunities.iloc[0], stake=25)        # when the bet is placed
    ledger.record_close(bet_id, closing_over_price=1.83,        # just before the game
                        closing_under_price=2.02)
    ledger.settle(bet_id, actual=31)                            # when the game is final
    summarize(ledger.to_frame())

Conventions
-----------
Prices are **decimal** odds everywhere (stake included, always > 1); American
odds appear only as a clearly named convenience (``price_american`` when
adding a bet, ``avg_price_american`` in a summary).  ``stake`` and ``pnl``
are in one unit of the user's choosing (dollars or betting units) - the
ledger never converts, so use the same unit for every bet.  ``fair_prob`` and
``closing_fair_prob`` are the unconditional probability that the bet *wins*;
the probability of a push (whole-number lines) is carried separately in
``push_prob``, exactly as in the scanner's opportunities table.

Ledger columns
--------------
One row per placed bet, in this order (:data:`LEDGER_COLUMNS`):

======================= =========================================================
``bet_id``              short unique id assigned by :meth:`BetLedger.add`
``placed_at``           when the bet was recorded (UTC, ISO 8601)
``sport``               short sport code (``nba`` ...)
``event_id``            provider's id for the game (may be blank)
``event``               ``"Away @ Home"``
``commence_time``       scheduled start (UTC, ISO 8601) or blank
``player``              player name as displayed
``player_key``          canonical join key (:func:`evmodel.names.player_key`)
``market``              canonical market key (``player_points`` ...)
``line``                the prop line that was bet
``side``                ``over`` | ``under``
``book``                bookmaker key, lower case
``price``               decimal odds obtained
``stake``               amount risked
``fair_prob``           P(win) estimated at bet time (blank if unknown)
``push_prob``           P(push) estimated at bet time (0 when not given)
``ev``                  expected profit per unit staked at bet time
``source``              where the probability came from (``market`` | ``model`` ...)
``flags``               comma separated cautions carried over from the scanner
``closing_line``        the line at close
``closing_price``       closing decimal odds of our side
``closing_other_price`` closing decimal odds of the other side
``closing_fair_prob``   de-vigged closing P(win) of our bet, at the line we bet
``clv_price``           ``price / closing_price - 1`` (same line only)
``clv_ev``              EV per unit of our bet at the closing fair probability
``result``              blank while open, else ``win`` | ``loss`` | ``push`` | ``void``
``actual``              final stat value (blank if settled by explicit result)
``pnl``                 profit in stake units
``settled_at``          when the bet was settled (UTC, ISO 8601)
``notes``               free text
======================= =========================================================

The file is ordinary CSV and may be inspected or edited by hand: blank cells
mean "unknown", extra columns are preserved, and a settled row whose ``pnl``
cell is blank gets it recomputed on the next read.  If you change a
``result`` by hand, clear its ``pnl`` cell too (or re-settle with
``overwrite=True``).
"""

from __future__ import annotations

import csv
import math
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from .devig import devig_two_way
from .distributions import FamilySpec
from .ev import bet_variance, expected_value
from .names import player_key as canonical_player_key
from .odds import american_to_decimal, decimal_to_american

__all__ = [
    "LEDGER_COLUMNS",
    "SIDES",
    "RESULTS",
    "grade",
    "settle_pnl",
    "clv_price",
    "closing_fair_prob",
    "clv_ev",
    "clv_with_line_move",
    "DuplicateBetError",
    "BetLedger",
    "bets_needed",
    "summarize",
    "breakdown",
    "calibration_table",
]

#: Columns of the ledger file and of :meth:`BetLedger.to_frame`, in order.
LEDGER_COLUMNS: List[str] = [
    "bet_id",
    "placed_at",
    "sport",
    "event_id",
    "event",
    "commence_time",
    "player",
    "player_key",
    "market",
    "line",
    "side",
    "book",
    "price",
    "stake",
    "fair_prob",
    "push_prob",
    "ev",
    "source",
    "flags",
    "closing_line",
    "closing_price",
    "closing_other_price",
    "closing_fair_prob",
    "clv_price",
    "clv_ev",
    "result",
    "actual",
    "pnl",
    "settled_at",
    "notes",
]

#: The two sides of a prop.
SIDES: Tuple[str, ...] = ("over", "under")

#: Possible results of a settled bet.  A ``void`` bet (player did not play)
#: returns the stake and is left out of every performance statistic.
RESULTS: Tuple[str, ...] = ("win", "loss", "push", "void")

_NUMERIC_COLUMNS = frozenset(
    {
        "line",
        "price",
        "stake",
        "fair_prob",
        "push_prob",
        "ev",
        "closing_line",
        "closing_price",
        "closing_other_price",
        "closing_fair_prob",
        "clv_price",
        "clv_ev",
        "actual",
        "pnl",
    }
)

# Fields a caller may set when adding a bet (``stake`` is a named argument).
_ADD_FIELDS = frozenset(
    {
        "placed_at",
        "sport",
        "event_id",
        "event",
        "commence_time",
        "player",
        "player_key",
        "market",
        "line",
        "side",
        "book",
        "price",
        "price_american",
        "fair_prob",
        "push_prob",
        "ev",
        "source",
        "flags",
        "notes",
    }
)

_LINE_TOL = 1e-9
_PROB_TOL = 1e-9
_FIT_TOL = 1e-6  # largest acceptable error of a fitted family at the closing price
_BOOT_CELLS = 2_000_000  # cap on resampled indices held in memory at once

Bet = Dict[str, Any]
ResultsLookup = Callable[[pd.Series], Union[None, float, str]]


# ---------------------------------------------------------------------------
# Input cleaning
# ---------------------------------------------------------------------------


def _is_missing(value: object) -> bool:
    """True for ``None``, NaN, NaT, ``pd.NA`` and blank strings."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _text(value: object) -> str:
    return "" if _is_missing(value) else str(value).strip()


def _number(value: object, name: str) -> float:
    """``float(value)``; NaN when missing, ``ValueError`` when not a finite number."""
    if _is_missing(value):
        return math.nan
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc
    if math.isinf(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _check_side(side: object) -> str:
    text = _text(side).lower()
    if text not in SIDES:
        raise ValueError(f"side must be 'over' or 'under', got {side!r}")
    return text


def _check_result(result: object) -> str:
    text = _text(result).lower()
    if text not in RESULTS:
        raise ValueError(f"result must be one of {RESULTS}, got {result!r}")
    return text


def _check_line(value: object, name: str = "line") -> float:
    line = _number(value, name)
    if math.isnan(line):
        raise ValueError(f"{name} is required")
    return line


def _check_price(value: object, name: str = "price") -> float:
    price = _number(value, name)
    if not price > 1.0:
        raise ValueError(
            f"{name} must be decimal odds > 1 (stake included), got {value!r}; "
            "convert American odds with evmodel.odds.american_to_decimal"
        )
    return price


def _optional_price(value: object, name: str) -> float:
    return math.nan if _is_missing(value) else _check_price(value, name)


def _check_stake(value: object) -> float:
    stake = _number(value, "stake")
    if not stake > 0.0:
        raise ValueError(f"stake must be > 0, got {value!r}")
    return stake


def _same_line(a: float, b: float) -> bool:
    return abs(a - b) < _LINE_TOL


def _utc_iso(value: object, name: str) -> str:
    """Normalise a timestamp to a UTC ISO-8601 string (``""`` when missing).

    Timestamps without a time zone are taken to be UTC already.
    """
    if _is_missing(value):
        return ""
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not a timestamp: {value!r}") from exc
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return str(stamp.isoformat(timespec="seconds"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Grading and closing-line value
# ---------------------------------------------------------------------------


def grade(side: str, line: float, actual: Optional[float]) -> str:
    """Result of an over/under bet given the final stat value.

    ``"win"``, ``"loss"`` or ``"push"`` (the stat landed exactly on a
    whole-number line).  A missing ``actual`` (``None`` / NaN: the player
    recorded no stat because he did not play) grades as ``"void"``.
    """
    side = _check_side(side)
    line = _check_line(line)
    value = _number(actual, "actual")
    if math.isnan(value):
        return "void"
    if _same_line(value, line):
        return "push"
    if (value > line) == (side == "over"):
        return "win"
    return "loss"


def settle_pnl(result: str, stake: float, decimal_price: float) -> float:
    """Profit of a settled bet in stake units.

    A win pays ``stake * (decimal_price - 1)``, a loss costs the stake, and a
    push or void returns it (profit 0).
    """
    result = _check_result(result)
    stake = _check_stake(stake)
    price = _check_price(decimal_price, "decimal_price")
    if result == "win":
        return stake * (price - 1.0)
    if result == "loss":
        return -stake
    return 0.0


def clv_price(bet_decimal: float, closing_decimal: float) -> float:
    """Price CLV: ``bet_decimal / closing_decimal - 1``.

    Positive when we were paid more than the closing price *of the same side
    at the same line*.  It still contains the closing book's margin, so a
    small negative value does not mean the bet was -EV; :func:`clv_ev` is the
    margin-free number.
    """
    return _check_price(bet_decimal, "bet_decimal") / _check_price(closing_decimal, "closing_decimal") - 1.0


def closing_fair_prob(
    closing_side_decimal: float, closing_other_decimal: float, method: str = "multiplicative"
) -> float:
    """De-vigged closing probability of our side from both closing prices.

    Like every de-vigged two-way price this is conditional on the bet not
    pushing; on a half-point line that is simply the win probability.
    """
    ours = _check_price(closing_side_decimal, "closing_side_decimal")
    other = _check_price(closing_other_decimal, "closing_other_decimal")
    return devig_two_way(ours, other, method)[0]


def clv_ev(bet_decimal: float, closing_fair_prob: float, push_prob: float = 0.0) -> float:
    """Expected value per unit of our bet, evaluated at the closing line.

    ``closing_fair_prob`` is the probability that the bet wins according to
    the closing market and ``push_prob`` the probability that it pushes
    (:func:`evmodel.ev.expected_value`).  On a whole-number line multiply the
    de-vigged closing probability by ``1 - push_prob`` first, because a
    two-way price says nothing about pushes.

    This is the most meaningful CLV number: the closing line is the sharpest
    estimate available, so the average ``clv_ev`` of many bets is close to
    their true edge.
    """
    return expected_value(closing_fair_prob, _check_price(bet_decimal, "bet_decimal"), push_prob)


def clv_with_line_move(
    spec: FamilySpec,
    side: str,
    bet_line: float,
    bet_decimal: float,
    closing_line: float,
    closing_over_decimal: float,
    closing_under_decimal: float,
    method: str = "multiplicative",
) -> Dict[str, float]:
    """CLV of a bet whose line (not just its price) moved before the close.

    Prop lines move (24.5 -> 25.5), so the closing price often belongs to a
    different line than the one we bet.  The closing over/under prices are
    de-vigged, the member of ``spec`` that reproduces them at ``closing_line``
    is found (:meth:`FamilySpec.solve_mean`) and that distribution is
    evaluated at ``bet_line``.

    Returns a dict with

    ``closing_fair_prob``
        probability that our side wins at ``bet_line`` according to the close;
    ``push_prob``
        probability of a push at ``bet_line`` (zero for half-point lines);
    ``clv_ev``
        EV per unit of our bet at those probabilities;
    ``implied_mean``
        the stat mean implied by the closing market.

    When the two lines are equal this reduces to :func:`clv_ev` at the
    de-vigged closing probability (with the family's push mass on a
    whole-number line).  Raises ``ValueError`` if no member of the family
    reproduces the closing price, since any translation would then be
    fiction.
    """
    side = _check_side(side)
    bet_line = _check_line(bet_line, "bet_line")
    closing_line = _check_line(closing_line, "closing_line")
    price = _check_price(bet_decimal, "bet_decimal")
    over_price = _check_price(closing_over_decimal, "closing_over_decimal")
    under_price = _check_price(closing_under_decimal, "closing_under_decimal")

    p_over, _ = devig_two_way(over_price, under_price, method)
    mean = spec.solve_mean(closing_line, p_over)
    dist = spec.dist(mean)
    if abs(dist.prob_over_no_push(closing_line) - p_over) > _FIT_TOL:
        raise ValueError(
            f"the {spec.family} family cannot reproduce a fair over probability of {p_over:.4f} "
            f"at line {closing_line:g}, so the close cannot be translated to line {bet_line:g}"
        )
    over, push, under = dist.probs(bet_line)
    p_win = over if side == "over" else under
    return {
        "closing_fair_prob": p_win,
        "push_prob": push,
        "clv_ev": expected_value(p_win, price, push),
        "implied_mean": mean,
    }


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


class DuplicateBetError(ValueError):
    """Raised when an identical bet is already in the ledger."""


def _blank_bet() -> Bet:
    return {column: (math.nan if column in _NUMERIC_COLUMNS else "") for column in LEDGER_COLUMNS}


def _row_to_dict(row: object) -> Dict[str, object]:
    if row is None:
        return {}
    if isinstance(row, (pd.Series, Mapping)):
        return {str(key): value for key, value in row.items()}
    raise TypeError(f"row must be a mapping or a pandas Series, got {type(row).__name__}")


def _resolve_price(price: object, price_american: object) -> float:
    """Decimal price of a new bet; American odds are used only if no decimal is given."""
    if _is_missing(price):
        if _is_missing(price_american):
            raise ValueError("price is required (decimal odds), or pass price_american")
        return american_to_decimal(_number(price_american, "price_american"))
    return _check_price(price)


def _new_bet(values: Mapping[str, object]) -> Bet:
    """Validate the fields of a new bet and lay them out as a ledger row (no id yet)."""
    player = _text(values.get("player"))
    key = _text(values.get("player_key")) or canonical_player_key(player)
    market = _text(values.get("market")).lower()
    if not key:
        raise ValueError("a bet needs a player (or player_key)")
    if not market:
        raise ValueError("a bet needs a market")
    if _is_missing(values.get("stake")):
        raise ValueError(
            "stake is required; pass stake=... (for example the row's stake_units "
            "to follow the scanner's recommendation)"
        )

    price = _resolve_price(values.get("price"), values.get("price_american"))
    fair_prob = _number(values.get("fair_prob"), "fair_prob")
    push_prob = _number(values.get("push_prob"), "push_prob")
    if math.isnan(push_prob):
        push_prob = 0.0
    if not 0.0 <= push_prob < 1.0:
        raise ValueError(f"push_prob must be in [0, 1), got {push_prob!r}")
    if math.isnan(fair_prob):
        ev = _number(values.get("ev"), "ev")
        if not math.isnan(ev) and not -1.0 <= ev <= price - 1.0:
            raise ValueError(f"ev per unit must be in [-1, price - 1], got {ev!r}")
    else:
        if not 0.0 <= fair_prob <= 1.0 - push_prob + _PROB_TOL:
            raise ValueError(
                f"fair_prob must be in [0, 1 - push_prob], got {fair_prob!r} with push_prob {push_prob!r}"
            )
        fair_prob = min(fair_prob, 1.0 - push_prob)
        # Recomputed rather than copied so that it matches the price actually obtained.
        ev = expected_value(fair_prob, price, push_prob)

    bet = _blank_bet()
    bet.update(
        placed_at=_utc_iso(values.get("placed_at"), "placed_at") or _now(),
        sport=_text(values.get("sport")).lower(),
        event_id=_text(values.get("event_id")),
        event=_text(values.get("event")),
        commence_time=_utc_iso(values.get("commence_time"), "commence_time"),
        player=player,
        player_key=key,
        market=market,
        line=_check_line(values.get("line")),
        side=_check_side(values.get("side")),
        book=_text(values.get("book")).lower(),
        price=price,
        stake=_check_stake(values.get("stake")),
        fair_prob=fair_prob,
        push_prob=push_prob,
        ev=ev,
        source=_text(values.get("source")),
        flags=_text(values.get("flags")),
        notes=_text(values.get("notes")),
    )
    return bet


def _is_duplicate(new: Bet, existing: Bet) -> bool:
    if not new["event_id"] and existing["result"]:
        # Without an event id a settled bet can only belong to an earlier game.
        return False
    return (
        new["event_id"] == existing["event_id"]
        and new["player_key"] == existing["player_key"]
        and new["market"] == existing["market"]
        and _same_line(new["line"], existing["line"])
        and new["side"] == existing["side"]
        and new["book"] == existing["book"]
    )


def _settle(bet: Bet, result: Optional[str], actual: Optional[float], overwrite: bool) -> None:
    """Settle one ledger row in place."""
    if bet["result"] and not overwrite:
        raise ValueError(
            f"bet {bet['bet_id']!r} is already settled ({bet['result']}); pass overwrite=True to re-settle"
        )
    if result is None:
        if actual is None:
            raise ValueError("settle needs a result or the actual stat value")
        outcome = grade(bet["side"], bet["line"], actual)
    else:
        outcome = _check_result(result)
    bet["result"] = outcome
    bet["actual"] = _number(actual, "actual")
    bet["pnl"] = settle_pnl(outcome, bet["stake"], bet["price"])
    bet["settled_at"] = _now()


def _results_lookup(results: pd.DataFrame) -> ResultsLookup:
    """Turn a results table into a function giving the final stat value of a bet."""
    if not isinstance(results, pd.DataFrame):
        raise TypeError("results must be a pandas DataFrame or a callable")
    missing = [c for c in ("market", "actual") if c not in results.columns]
    if "player_key" not in results.columns and "player" not in results.columns:
        missing.append("player_key")
    if missing:
        raise ValueError(f"results table is missing required columns: {missing}")

    has_key = "player_key" in results.columns
    has_event = "event_id" in results.columns
    table: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}
    for record in results.to_dict("records"):
        actual = _number(record["actual"], "actual")
        key = _text(record["player_key"]) if has_key else ""
        key = key or canonical_player_key(record.get("player"))
        if math.isnan(actual) or not key:
            continue
        event_id = _text(record["event_id"]) if has_event else ""
        table.setdefault((key, _text(record["market"]).lower()), []).append((event_id, actual))

    def lookup(bet: pd.Series) -> Optional[float]:
        candidates = {
            actual
            for event_id, actual in table.get((bet["player_key"], bet["market"]), [])
            if not event_id or not bet["event_id"] or event_id == bet["event_id"]
        }
        if len(candidates) > 1:
            raise ValueError(
                f"bet {bet['bet_id']!r} ({bet['player_key']} {bet['market']}) matches {len(candidates)} "
                "different results; give both the bet and the results an event_id, or pass one game only"
            )
        return candidates.pop() if candidates else None

    return lookup


class BetLedger:
    """A persistent ledger of placed bets, stored as a CSV file.

    The file is the single source of truth: every method reads it afresh and
    every change rewrites it atomically (temporary file in the same directory
    followed by ``os.replace``), so an interrupted write can never leave a
    half-written ledger behind.  There is no locking; use one writer at a
    time.

    Parameters
    ----------
    path:
        Location of the CSV file.  Missing parent directories are created;
        the file itself appears with the first bet.
    """

    def __init__(self, path: Union[str, "os.PathLike[str]"]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"BetLedger({str(self.path)!r})"

    # -- storage -------------------------------------------------------------
    def _parse_row(self, raw: Mapping[Optional[str], Optional[str]], line_number: int) -> Bet:
        """One line of the file as a typed, validated ledger row."""
        where = f"{self.path} line {line_number}"
        if None in raw:
            raise ValueError(f"{where}: more fields than the header has columns")
        bet = _blank_bet()
        try:
            for name, cell in raw.items():
                column = str(name)
                if column in _NUMERIC_COLUMNS:
                    bet[column] = _number(cell, column)
                else:
                    bet[column] = "" if cell is None else str(cell)
            bet["bet_id"] = bet["bet_id"].strip()
            if not bet["bet_id"]:
                raise ValueError("bet_id is blank")
            bet["side"] = _check_side(bet["side"])
            bet["result"] = _check_result(bet["result"]) if bet["result"].strip() else ""
            bet["line"] = _check_line(bet["line"])
            bet["price"] = _check_price(bet["price"])
            bet["stake"] = _check_stake(bet["stake"])
            if math.isnan(bet["push_prob"]):
                bet["push_prob"] = 0.0
            if bet["result"] and math.isnan(bet["pnl"]):
                bet["pnl"] = settle_pnl(bet["result"], bet["stake"], bet["price"])
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from exc
        return bet

    def _load(self) -> Tuple[List[Bet], List[str]]:
        """Read the file: ``(rows, names of extra columns to preserve)``."""
        if not self.path.exists():
            return [], []
        with self.path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            extras = [c for c in (reader.fieldnames or []) if c not in LEDGER_COLUMNS]
            rows = [self._parse_row(raw, reader.line_num) for raw in reader]
        seen: Set[str] = set()
        for bet in rows:
            if bet["bet_id"] in seen:
                raise ValueError(f"{self.path}: bet_id {bet['bet_id']!r} appears more than once")
            seen.add(bet["bet_id"])
        return rows, extras

    def _save(self, rows: List[Bet], extras: List[str]) -> None:
        """Atomically replace the file with ``rows``."""
        columns = LEDGER_COLUMNS + extras
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("x", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(columns)
                for bet in rows:
                    writer.writerow([_cell(bet.get(column, "")) for column in columns])
                handle.flush()
                os.fsync(handle.fileno())
            if self.path.exists():
                shutil.copymode(self.path, temporary)
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)  # only still there if the write failed

    def _find(self, rows: List[Bet], bet_id: str) -> Bet:
        for bet in rows:
            if bet["bet_id"] == bet_id:
                return bet
        raise KeyError(f"no bet with id {bet_id!r} in {self.path}")

    # -- recording -----------------------------------------------------------
    def add(
        self,
        row: Union[Mapping[str, object], pd.Series, None] = None,
        stake: Optional[float] = None,
        *,
        allow_duplicate: bool = False,
        **fields: object,
    ) -> str:
        """Record a placed bet and return its ``bet_id``.

        Either pass the bet as keyword fields::

            ledger.add(player="Nikola Jokic", market="player_points", line=26.5,
                       side="over", book="draftkings", price=1.91, stake=25)

        or pass a row of the scanner's opportunities table (a mapping or a
        pandas Series; columns the ledger does not store are ignored) plus the
        stake, overriding any field that differed when the bet was struck::

            ledger.add(opportunities.iloc[0], stake=25, price=1.87)

        A keyword field left as ``None`` counts as not given and never blanks
        the row's value.

        Required: ``player`` (or ``player_key``), ``market``, ``line``,
        ``side`` and ``stake``, and the price as decimal odds in ``price``
        (``price_american`` is converted when no decimal price is given).
        Optional: ``sport``, ``event_id``, ``event``, ``commence_time``,
        ``book``, ``fair_prob``, ``push_prob``, ``ev``, ``source``, ``flags``,
        ``notes`` and ``placed_at`` (defaults to now).  Timestamps without a
        time zone are read as UTC.

        ``ev`` is stored per unit staked.  When ``fair_prob`` is known it is
        recomputed at the recorded price, so a bet struck at a different price
        than the scanner showed still carries the right expectation.

        Raises :class:`DuplicateBetError` if a bet with the same event,
        player, market, line, side and book is already recorded, unless
        ``allow_duplicate`` is true.  When the new bet has no ``event_id``
        only open bets count, since the event cannot be told apart from an
        earlier game.
        """
        unknown = sorted(set(fields) - _ADD_FIELDS)
        if unknown:
            raise TypeError(f"add() got unexpected field(s) {unknown}; allowed: {sorted(_ADD_FIELDS)}")
        values = _row_to_dict(row)
        given = {name: value for name, value in fields.items() if value is not None}
        if "price" in given and "price_american" not in given:
            values.pop("price_american", None)  # the row's American price describes the old price
        values.update(given)
        if stake is not None:
            values["stake"] = stake
        bet = _new_bet(values)

        rows, extras = self._load()
        if not allow_duplicate:
            for existing in rows:
                if _is_duplicate(bet, existing):
                    raise DuplicateBetError(
                        f"bet {existing['bet_id']!r} is already recorded for {bet['player_key']} "
                        f"{bet['market']} {bet['side']} {bet['line']:g} at {bet['book'] or 'this book'}; "
                        "pass allow_duplicate=True to add to the position"
                    )
        taken = {existing["bet_id"] for existing in rows}
        bet_id = _new_id()
        while bet_id in taken:
            bet_id = _new_id()
        bet["bet_id"] = bet_id
        rows.append(bet)
        self._save(rows, extras)
        return bet_id

    def record_close(
        self,
        bet_id: str,
        closing_over_price: Optional[float] = None,
        closing_under_price: Optional[float] = None,
        closing_line: Optional[float] = None,
        spec: Optional[FamilySpec] = None,
        method: str = "multiplicative",
    ) -> Dict[str, float]:
        """Store the closing market of a bet and compute its closing-line value.

        ``closing_over_price`` / ``closing_under_price`` are the last decimal
        prices before the game started (ideally from a sharp book) and
        ``closing_line`` the line they belong to (default: the line we bet).
        Calling again replaces the previous closing numbers.

        * ``clv_price`` needs our side's closing price at the line we bet.
        * ``clv_ev`` needs both closing prices.  At the line we bet it is our
          price's EV at the de-vigged closing probability, using the push
          probability recorded with the bet.  If the line moved it is computed
          by :func:`clv_with_line_move`, which needs ``spec``, the
          distribution family of the stat; without one it is left blank.

        Whatever cannot be computed from what was given is stored as NaN.
        Returns ``{"closing_fair_prob", "clv_price", "clv_ev"}`` as stored.
        """
        over = _optional_price(closing_over_price, "closing_over_price")
        under = _optional_price(closing_under_price, "closing_under_price")
        if math.isnan(over) and math.isnan(under):
            raise ValueError("record_close needs at least one closing price")

        rows, extras = self._load()
        bet = self._find(rows, bet_id)
        line = bet["line"] if _is_missing(closing_line) else _check_line(closing_line, "closing_line")
        ours, other = (over, under) if bet["side"] == "over" else (under, over)
        moved = not _same_line(line, bet["line"])

        fair = value = price_clv = math.nan
        if not moved and not math.isnan(ours):
            price_clv = clv_price(bet["price"], ours)
        if not math.isnan(over) and not math.isnan(under):
            if not moved:
                push = bet["push_prob"]
                fair = closing_fair_prob(ours, other, method) * (1.0 - push)
                value = clv_ev(bet["price"], fair, push)
            elif spec is not None:
                translated = clv_with_line_move(
                    spec, bet["side"], bet["line"], bet["price"], line, over, under, method
                )
                fair, value = translated["closing_fair_prob"], translated["clv_ev"]

        bet.update(
            closing_line=line,
            closing_price=ours,
            closing_other_price=other,
            closing_fair_prob=fair,
            clv_price=price_clv,
            clv_ev=value,
        )
        self._save(rows, extras)
        return {"closing_fair_prob": fair, "clv_price": price_clv, "clv_ev": value}

    # -- settlement ----------------------------------------------------------
    def settle(
        self,
        bet_id: str,
        result: Optional[str] = None,
        actual: Optional[float] = None,
        overwrite: bool = False,
    ) -> float:
        """Settle a bet and return its profit.

        Give the final stat value in ``actual`` and the bet is graded with
        :func:`grade` (a NaN ``actual`` voids it), or give ``result``
        (``win`` / ``loss`` / ``push`` / ``void``) directly.  If both are
        given the explicit result stands - books occasionally grade against a
        stat that is corrected later - and ``actual`` is kept for reference.

        A bet that is already settled is only re-settled with
        ``overwrite=True``.
        """
        rows, extras = self._load()
        bet = self._find(rows, bet_id)
        _settle(bet, result, actual, overwrite)
        self._save(rows, extras)
        return float(bet["pnl"])

    def settle_from_results(self, results: Union[pd.DataFrame, ResultsLookup]) -> int:
        """Grade every open bet for which a final stat is available.

        ``results`` is either

        * a DataFrame with columns ``player_key`` (or ``player``), ``market``
          and ``actual``, and optionally ``event_id``.  A bet matches a row
          with the same player and market; event ids must agree where both
          the bet and the row have one.  Without event ids the match is by
          player and market alone, so pass only the results of the games the
          open bets were placed on.  If a bet matches rows with different
          values a ``ValueError`` is raised and nothing is settled; or
        * a callable receiving each open bet as a pandas Series and returning
          the final stat value, a result string (for example ``"void"``), or
          ``None`` when the bet cannot be settled yet.

        Bets without a result stay open: a missing value (no row, NaN,
        ``None``) never voids a bet here, because in a bulk table it usually
        means "not available yet".  Void bets explicitly with :meth:`settle`.

        Returns the number of bets settled.
        """
        lookup = results if callable(results) else _results_lookup(results)
        rows, extras = self._load()
        settled = 0
        for bet in rows:
            if bet["result"]:
                continue
            outcome = lookup(pd.Series(bet))
            if _is_missing(outcome):
                continue
            if isinstance(outcome, str):
                _settle(bet, outcome, None, overwrite=False)
            else:
                _settle(bet, None, outcome, overwrite=False)
            settled += 1
        if settled:
            self._save(rows, extras)
        return settled

    def remove(self, bet_id: str) -> None:
        """Delete a bet (one recorded by mistake) from the ledger."""
        rows, extras = self._load()
        removed = self._find(rows, bet_id)
        self._save([bet for bet in rows if bet is not removed], extras)

    # -- views ---------------------------------------------------------------
    def to_frame(self) -> pd.DataFrame:
        """Every bet as a DataFrame with :data:`LEDGER_COLUMNS` (then any extra columns).

        Numeric columns are float (NaN = unknown), the rest are strings
        (``""`` = blank).
        """
        rows, extras = self._load()
        columns = LEDGER_COLUMNS + extras
        frame = pd.DataFrame(rows, columns=columns)
        for column in columns:
            frame[column] = frame[column].astype(float if column in _NUMERIC_COLUMNS else str)
        return frame

    def open_bets(self) -> pd.DataFrame:
        """Bets that have not been settled yet."""
        frame = self.to_frame()
        return frame[frame["result"] == ""].reset_index(drop=True)

    def settled_bets(self) -> pd.DataFrame:
        """Bets with a result (including voids)."""
        frame = self.to_frame()
        return frame[frame["result"] != ""].reset_index(drop=True)


def _new_id() -> str:
    # The leading letter keeps spreadsheet programs from reading an id as a number.
    return "b" + uuid.uuid4().hex[:8]


def _cell(value: object) -> str:
    """CSV text of one ledger value: floats round-trip exactly, NaN is blank."""
    if isinstance(value, float):
        return "" if math.isnan(value) else repr(value)
    # A bare carriage return is not quoted by every csv version and would split the row.
    return str(value).replace("\r\n", "\n").replace("\r", "\n")


# ---------------------------------------------------------------------------
# Performance statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Arrays:
    """Validated NumPy view of a bets table, with gaps filled where the maths allows."""

    result: np.ndarray
    stake: np.ndarray
    price: np.ndarray
    fair_prob: np.ndarray
    push_prob: np.ndarray
    ev: np.ndarray
    pnl: np.ndarray
    clv_price: np.ndarray
    clv_ev: np.ndarray


def _numeric_column(bets: pd.DataFrame, name: str) -> np.ndarray:
    """Column as a fresh float array; absent column, blanks and infinities become NaN."""
    if name not in bets.columns:
        return np.full(len(bets), np.nan)
    try:
        values = pd.to_numeric(bets[name], errors="raise").to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"column {name!r} must be numeric: {exc}") from exc
    return np.where(np.isfinite(values), values, np.nan)


def _prepare(bets: pd.DataFrame) -> _Arrays:
    """Validate a bets table and derive what the statistics need.

    ``ev`` is filled from ``fair_prob`` where blank (and the reverse), and
    ``pnl`` from the result where blank, so no NaN reaches a sum.
    """
    if not isinstance(bets, pd.DataFrame):
        raise TypeError("bets must be a pandas DataFrame (for a ledger use BetLedger.to_frame())")
    count = len(bets)
    if "result" in bets.columns:
        result = np.array([_text(value).lower() for value in bets["result"].tolist()], dtype=object)
    else:
        result = np.full(count, "", dtype=object)
    unknown = sorted(set(result.tolist()) - set(RESULTS) - {""})
    if unknown:
        raise ValueError(f"unknown result value(s) {unknown}; expected one of {RESULTS} or blank")

    stake = _numeric_column(bets, "stake")
    price = _numeric_column(bets, "price")
    invalid = ~(stake > 0.0) | ~(price > 1.0)
    if invalid.any():
        raise ValueError(
            f"{int(invalid.sum())} bet(s) lack a valid stake (> 0) or decimal price (> 1); "
            f"first at position {int(np.flatnonzero(invalid)[0])}"
        )

    push = _numeric_column(bets, "push_prob")
    push = np.where(np.isnan(push), 0.0, push)
    fair = _numeric_column(bets, "fair_prob")
    ev = _numeric_column(bets, "ev")
    has_fair = ~np.isnan(fair)
    if ((push < 0.0) | (push >= 1.0)).any() or ((fair < 0.0) | (fair + push > 1.0 + _PROB_TOL)).any():
        raise ValueError("fair_prob must be in [0, 1 - push_prob] and push_prob in [0, 1) for every bet")
    # A bet recorded with an EV but no probability: invert EV = p * d - 1 + push.
    implied = (ev + 1.0 - push) / price
    recoverable = ~has_fair & (implied >= 0.0) & (implied <= 1.0 - push + _PROB_TOL)
    fair = np.minimum(np.where(recoverable, implied, fair), 1.0 - push)
    for i in np.flatnonzero(has_fair & np.isnan(ev)):
        ev[i] = expected_value(fair[i], price[i], push[i])

    stored_pnl = _numeric_column(bets, "pnl")
    graded_pnl = np.select([result == "win", result == "loss"], [stake * (price - 1.0), -stake], default=0.0)
    return _Arrays(
        result=result,
        stake=stake,
        price=price,
        fair_prob=fair,
        push_prob=push,
        ev=ev,
        pnl=np.where(np.isnan(stored_pnl), graded_pnl, stored_pnl),
        clv_price=_numeric_column(bets, "clv_price"),
        clv_ev=_numeric_column(bets, "clv_ev"),
    )


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else math.nan


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return _ratio(float(np.sum(values * weights)), float(np.sum(weights)))


def _bootstrap_roi_ci(pnl: np.ndarray, stake: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    """Percentile 95% interval of ROI when the bets are resampled with replacement."""
    count = len(pnl)
    if count < 2 or n_boot < 1:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    rois = np.empty(n_boot)
    chunk = max(1, _BOOT_CELLS // count)
    for start in range(0, n_boot, chunk):
        draws = rng.integers(0, count, size=(min(chunk, n_boot - start), count))
        rois[start : start + len(draws)] = pnl[draws].sum(axis=1) / stake[draws].sum(axis=1)
    low, high = np.percentile(rois, [2.5, 97.5])
    return float(low), float(high)


def bets_needed(edge: float, sd: float, n_sigma: float = 2.0) -> float:
    """Approximate number of bets before an edge stands clear of zero.

    ``edge`` is the mean profit per bet and ``sd`` the standard deviation of
    the profit of one bet, in the same units.  After ``n`` independent bets
    the total profit has mean ``n * edge`` and standard deviation
    ``sqrt(n) * sd``, so it is ``n_sigma`` standard deviations away from zero
    once ``n >= (n_sigma * sd / edge) ** 2``.  A typical prop bet has ``sd``
    near 1 unit, so a 3% edge needs about ``(2 / 0.03) ** 2`` = 4,400 bets.

    Returns ``inf`` for a zero edge and NaN for unusable input.
    """
    if not (math.isfinite(edge) and math.isfinite(sd) and sd >= 0.0 and n_sigma > 0.0):
        return math.nan
    if edge == 0.0:
        return math.inf
    return float(math.ceil((n_sigma * sd / edge) ** 2))


def summarize(bets: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> Dict[str, Any]:
    """Performance of a set of bets, next to what the model said to expect.

    ``bets`` is :meth:`BetLedger.to_frame` or any frame with (a subset of)
    its columns; only ``stake`` and ``price`` are mandatory.  Nothing here
    raises on an empty or partly settled ledger: a statistic that cannot be
    computed is NaN.

    "Graded" below means settled as win, loss or push.  Void bets returned
    the stake and are left out of everything except the counts.

    Counts
        ``n_bets``, ``n_settled`` (has a result, voids included), ``n_open``,
        ``wins``, ``losses``, ``pushes``, ``voids``.
    Money
        ``total_staked`` (stakes of graded bets - the ROI denominator),
        ``open_stake`` (stakes still at risk), ``pnl`` (profit of graded
        bets; taken from the ``pnl`` column, or derived from the result where
        that is blank), ``roi`` = ``pnl / total_staked``, ``win_rate`` =
        ``wins / (wins + losses)``.
    Prices
        ``avg_price``: stake-weighted mean decimal price of all non-void
        bets, i.e. the price of the whole book of bets as one wager;
        ``avg_price_american`` is that same price in American form.
    Results versus expectation
        Over the ``n_model`` graded bets that carry a model expectation (a
        ``fair_prob``, or an ``ev`` from which it is recovered):
        ``expected_pnl`` = sum of ``stake * ev`` at bet time,
        ``expected_roi`` = ``expected_pnl`` per unit staked on those bets,
        ``luck`` = their profit minus ``expected_pnl``, and ``z_score`` =
        ``luck`` divided by its standard deviation under the model, built
        from each bet's own variance (:func:`evmodel.ev.bet_variance` at its
        ``fair_prob``).  If the fair probabilities are right ``z_score`` is
        roughly standard normal: beyond about -2 the results are worse than
        luck plausibly explains and the claimed edge is in doubt.
    Significance
        ``roi_ci``: ``(low, high)`` bootstrap 95% interval of ROI, resampling
        graded bets ``n_boot`` times with ``seed`` (NaN when ``n_boot`` is 0
        or there are fewer than two graded bets).  ``p_value_profit``:
        one-sided normal-approximation p-value of the hypothesis "true ROI
        <= 0" from the per-bet profits; small means the profit is unlikely to
        be luck.  ``bets_needed``: :func:`bets_needed` for the model's
        average edge and variance per bet (for the observed profits when no
        bet carries a model expectation) - how many bets of this kind it
        takes before the edge is two standard deviations from zero.
    Closing-line value
        ``avg_clv_price`` and ``avg_clv_ev`` are stake-weighted means over
        the non-void bets that have them; ``clv_beat_rate`` is the share of
        bets with ``clv_ev > 0`` and ``clv_n`` how many have a ``clv_ev``.
    """
    data = _prepare(bets)
    won = data.result == "win"
    lost = data.result == "loss"
    pushed = data.result == "push"
    void = data.result == "void"
    is_open = data.result == ""
    graded = won | lost | pushed

    wins, losses = int(won.sum()), int(lost.sum())
    total_staked = float(data.stake[graded].sum())
    profits = data.pnl[graded]
    pnl = float(profits.sum())
    n_graded = len(profits)

    avg_price = _weighted_mean(data.price[~void], data.stake[~void])

    model = graded & ~np.isnan(data.ev) & ~np.isnan(data.fair_prob)
    n_model = int(model.sum())
    expected_pnl = expected_roi = luck = z_score = math.nan
    needed = math.nan
    if n_model:
        stakes = data.stake[model]
        expected = stakes * data.ev[model]
        variances = stakes**2 * np.array(
            [
                bet_variance(p, d, q)
                for p, d, q in zip(data.fair_prob[model], data.price[model], data.push_prob[model])
            ]
        )
        expected_pnl = float(expected.sum())
        expected_roi = _ratio(expected_pnl, float(stakes.sum()))
        luck = float(data.pnl[model].sum()) - expected_pnl
        total_variance = float(variances.sum())
        if total_variance > 0.0:
            z_score = luck / math.sqrt(total_variance)
        needed = bets_needed(float(expected.mean()), math.sqrt(float(variances.mean())))

    p_value = math.nan
    if n_graded >= 2:
        spread = float(profits.std(ddof=1))
        if spread > 0.0:
            t_stat = float(profits.mean()) / (spread / math.sqrt(n_graded))
            p_value = 0.5 * math.erfc(t_stat / math.sqrt(2.0))
        if not n_model:
            needed = bets_needed(float(profits.mean()), spread)

    has_clv_price = ~void & ~np.isnan(data.clv_price)
    has_clv_ev = ~void & ~np.isnan(data.clv_ev)
    clv_n = int(has_clv_ev.sum())

    return {
        "n_bets": len(data.result),
        "n_settled": int((~is_open).sum()),
        "n_open": int(is_open.sum()),
        "wins": wins,
        "losses": losses,
        "pushes": int(pushed.sum()),
        "voids": int(void.sum()),
        "total_staked": total_staked,
        "open_stake": float(data.stake[is_open].sum()),
        "pnl": pnl,
        "roi": _ratio(pnl, total_staked),
        "win_rate": _ratio(wins, wins + losses),
        "avg_price": avg_price,
        "avg_price_american": decimal_to_american(avg_price) if avg_price > 1.0 else math.nan,
        "n_model": n_model,
        "expected_pnl": expected_pnl,
        "expected_roi": expected_roi,
        "luck": luck,
        "z_score": z_score,
        "roi_ci": _bootstrap_roi_ci(profits, data.stake[graded], n_boot, seed),
        "p_value_profit": p_value,
        "bets_needed": needed,
        "avg_clv_price": _weighted_mean(data.clv_price[has_clv_price], data.stake[has_clv_price]),
        "avg_clv_ev": _weighted_mean(data.clv_ev[has_clv_ev], data.stake[has_clv_ev]),
        "clv_beat_rate": _ratio(float((data.clv_ev[has_clv_ev] > 0.0).sum()), clv_n),
        "clv_n": clv_n,
    }


_BREAKDOWN_STATS: List[str] = [
    "n_bets",
    "n_settled",
    "wins",
    "losses",
    "pushes",
    "total_staked",
    "pnl",
    "roi",
    "win_rate",
    "avg_price",
    "expected_roi",
    "z_score",
    "avg_clv_ev",
    "clv_beat_rate",
    "clv_n",
]


def breakdown(bets: pd.DataFrame, by: Union[str, Sequence[str]] = "market") -> pd.DataFrame:
    """The main :func:`summarize` statistics for each group of bets.

    ``by`` is a column of ``bets`` (``"market"``, ``"book"``, ``"sport"``,
    ``"side"``, ``"source"`` ...) or a list of columns.  Returns one row per
    group - the group columns followed by counts, staked amount, profit, ROI,
    win rate, average price, expected ROI, z-score and CLV - with the
    largest groups first.  Small groups are noisy: read ``z_score`` and
    ``n_settled`` before concluding that a market or book is good or bad.
    """
    if not isinstance(bets, pd.DataFrame):
        raise TypeError("bets must be a pandas DataFrame (for a ledger use BetLedger.to_frame())")
    keys = [by] if isinstance(by, str) else list(by)
    missing = [key for key in keys if key not in bets.columns]
    if not keys or missing:
        raise ValueError(f"cannot break down by {missing or 'no columns'}; available: {list(bets.columns)}")

    rows: List[Dict[str, Any]] = []
    for group_key, group in bets.groupby(keys, dropna=False, sort=True):
        stats = summarize(group, n_boot=0)
        rows.append({**dict(zip(keys, group_key)), **{name: stats[name] for name in _BREAKDOWN_STATS}})
    table = pd.DataFrame(rows, columns=keys + _BREAKDOWN_STATS)
    return table.sort_values("n_bets", ascending=False, kind="stable").reset_index(drop=True)


_CALIBRATION_COLUMNS: List[str] = ["prob_low", "prob_high", "n", "wins", "predicted", "actual", "diff", "z"]


def _bin_edges(bins: Union[int, Sequence[float]]) -> np.ndarray:
    if isinstance(bins, (int, np.integer)):
        if bins < 1:
            raise ValueError("bins must be a positive number of buckets or a sequence of edges")
        return np.arange(bins + 1) / bins  # exact decimal edges, unlike linspace
    edges = np.asarray(list(bins), dtype=float)
    if len(edges) < 2 or not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0.0):
        raise ValueError("bin edges must be at least two finite, strictly increasing numbers")
    return edges


def calibration_table(bets: pd.DataFrame, bins: Union[int, Sequence[float]] = 10) -> pd.DataFrame:
    """Predicted win probability against realised win frequency.

    Uses settled bets that were decided (wins and losses; pushes and voids
    are excluded) and have a ``fair_prob``.  Because pushes are excluded the
    prediction compared is the probability of winning given no push,
    ``fair_prob / (1 - push_prob)``.

    ``bins`` is a number of equal-width buckets on [0, 1] or an explicit
    sequence of bucket edges; a bucket contains its lower edge, and the last
    bucket its upper edge too.  One row per non-empty bucket:

    ``prob_low``, ``prob_high``  bucket edges
    ``n``, ``wins``              decided bets and winners in the bucket
    ``predicted``                mean predicted probability
    ``actual``                   realised win frequency
    ``diff``                     ``actual - predicted``
    ``z``                        ``(wins - expected wins) / sd`` under the
                                 predictions; beyond +-2 the bucket is
                                 miscalibrated by more than chance explains
    """
    data = _prepare(bets)
    edges = _bin_edges(bins)
    decided = ((data.result == "win") | (data.result == "loss")) & ~np.isnan(data.fair_prob)
    predicted = np.clip(data.fair_prob[decided] / (1.0 - data.push_prob[decided]), 0.0, 1.0)
    won = (data.result[decided] == "win").astype(float)

    last = len(edges) - 2
    bucket = np.searchsorted(edges, predicted, side="right") - 1
    bucket[predicted == edges[-1]] = last

    rows: List[Dict[str, float]] = []
    for index in range(last + 1):
        inside = bucket == index
        count = int(inside.sum())
        if not count:
            continue
        probs = predicted[inside]
        wins = float(won[inside].sum())
        spread = math.sqrt(float(np.sum(probs * (1.0 - probs))))
        rows.append(
            {
                "prob_low": float(edges[index]),
                "prob_high": float(edges[index + 1]),
                "n": count,
                "wins": int(wins),
                "predicted": float(probs.mean()),
                "actual": wins / count,
                "diff": wins / count - float(probs.mean()),
                "z": (wins - float(probs.sum())) / spread if spread > 0.0 else math.nan,
            }
        )
    return pd.DataFrame(rows, columns=_CALIBRATION_COLUMNS)
