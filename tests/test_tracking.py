"""Tests for evmodel.tracking: grading, CLV, the CSV ledger and the statistics."""

from __future__ import annotations

import math
import os
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, poisson

from evmodel import tracking
from evmodel.distributions import FamilySpec
from evmodel.ev import bet_variance, expected_value
from evmodel.schema import OPPORTUNITY_COLUMNS
from evmodel.tracking import (
    LEDGER_COLUMNS,
    BetLedger,
    DuplicateBetError,
    bets_needed,
    breakdown,
    calibration_table,
    closing_fair_prob,
    clv_ev,
    clv_price,
    clv_with_line_move,
    grade,
    settle_pnl,
    summarize,
)

# Any warning (NumPy invalid-value, pandas deprecation ...) is a failure.
pytestmark = pytest.mark.filterwarnings("error")

NEGBIN = FamilySpec("negbin", phi=1.0, alpha=0.05)
MINUS_110 = 1.0 + 100.0 / 110.0


def _fields(**overrides: object) -> Dict[str, object]:
    """Keyword fields of a valid bet."""
    fields: Dict[str, object] = dict(
        sport="NBA",
        event_id="evt1",
        event="Nuggets @ Lakers",
        player="Nikola Jokić",
        market="player_points",
        line=26.5,
        side="over",
        book="DraftKings",
        price=2.0,
        fair_prob=0.55,
    )
    fields.update(overrides)
    return fields


@pytest.fixture
def ledger(tmp_path: Path) -> BetLedger:
    return BetLedger(tmp_path / "ledger.csv")


# ---------------------------------------------------------------------------
# grade / settle_pnl
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "side, line, actual, expected",
    [
        ("over", 24.5, 25, "win"),
        ("over", 24.5, 24, "loss"),
        ("under", 24.5, 24, "win"),
        ("under", 24.5, 25, "loss"),
        ("over", 25, 26, "win"),
        ("over", 25, 25, "push"),
        ("over", 25, 24, "loss"),
        ("under", 25, 24, "win"),
        ("under", 25, 25, "push"),
        ("under", 25, 26, "loss"),
        ("over", 0.5, 0, "loss"),
        ("under", 0.5, 0, "win"),
        ("over", 0.5, 1, "win"),
    ],
)
def test_grade_half_and_whole_lines(side: str, line: float, actual: float, expected: str) -> None:
    assert grade(side, line, actual) == expected


def test_grade_missing_actual_is_void() -> None:
    assert grade("over", 24.5, None) == "void"
    assert grade("under", 24.5, float("nan")) == "void"
    assert grade("over", 24.5, np.nan) == "void"


def test_grade_normalises_side_and_rejects_bad_input() -> None:
    assert grade(" Over ", 24.5, 30) == "win"
    with pytest.raises(ValueError, match="side"):
        grade("yes", 24.5, 30)
    with pytest.raises(ValueError, match="line"):
        grade("over", float("nan"), 30)
    with pytest.raises(ValueError, match="actual"):
        grade("over", 24.5, float("inf"))
    with pytest.raises(ValueError, match="actual"):
        grade("over", 24.5, "thirty")  # type: ignore[arg-type]


def test_settle_pnl() -> None:
    assert settle_pnl("win", 10.0, 2.5) == pytest.approx(15.0)
    assert settle_pnl("win", 110.0, MINUS_110) == pytest.approx(100.0)
    assert settle_pnl("loss", 10.0, 2.5) == -10.0
    assert settle_pnl("push", 10.0, 2.5) == 0.0
    assert settle_pnl("void", 10.0, 2.5) == 0.0


def test_settle_pnl_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="result"):
        settle_pnl("won", 10.0, 2.0)
    with pytest.raises(ValueError, match="stake"):
        settle_pnl("win", 0.0, 2.0)
    with pytest.raises(ValueError, match="stake"):
        settle_pnl("win", float("nan"), 2.0)
    # American odds must never be mistaken for a decimal price.
    with pytest.raises(ValueError, match="decimal odds"):
        settle_pnl("win", 10.0, -110)
    with pytest.raises(ValueError, match="decimal odds"):
        settle_pnl("win", 10.0, 1.0)


# ---------------------------------------------------------------------------
# CLV functions
# ---------------------------------------------------------------------------


def test_clv_price() -> None:
    assert clv_price(2.10, 2.00) == pytest.approx(0.05)
    assert clv_price(1.90, 2.00) == pytest.approx(-0.05)
    assert clv_price(1.95, 1.95) == 0.0
    with pytest.raises(ValueError):
        clv_price(2.0, 1.0)
    with pytest.raises(ValueError):
        clv_price(-110, 2.0)


def test_closing_fair_prob() -> None:
    assert closing_fair_prob(MINUS_110, MINUS_110) == pytest.approx(0.5)
    # multiplicative: (1/1.8) / (1/1.8 + 1/2.1) = 2.1 / 3.9
    assert closing_fair_prob(1.8, 2.1) == pytest.approx(2.1 / 3.9)
    assert closing_fair_prob(2.1, 1.8) == pytest.approx(1.8 / 3.9)
    # additive: q - overround / 2 with q = 1/1.8 and overround = 1/1.8 + 1/2.1 - 1
    overround = 1 / 1.8 + 1 / 2.1 - 1
    assert closing_fair_prob(1.8, 2.1, method="additive") == pytest.approx(1 / 1.8 - overround / 2)
    with pytest.raises(ValueError):
        closing_fair_prob(1.8, 2.1, method="nonsense")
    with pytest.raises(ValueError):
        closing_fair_prob(1.8, 0.9)


def test_clv_ev() -> None:
    assert clv_ev(2.0, 0.55) == pytest.approx(0.10)
    assert clv_ev(MINUS_110, 0.5) == pytest.approx(-1 / 22)  # 0.5 * 100/110 - 0.5
    # push: win 0.45 at +100, lose 0.45, push 0.10 -> exactly break-even
    assert clv_ev(2.0, 0.45, push_prob=0.10) == pytest.approx(0.0)
    assert clv_ev(2.5, 0.40, push_prob=0.10) == pytest.approx(0.40 * 1.5 - 0.50)
    assert clv_ev(2.0, closing_fair_prob(1.8, 2.1)) == pytest.approx(2 * 2.1 / 3.9 - 1)
    with pytest.raises(ValueError):
        clv_ev(2.0, float("nan"))
    with pytest.raises(ValueError):
        clv_ev(2.0, 0.6, push_prob=0.5)


@pytest.mark.parametrize("side", ["over", "under"])
def test_line_move_clv_equals_plain_when_lines_match(side: str) -> None:
    closing = {"over": 1.8, "under": 2.1}
    other = "under" if side == "over" else "over"
    out = clv_with_line_move(NEGBIN, side, 24.5, 2.0, 24.5, closing["over"], closing["under"])
    fair = closing_fair_prob(closing[side], closing[other])
    assert out["push_prob"] == 0.0
    assert out["closing_fair_prob"] == pytest.approx(fair, abs=1e-8)
    assert out["clv_ev"] == pytest.approx(clv_ev(2.0, fair), abs=1e-8)


def test_line_move_clv_same_whole_line_carries_push_mass() -> None:
    out = clv_with_line_move(NEGBIN, "over", 25, 2.0, 25, 1.8, 2.1)
    conditional = closing_fair_prob(1.8, 2.1)
    push = out["push_prob"]
    assert 0.02 < push < 0.10
    assert out["closing_fair_prob"] == pytest.approx(conditional * (1 - push), abs=1e-8)
    assert out["clv_ev"] == pytest.approx(clv_ev(2.0, conditional * (1 - push), push), abs=1e-8)
    assert out["clv_ev"] == pytest.approx((1 - push) * (conditional * 2.0 - 1), abs=1e-8)


def test_line_move_in_our_favour_is_positive_clv() -> None:
    # Bet over 24.5 at -110; the market closes a full point higher at -110 / -110.
    favourable = clv_with_line_move(NEGBIN, "over", 24.5, MINUS_110, 25.5, MINUS_110, MINUS_110)
    unchanged = clv_with_line_move(NEGBIN, "over", 24.5, MINUS_110, 24.5, MINUS_110, MINUS_110)
    against = clv_with_line_move(NEGBIN, "over", 24.5, MINUS_110, 23.5, MINUS_110, MINUS_110)
    assert favourable["closing_fair_prob"] > 0.5 > against["closing_fair_prob"]
    assert favourable["clv_ev"] > 0.0
    assert unchanged["clv_ev"] == pytest.approx(-1 / 22, abs=1e-8)  # just the vig we paid
    assert against["clv_ev"] < unchanged["clv_ev"] < favourable["clv_ev"]
    assert favourable["implied_mean"] > unchanged["implied_mean"] > against["implied_mean"]

    # For an under the favourable direction is down.
    under_good = clv_with_line_move(NEGBIN, "under", 24.5, MINUS_110, 23.5, MINUS_110, MINUS_110)
    under_bad = clv_with_line_move(NEGBIN, "under", 24.5, MINUS_110, 25.5, MINUS_110, MINUS_110)
    assert under_good["clv_ev"] > 0.0 > under_bad["clv_ev"]
    # Over and under at the same line see complementary probabilities.
    assert favourable["closing_fair_prob"] + under_bad["closing_fair_prob"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "spec, line, step",
    [
        (FamilySpec("poisson"), 2.5, 1.0),
        (FamilySpec("negbin", phi=1.0, alpha=0.08), 24.5, 1.0),
        (FamilySpec("gamma", phi=10.0, alpha=0.25), 62.5, 5.0),
        (FamilySpec("normal", phi=0.0, alpha=0.06, c=1500.0, lower=None), 249.5, 10.0),
        (FamilySpec("negbin", phi=1.0, alpha=0.3, zero_inflation=0.1), 4.5, 1.0),
    ],
)
def test_line_move_direction_holds_for_every_family(spec: FamilySpec, line: float, step: float) -> None:
    def over_clv(closing_line: float) -> float:
        return clv_with_line_move(spec, "over", line, 2.0, closing_line, 1.9, 1.9)["clv_ev"]

    def under_clv(closing_line: float) -> float:
        return clv_with_line_move(spec, "under", line, 2.0, closing_line, 1.9, 1.9)["clv_ev"]

    # At an unchanged line a 2.0 price against a 50% close is exactly break-even.
    assert over_clv(line) == pytest.approx(0.0, abs=1e-7)
    assert under_clv(line) == pytest.approx(0.0, abs=1e-7)
    assert over_clv(line + step) > 0.0 > over_clv(line - step)
    assert under_clv(line - step) > 0.0 > under_clv(line + step)
    # At a half-point line nothing pushes, so the two sides mirror each other.
    assert over_clv(line + step) == pytest.approx(-under_clv(line + step), abs=1e-9)


def test_line_move_clv_against_poisson_by_hand() -> None:
    # A vig-free closing market at 4.5 generated by a Poisson with mean 5.
    spec = FamilySpec("poisson")
    p_over_close = float(poisson.sf(4, 5.0))
    over_price, under_price = 1 / p_over_close, 1 / (1 - p_over_close)

    half = clv_with_line_move(spec, "over", 5.5, 2.4, 4.5, over_price, under_price)
    assert half["implied_mean"] == pytest.approx(5.0, abs=1e-6)
    assert half["closing_fair_prob"] == pytest.approx(float(poisson.sf(5, 5.0)), abs=1e-7)
    assert half["push_prob"] == 0.0
    assert half["clv_ev"] == pytest.approx(float(poisson.sf(5, 5.0)) * 2.4 - 1, abs=1e-6)

    whole = clv_with_line_move(spec, "under", 5, 2.4, 4.5, over_price, under_price)
    p_under, p_push = float(poisson.cdf(4, 5.0)), float(poisson.pmf(5, 5.0))
    assert whole["closing_fair_prob"] == pytest.approx(p_under, abs=1e-7)
    assert whole["push_prob"] == pytest.approx(p_push, abs=1e-7)
    assert whole["clv_ev"] == pytest.approx(p_under * 1.4 - (1 - p_under - p_push), abs=1e-6)


class _StuckSpec(FamilySpec):
    """A family whose mean solver gives up and returns the same mean for any price."""

    def solve_mean(self, line: float, p_over: float, exclude_push: bool = True) -> float:
        return 20.0


def test_line_move_clv_refuses_a_family_that_does_not_fit_the_close() -> None:
    stuck = _StuckSpec("poisson")
    with pytest.raises(ValueError, match="cannot reproduce a fair over probability of 0.5000 at line 25.5"):
        clv_with_line_move(stuck, "over", 24.5, 2.0, 25.5, 1.9, 1.9)
    # The same call is fine when the price happens to be the one that mean produces.
    p_over = float(poisson.sf(25, 20.0))
    out = clv_with_line_move(stuck, "over", 24.5, 2.0, 25.5, 1 / p_over, 1 / (1 - p_over))
    assert out["closing_fair_prob"] == pytest.approx(float(poisson.sf(24, 20.0)))


def test_line_move_clv_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="side"):
        clv_with_line_move(NEGBIN, "home", 24.5, 2.0, 25.5, 1.9, 1.9)
    with pytest.raises(ValueError, match="bet_decimal"):
        clv_with_line_move(NEGBIN, "over", 24.5, -110, 25.5, 1.9, 1.9)
    with pytest.raises(ValueError, match="closing_under_decimal"):
        clv_with_line_move(NEGBIN, "over", 24.5, 2.0, 25.5, 1.9, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="closing_line"):
        clv_with_line_move(NEGBIN, "over", 24.5, 2.0, float("nan"), 1.9, 1.9)


# ---------------------------------------------------------------------------
# BetLedger: adding bets
# ---------------------------------------------------------------------------


def test_new_ledger_creates_directories_and_is_empty(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "er" / "ledger.csv"
    book = BetLedger(path)
    assert path.parent.is_dir()
    assert not path.exists()  # nothing written until there is a bet
    frame = book.to_frame()
    assert list(frame.columns) == LEDGER_COLUMNS
    assert len(frame) == 0
    assert frame["stake"].dtype == float and frame["pnl"].dtype == float
    assert len(book.open_bets()) == 0 and len(book.settled_bets()) == 0


def test_add_with_keyword_fields(ledger: BetLedger) -> None:
    bet_id = ledger.add(stake=25, **_fields(price=1.91, commence_time="2026-11-13 00:30"))
    assert isinstance(bet_id, str) and bet_id[0].isalpha() and len(bet_id) <= 12

    frame = ledger.to_frame()
    assert list(frame.columns) == LEDGER_COLUMNS
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["bet_id"] == bet_id
    assert row["sport"] == "nba" and row["book"] == "draftkings"
    assert row["player"] == "Nikola Jokić" and row["player_key"] == "nikola jokic"
    assert row["market"] == "player_points" and row["side"] == "over"
    assert row["line"] == 26.5 and row["price"] == 1.91 and row["stake"] == 25.0
    assert row["fair_prob"] == 0.55 and row["push_prob"] == 0.0
    assert row["ev"] == pytest.approx(0.55 * 1.91 - 1)
    assert row["commence_time"] == "2026-11-13T00:30:00+00:00"
    placed = datetime.fromisoformat(row["placed_at"])
    assert placed.utcoffset() == timedelta(0)
    assert abs(datetime.now(timezone.utc) - placed) < timedelta(hours=1)
    assert row["result"] == "" and row["settled_at"] == "" and row["notes"] == ""
    for column in ("closing_line", "closing_price", "clv_price", "clv_ev", "actual", "pnl"):
        assert math.isnan(row[column])


def test_add_from_opportunity_row(ledger: BetLedger) -> None:
    opportunity = pd.Series(
        {
            **{column: np.nan for column in OPPORTUNITY_COLUMNS},
            "sport": "nba",
            "event_id": "abc123",
            "commence_time": pd.Timestamp("2026-11-12 19:30", tz="America/New_York"),
            "event": "Nuggets @ Lakers",
            "player": "LeBron James Jr.",
            "player_key": "lebron james",
            "market": "player_rebounds",
            "line": 8.0,
            "side": "under",
            "book": "fanduel",
            "price": 2.05,
            "price_american": 105.0,
            "fair_prob": 0.46,
            "push_prob": 0.10,
            "ev": expected_value(0.46, 2.05, 0.10),
            "kelly": 0.04,
            "stake_units": 0.8,
            "source": "blend",
            "flags": "single_ref,alt_line",
        }
    )
    assert list(opportunity.index) == OPPORTUNITY_COLUMNS
    bet_id = ledger.add(opportunity, stake=20)
    row = ledger.to_frame().set_index("bet_id").loc[bet_id]
    assert row["event_id"] == "abc123" and row["event"] == "Nuggets @ Lakers"
    assert row["commence_time"] == "2026-11-13T00:30:00+00:00"  # converted to UTC
    assert row["player_key"] == "lebron james" and row["line"] == 8.0 and row["side"] == "under"
    assert row["price"] == 2.05 and row["stake"] == 20.0  # the explicit stake, not stake_units
    assert row["fair_prob"] == 0.46 and row["push_prob"] == 0.10
    assert row["ev"] == pytest.approx(0.46 * 1.05 - 0.44)
    assert row["source"] == "blend" and row["flags"] == "single_ref,alt_line"

    # A plain mapping works too, the stake may be positional, and an overridden
    # price gets its own EV instead of the row's stale one.
    other = dict(opportunity, book="betmgm")
    second = ledger.add(other, 10.0, price=1.95)
    row = ledger.to_frame().set_index("bet_id").loc[second]
    assert row["price"] == 1.95 and row["stake"] == 10.0
    assert row["ev"] == pytest.approx(0.46 * 0.95 - 0.44)

    # Optional overrides left as None do not blank what the row says.
    third = ledger.add(dict(opportunity, book="caesars"), stake=5, price=None, notes=None, book=None)
    row = ledger.to_frame().set_index("bet_id").loc[third]
    assert row["price"] == 2.05 and row["book"] == "caesars" and row["notes"] == ""


def test_add_normalises_text_and_timestamps(ledger: BetLedger) -> None:
    bet_id = ledger.add(
        stake=10,
        **_fields(
            placed_at=pd.Timestamp("2026-01-05 09:15:30.75", tz="Europe/London"),
            commence_time="2026-01-05T19:00:00-05:00",
            player="  Nikola Jokić ",
            market=" Player_Points ",
            notes="first line\rsecond line",
        ),
    )
    frame = ledger.to_frame()
    assert len(frame) == 1  # the bare carriage return did not split the row
    row = frame.iloc[0]
    assert row["bet_id"] == bet_id
    assert row["placed_at"] == "2026-01-05T09:15:30+00:00"
    assert row["commence_time"] == "2026-01-06T00:00:00+00:00"
    assert row["player"] == "Nikola Jokić" and row["market"] == "player_points"
    assert row["notes"] == "first line\nsecond line"


def test_add_requires_an_explicit_stake(ledger: BetLedger) -> None:
    with pytest.raises(ValueError, match="stake is required"):
        ledger.add({**_fields(), "stake_units": 1.5})
    assert not ledger.path.exists()


def test_add_accepts_american_price_only_by_name(ledger: BetLedger) -> None:
    bet_id = ledger.add(stake=11, **_fields(price=None, price_american=-110))
    row = ledger.to_frame().set_index("bet_id").loc[bet_id]
    assert row["price"] == pytest.approx(MINUS_110)
    assert row["ev"] == pytest.approx(0.55 * MINUS_110 - 1)
    # American odds passed where decimal odds belong are refused, not reinterpreted.
    with pytest.raises(ValueError, match="decimal odds"):
        ledger.add(stake=11, **_fields(price=-110, book="other"))
    with pytest.raises(ValueError, match="price is required"):
        ledger.add(stake=11, **_fields(price=None, book="other"))


def test_add_absorbs_rounding_noise_in_probabilities(ledger: BetLedger) -> None:
    # over + push from a distribution can exceed 1 by a few ulps; that is not bad input.
    bet_id = ledger.add(stake=5, **_fields(fair_prob=1.0 + 5e-10))
    whole = ledger.add(stake=5, **_fields(line=27, fair_prob=0.9 + 5e-10, push_prob=0.1))
    rows = ledger.to_frame().set_index("bet_id")
    assert rows.loc[bet_id, "fair_prob"] == 1.0 and rows.loc[bet_id, "ev"] == pytest.approx(1.0)
    assert rows.loc[whole, "fair_prob"] + rows.loc[whole, "push_prob"] <= 1.0
    assert summarize(ledger.to_frame())["n_bets"] == 2  # and the statistics accept what was stored


def test_add_keeps_ev_when_no_probability_is_known(ledger: BetLedger) -> None:
    bet_id = ledger.add(stake=5, **_fields(fair_prob=None, ev=0.04))
    row = ledger.to_frame().set_index("bet_id").loc[bet_id]
    assert math.isnan(row["fair_prob"]) and row["ev"] == 0.04
    with pytest.raises(ValueError, match="ev per unit"):
        ledger.add(stake=5, **_fields(fair_prob=None, ev=4.0, book="other"))


@pytest.mark.parametrize(
    "overrides, stake, message",
    [
        ({"side": "yes"}, 10, "side"),
        ({"side": None}, 10, "side"),
        ({"price": 1.0}, 10, "decimal odds"),
        ({"price": float("inf")}, 10, "finite"),
        ({"price": "abc"}, 10, "must be a number"),
        ({}, 0, "stake"),
        ({}, -5, "stake"),
        ({}, None, "stake is required"),
        ({"player": ""}, 10, "player"),
        ({"market": None}, 10, "market"),
        ({"line": None}, 10, "line"),
        ({"fair_prob": 1.2}, 10, "fair_prob"),
        ({"fair_prob": 0.95, "push_prob": 0.1}, 10, "fair_prob"),
        ({"push_prob": 1.0}, 10, "push_prob"),
        ({"commence_time": "not a date"}, 10, "commence_time"),
    ],
)
def test_add_rejects_bad_input(ledger: BetLedger, overrides: dict, stake: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ledger.add(stake=stake, **_fields(**overrides))  # type: ignore[arg-type]
    assert not ledger.path.exists()  # nothing was written


def test_add_rejects_unknown_fields_and_row_types(ledger: BetLedger) -> None:
    with pytest.raises(TypeError, match="prise"):
        ledger.add(stake=10, **_fields(prise=1.9))
    with pytest.raises(TypeError, match="row must be"):
        ledger.add([1, 2, 3], stake=10)  # type: ignore[arg-type]


def test_bet_ids_are_unique(ledger: BetLedger) -> None:
    ids = [ledger.add(stake=1, **_fields(line=10.5 + i)) for i in range(40)]
    assert len(set(ids)) == 40
    assert list(ledger.to_frame()["bet_id"]) == ids  # insertion order is kept


def test_duplicate_bets_are_refused(ledger: BetLedger) -> None:
    first = ledger.add(stake=10, **_fields())
    with pytest.raises(DuplicateBetError, match=first):
        ledger.add(stake=10, **_fields())
    # Same bet however it is spelled: the canonical key decides.
    with pytest.raises(DuplicateBetError):
        ledger.add(stake=5, **_fields(player="NIKOLA JOKIC", book="draftkings", side="OVER", price=2.1))
    assert len(ledger.to_frame()) == 1

    ledger.add(stake=10, allow_duplicate=True, **_fields())
    assert len(ledger.to_frame()) == 2

    # Anything that makes it a different bet is fine.
    for change in ({"line": 27.5}, {"side": "under"}, {"book": "fanduel"}, {"event_id": "evt2"},
                   {"market": "player_assists"}, {"player": "Jamal Murray"}):
        ledger.add(stake=10, **_fields(**change))
    assert len(ledger.to_frame()) == 8


def test_duplicate_check_and_settled_bets(ledger: BetLedger) -> None:
    # With an event id the same bet is a duplicate even after it settled ...
    first = ledger.add(stake=10, **_fields())
    ledger.settle(first, actual=30)
    with pytest.raises(DuplicateBetError):
        ledger.add(stake=10, **_fields())
    # ... without one, a settled bet must belong to an earlier game.
    anonymous = _fields(event_id="", player="Jamal Murray")
    second = ledger.add(stake=10, **anonymous)
    with pytest.raises(DuplicateBetError):
        ledger.add(stake=10, **anonymous)
    ledger.settle(second, actual=30)
    ledger.add(stake=10, **anonymous)
    assert len(ledger.to_frame()) == 3


# ---------------------------------------------------------------------------
# BetLedger: persistence
# ---------------------------------------------------------------------------


def _populate(book: BetLedger) -> List[str]:
    ids = [
        book.add(stake=25, **_fields(price=1.91, notes='said "lock", allegedly\nsecond line')),
        book.add(stake=10, **_fields(player="LeBron James", market="player_rebounds", line=8, side="under",
                                     price=2.05, fair_prob=0.46, push_prob=0.1, flags="single_ref,alt_line")),
        book.add(stake=1 / 3, **_fields(player="Luka Dončić", event_id="007", price_american=135, price=None,
                                        fair_prob=None)),
    ]
    book.record_close(ids[0], closing_over_price=1.8, closing_under_price=2.1)
    book.record_close(ids[1], 1.95, 1.87, closing_line=7.5, spec=NEGBIN)
    book.settle(ids[0], actual=31)
    book.settle(ids[1], result="push")
    return ids


def test_round_trip_through_disk(ledger: BetLedger) -> None:
    ids = _populate(ledger)
    before = ledger.to_frame()
    reloaded = BetLedger(ledger.path).to_frame()
    pd.testing.assert_frame_equal(before, reloaded, check_exact=True)

    assert list(reloaded["bet_id"]) == ids
    rows = reloaded.set_index("bet_id")
    assert rows.loc[ids[0], "notes"] == 'said "lock", allegedly\nsecond line'
    assert rows.loc[ids[1], "flags"] == "single_ref,alt_line"
    assert rows.loc[ids[2], "player"] == "Luka Dončić"
    assert rows.loc[ids[2], "event_id"] == "007"  # ids stay text: no leading zero lost
    assert rows.loc[ids[2], "stake"] == 1 / 3  # floats survive exactly
    assert rows.loc[ids[2], "price"] == 2.35
    assert rows.loc[ids[0], "pnl"] == 25 * (1.91 - 1)
    assert rows.loc[ids[0], "result"] == "win" and rows.loc[ids[1], "result"] == "push"

    # The file itself is plain CSV with the documented header.
    raw = pd.read_csv(ledger.path, dtype=str, keep_default_na=False)
    assert list(raw.columns) == LEDGER_COLUMNS
    assert len(raw) == 3
    assert raw.loc[2, "result"] == "" and raw.loc[2, "fair_prob"] == ""  # unknown = blank cell


def test_two_handles_on_one_file_see_each_other(ledger: BetLedger) -> None:
    other = BetLedger(ledger.path)
    first = ledger.add(stake=10, **_fields())
    second = other.add(stake=10, **_fields(line=27.5))
    ledger.settle(second, actual=30)  # written by the other handle
    assert list(ledger.to_frame()["bet_id"]) == [first, second]
    assert list(other.settled_bets()["bet_id"]) == [second]


def test_writes_are_atomic_and_leave_no_temp_files(ledger: BetLedger) -> None:
    ids = _populate(ledger)
    ledger.remove(ids[2])
    assert sorted(p.name for p in ledger.path.parent.iterdir()) == ["ledger.csv"]


def test_failed_write_keeps_the_old_file(ledger: BetLedger, monkeypatch: pytest.MonkeyPatch) -> None:
    first = ledger.add(stake=10, **_fields())
    original = ledger.path.read_bytes()

    def refuse(src: object, dst: object) -> None:
        raise OSError("disk says no")

    monkeypatch.setattr(tracking.os, "replace", refuse)
    with pytest.raises(OSError, match="disk says no"):
        ledger.add(stake=10, **_fields(line=30.5))
    with pytest.raises(OSError, match="disk says no"):
        ledger.settle(first, actual=40)
    monkeypatch.undo()

    assert ledger.path.read_bytes() == original
    assert sorted(p.name for p in ledger.path.parent.iterdir()) == ["ledger.csv"]
    assert list(ledger.open_bets()["bet_id"]) == [first]


def test_crash_while_writing_keeps_the_old_file(ledger: BetLedger, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger.add(stake=10, **_fields())
    original = ledger.path.read_bytes()

    def explode(value: object) -> str:
        raise RuntimeError("crash mid-write")

    monkeypatch.setattr(tracking, "_cell", explode)
    with pytest.raises(RuntimeError, match="crash mid-write"):
        ledger.add(stake=10, **_fields(line=30.5))
    monkeypatch.undo()

    assert ledger.path.read_bytes() == original
    assert sorted(p.name for p in ledger.path.parent.iterdir()) == ["ledger.csv"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_rewrite_preserves_file_mode(ledger: BetLedger) -> None:
    ledger.add(stake=10, **_fields())
    os.chmod(ledger.path, 0o600)
    ledger.add(stake=10, **_fields(line=30.5))
    assert stat.S_IMODE(ledger.path.stat().st_mode) == 0o600


def test_hand_edited_file_is_read_and_extra_columns_survive(ledger: BetLedger) -> None:
    first = ledger.add(stake=10, **_fields())
    # A person appends a settled bet in a spreadsheet: BOM, own column, blank pnl.
    header = LEDGER_COLUMNS + ["my_tag"]
    manual = {column: "" for column in header}
    manual.update(bet_id="manual1", player="Jamal Murray", player_key="jamal murray", market="player_assists",
                  line="6.5", side="Under", price="1.8", stake="20", result="WIN", my_tag="gut feel")
    text = ledger.path.read_text(encoding="utf-8")
    lines = text.splitlines()
    lines[0] += ",my_tag"
    lines[1] += ","
    lines.append(",".join(manual[column] for column in header))
    ledger.path.write_text("﻿" + "\n".join(lines) + "\n", encoding="utf-8")

    frame = ledger.to_frame()
    assert list(frame.columns) == header
    row = frame.set_index("bet_id").loc["manual1"]
    assert row["side"] == "under" and row["result"] == "win"
    assert row["pnl"] == pytest.approx(16.0)  # recomputed from the result
    assert row["push_prob"] == 0.0 and row["my_tag"] == "gut feel"

    ledger.settle(first, actual=20)  # any rewrite keeps the extra column
    frame = BetLedger(ledger.path).to_frame()
    assert list(frame.columns) == header
    assert frame.set_index("bet_id").loc["manual1", "my_tag"] == "gut feel"
    assert frame.set_index("bet_id").loc[first, "my_tag"] == ""
    assert summarize(frame)["pnl"] == pytest.approx(16.0 - 10.0)


@pytest.mark.parametrize(
    "column, value, message",
    [
        ("stake", "ten", "line 2: stake must be a number"),
        ("price", "-110", "line 2: price must be decimal odds"),
        ("side", "home", "line 2: side"),
        ("result", "won", "line 2: result"),
        ("bet_id", " ", "line 2: bet_id is blank"),
        ("line", "", "line 2: line is required"),
    ],
)
def test_corrupt_file_raises_a_located_error(ledger: BetLedger, column: str, value: str, message: str) -> None:
    ledger.add(stake=10, **_fields())
    raw = pd.read_csv(ledger.path, dtype=str, keep_default_na=False)
    raw.loc[0, column] = value
    raw.to_csv(ledger.path, index=False)
    with pytest.raises(ValueError, match=message):
        ledger.to_frame()


def test_corrupt_structure_raises(ledger: BetLedger) -> None:
    ledger.add(stake=10, **_fields())
    good = ledger.path.read_text(encoding="utf-8")
    ledger.path.write_text(good + good.splitlines()[1] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="appears more than once"):
        ledger.to_frame()
    ledger.path.write_text(good.rstrip("\n") + ",stray,cells\n", encoding="utf-8")
    with pytest.raises(ValueError, match="more fields than the header"):
        ledger.to_frame()


# ---------------------------------------------------------------------------
# BetLedger: closing line
# ---------------------------------------------------------------------------


def test_record_close_same_line(ledger: BetLedger) -> None:
    over = ledger.add(stake=10, **_fields())
    under = ledger.add(stake=10, **_fields(side="under", fair_prob=0.5))
    out = ledger.record_close(over, closing_over_price=1.8, closing_under_price=2.1)
    ledger.record_close(under, closing_over_price=1.8, closing_under_price=2.1)

    rows = ledger.to_frame().set_index("bet_id")
    assert rows.loc[over, "closing_line"] == 26.5
    assert rows.loc[over, "closing_price"] == 1.8 and rows.loc[over, "closing_other_price"] == 2.1
    assert rows.loc[over, "clv_price"] == pytest.approx(2.0 / 1.8 - 1)
    assert rows.loc[over, "closing_fair_prob"] == pytest.approx(2.1 / 3.9)
    assert rows.loc[over, "clv_ev"] == pytest.approx(2 * 2.1 / 3.9 - 1)
    assert out == {
        "closing_fair_prob": rows.loc[over, "closing_fair_prob"],
        "clv_price": rows.loc[over, "clv_price"],
        "clv_ev": rows.loc[over, "clv_ev"],
    }
    # The under's "own" closing price is the under price.
    assert rows.loc[under, "closing_price"] == 2.1 and rows.loc[under, "closing_other_price"] == 1.8
    assert rows.loc[under, "clv_price"] == pytest.approx(2.0 / 2.1 - 1)
    assert rows.loc[under, "closing_fair_prob"] == pytest.approx(1.8 / 3.9)
    assert rows.loc[under, "clv_ev"] == pytest.approx(2 * 1.8 / 3.9 - 1)


def test_record_close_whole_line_uses_recorded_push_probability(ledger: BetLedger) -> None:
    bet_id = ledger.add(stake=10, **_fields(line=25, fair_prob=0.45, push_prob=0.10))
    out = ledger.record_close(bet_id, 1.8, 2.1, closing_line=25.0)
    conditional = 2.1 / 3.9
    assert out["closing_fair_prob"] == pytest.approx(0.9 * conditional)
    assert out["clv_ev"] == pytest.approx(0.9 * (conditional * 2.0 - 1))
    assert out["clv_price"] == pytest.approx(2.0 / 1.8 - 1)


def test_record_close_with_one_price(ledger: BetLedger) -> None:
    bet_id = ledger.add(stake=10, **_fields())
    ours_only = ledger.record_close(bet_id, closing_over_price=1.9)
    assert ours_only["clv_price"] == pytest.approx(2.0 / 1.9 - 1)
    assert math.isnan(ours_only["clv_ev"]) and math.isnan(ours_only["closing_fair_prob"])

    # A later call replaces everything, including the price CLV that no longer applies.
    other_only = ledger.record_close(bet_id, closing_under_price=1.9)
    assert all(math.isnan(value) for value in other_only.values())
    row = ledger.to_frame().iloc[0]
    assert math.isnan(row["closing_price"]) and row["closing_other_price"] == 1.9
    assert math.isnan(row["clv_price"])


def test_record_close_when_the_line_moved(ledger: BetLedger) -> None:
    bet_id = ledger.add(stake=10, **_fields(line=24.5, price=MINUS_110))
    # Without a distribution family nothing can be said across lines.
    blind = ledger.record_close(bet_id, MINUS_110, MINUS_110, closing_line=25.5)
    assert all(math.isnan(value) for value in blind.values())
    row = ledger.to_frame().iloc[0]
    assert row["closing_line"] == 25.5 and row["closing_price"] == pytest.approx(MINUS_110)

    out = ledger.record_close(bet_id, MINUS_110, MINUS_110, closing_line=25.5, spec=NEGBIN)
    expected = clv_with_line_move(NEGBIN, "over", 24.5, MINUS_110, 25.5, MINUS_110, MINUS_110)
    assert out["clv_ev"] == expected["clv_ev"] > 0.0
    assert out["closing_fair_prob"] == expected["closing_fair_prob"] > 0.5
    assert math.isnan(out["clv_price"])  # prices at different lines are not comparable
    row = ledger.to_frame().iloc[0]
    assert row["clv_ev"] == expected["clv_ev"] and row["closing_line"] == 25.5

    # A family that cannot reproduce the close is an error, and the stored close is kept.
    with pytest.raises(ValueError, match="cannot reproduce"):
        ledger.record_close(bet_id, 1.5, 2.6, closing_line=23.5, spec=_StuckSpec("poisson"))
    row = ledger.to_frame().iloc[0]
    assert row["clv_ev"] == expected["clv_ev"] and row["closing_line"] == 25.5

    # A spec is harmless when the line did not move.
    same = ledger.record_close(bet_id, 1.8, 2.1, spec=NEGBIN)
    assert same["clv_ev"] == pytest.approx(MINUS_110 * 2.1 / 3.9 - 1)
    assert same["clv_price"] == pytest.approx(MINUS_110 / 1.8 - 1)
    same = ledger.record_close(bet_id, 1.8, 2.1, spec=_StuckSpec("poisson"))  # never consulted
    assert same["clv_ev"] == pytest.approx(MINUS_110 * 2.1 / 3.9 - 1)


def test_record_close_rejects_bad_input(ledger: BetLedger) -> None:
    bet_id = ledger.add(stake=10, **_fields())
    with pytest.raises(ValueError, match="at least one closing price"):
        ledger.record_close(bet_id)
    with pytest.raises(ValueError, match="closing_over_price"):
        ledger.record_close(bet_id, closing_over_price=-120, closing_under_price=2.0)
    with pytest.raises(ValueError, match="de-vig method"):
        ledger.record_close(bet_id, 1.9, 1.9, method="nonsense")
    with pytest.raises(KeyError, match="nope"):
        ledger.record_close("nope", 1.9, 1.9)
    assert math.isnan(ledger.to_frame().iloc[0]["closing_price"])  # nothing was stored


# ---------------------------------------------------------------------------
# BetLedger: settlement
# ---------------------------------------------------------------------------


def test_settle_from_actual_and_from_result(ledger: BetLedger) -> None:
    win = ledger.add(stake=10, **_fields(price=1.91))
    loss = ledger.add(stake=20, **_fields(side="under"))
    push = ledger.add(stake=30, **_fields(line=27))
    by_result = ledger.add(stake=40, **_fields(line=28.5, price=2.5))
    void = ledger.add(stake=50, **_fields(line=29.5))
    still_open = ledger.add(stake=60, **_fields(line=30.5))

    assert ledger.settle(win, actual=27) == pytest.approx(9.1)
    assert ledger.settle(loss, actual=27) == -20.0
    assert ledger.settle(push, actual=27) == 0.0
    assert ledger.settle(by_result, result="WIN") == pytest.approx(60.0)
    assert ledger.settle(void, actual=float("nan")) == 0.0

    rows = ledger.to_frame().set_index("bet_id")
    assert list(rows.loc[[win, loss, push, by_result, void, still_open], "result"]) == [
        "win", "loss", "push", "win", "void", "",
    ]
    assert rows.loc[win, "actual"] == 27.0 and math.isnan(rows.loc[by_result, "actual"])
    assert rows.loc[win, "pnl"] == pytest.approx(9.1) and rows.loc[void, "pnl"] == 0.0
    assert math.isnan(rows.loc[still_open, "pnl"]) and rows.loc[still_open, "settled_at"] == ""
    assert datetime.fromisoformat(rows.loc[win, "settled_at"]).utcoffset() == timedelta(0)

    assert list(ledger.open_bets()["bet_id"]) == [still_open]
    assert list(ledger.settled_bets()["bet_id"]) == [win, loss, push, by_result, void]


def test_settle_twice_needs_overwrite(ledger: BetLedger) -> None:
    bet_id = ledger.add(stake=10, **_fields())
    ledger.settle(bet_id, actual=30)
    with pytest.raises(ValueError, match="already settled"):
        ledger.settle(bet_id, actual=30)
    with pytest.raises(ValueError, match="already settled"):
        ledger.settle(bet_id, result="loss")
    assert ledger.to_frame().iloc[0]["pnl"] == 10.0

    # A stat correction: 30 becomes 26, and the old actual does not linger.
    assert ledger.settle(bet_id, actual=26, overwrite=True) == -10.0
    assert ledger.settle(bet_id, result="void", overwrite=True) == 0.0
    row = ledger.to_frame().iloc[0]
    assert row["result"] == "void" and math.isnan(row["actual"]) and row["pnl"] == 0.0


def test_settle_explicit_result_stands_over_actual(ledger: BetLedger) -> None:
    bet_id = ledger.add(stake=10, **_fields())
    assert ledger.settle(bet_id, result="win", actual=26) == 10.0  # the book paid before a correction
    row = ledger.to_frame().iloc[0]
    assert row["result"] == "win" and row["actual"] == 26.0


def test_settle_rejects_bad_input(ledger: BetLedger) -> None:
    bet_id = ledger.add(stake=10, **_fields())
    with pytest.raises(ValueError, match="result or the actual"):
        ledger.settle(bet_id)
    with pytest.raises(ValueError, match="result must be"):
        ledger.settle(bet_id, result="won")
    with pytest.raises(KeyError, match="missing"):
        ledger.settle("missing", actual=3)
    assert len(ledger.open_bets()) == 1


def test_settle_from_results_table(ledger: BetLedger) -> None:
    jokic = ledger.add(stake=10, **_fields())  # evt1, over 26.5
    other_game = ledger.add(stake=10, **_fields(event_id="evt2"))
    rebounds = ledger.add(stake=10, **_fields(market="player_rebounds", line=12, side="under"))
    no_event = ledger.add(stake=10, **_fields(event_id="", player="Jamal Murray", line=20.5))
    unmatched = ledger.add(stake=10, **_fields(player="LeBron James"))
    not_final = ledger.add(stake=10, **_fields(player="Aaron Gordon"))
    done = ledger.add(stake=10, **_fields(player="Austin Reaves"))
    ledger.settle(done, result="loss")

    results = pd.DataFrame(
        [
            {"event_id": "evt1", "player_key": "nikola jokic", "market": "player_points", "actual": 31},
            {"event_id": "evt1", "player_key": "nikola jokic", "market": "player_rebounds", "actual": 12},
            {"event_id": "evt1", "player_key": "jamal murray", "market": "player_points", "actual": 18},
            {"event_id": "evt1", "player_key": "aaron gordon", "market": "player_points", "actual": np.nan},
            {"event_id": "evt1", "player_key": "austin reaves", "market": "player_points", "actual": 40},
            {"event_id": "evt9", "player_key": "lebron james", "market": "player_points", "actual": 40},
        ]
    )
    assert ledger.settle_from_results(results) == 3

    rows = ledger.to_frame().set_index("bet_id")
    assert rows.loc[jokic, "result"] == "win" and rows.loc[jokic, "actual"] == 31.0
    assert rows.loc[rebounds, "result"] == "push" and rows.loc[rebounds, "pnl"] == 0.0
    assert rows.loc[no_event, "result"] == "loss" and rows.loc[no_event, "pnl"] == -10.0
    assert rows.loc[other_game, "result"] == ""  # a different event
    assert rows.loc[unmatched, "result"] == ""  # that player's row belongs to another event
    assert rows.loc[not_final, "result"] == ""  # NaN never voids in bulk
    assert rows.loc[done, "result"] == "loss"  # already settled: untouched
    assert ledger.settle_from_results(results) == 0  # nothing left to do


def test_settle_from_results_without_event_ids_and_by_player_name(ledger: BetLedger) -> None:
    jokic = ledger.add(stake=10, **_fields())
    doncic = ledger.add(stake=10, **_fields(player="Luka Dončić", side="under"))
    results = pd.DataFrame(
        {"player": ["Nikola Jokic", "Doncic, Luka"], "market": ["PLAYER_POINTS", "player_points"],
         "actual": ["24", 24.0]}
    )
    assert ledger.settle_from_results(results) == 2
    rows = ledger.to_frame().set_index("bet_id")
    assert rows.loc[jokic, "result"] == "loss" and rows.loc[doncic, "result"] == "win"


def test_settle_from_results_refuses_ambiguous_matches(ledger: BetLedger) -> None:
    clear = ledger.add(stake=10, **_fields(player="Jamal Murray"))
    ambiguous = ledger.add(stake=10, **_fields(event_id=""))
    results = pd.DataFrame(
        [
            {"event_id": "evt1", "player_key": "jamal murray", "market": "player_points", "actual": 30},
            {"event_id": "evt1", "player_key": "nikola jokic", "market": "player_points", "actual": 31},
            {"event_id": "evt2", "player_key": "nikola jokic", "market": "player_points", "actual": 19},
        ]
    )
    with pytest.raises(ValueError, match=f"{ambiguous}.*2 different results"):
        ledger.settle_from_results(results)
    assert len(ledger.open_bets()) == 2  # all or nothing: the clear bet was not settled either

    # Two rows that agree are not ambiguous.
    results.loc[2, "actual"] = 31
    assert ledger.settle_from_results(results) == 2
    assert list(ledger.to_frame().set_index("bet_id").loc[[clear, ambiguous], "result"]) == ["win", "win"]


def test_settle_from_results_writes_nothing_when_nothing_settles(ledger: BetLedger) -> None:
    results = pd.DataFrame({"player_key": ["nikola jokic"], "market": ["player_points"], "actual": [30]})
    assert ledger.settle_from_results(results) == 0
    assert not ledger.path.exists()  # an empty ledger stays a missing file

    ledger.add(stake=10, **_fields(player="Jamal Murray"))
    before = ledger.path.read_bytes()
    assert ledger.settle_from_results(results) == 0
    assert ledger.settle_from_results(lambda bet: None) == 0
    assert ledger.path.read_bytes() == before


def test_settle_from_results_rejects_bad_tables(ledger: BetLedger) -> None:
    ledger.add(stake=10, **_fields())
    with pytest.raises(ValueError, match="actual"):
        ledger.settle_from_results(pd.DataFrame({"player_key": ["nikola jokic"], "market": ["player_points"]}))
    with pytest.raises(ValueError, match="player_key"):
        ledger.settle_from_results(pd.DataFrame({"market": ["player_points"], "actual": [3]}))
    with pytest.raises(TypeError):
        ledger.settle_from_results([("nikola jokic", "player_points", 30)])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be a number"):
        ledger.settle_from_results(
            pd.DataFrame({"player_key": ["nikola jokic"], "market": ["player_points"], "actual": ["DNP"]})
        )
    assert len(ledger.open_bets()) == 1


def test_settle_from_results_with_callable(ledger: BetLedger) -> None:
    graded = ledger.add(stake=10, **_fields())
    voided = ledger.add(stake=10, **_fields(player="Jamal Murray"))
    waiting = ledger.add(stake=10, **_fields(player="Aaron Gordon"))
    nan_actual = ledger.add(stake=10, **_fields(player="Austin Reaves"))
    seen: List[str] = []

    def lookup(bet: pd.Series) -> object:
        seen.append(bet["bet_id"])
        assert bet["market"] == "player_points" and bet["line"] == 26.5
        return {"nikola jokic": 27, "jamal murray": "void", "austin reaves": float("nan")}.get(bet["player_key"])

    assert ledger.settle_from_results(lookup) == 2
    assert seen == [graded, voided, waiting, nan_actual]
    rows = ledger.to_frame().set_index("bet_id")
    assert list(rows.loc[[graded, voided, waiting, nan_actual], "result"]) == ["win", "void", "", ""]
    assert rows.loc[graded, "actual"] == 27.0 and rows.loc[graded, "pnl"] == 10.0


def test_remove(ledger: BetLedger) -> None:
    keep = ledger.add(stake=10, **_fields())
    mistake = ledger.add(stake=10, **_fields(line=30.5))
    ledger.remove(mistake)
    assert list(ledger.to_frame()["bet_id"]) == [keep]
    with pytest.raises(KeyError, match=mistake):
        ledger.remove(mistake)
    ledger.add(stake=10, **_fields(line=30.5))  # no longer a duplicate
    assert len(ledger.to_frame()) == 2


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


def _hand_ledger() -> pd.DataFrame:
    """Six bets whose statistics are worked out by hand in the tests."""
    rows = [
        # stake price fair  push  result  clv_price clv_ev   market           book
        (10.0, 2.0, 0.55, 0.00, "win", 0.020, 0.03, "player_points", "draftkings"),
        (20.0, 1.8, 0.60, 0.00, "loss", -0.015, -0.01, "player_points", "fanduel"),
        (10.0, 2.5, 0.40, 0.10, "push", np.nan, np.nan, "player_rebounds", "draftkings"),
        (10.0, 3.0, 0.40, 0.00, "win", np.nan, np.nan, "player_rebounds", "fanduel"),
        (10.0, 2.0, 0.52, 0.00, "", np.nan, 0.05, "player_points", "draftkings"),
        (10.0, 1.9, 0.60, 0.00, "void", 0.500, 0.50, "player_assists", "draftkings"),
    ]
    frame = pd.DataFrame(
        rows, columns=["stake", "price", "fair_prob", "push_prob", "result", "clv_price", "clv_ev", "market", "book"]
    )
    frame["ev"] = [expected_value(p, d, q) for p, d, q in zip(frame.fair_prob, frame.price, frame.push_prob)]
    return frame


def test_summarize_hand_computed_ledger() -> None:
    frame = _hand_ledger()
    before = frame.copy()
    stats = summarize(frame, n_boot=500, seed=1)
    pd.testing.assert_frame_equal(frame, before)  # the input is not modified

    assert (stats["n_bets"], stats["n_settled"], stats["n_open"]) == (6, 5, 1)
    assert (stats["wins"], stats["losses"], stats["pushes"], stats["voids"]) == (2, 1, 1, 1)
    assert stats["total_staked"] == 50.0  # graded bets only: no open, no void
    assert stats["open_stake"] == 10.0
    assert stats["pnl"] == pytest.approx(10 - 20 + 0 + 20)
    assert stats["roi"] == pytest.approx(0.20)
    assert stats["win_rate"] == pytest.approx(2 / 3)

    # Stake-weighted price of the five non-void bets: (20 + 36 + 25 + 30 + 20) / 60.
    assert stats["avg_price"] == pytest.approx(131 / 60)
    assert stats["avg_price_american"] == pytest.approx((131 / 60 - 1) * 100)

    # EV per unit: 0.10, 0.08, 0.10, 0.20 -> expected 1 + 1.6 + 1 + 2.
    assert stats["n_model"] == 4
    assert stats["expected_pnl"] == pytest.approx(5.6)
    assert stats["expected_roi"] == pytest.approx(0.112)
    assert stats["luck"] == pytest.approx(4.4)
    assert stats["luck"] == pytest.approx(stats["pnl"] - stats["expected_pnl"])
    # Variance per unit: 0.99, 0.7776, 1.39, 2.16; times stake^2 -> 99 + 311.04 + 139 + 216.
    assert stats["z_score"] == pytest.approx(4.4 / math.sqrt(765.04))
    # 4 * mean variance / mean edge^2 = 4 * 191.26 / 1.4^2 = 390.3 -> 391 bets.
    assert stats["bets_needed"] == 391.0

    # Profits 10, -20, 0, 20: mean 2.5, sample variance 875 / 3.
    t_stat = 2.5 / math.sqrt(875 / 3 / 4)
    assert stats["p_value_profit"] == pytest.approx(float(norm.sf(t_stat)))
    assert stats["p_value_profit"] == pytest.approx(0.3848, abs=5e-4)

    low, high = stats["roi_ci"]
    assert low < stats["roi"] < high
    assert -1.0 <= low and high <= 2.0  # ROI of any resample lies between the worst and best bet

    # CLV: void bet excluded; EV-CLV on bets 1, 2 and the open one, price-CLV on 1 and 2.
    assert stats["clv_n"] == 3
    assert stats["avg_clv_ev"] == pytest.approx((10 * 0.03 - 20 * 0.01 + 10 * 0.05) / 40)
    assert stats["clv_beat_rate"] == pytest.approx(2 / 3)
    assert stats["avg_clv_price"] == pytest.approx((10 * 0.02 - 20 * 0.015) / 30)


def test_summarize_bootstrap_is_seeded() -> None:
    frame = _simulated_ledger(300, seed=5)
    first = summarize(frame, n_boot=300, seed=7)["roi_ci"]
    assert summarize(frame, n_boot=300, seed=7)["roi_ci"] == first
    assert summarize(frame, n_boot=300, seed=8)["roi_ci"] != first
    assert first[0] < summarize(frame, n_boot=0)["roi"] < first[1]
    skipped = summarize(frame, n_boot=0)["roi_ci"]
    assert math.isnan(skipped[0]) and math.isnan(skipped[1])


def test_summarize_bootstrap_interval_matches_normal_theory() -> None:
    # 400 even-money unit bets, 220 winners: ROI 0.10 with standard error ~0.0497.
    frame = pd.DataFrame({"stake": 1.0, "price": 2.0, "result": ["win"] * 220 + ["loss"] * 180})
    stats = summarize(frame, n_boot=4000, seed=3)
    low, high = stats["roi_ci"]
    assert stats["roi"] == pytest.approx(0.10)
    assert low == pytest.approx(0.10 - 1.96 * 0.0497, abs=0.012)
    assert high == pytest.approx(0.10 + 1.96 * 0.0497, abs=0.012)
    assert stats["p_value_profit"] == pytest.approx(float(norm.sf(0.10 / 0.0498)), abs=2e-3)
    # No model expectation anywhere: those statistics are NaN, not zero ...
    assert stats["n_model"] == 0
    for key in ("expected_pnl", "expected_roi", "luck", "z_score"):
        assert math.isnan(stats[key])
    # ... and bets_needed falls back to the observed edge: (2 * 0.996 / 0.10)^2.
    assert stats["bets_needed"] == pytest.approx(397, abs=2)


def test_summarize_empty_inputs() -> None:
    for frame in (pd.DataFrame(), pd.DataFrame(columns=LEDGER_COLUMNS)):
        stats = summarize(frame)
        for key in ("n_bets", "n_settled", "n_open", "wins", "losses", "pushes", "voids", "n_model", "clv_n"):
            assert stats[key] == 0
        for key in ("total_staked", "open_stake", "pnl"):
            assert stats[key] == 0.0
        for key in ("roi", "win_rate", "avg_price", "avg_price_american", "expected_pnl", "expected_roi", "luck",
                    "z_score", "p_value_profit", "bets_needed", "avg_clv_price", "avg_clv_ev", "clv_beat_rate"):
            assert math.isnan(stats[key]), key
        assert all(math.isnan(bound) for bound in stats["roi_ci"])


def test_summarize_empty_ledger_file(ledger: BetLedger) -> None:
    stats = summarize(ledger.to_frame())
    assert stats["n_bets"] == 0 and math.isnan(stats["roi"])
    assert breakdown(ledger.to_frame(), "market").empty
    assert calibration_table(ledger.to_frame()).empty


def test_summarize_open_bets_only(ledger: BetLedger) -> None:
    first = ledger.add(stake=10, **_fields())
    ledger.add(stake=30, **_fields(line=27.5, price=3.0))
    ledger.record_close(first, 1.8, 2.1)
    stats = summarize(ledger.to_frame())
    assert (stats["n_bets"], stats["n_open"], stats["n_settled"]) == (2, 2, 0)
    assert stats["total_staked"] == 0.0 and stats["open_stake"] == 40.0 and stats["pnl"] == 0.0
    assert math.isnan(stats["roi"]) and math.isnan(stats["win_rate"]) and math.isnan(stats["z_score"])
    assert math.isnan(stats["expected_pnl"]) and math.isnan(stats["p_value_profit"])
    assert stats["avg_price"] == pytest.approx((10 * 2.0 + 30 * 3.0) / 40)
    assert stats["clv_n"] == 1 and stats["avg_clv_ev"] == pytest.approx(2 * 2.1 / 3.9 - 1)
    assert stats["clv_beat_rate"] == 1.0


def test_summarize_single_settled_bet() -> None:
    frame = pd.DataFrame({"stake": [10.0], "price": [2.0], "fair_prob": [0.55], "result": ["win"]})
    stats = summarize(frame)
    assert stats["pnl"] == 10.0 and stats["roi"] == 1.0 and stats["win_rate"] == 1.0
    assert stats["expected_pnl"] == pytest.approx(1.0)
    assert stats["z_score"] == pytest.approx(9.0 / math.sqrt(99.0))
    assert math.isnan(stats["p_value_profit"]) and math.isnan(stats["roi_ci"][0])


def test_summarize_only_void_bets() -> None:
    frame = pd.DataFrame({"stake": [10.0, 5.0], "price": [2.0, 1.9], "result": ["void", "void"], "clv_ev": [0.1, 0.2]})
    stats = summarize(frame)
    assert stats["n_settled"] == 2 and stats["voids"] == 2 and stats["total_staked"] == 0.0
    assert math.isnan(stats["roi"]) and math.isnan(stats["avg_price"]) and stats["clv_n"] == 0


def test_summarize_pnl_column_versus_result() -> None:
    frame = pd.DataFrame({"stake": [10.0, 10.0, 10.0], "price": [2.0, 2.0, 2.0], "result": ["win", "loss", "win"]})
    assert summarize(frame, n_boot=0)["pnl"] == 10.0  # derived from the results
    frame["pnl"] = [10.0, -10.0, np.nan]
    assert summarize(frame, n_boot=0)["pnl"] == 10.0  # a blank cell never poisons the sum
    frame["pnl"] = [10.0, -10.0, 13.0]  # e.g. a profit boost entered by hand
    assert summarize(frame, n_boot=0)["pnl"] == 13.0
    frame["result"] = ["win", "loss", ""]  # pnl of an open bet is ignored
    assert summarize(frame, n_boot=0)["pnl"] == 0.0


def test_summarize_recovers_probability_from_ev() -> None:
    # Bets recorded with an EV% only (no fair probability) still get an expectation.
    with_prob = pd.DataFrame(
        {"stake": [10.0, 20.0], "price": [2.0, 2.5], "fair_prob": [0.55, 0.40], "push_prob": [0.0, 0.1],
         "result": ["win", "loss"]}
    )
    ev_only = with_prob.drop(columns="fair_prob")
    ev_only["ev"] = [0.10, 0.40 * 1.5 - 0.5]
    expected, recovered = summarize(with_prob, n_boot=0), summarize(ev_only, n_boot=0)
    assert recovered["n_model"] == 2
    for key in ("expected_pnl", "expected_roi", "luck", "z_score", "bets_needed"):
        assert recovered[key] == pytest.approx(expected[key])
    # An EV no probability can produce at that price is ignored rather than trusted.
    ev_only["ev"] = [0.10, 5.0]
    assert summarize(ev_only, n_boot=0)["n_model"] == 1


def test_summarize_partial_model_information() -> None:
    frame = pd.DataFrame(
        {"stake": [10.0, 10.0], "price": [2.0, 2.0], "fair_prob": [0.55, np.nan], "result": ["loss", "win"]}
    )
    stats = summarize(frame, n_boot=0)
    assert stats["pnl"] == 0.0 and stats["n_model"] == 1
    assert stats["expected_pnl"] == pytest.approx(1.0) and stats["expected_roi"] == pytest.approx(0.10)
    # Luck compares like with like: only the bet that has an expectation.
    assert stats["luck"] == pytest.approx(-11.0)
    assert stats["z_score"] == pytest.approx(-11.0 / math.sqrt(99.0))


def test_summarize_rejects_bad_tables() -> None:
    good = {"stake": [10.0], "price": [2.0], "result": ["win"]}
    with pytest.raises(TypeError, match="DataFrame"):
        summarize([good])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="valid stake"):
        summarize(pd.DataFrame({**good, "stake": [0.0]}))
    with pytest.raises(ValueError, match="valid stake"):
        summarize(pd.DataFrame({**good, "price": [-110.0]}))  # American odds in a decimal column
    with pytest.raises(ValueError, match="valid stake"):
        summarize(pd.DataFrame({"price": [2.0], "result": ["win"]}))  # no stake column at all
    with pytest.raises(ValueError, match="unknown result"):
        summarize(pd.DataFrame({**good, "result": ["won"]}))
    with pytest.raises(ValueError, match="'fair_prob' must be numeric"):
        summarize(pd.DataFrame({**good, "fair_prob": ["high"]}))
    with pytest.raises(ValueError, match="fair_prob must be in"):
        summarize(pd.DataFrame({**good, "fair_prob": [1.3]}))
    with pytest.raises(ValueError, match="fair_prob must be in"):
        summarize(pd.DataFrame({**good, "fair_prob": [0.95], "push_prob": [0.2]}))


def test_summarize_tolerates_missing_markers() -> None:
    frame = pd.DataFrame(
        {"stake": ["10", "10", "10"], "price": [2.0, 2.0, 2.0], "result": [None, np.nan, " Win "],
         "fair_prob": ["", None, "0.5"], "clv_ev": [np.inf, None, 0.02]}
    )
    stats = summarize(frame, n_boot=0)
    assert (stats["n_bets"], stats["n_open"], stats["wins"]) == (3, 2, 1)
    assert stats["pnl"] == 10.0 and stats["n_model"] == 1 and stats["clv_n"] == 1


def _simulated_ledger(n: int, seed: int, overconfidence: float = 0.0) -> pd.DataFrame:
    """Bets whose outcomes are drawn from ``fair_prob - overconfidence``."""
    rng = np.random.default_rng(seed)
    push = np.where(rng.random(n) < 0.2, 0.06, 0.0)
    fair = rng.uniform(0.30, 0.65, n) * (1.0 - push)
    target_ev = rng.uniform(0.01, 0.07, n)
    price = (1.0 + target_ev - push) / fair  # so that EV per unit equals target_ev
    stake = rng.choice([1.0, 2.0, 5.0], n)
    draw = rng.random(n)
    true = fair - overconfidence
    result = np.where(draw < true, "win", np.where(draw < true + push, "push", "loss"))
    return pd.DataFrame(
        {"stake": stake, "price": price, "fair_prob": fair, "push_prob": push, "ev": target_ev, "result": result,
         "market": rng.choice(["player_points", "player_rebounds", "player_assists"], n)}
    )


def test_summarize_simulated_ledger_is_consistent_with_its_edge() -> None:
    frame = _simulated_ledger(5000, seed=20261112)
    stats = summarize(frame)

    assert stats["n_bets"] == stats["n_settled"] == stats["n_model"] == 5000
    assert stats["wins"] + stats["losses"] + stats["pushes"] == 5000 and stats["pushes"] > 0
    assert stats["expected_roi"] == pytest.approx(0.04, abs=0.003)
    assert stats["expected_pnl"] == pytest.approx(float((frame.stake * frame.ev).sum()))
    assert stats["luck"] == pytest.approx(stats["pnl"] - stats["expected_pnl"])

    # Outcomes follow the fair probabilities, so the results must look like the model said.
    assert -3.0 < stats["z_score"] < 3.0
    assert stats["roi"] == pytest.approx(stats["expected_roi"], abs=0.05)
    low, high = stats["roi_ci"]
    assert low < stats["roi"] < high and 0.02 < high - low < 0.12
    assert low - 0.03 < stats["expected_roi"] < high + 0.03
    assert 0.0 <= stats["p_value_profit"] <= 1.0

    # The z-score is exactly the hand formula on each bet's own variance.
    won, lost = frame.result == "win", frame.result == "loss"
    pnl = np.where(won, frame.stake * (frame.price - 1), np.where(lost, -frame.stake, 0.0))
    variance = [bet_variance(p, d, q) for p, d, q in zip(frame.fair_prob, frame.price, frame.push_prob)]
    by_hand = (pnl.sum() - (frame.stake * frame.ev).sum()) / math.sqrt(float((frame.stake**2 * variance).sum()))
    assert stats["z_score"] == pytest.approx(by_hand)
    assert stats["pnl"] == pytest.approx(float(pnl.sum()))

    # A 4% edge at roughly even money needs thousands of bets to show.
    assert 1500 < stats["bets_needed"] < 6000


def test_summarize_z_score_is_standard_normal_across_simulations() -> None:
    scores = np.array(
        [summarize(_simulated_ledger(400, seed), n_boot=0)["z_score"] for seed in range(300)]
    )
    assert abs(scores.mean()) < 0.2
    assert 0.85 < scores.std(ddof=1) < 1.15
    assert np.mean(np.abs(scores) > 1.96) < 0.10


def test_summarize_z_score_exposes_an_overconfident_model() -> None:
    stats = summarize(_simulated_ledger(5000, seed=20261112, overconfidence=0.05), n_boot=0)
    assert stats["z_score"] < -4.0  # the claimed edge is not there
    assert stats["roi"] < 0.0 < stats["expected_roi"]
    assert stats["luck"] < 0.0 and stats["p_value_profit"] > 0.5


def test_bets_needed() -> None:
    assert bets_needed(0.05, 1.0) == 1600.0
    assert bets_needed(-0.05, 1.0) == 1600.0  # a losing edge takes as long to prove
    assert bets_needed(0.03, 0.95) == math.ceil((2 * 0.95 / 0.03) ** 2)
    assert bets_needed(0.05, 1.0, n_sigma=3.0) == 3600.0
    assert bets_needed(0.0, 1.0) == math.inf
    assert bets_needed(0.05, 0.0) == 0.0
    assert math.isnan(bets_needed(float("nan"), 1.0))
    assert math.isnan(bets_needed(0.05, float("nan")))
    assert math.isnan(bets_needed(0.05, -1.0))


# ---------------------------------------------------------------------------
# breakdown / calibration_table
# ---------------------------------------------------------------------------


def test_breakdown_matches_summarize_per_group() -> None:
    frame = _hand_ledger()
    table = breakdown(frame, by="market")
    assert list(table["market"]) == ["player_points", "player_rebounds", "player_assists"]  # largest first
    assert list(table["n_bets"]) == [3, 2, 1]
    assert {"n_settled", "wins", "losses", "pushes", "total_staked", "pnl", "roi", "win_rate", "avg_price",
            "expected_roi", "z_score", "avg_clv_ev", "clv_beat_rate", "clv_n"} <= set(table.columns)
    for _, row in table.iterrows():
        expected = summarize(frame[frame.market == row["market"]], n_boot=0)
        for column in table.columns.drop("market"):
            assert row[column] == pytest.approx(expected[column], nan_ok=True), (row["market"], column)

    points = table.set_index("market").loc["player_points"]
    assert points["pnl"] == pytest.approx(-10.0) and points["roi"] == pytest.approx(-10 / 30)
    assert points["total_staked"] == 30.0 and points["clv_n"] == 3
    assists = table.set_index("market").loc["player_assists"]  # a void bet only
    assert assists["n_settled"] == 1 and math.isnan(assists["roi"]) and assists["pnl"] == 0.0

    assert breakdown(frame, by="book")["pnl"].sum() == pytest.approx(summarize(frame, n_boot=0)["pnl"])


def test_breakdown_by_several_columns_and_bad_input() -> None:
    frame = _hand_ledger()
    table = breakdown(frame, by=["book", "market"])
    assert list(table.columns[:2]) == ["book", "market"]
    assert len(table) == 5 and table["n_bets"].sum() == 6
    top = table.iloc[0]
    assert (top["book"], top["market"], top["n_bets"]) == ("draftkings", "player_points", 2)

    with pytest.raises(ValueError, match="bookie"):
        breakdown(frame, by="bookie")
    with pytest.raises(TypeError):
        breakdown("market")  # type: ignore[arg-type]
    empty = breakdown(frame.iloc[:0], by="market")
    assert empty.empty and "roi" in empty.columns


def test_breakdown_of_a_real_ledger(ledger: BetLedger) -> None:
    ids = _populate(ledger)
    ledger.settle(ids[2], actual=20)
    table = breakdown(ledger.to_frame(), by="side").set_index("side")
    assert table.loc["over", "n_bets"] == 2 and table.loc["under", "n_bets"] == 1
    assert table.loc["over", "pnl"] == pytest.approx(25 * 0.91 - 1 / 3)
    assert table.loc["under", "pushes"] == 1


def _calibration_bets() -> pd.DataFrame:
    rows = (
        [(0.55, 0.0, "win")] * 6
        + [(0.55, 0.0, "loss")] * 4
        + [(0.72, 0.0, "win")] * 3
        + [(0.72, 0.0, "loss")] * 2
        + [(0.72, 0.0, "push"), (0.72, 0.0, "void"), (0.72, 0.0, "")]  # never counted
        + [(0.60, 0.0, "win")]  # an edge belongs to the bucket above it
        + [(0.54, 0.10, "loss")]  # 0.54 / 0.9 = 0.60 given no push
        + [(1.00, 0.0, "win")]  # top edge belongs to the last bucket
        + [(np.nan, 0.0, "win")]  # no prediction, nothing to calibrate
    )
    frame = pd.DataFrame(rows, columns=["fair_prob", "push_prob", "result"])
    frame["stake"], frame["price"] = 10.0, 1.9
    return frame


def test_calibration_table() -> None:
    table = calibration_table(_calibration_bets(), bins=10)
    assert list(table.columns) == ["prob_low", "prob_high", "n", "wins", "predicted", "actual", "diff", "z"]
    assert list(table["prob_low"]) == [0.5, 0.6, 0.7, 0.9]
    assert list(table["prob_high"]) == [0.6, 0.7, 0.8, 1.0]
    assert list(table["n"]) == [10, 2, 5, 1]
    assert list(table["wins"]) == [6, 1, 3, 1]
    assert table["n"].sum() == 18  # push, void, open and unpredicted bets are left out

    first = table.iloc[0]
    assert first["predicted"] == pytest.approx(0.55) and first["actual"] == pytest.approx(0.6)
    assert first["diff"] == pytest.approx(0.05)
    assert first["z"] == pytest.approx(0.5 / math.sqrt(10 * 0.55 * 0.45))
    assert table.iloc[1]["predicted"] == pytest.approx(0.60)
    third = table.iloc[2]
    assert third["actual"] == pytest.approx(0.6) and third["diff"] == pytest.approx(-0.12)
    assert third["z"] == pytest.approx((3 - 3.6) / math.sqrt(5 * 0.72 * 0.28))
    assert math.isnan(table.iloc[3]["z"])  # a certain prediction has no spread


def test_calibration_table_custom_bins_and_bad_input() -> None:
    bets = _calibration_bets()
    two = calibration_table(bets, bins=2)
    assert list(two["prob_low"]) == [0.5] and two.iloc[0]["n"] == 18 and two.iloc[0]["wins"] == 11

    edges = calibration_table(bets, bins=[0.5, 0.58, 0.75])
    assert list(edges["n"]) == [10, 7]  # the certain bet is outside the given edges
    assert list(edges["prob_high"]) == [0.58, 0.75]

    for bad in (0, [0.5], [0.6, 0.5], [0.1, float("nan")]):
        with pytest.raises(ValueError, match="bin"):
            calibration_table(bets, bins=bad)

    nothing = calibration_table(bets[bets.result == ""])
    assert nothing.empty and list(nothing.columns) == list(two.columns)


def test_calibration_of_a_simulated_ledger_is_good() -> None:
    table = calibration_table(_simulated_ledger(5000, seed=20261112), bins=10)
    assert table["n"].sum() == 5000 - (_simulated_ledger(5000, seed=20261112).result == "push").sum()
    assert (table["z"].abs() < 3.5).all()
    assert (table.loc[table["n"] > 200, "diff"].abs() < 0.06).all()
    biased = calibration_table(_simulated_ledger(5000, seed=20261112, overconfidence=0.05), bins=10)
    assert (biased.loc[biased["n"] > 200, "diff"] < 0.0).all()  # wins fall short of predictions everywhere


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_ledger_to_summary_end_to_end(ledger: BetLedger) -> None:
    over = ledger.add(stake=10, **_fields())  # over 26.5 at 2.0, fair 0.55
    under = ledger.add(stake=20, **_fields(player="Jamal Murray", line=20.5, side="under", price=1.8,
                                           fair_prob=0.60))
    whole = ledger.add(stake=10, **_fields(player="Aaron Gordon", line=14, price=2.5, fair_prob=0.40,
                                           push_prob=0.10))
    ledger.add(stake=10, **_fields(player="LeBron James", event_id="evt2", fair_prob=0.52))
    ledger.record_close(over, 1.8, 2.1)
    ledger.record_close(under, 1.8, 2.1)
    results = pd.DataFrame(
        {"event_id": "evt1", "player_key": ["nikola jokic", "jamal murray", "aaron gordon"],
         "market": "player_points", "actual": [30, 24, 14]}
    )
    assert ledger.settle_from_results(results) == 3

    stats = summarize(BetLedger(ledger.path).to_frame(), n_boot=200)
    assert (stats["n_bets"], stats["n_open"], stats["wins"], stats["losses"], stats["pushes"]) == (4, 1, 1, 1, 1)
    assert stats["total_staked"] == 40.0 and stats["pnl"] == pytest.approx(-10.0)
    assert stats["roi"] == pytest.approx(-0.25)
    assert stats["expected_pnl"] == pytest.approx(10 * 0.10 + 20 * 0.08 + 10 * 0.10)
    assert stats["luck"] == pytest.approx(-13.6)
    assert stats["z_score"] == pytest.approx(-13.6 / math.sqrt(99 + 311.04 + 139))
    assert stats["clv_n"] == 2
    assert stats["avg_clv_ev"] == pytest.approx((10 * (2 * 2.1 / 3.9 - 1) + 20 * (1.8 * 1.8 / 3.9 - 1)) / 30)
    assert stats["clv_beat_rate"] == 0.5
    assert whole in set(ledger.settled_bets()["bet_id"])
