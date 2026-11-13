"""Fee schedules and settlement rules, as data.

A price only means something together with the rules it settles under and
the fees charged on it.  Both are kept here as named, dated records so that
an evaluation can say exactly which assumptions it used - and so they can be
edited when a venue changes its terms.

**Verify before relying on any of this.**  Fee schedules were read from the
venues' published pages on the date shown; venues change them, and Kalshi and
Polymarket publish per-series / per-market parameters that override the
defaults.  Settlement rules differ between sportsbooks; the rule sets below
are the common conventions, selectable per quote with ``rules_key``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from ..core.results import flag, regulation_result

__all__ = ["FeeSchedule", "FEES", "fee_for", "SettlementRules", "RULES", "comparable", "settle_outcome"]


@dataclass(frozen=True)
class FeeSchedule:
    key: str
    venue: str
    kind: str                      # none | quadratic | commission
    rate: float = 0.0              # quadratic: fee = rate * contracts * p * (1 - p); commission: share of net winnings
    round_up_cent: bool = False
    source: str = ""
    verified: str = ""             # date the source was read
    note: str = ""

    def per_contract(self, price: float, contracts: float = 1.0) -> float:
        """Fee per contract when buying ``contracts`` at ``price`` (0..1)."""
        if self.kind != "quadratic":
            return 0.0
        total = self.rate * contracts * price * (1.0 - price)
        if self.round_up_cent:
            total = math.ceil(total * 100.0 - 1e-9) / 100.0
        return total / contracts


FEES: Dict[str, FeeSchedule] = {f.key: f for f in (
    FeeSchedule("sportsbook", "sportsbook", "none",
                note="The margin is inside the odds; no separate fee."),
    FeeSchedule("kalshi_taker", "kalshi", "quadratic", rate=0.07, round_up_cent=True,
                source="https://kalshi.com/fee-schedule", verified="2026-11-12",
                note="Taker fee = round up to the cent(0.07 x contracts x price x (1 - price)), times a per-series "
                     "multiplier (default 1). Some series also charge maker fees; read fee_type and the multiplier "
                     "from the series before relying on this default."),
    FeeSchedule("polymarket_us_taker", "polymarket_us", "quadratic", rate=0.0695,
                source="https://docs.polymarket.us/fees", verified="2026-11-12",
                note="Taker fee = 0.0695 x contracts x price x (1 - price). Volume rebates are ignored."),
    FeeSchedule("polymarket_intl_sports_taker", "polymarket", "quadratic", rate=0.05,
                source="https://docs.polymarket.com/trading/fees", verified="2026-11-12",
                note="Sports taker fee = 0.05 x shares x price x (1 - price); makers pay nothing. US persons may "
                     "not trade on the international platform. Read the market's own fee parameters."),
    FeeSchedule("exchange_2pct", "exchange", "commission", rate=0.02,
                note="Illustrative exchange commission of 2% of net winnings; set the rate your exchange charges."),
)}


def fee_for(key: Optional[str]) -> FeeSchedule:
    if not key:
        return FEES["sportsbook"]
    if key not in FEES:
        raise KeyError(f"unknown fee schedule {key!r}; known: {sorted(FEES)}")
    return FEES[key]


@dataclass(frozen=True)
class SettlementRules:
    key: str
    description: str
    market: str                    # the prediction market key these rules can be compared with
    includes_overtime: Optional[bool] = None
    draw: str = "push"             # winner markets when the event is drawn: push | loss | half | n/a
    exact_line: str = "push"       # spread / total landing exactly on the line: push | loss
    retirement: str = "void"       # tennis: void | action | void_before_first_set
    canceled: str = "void"         # void | half (prediction markets often settle 50-50)
    dead_heat: str = "divide"      # field events
    note: str = ""
    sports: Tuple[str, ...] = ()   # sports these rules belong to; empty = any sport


RULES: Dict[str, SettlementRules] = {r.key: r for r in (
    SettlementRules("moneyline_incl_ot", "Game winner, overtime and shootout included; a tie is a push.",
                    market="winner", includes_overtime=True, draw="push"),
    SettlementRules("moneyline_incl_ot_tie_loses", "Game winner incl. overtime; a tie loses (some books, NFL).",
                    market="winner", includes_overtime=True, draw="loss"),
    SettlementRules("three_way_regulation", "Home / draw / away at the end of regulation time.",
                    market="winner_3way", includes_overtime=False, draw="n/a"),
    SettlementRules("spread_incl_ot", "Point spread, overtime included; landing on the line is a push.",
                    market="spread", includes_overtime=True),
    SettlementRules("total_incl_ot", "Total, overtime included; landing on the line is a push.",
                    market="total", includes_overtime=True),
    SettlementRules("tennis_void_on_retirement", "Match winner; any retirement or walkover voids the bet.",
                    market="winner", retirement="void", sports=("tennis",)),
    SettlementRules("tennis_action_after_first_set",
                    "Match winner; a retirement after one completed set stands, before that the bet is void.",
                    market="winner", retirement="void_before_first_set", sports=("tennis",)),
    SettlementRules("tennis_action_on_retirement", "Match winner; the player who advances wins, however it ended.",
                    market="winner", retirement="action", sports=("tennis",)),
    SettlementRules("mma_draw_void", "Fight winner; a draw or no contest voids the bet.", market="winner", draw="push",
                    sports=("mma", "boxing")),
    SettlementRules("prediction_market_winner_incl_ot",
                    "Winner contract, overtime included; a tie or a cancelled event settles 50-50.",
                    market="winner", includes_overtime=True, draw="half", canceled="half",
                    note="Kalshi and Polymarket publish per-market rules (postponement windows differ); read them."),
    SettlementRules("player_prop_void_if_dnp", "Player statistic over/under; void if the player does not play.",
                    market="player_prop"),
)}


def comparable(prediction_market: str, rules: SettlementRules, sport: Optional[str] = None) -> Tuple[bool, str]:
    """Can a quote governed by ``rules`` be compared with a prediction of
    ``prediction_market``?  A model probability for "wins including overtime"
    says nothing about a regulation-time three-way price, and vice versa.

    Two things are checked: the market family, and - for rule sets written
    for particular sports - the sport, when it is given.  Nothing else is:
    whether the venue's own house rules match the rule set is for the person
    supplying the quote to verify."""
    family = prediction_market.split(":", 1)[0]
    if family != rules.market:
        return False, (f"the quote settles as {rules.market!r} ({rules.description}) but the prediction is for "
                       f"{prediction_market!r}")
    if sport is not None and rules.sports and sport not in rules.sports:
        return False, (f"the rule set {rules.key!r} is written for {' / '.join(rules.sports)}, not for {sport}")
    return True, ""


def settle_outcome(rules: SettlementRules, market: str, selection: str, line: Optional[float],
                   event: Mapping[str, Any]) -> Tuple[str, str]:
    """Outcome of a position under ``rules``: ``win | loss | push | void | half``
    plus a note, or ``("unsettled", reason)`` when the data on file cannot
    decide it.  ``event`` is an event row joined with its result."""
    status = event.get("status")
    if status in ("canceled", "postponed"):
        return ("half" if rules.canceled == "half" else "void"), f"event {status}"
    if status != "final":
        return "unsettled", f"event is {status}"
    family = market.split(":", 1)[0]
    method = str(event.get("method") or "").upper()
    hs, as_ = event.get("home_score"), event.get("away_score")
    have_scores = hs is not None and hs == hs and as_ is not None and as_ == as_
    if method in ("W/O", "WO", "WALKOVER"):
        return "void", "walkover: no match was played"
    if method in ("NC", "NO CONTEST"):
        return "void", "no contest"
    if method.startswith("RET"):
        if rules.retirement == "void":
            return "void", "retirement voids the bet under these rules"
        if rules.retirement == "void_before_first_set":
            detail = event.get("detail_json")
            sets = None
            if isinstance(detail, str) and "sets_completed" in detail:
                import json
                sets = json.loads(detail).get("sets_completed")
            if sets is None:
                return "unsettled", "retirement: completed sets are not on file, settle by hand"
            if int(sets) < 1:
                return "void", "retirement before a set was completed"
    if family == "winner":
        if flag(event.get("is_draw")):
            return {"push": "push", "loss": "loss", "half": "half"}.get(rules.draw, "push"), "event was drawn"
        if not event.get("winner_id"):
            return "unsettled", "no winner on file"
        return ("win" if selection == event["winner_id"] else "loss"), ""
    if family == "winner_3way":
        actual, basis = regulation_result(event)
        if actual is None:
            return "unsettled", basis
        return ("win" if selection == actual else "loss"), f"regulation result: {actual} ({basis})"
    if family in ("spread", "total"):
        if not have_scores or line is None or line != line:
            return "unsettled", "score or line missing"
        if family == "spread":
            value = float(hs) - float(as_) + float(line)          # line is the home side's handicap
            if value == 0:
                return ("push" if rules.exact_line == "push" else "loss"), "landed on the line"
            return ("win" if (selection == event["home_id"]) == (value > 0) else "loss"), ""
        total = float(hs) + float(as_)
        if total == float(line):
            return ("push" if rules.exact_line == "push" else "loss"), "landed on the line"
        return ("win" if (selection == "over") == (total > float(line)) else "loss"), ""
    return "unsettled", f"no settlement logic for market {market!r}"
