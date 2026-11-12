"""Optional price comparison (EV) module - separate from the prediction engine.

Sports-based probabilities are produced first, by code that cannot import
this package.  Only afterwards, and only if prices are supplied, are they
compared with timestamped executable quotes.  Paper tracking only: nothing in
this package can place a bet.
"""

from .core import (
    PRICE_TEMPLATE,
    PriceImportError,
    evaluate,
    evaluate_one,
    import_prices,
    no_vig,
    open_paper_positions,
    paper_report,
    raw_implied,
    settle_paper_positions,
)
from .rules import FEES, RULES, FeeSchedule, SettlementRules, comparable, fee_for, settle_outcome

__all__ = [
    "PRICE_TEMPLATE", "PriceImportError", "evaluate", "evaluate_one", "import_prices", "no_vig",
    "open_paper_positions", "paper_report", "raw_implied", "settle_paper_positions", "FEES", "RULES",
    "FeeSchedule", "SettlementRules", "comparable", "fee_for", "settle_outcome",
]
