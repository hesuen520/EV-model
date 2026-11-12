"""evmodel - find +EV player-prop bets.

The pieces, in the order a bet flows through them:

* :mod:`evmodel.odds`, :mod:`evmodel.devig` - prices and vig removal;
* :mod:`evmodel.market` - fair probabilities from sharp books' prices;
* :mod:`evmodel.projection` - a statistical second opinion from game logs;
* :mod:`evmodel.scanner` - expected value, ranking and stake sizing;
* :mod:`evmodel.tracking` - the ledger that tells you whether it works.

Nothing here guarantees profit.  Expected value is an estimate that is only
as good as its inputs, sportsbooks limit winning accounts, and betting should
only ever be done with money you can afford to lose.
"""

__version__ = "0.1.0"

from .config import Config, load_config
from .devig import devig, devig_two_way
from .distributions import FamilySpec
from .ev import expected_value, fractional_kelly, kelly_fraction
from .market import MarketEngine
from .odds import american_to_decimal, decimal_to_american, parse_odds
from .scanner import evaluate, scan

__all__ = [
    "__version__",
    "Config",
    "load_config",
    "devig",
    "devig_two_way",
    "FamilySpec",
    "expected_value",
    "kelly_fraction",
    "fractional_kelly",
    "MarketEngine",
    "american_to_decimal",
    "decimal_to_american",
    "parse_odds",
    "evaluate",
    "scan",
]
