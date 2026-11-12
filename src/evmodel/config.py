"""Configuration: every tunable default in one place.

``Config()`` gives sensible defaults.  Override them in code, or keep a TOML
file and load it with :func:`load_config`::

    [consensus]
    devig_method = "power"
    min_ref_books = 2

    [consensus.book_weights]
    pinnacle = 1.0
    fanduel = 0.8

    [scan]
    min_ev = 0.03
    target_books = ["draftkings", "betmgm"]

    [staking]
    kelly_fraction = 0.25
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

__all__ = [
    "ConsensusConfig",
    "ScanConfig",
    "StakingConfig",
    "Config",
    "load_config",
    "DEFAULT_BOOK_WEIGHTS",
    "DEFAULT_DFS_BOOKS",
]

#: How much each book counts towards the "fair" consensus.  Higher = sharper
#: (prices that are harder to beat).  Books not listed get weight 0: they are
#: scanned for value but do not define the fair price.  These are judgement
#: calls - tune them to the sports and markets you bet.
DEFAULT_BOOK_WEIGHTS: Dict[str, float] = {
    "pinnacle": 1.0,
    "circasports": 1.0,
    "bookmaker": 0.9,
    "novig": 0.8,
    "prophetx": 0.8,
    "sporttrade": 0.7,
    "fanduel": 0.7,
    "betonlineag": 0.6,
    "draftkings": 0.4,
    "lowvig": 0.4,
    "betrivers": 0.2,
    "betmgm": 0.2,
    "williamhill_us": 0.2,
}

#: Daily-fantasy pick'em operators.  They have no real per-leg price, so a leg
#: is valued against the per-leg break-even probability of the entry type you
#: intend to play; the number stored here is that break-even.  See
#: :mod:`evmodel.dfs` for how these are derived from payout tables.
DEFAULT_DFS_BOOKS: Dict[str, float] = {
    "prizepicks": 0.5425,
    "underdog": 0.5434,
    "sleeper": 0.5434,
    "betr_us_dfs": 0.5434,
    "pick6": 0.5434,
    "parlayplay": 0.5434,
    "dabble_au": 0.5434,
}


@dataclass
class ConsensusConfig:
    """How fair probabilities are built from reference books."""

    book_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_BOOK_WEIGHTS))
    #: de-vig method applied to each reference book's two-way price.  For a
    #: two-way market "shin" equals "additive" and sits in the middle of the
    #: range the methods span.
    devig_method: str = "shin"
    #: minimum number of weighted reference books (excluding the book being
    #: evaluated) for a "sharp" consensus
    min_ref_books: int = 1
    #: minimum total weight of those reference books
    min_ref_weight: float = 0.6
    #: once other books support a consensus, also pool in the evaluated book's
    #: own de-vigged quote at its own weight.  This shrinks apparent edges
    #: against sharp books (whose price is itself strong evidence) and leaves
    #: zero-weight soft books untouched.
    include_target_book: bool = True
    #: if no sharp consensus is available, fall back to the average of all
    #: other books ...
    fallback_market_average: bool = True
    #: ... provided at least this many of them quote the prop
    fallback_min_books: int = 3
    #: reference quotes with a hold above this are ignored (illiquid or
    #: alternate-style pricing says little about the fair price)
    max_ref_hold: float = 0.10
    #: reference quotes whose implied probabilities sum below this are treated
    #: as bad data (a two-way market cannot be that far under 100%)
    min_ref_total: float = 0.98
    #: references at a different line are translated through the market's
    #: distribution; the gap is measured in standard deviations of the stat
    max_line_gap_sd: float = 0.35
    #: weight multiplier for a translated reference: exp(-gap_sd / scale)
    line_gap_scale: float = 0.25
    #: drop reference quotes older than this many minutes relative to the
    #: newest quote in the snapshot (None = keep all)
    max_quote_age_minutes: Optional[float] = None
    #: books that are never used as references (DFS operators are added
    #: automatically)
    exclude_reference_books: List[str] = field(default_factory=list)


@dataclass
class ScanConfig:
    """Which opportunities are reported."""

    #: minimum expected value per unit staked (0.02 = +2%)
    min_ev: float = 0.02
    #: stricter threshold when the fair price had to be translated from a
    #: different line or comes from the market-average fallback
    min_ev_low_confidence: float = 0.04
    #: a bet must also clear this EV under whichever de-vig method is least
    #: favourable to it, so no edge exists merely because of how the vig was
    #: removed (set to None to disable)
    min_ev_worst_case: Optional[float] = 0.0
    #: only prices within this American-odds window are considered
    min_price_american: float = -300.0
    max_price_american: float = 300.0
    #: books to look for bets at (None = every book in the snapshot)
    target_books: Optional[List[str]] = None
    #: skip alternate lines when scanning for bets
    include_alternates: bool = True
    #: weight of the projection model when a market consensus also exists
    model_weight: float = 0.2
    #: model weight is scaled down below this many games of history
    model_full_weight_games: int = 20
    #: report bets supported only by the projection model (no consensus)
    allow_model_only: bool = False
    #: minimum EV for model-only bets
    min_ev_model_only: float = 0.06
    #: keep only the best price for each (prop, side)
    best_price_only: bool = True
    #: flag consensus built from references that disagree by more than this
    #: (weighted sd, probability points)
    disagreement_flag: float = 0.03
    #: ignore events starting within this many minutes (lines move fast and a
    #: snapshot is likely stale by the time a bet is placed); None disables
    min_minutes_to_start: Optional[float] = None
    #: DFS pick'em books and their per-leg break-even probability
    dfs_books: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_DFS_BOOKS))


@dataclass
class StakingConfig:
    """Bet sizing."""

    #: fraction of full Kelly to stake
    kelly_fraction: float = 0.25
    #: one unit as a fraction of bankroll (0.01 = 1%)
    unit_size: float = 0.01
    #: hard cap per bet, in units
    max_units: float = 2.0
    #: smallest stake worth placing, in units (smaller is rounded to 0)
    min_units: float = 0.1
    #: stakes are rounded to this increment, in units
    round_to: float = 0.05
    #: extra multiplier for low-confidence bets (translated line, market
    #: average, single reference, model only)
    low_confidence_multiplier: float = 0.5
    #: maximum total units on one player and on one game
    max_units_per_player: float = 2.0
    max_units_per_event: float = 4.0
    #: maximum total units across one scan (None = unlimited)
    max_units_total: Optional[float] = 25.0


@dataclass
class Config:
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)
    staking: StakingConfig = field(default_factory=StakingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: _as_dict(getattr(self, f.name)) for f in fields(self)}


def _as_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _as_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {k: _as_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_as_dict(v) for v in obj]
    return obj


def _update(section: Any, values: Mapping[str, Any], path: str) -> None:
    known = {f.name for f in fields(section)}
    for key, value in values.items():
        if key not in known:
            raise KeyError(f"unknown config option [{path}] {key!r}; known: {sorted(known)}")
        current = getattr(section, key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            # tables replace the default table entirely so a user can drop books
            setattr(section, key, {str(k).lower(): float(v) for k, v in value.items()})
        else:
            setattr(section, key, value)


def config_from_dict(data: Mapping[str, Any]) -> Config:
    """Build a :class:`Config` from a nested mapping (as parsed from TOML)."""
    config = Config()
    for name, values in data.items():
        if not hasattr(config, name):
            raise KeyError(f"unknown config section [{name}]; known: consensus, scan, staking")
        if not isinstance(values, Mapping):
            raise TypeError(f"config section [{name}] must be a table")
        _update(getattr(config, name), values, name)
    return config


def load_config(path: Union[str, Path, None]) -> Config:
    """Load configuration from a TOML file (``None`` returns the defaults)."""
    if path is None:
        return Config()
    with open(path, "rb") as fh:
        return config_from_dict(tomllib.load(fh))
