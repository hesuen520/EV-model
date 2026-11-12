"""Registry of player-prop markets.

A *market* is a kind of prop ("player points", "receiving yards").  Each
:class:`MarketDef` records

* the canonical key (The Odds API's market key is used as the canonical
  name, e.g. ``player_points``; other feeds are mapped onto it via aliases),
* which game-log stat columns add up to the prop's result,
* the predictive :class:`~evmodel.distributions.FamilySpec` used for that
  stat (shape and how variance grows with the mean).

The dispersion numbers for NBA and NFL markets were fitted on real game logs
with :mod:`evmodel.projection.calibrate`; they live in
``evmodel/data/market_specs.json`` and can be re-fitted with
``evmodel calibrate``.  Markets marked ``calibrated=False`` (NHL, MLB) use
reasonable generic shapes that have **not** been fitted - the market engine
only relies on them when it must translate between different lines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from importlib import resources
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import pandas as pd

from .distributions import FamilySpec

__all__ = [
    "MarketDef",
    "MARKETS",
    "get_market",
    "find_market",
    "canonical_market_key",
    "markets_for_sport",
    "stat_value",
    "load_spec_overrides",
    "DEFAULT_SPEC",
]

#: Used for markets the registry does not know: a mildly over-dispersed count.
DEFAULT_SPEC = FamilySpec("negbin", phi=1.0, alpha=0.08)

_ALT_SUFFIX = "_alternate"


@dataclass(frozen=True)
class MarketDef:
    """Definition of one prop market for one sport."""

    sport: str
    key: str
    label: str
    stats: Tuple[str, ...]
    spec: FamilySpec = DEFAULT_SPEC
    yes_no: bool = False  # "anytime TD"-style: line 0.5, over == yes
    aliases: Tuple[str, ...] = ()
    calibrated: bool = False
    #: positions (provider codes) for which the market is normally offered
    positions: Tuple[str, ...] = ()
    #: multiplier applied to the walk-forward projection's mean, fitted on
    #: history (players who take the floor slightly out-produce their
    #: weighted past average)
    mean_scale: float = 1.0
    #: books rarely post the market for players projected below this mean;
    #: used to pick realistic player-games when back-testing calibration
    min_mean: float = 0.0

    def value(self, row: Mapping[str, object]) -> float:
        """Result of the prop for one game-log row (sum of its stat columns)."""
        total = 0.0
        for col in self.stats:
            v = row.get(col) if hasattr(row, "get") else row[col]
            if v is None or pd.isna(v):
                return float("nan")
            total += float(v)
        return total


def _nb(alpha: float, phi: float = 1.0) -> FamilySpec:
    return FamilySpec("negbin", phi=phi, alpha=alpha)


def _poisson() -> FamilySpec:
    return FamilySpec("poisson")


def _m(sport, key, label, stats, spec, **kw) -> MarketDef:
    if isinstance(stats, str):
        stats = (stats,)
    return MarketDef(sport=sport, key=key, label=label, stats=tuple(stats), spec=spec, **kw)


# Provisional shapes; fitted values from market_specs.json override them at
# import time (see _apply_packaged_overrides below).
_DEFS: List[MarketDef] = [
    # ---------------------------------------------------------------- NBA
    _m("nba", "player_points", "Points", "points", _nb(0.03, 1.5)),
    _m("nba", "player_rebounds", "Rebounds", "rebounds", _nb(0.03, 1.2)),
    _m("nba", "player_assists", "Assists", "assists", _nb(0.04, 1.2)),
    _m("nba", "player_threes", "3-Pointers Made", "threes", _nb(0.04, 1.1)),
    _m("nba", "player_steals", "Steals", "steals", _nb(0.03)),
    _m("nba", "player_blocks", "Blocks", "blocks", _nb(0.10)),
    _m("nba", "player_turnovers", "Turnovers", "turnovers", _nb(0.03)),
    _m("nba", "player_blocks_steals", "Steals + Blocks", ("steals", "blocks"), _nb(0.04)),
    _m("nba", "player_points_rebounds_assists", "Pts + Reb + Ast",
       ("points", "rebounds", "assists"), _nb(0.025, 1.6)),
    _m("nba", "player_points_rebounds", "Pts + Reb", ("points", "rebounds"), _nb(0.025, 1.6)),
    _m("nba", "player_points_assists", "Pts + Ast", ("points", "assists"), _nb(0.025, 1.6)),
    _m("nba", "player_rebounds_assists", "Reb + Ast", ("rebounds", "assists"), _nb(0.03, 1.3)),
    # ---------------------------------------------------------------- NFL
    _m("nfl", "player_pass_yds", "Passing Yards", "passing_yards",
       FamilySpec("normal", phi=0.0, alpha=0.06, c=1500.0, lower=None), positions=("QB",)),
    _m("nfl", "player_pass_attempts", "Pass Attempts", "pass_attempts", _nb(0.02, 1.0), positions=("QB",)),
    _m("nfl", "player_pass_completions", "Completions", "completions", _nb(0.02, 1.0), positions=("QB",)),
    _m("nfl", "player_pass_tds", "Passing TDs", "passing_tds", _poisson(), positions=("QB",)),
    _m("nfl", "player_pass_interceptions", "Interceptions", "interceptions", _poisson(), positions=("QB",)),
    _m("nfl", "player_rush_yds", "Rushing Yards", "rushing_yards",
       FamilySpec("gamma", phi=8.0, alpha=0.15), positions=("RB", "QB", "WR")),
    _m("nfl", "player_rush_attempts", "Rush Attempts", "carries", _nb(0.04, 1.0), positions=("RB", "QB")),
    _m("nfl", "player_receptions", "Receptions", "receptions", _nb(0.05, 1.0), positions=("WR", "TE", "RB")),
    _m("nfl", "player_reception_yds", "Receiving Yards", "receiving_yards",
       FamilySpec("gamma", phi=10.0, alpha=0.25), positions=("WR", "TE", "RB")),
    _m("nfl", "player_rush_reception_yds", "Rush + Rec Yards",
       ("rushing_yards", "receiving_yards"), FamilySpec("gamma", phi=10.0, alpha=0.15),
       positions=("RB", "WR", "TE")),
    _m("nfl", "player_pass_rush_yds", "Pass + Rush Yards",
       ("passing_yards", "rushing_yards"),
       FamilySpec("normal", phi=0.0, alpha=0.06, c=1500.0, lower=None), positions=("QB",)),
    _m("nfl", "player_anytime_td", "Anytime TD", ("rushing_tds", "receiving_tds"), _poisson(),
       yes_no=True, positions=("RB", "WR", "TE", "QB")),
    # ---------------------------------------------------------------- NHL
    _m("nhl", "player_shots_on_goal", "Shots on Goal", "shots_on_goal", _nb(0.05)),
    _m("nhl", "player_points", "Points", "points", _poisson()),
    _m("nhl", "player_goals", "Goals", "goals", _poisson()),
    _m("nhl", "player_assists", "Assists", "assists", _poisson()),
    _m("nhl", "player_blocked_shots", "Blocked Shots", "blocked_shots", _nb(0.08)),
    _m("nhl", "player_total_saves", "Saves", "saves", _nb(0.02)),
    _m("nhl", "player_goal_scorer_anytime", "Anytime Goal", "goals", _poisson(), yes_no=True),
    # ---------------------------------------------------------------- MLB
    _m("mlb", "pitcher_strikeouts", "Pitcher Strikeouts", "pitcher_strikeouts", _nb(0.03)),
    _m("mlb", "pitcher_outs", "Pitcher Outs", "outs_recorded", _nb(0.02)),
    _m("mlb", "pitcher_hits_allowed", "Hits Allowed", "hits_allowed", _nb(0.03)),
    _m("mlb", "pitcher_earned_runs", "Earned Runs", "earned_runs", _nb(0.15)),
    _m("mlb", "pitcher_walks", "Walks Allowed", "walks_allowed", _nb(0.05)),
    _m("mlb", "batter_hits", "Hits", "hits", _poisson()),
    _m("mlb", "batter_total_bases", "Total Bases", "total_bases", _nb(0.6)),
    _m("mlb", "batter_home_runs", "Home Runs", "home_runs", _poisson()),
    _m("mlb", "batter_rbis", "RBIs", "rbis", _nb(0.5)),
    _m("mlb", "batter_runs_scored", "Runs", "runs", _nb(0.1)),
    _m("mlb", "batter_strikeouts", "Batter Strikeouts", "batter_strikeouts", _poisson()),
    _m("mlb", "batter_hits_runs_rbis", "Hits + Runs + RBIs", ("hits", "runs", "rbis"), _nb(0.35)),
]

#: Minimum projected mean at which each market is typically offered.
_MIN_MEANS: Dict[Tuple[str, str], float] = {
    ("nba", "player_points"): 8.0,
    ("nba", "player_rebounds"): 3.0,
    ("nba", "player_assists"): 2.0,
    ("nba", "player_threes"): 0.8,
    ("nba", "player_steals"): 0.7,
    ("nba", "player_blocks"): 0.6,
    ("nba", "player_turnovers"): 1.2,
    ("nba", "player_blocks_steals"): 1.2,
    ("nba", "player_points_rebounds_assists"): 14.0,
    ("nba", "player_points_rebounds"): 11.0,
    ("nba", "player_points_assists"): 10.0,
    ("nba", "player_rebounds_assists"): 5.0,
    ("nfl", "player_pass_yds"): 150.0,
    ("nfl", "player_pass_attempts"): 20.0,
    ("nfl", "player_pass_completions"): 12.0,
    ("nfl", "player_pass_tds"): 0.8,
    ("nfl", "player_pass_interceptions"): 0.4,
    ("nfl", "player_rush_yds"): 20.0,
    ("nfl", "player_rush_attempts"): 6.0,
    ("nfl", "player_receptions"): 2.0,
    ("nfl", "player_reception_yds"): 20.0,
    ("nfl", "player_rush_reception_yds"): 30.0,
    ("nfl", "player_pass_rush_yds"): 160.0,
    ("nfl", "player_anytime_td"): 0.15,
    ("nhl", "player_shots_on_goal"): 1.5,
    ("nhl", "player_points"): 0.4,
    ("nhl", "player_goals"): 0.2,
    ("nhl", "player_assists"): 0.25,
    ("nhl", "player_blocked_shots"): 0.8,
    ("nhl", "player_total_saves"): 18.0,
    ("nhl", "player_goal_scorer_anytime"): 0.2,
}

MARKETS: Dict[Tuple[str, str], MarketDef] = {
    (m.sport, m.key): replace(m, min_mean=_MIN_MEANS.get((m.sport, m.key), 0.0)) for m in _DEFS
}


def canonical_market_key(key: str) -> Tuple[str, bool]:
    """Strip the alternate-line suffix: ``(base_key, is_alternate)``."""
    key = str(key).strip().lower()
    if key.endswith(_ALT_SUFFIX):
        return key[: -len(_ALT_SUFFIX)], True
    return key, False


def find_market(sport: str, key: str) -> Optional[MarketDef]:
    """Look up a market by sport and (possibly alternate / aliased) key."""
    sport = str(sport).strip().lower()
    base, _ = canonical_market_key(key)
    market = MARKETS.get((sport, base))
    if market is not None:
        return market
    for (s, _k), m in MARKETS.items():
        if s == sport and base in m.aliases:
            return m
    return None


def get_market(sport: str, key: str) -> MarketDef:
    """Like :func:`find_market` but raises ``KeyError`` for unknown markets."""
    market = find_market(sport, key)
    if market is None:
        raise KeyError(f"unknown market {key!r} for sport {sport!r}")
    return market


def markets_for_sport(sport: str) -> List[MarketDef]:
    """All registered markets of a sport, in registry order."""
    sport = str(sport).strip().lower()
    return [m for (s, _), m in MARKETS.items() if s == sport]


def stat_value(logs: pd.DataFrame, market: MarketDef) -> pd.Series:
    """Per-row result of ``market`` for a game-log table (NaN if a stat is missing)."""
    missing = [c for c in market.stats if c not in logs.columns]
    if missing:
        raise KeyError(f"game logs lack columns {missing} needed for {market.key}")
    return logs[list(market.stats)].sum(axis=1, skipna=False)


def load_spec_overrides(overrides: Mapping[str, Mapping[str, object]]) -> List[str]:
    """Replace the specs of registered markets.

    ``overrides`` maps ``"sport/market_key"`` to a :class:`FamilySpec` dict
    (see :meth:`FamilySpec.to_dict`).  Returns the keys that were applied.
    Markets updated this way are marked ``calibrated``.
    """
    applied = []
    for name, spec_dict in overrides.items():
        sport, _, key = str(name).partition("/")
        current = MARKETS.get((sport, key))
        if current is None:
            continue
        MARKETS[(sport, key)] = replace(
            current,
            spec=FamilySpec.from_dict(dict(spec_dict)),
            mean_scale=float(spec_dict.get("mean_scale", 1.0)),
            calibrated=True,
        )
        applied.append(name)
    return applied


def _apply_packaged_overrides() -> None:
    try:
        text = resources.files("evmodel").joinpath("data/market_specs.json").read_text("utf-8")
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        return
    payload = json.loads(text)
    load_spec_overrides(payload.get("specs", payload))


_apply_packaged_overrides()
