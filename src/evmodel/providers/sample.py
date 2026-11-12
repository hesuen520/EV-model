"""A simulated sportsbook world - **not real odds**.

This generates fictional games, fictional players and the prices a set of
simulated sportsbooks would post for them.  It exists for three reasons:

* the demo (``evmodel demo``) works without an API key;
* tests have a realistic, deterministic slate to run on;
* because the *true* probability of every prop is known here, it lets us
  check that the scanner's expected-value numbers mean what they claim
  (see :func:`evmodel.backtest.simulate_market`).

How the world works.  Every prop has a true mean; the true outcome
distribution is the market's :class:`~evmodel.distributions.FamilySpec` at
that mean.  Each book believes in a noisy version of the truth::

    book_mean = true_mean * exp(market_error + book_error)

``market_error`` is shared by every book (news nobody has priced yet - no
amount of de-vigging can see it) and ``book_error`` is the book's own
mistake, small for sharp books and larger for soft ones.  A book posts the
half-point line nearest its median, prices both sides from its belief, adds
its margin and rounds to its price grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..markets import MarketDef, get_market
from ..names import player_key
from ..odds import american_to_decimal, decimal_to_american
from ..schema import normalize_quotes

__all__ = ["BookProfile", "DEFAULT_BOOKS", "SampleWorld", "make_slate", "simulate_outcomes"]


@dataclass(frozen=True)
class BookProfile:
    """Behaviour of one simulated sportsbook."""

    key: str
    noise: float  # sd of the book's own error in log(mean)
    overround: float  # sum of implied probabilities minus one
    post_prob: float = 0.9  # chance it posts any given prop
    stale_prob: float = 0.0  # chance its number is badly off (extra noise)
    stale_noise: float = 0.035
    price_grid: int = 5  # American prices are rounded to this many "cents"
    dfs: bool = False  # pick'em operator: fixed placeholder price both ways


#: A plausible mix of sharp and soft books.  Purely illustrative numbers.
DEFAULT_BOOKS: Tuple[BookProfile, ...] = (
    BookProfile("pinnacle", noise=0.006, overround=0.045, post_prob=0.85, price_grid=1),
    BookProfile("fanduel", noise=0.009, overround=0.060, post_prob=0.95),
    BookProfile("draftkings", noise=0.012, overround=0.065, post_prob=0.95, stale_prob=0.03),
    BookProfile("betmgm", noise=0.015, overround=0.075, post_prob=0.90, stale_prob=0.04),
    BookProfile("williamhill_us", noise=0.015, overround=0.075, post_prob=0.85, stale_prob=0.04),
    BookProfile("espnbet", noise=0.018, overround=0.080, post_prob=0.80, stale_prob=0.05),
    BookProfile("prizepicks", noise=0.015, overround=0.0, post_prob=0.75, stale_prob=0.04, dfs=True),
)

_FIRST = (
    "Marcus Dario Elias Tobin Rafael Jonah Kellan Idris Mateo Soren Quentin Davon Emeric Lucan "
    "Tavian Orrin Cassius Bram Nikolai Zeke Anders Malik Reuben Cyrus Leander Osei Thane Ravi"
).split()
_LAST = (
    "Ellery Vantongeren Okafor-Reyes Halvorsen Brightwater Castellan Dunmore Ashgrove Pennick "
    "Calloway-Imre Sturridge Marchetti-Lowe Abernethy Kovalenko-Price Thistlewood Varga Ojukwu "
    "Lindqvist Harrowgate Maddox-Bell Okonkwo-Hart Feversham Delacroix-Yu Stanhope Greaves"
).split()
_CITIES = (
    "Harbor City;Red Mesa;Lakeshore;Iron Valley;Summit Ridge;Coral Bay;North Fork;Stonebridge;"
    "Elm Hollow;Port Meridian;Silver Plains;Cedar Falls East"
).split(";")
_MASCOTS = "Gulls Coyotes Lanterns Forge Kestrels Tide Timberjacks Owls Comets Pilots Bison Herons".split()

#: Typical true means used to draw props for each market, as (low, high).
_MEAN_RANGES: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("nba", "player_points"): (9.0, 31.0),
    ("nba", "player_rebounds"): (3.0, 12.5),
    ("nba", "player_assists"): (2.0, 10.5),
    ("nba", "player_threes"): (0.9, 4.2),
    ("nba", "player_points_rebounds_assists"): (16.0, 48.0),
    ("nfl", "player_pass_yds"): (185.0, 290.0),
    ("nfl", "player_rush_yds"): (28.0, 95.0),
    ("nfl", "player_reception_yds"): (22.0, 90.0),
    ("nfl", "player_receptions"): (2.2, 7.5),
    ("nfl", "player_pass_attempts"): (27.0, 39.0),
}

#: A pick'em operator pays the same on both sides, so it only posts a line
#: when it believes the two sides are within this distance of 50/50.
_DFS_MAX_IMBALANCE = 0.06

DEFAULT_MARKETS: Dict[str, Tuple[str, ...]] = {
    "nba": ("player_points", "player_rebounds", "player_assists", "player_threes", "player_points_rebounds_assists"),
    "nfl": ("player_pass_yds", "player_rush_yds", "player_reception_yds", "player_receptions"),
}


@dataclass
class SampleWorld:
    """A generated slate: what books quote, and the truth behind it."""

    quotes: pd.DataFrame
    #: one row per (event_id, player_key, market): true_mean and the spec used
    truth: pd.DataFrame
    sport: str
    _markets: Dict[str, MarketDef] = field(default_factory=dict, repr=False)

    def true_probs(self, player_key_: str, market: str, line: float, event_id: Optional[str] = None) -> Tuple[float, float, float]:
        """True ``(P(over), P(push), P(under))`` of a prop at a line."""
        t = self.truth
        mask = (t["player_key"] == player_key_) & (t["market"] == market)
        if event_id is not None:
            mask &= t["event_id"] == event_id
        if not mask.any():
            raise KeyError(f"no such prop in the sample world: {player_key_!r} {market!r}")
        mean = float(t.loc[mask, "true_mean"].iloc[0])
        return self._markets[market].spec.dist(mean).probs(line)

    def true_ev(self, opportunities: pd.DataFrame) -> pd.Series:
        """True expected value per unit of each row of a scanner result."""
        values = []
        for row in opportunities.itertuples(index=False):
            over, push, under = self.true_probs(row.player_key, row.market, row.line, row.event_id)
            win, lose = (over, under) if row.side == "over" else (under, over)
            values.append(win * (row.price - 1.0) - lose)
        return pd.Series(values, index=opportunities.index, dtype=float, name="true_ev")


def _name(rng: np.random.Generator, used: set) -> str:
    for _ in range(1000):
        name = f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"
        if name not in used:
            used.add(name)
            return name
    raise RuntimeError("ran out of fictional names")  # pragma: no cover


def _round_american(decimal: float, grid: int) -> float:
    american = decimal_to_american(decimal)
    rounded = round(american / grid) * grid
    if -100 < rounded < 100:
        rounded = 100 if american > 0 else -100
    return american_to_decimal(rounded)


def _book_line(dist_median: int, mean: float) -> float:
    # Books hang half-point lines next to their median.
    return dist_median + 0.5 if mean >= dist_median else dist_median - 0.5


def make_slate(
    sport: str = "nba",
    n_games: int = 6,
    players_per_team: int = 5,
    markets: Optional[Sequence[str]] = None,
    books: Sequence[BookProfile] = DEFAULT_BOOKS,
    market_noise: float = 0.015,
    seed: int = 0,
    start: str = "2026-01-15T00:00:00Z",
) -> SampleWorld:
    """Generate a fictional slate of props and the books' prices for it.

    Parameters
    ----------
    market_noise:
        sd of the error in log(mean) shared by all books (see module docs).
    seed:
        Seed for the random generator; the same seed gives the same slate.
    """
    sport = sport.lower()
    market_keys = tuple(markets) if markets else DEFAULT_MARKETS.get(sport, ())
    if not market_keys:
        raise ValueError(f"no sample markets for sport {sport!r}; pass markets=...")
    market_defs = {key: get_market(sport, key) for key in market_keys}
    for key in market_keys:
        if (sport, key) not in _MEAN_RANGES:
            raise ValueError(f"the sample generator has no mean range for {sport}/{key}")

    rng = np.random.default_rng(seed)
    used_names: set = set()
    team_ids = rng.permutation(len(_CITIES))
    start_ts = pd.Timestamp(start)
    snapshot_ts = start_ts - pd.Timedelta(hours=3)

    quote_rows: List[Dict[str, object]] = []
    truth_rows: List[Dict[str, object]] = []
    for g in range(n_games):
        home_i, away_i = team_ids[(2 * g) % len(team_ids)], team_ids[(2 * g + 1) % len(team_ids)]
        home = f"{_CITIES[home_i]} {_MASCOTS[home_i]}"
        away = f"{_CITIES[away_i]} {_MASCOTS[away_i]}"
        event_id = f"sim-{seed}-{g:03d}"
        commence = start_ts + pd.Timedelta(minutes=30 * g)
        for team in (home, away):
            for _ in range(players_per_team):
                player = _name(rng, used_names)
                for key in market_keys:
                    if rng.random() < 0.15:
                        continue  # not every player has every market
                    definition = market_defs[key]
                    low, high = _MEAN_RANGES[(sport, key)]
                    true_mean = float(rng.uniform(low, high))
                    truth_rows.append(
                        {
                            "sport": sport,
                            "event_id": event_id,
                            "player": player,
                            "player_key": player_key(player),
                            "team": team,
                            "market": key,
                            "true_mean": true_mean,
                        }
                    )
                    market_error = rng.normal(0.0, market_noise)
                    for book in books:
                        if rng.random() > book.post_prob:
                            continue
                        error = market_error + rng.normal(0.0, book.noise)
                        if rng.random() < book.stale_prob:
                            error += rng.normal(0.0, book.stale_noise)
                        belief = definition.spec.dist(true_mean * math.exp(error))
                        line = max(_book_line(belief.median, belief.mean), 0.5)
                        p_over = min(max(belief.prob_over(line), 0.02), 0.98)
                        if book.dfs and abs(p_over - 0.5) > _DFS_MAX_IMBALANCE:
                            continue  # pick'em operators only post roughly even lines
                        if book.dfs:
                            over_price = under_price = american_to_decimal(-119)
                        else:
                            scale = 1.0 + book.overround
                            over_price = _round_american(1.0 / min(p_over * scale, 0.985), book.price_grid)
                            under_price = _round_american(1.0 / min((1.0 - p_over) * scale, 0.985), book.price_grid)
                        age = pd.Timedelta(minutes=float(rng.uniform(1, 90)))
                        quote_rows.append(
                            {
                                "sport": sport,
                                "event_id": event_id,
                                "commence_time": commence,
                                "home_team": home,
                                "away_team": away,
                                "book": book.key,
                                "market": key,
                                "player": player,
                                "line": line,
                                "over_price": over_price,
                                "under_price": under_price,
                                "last_update": snapshot_ts - age,
                                "is_alternate": False,
                            }
                        )
    quotes = normalize_quotes(pd.DataFrame(quote_rows))
    truth = pd.DataFrame(truth_rows)
    return SampleWorld(quotes=quotes, truth=truth, sport=sport, _markets=market_defs)


def simulate_outcomes(world: SampleWorld, seed: int = 0) -> pd.DataFrame:
    """Draw one realised stat value for every prop in a sample world.

    Returns the truth table with an ``actual`` column.  Each market is drawn
    independently from its own distribution (combo markets are not forced to
    equal the sum of their components).
    """
    rng = np.random.default_rng(seed)
    out = world.truth.copy()
    actual = []
    for row in out.itertuples(index=False):
        dist = world._markets[row.market].spec.dist(row.true_mean)
        actual.append(dist.quantile(float(np.clip(rng.random(), 1e-9, 1 - 1e-9))))
    out["actual"] = np.asarray(actual, dtype=float)
    return out
