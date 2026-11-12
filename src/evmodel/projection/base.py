"""The projection engine's output type."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Tuple

from ..distributions import Dist, FamilySpec

__all__ = ["Projection", "ProjectionKey"]

#: Projections are looked up by ``(player_key, canonical market key)``.
ProjectionKey = Tuple[str, str]


@dataclass(frozen=True)
class Projection:
    """A predictive distribution for one player's stat in their next game."""

    sport: str
    player_key: str
    player: str
    market: str
    mean: float
    spec: FamilySpec
    n_games: int
    team: str = ""
    details: Mapping[str, float] = field(default_factory=dict)

    @property
    def dist(self) -> Dist:
        return self.spec.dist(self.mean)

    @property
    def key(self) -> ProjectionKey:
        return (self.player_key, self.market)

    def probs(self, line: float) -> Tuple[float, float, float]:
        """``(P(over), P(push), P(under))`` at a line."""
        return self.dist.probs(line)

    def to_dict(self) -> Dict[str, object]:
        dist = self.dist
        low, high = dist.interval(0.8)
        return {
            "sport": self.sport,
            "player": self.player,
            "player_key": self.player_key,
            "team": self.team,
            "market": self.market,
            "mean": self.mean,
            "median": dist.median,
            "p10": low,
            "p90": high,
            "sd": dist.sd,
            "n_games": self.n_games,
            **{f"detail_{k}": v for k, v in self.details.items()},
        }
