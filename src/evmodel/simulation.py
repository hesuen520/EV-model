"""Check the scanner against a world where the truth is known.

Real betting results are far too noisy to tell quickly whether an EV
calculation is right: at typical prop prices it takes thousands of bets.
In the simulated world of :mod:`evmodel.providers.sample` the true
probability of every prop is known, so the scanner's claims can be checked
exactly:

* is the *predicted* EV of the recommended bets close to their *true* EV?
* how much of the predicted edge is real once the reference books are
  themselves noisy (the "winner's curse": the bets that look best are
  partly the ones where the reference happens to be wrong)?
* does staking the recommended units actually grow a bankroll?

This validates the arithmetic and the selection logic.  It says nothing
about whether real sportsbooks behave like the simulated ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .config import Config
from .providers.sample import DEFAULT_BOOKS, BookProfile, make_slate, simulate_outcomes
from .scanner import scan

__all__ = ["SimulationResult", "simulate_market"]


@dataclass
class SimulationResult:
    """Outcome of :func:`simulate_market`."""

    bets: pd.DataFrame  # scanner output + true_ev, actual, result, pnl_units
    n_slates: int

    def summary(self) -> Dict[str, float]:
        b = self.bets
        if b.empty:
            return {"n_slates": self.n_slates, "n_bets": 0}
        staked = float(b["stake_units"].sum())
        decided = b[b["result"].isin(["win", "loss"])]
        return {
            "n_slates": self.n_slates,
            "n_bets": int(len(b)),
            "bets_per_slate": len(b) / self.n_slates,
            "units_staked": staked,
            "predicted_ev": float(np.average(b["ev"], weights=b["stake_units"])),
            "true_ev": float(np.average(b["true_ev"], weights=b["stake_units"])),
            "share_truly_positive": float((b["true_ev"] > 0).mean()),
            "realized_roi": float(b["pnl_units"].sum() / staked) if staked > 0 else float("nan"),
            "pnl_units": float(b["pnl_units"].sum()),
            "win_rate": float((decided["result"] == "win").mean()) if len(decided) else float("nan"),
        }

    def by_ev_bucket(self, edges: Sequence[float] = (0.0, 0.03, 0.05, 0.08, 0.12, 1.0)) -> pd.DataFrame:
        """Predicted vs true EV by size of the predicted edge."""
        b = self.bets
        if b.empty:
            return pd.DataFrame(columns=["predicted_ev", "true_ev", "realized_roi", "n"])
        buckets = pd.cut(b["ev"], list(edges), include_lowest=True)
        grouped = b.groupby(buckets, observed=True)
        out = pd.DataFrame(
            {
                "predicted_ev": grouped["ev"].mean(),
                "true_ev": grouped["true_ev"].mean(),
                "realized_roi": grouped["pnl_units"].sum() / grouped["stake_units"].sum(),
                "n": grouped.size(),
            }
        )
        out.index = out.index.astype(str)
        return out

    def by(self, column: str) -> pd.DataFrame:
        """Predicted vs true EV grouped by any column (``book``, ``market`` ...)."""
        b = self.bets
        if b.empty:
            return pd.DataFrame(columns=["predicted_ev", "true_ev", "realized_roi", "n"])
        grouped = b.groupby(column, observed=True)
        return pd.DataFrame(
            {
                "predicted_ev": grouped["ev"].mean(),
                "true_ev": grouped["true_ev"].mean(),
                "realized_roi": grouped["pnl_units"].sum() / grouped["stake_units"].sum(),
                "n": grouped.size(),
            }
        ).sort_values("n", ascending=False)


def simulate_market(
    n_slates: int = 50,
    sport: str = "nba",
    config: Optional[Config] = None,
    books: Sequence[BookProfile] = DEFAULT_BOOKS,
    market_noise: float = 0.015,
    n_games: int = 6,
    seed: int = 0,
) -> SimulationResult:
    """Run the scanner over many simulated slates and settle every bet.

    Each slate is generated, scanned with ``config``, and its props are then
    played out once from their true distributions.
    """
    config = config or Config()
    frames = []
    for i in range(n_slates):
        world = make_slate(
            sport, n_games=n_games, books=books, market_noise=market_noise, seed=seed * 100_003 + i
        )
        picks = scan(world.quotes, config)
        if picks.empty:
            continue
        picks = picks.copy()
        picks["slate"] = i
        picks["true_ev"] = world.true_ev(picks)
        outcomes = simulate_outcomes(world, seed=seed * 7_919 + i)
        actual = outcomes.set_index(["event_id", "player_key", "market"])["actual"]
        keys = list(zip(picks["event_id"], picks["player_key"], picks["market"]))
        picks["actual"] = [float(actual.loc[k]) for k in keys]
        over = picks["side"] == "over"
        diff = picks["actual"] - picks["line"]
        win = (over & (diff > 0)) | (~over & (diff < 0))
        push = diff == 0
        picks["result"] = np.where(push, "push", np.where(win, "win", "loss"))
        picks["pnl_units"] = np.where(
            push, 0.0, np.where(win, picks["stake_units"] * (picks["price"] - 1.0), -picks["stake_units"])
        )
        frames.append(picks)
    bets = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return SimulationResult(bets=bets, n_slates=n_slates)
