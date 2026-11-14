"""A Poisson goals model with the Dixon-Coles low-score correction.

Two steps, both fitted on the training rows of a fold and nothing else:

1. A Poisson regression of each side's goals on its point-in-time attack
   rating, the opponent's defence rating, both sides' recent-form deviations
   and home advantage.  That turns the running ratings (which are on an
   additive goals scale) into a scoring rate for each side.
2. A single dependence parameter ``rho`` (Dixon and Coles, 1997) chosen by
   maximum likelihood.  Independent Poisson scores under-count 0-0 and 1-1
   and over-count 1-0 and 0-1; ``rho`` moves probability between those four
   scores and is what adjusts the draw probability.

The home / draw / away probabilities are then read off the score distribution.
The model sees goals of past matches only: no odds, no line-ups.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

from ..base import Candidate
from .features import GOAL_CAP, GOAL_MODEL_INPUTS

__all__ = ["DixonColesCandidate", "outcome_probabilities", "fit_rho"]

_MAX_GOALS = 20                 # the score distribution is summed up to this many goals a side
_RATE_RANGE = (0.05, 8.0)       # scoring rates outside this range are clipped
_RHO_RANGE = (-0.2, 0.2)


def _bounded_rho(rho: float, lam: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """``rho`` shrunk row by row so that every corrected score probability
    stays positive whatever the two scoring rates are."""
    low = -0.9 / np.maximum(np.maximum(lam, mu), 1.0)
    high = 0.9 / np.maximum(lam * mu, 1.0)
    return np.clip(rho, low, high)


def outcome_probabilities(lam: np.ndarray, mu: np.ndarray, rho: float = 0.0) -> np.ndarray:
    """``(n, 3)`` probabilities of home win, draw, away win when the home side
    scores at rate ``lam`` and the away side at rate ``mu``."""
    lam, mu = np.asarray(lam, dtype=float), np.asarray(mu, dtype=float)
    goals = np.arange(_MAX_GOALS + 1)
    pa = poisson.pmf(goals[None, :], lam[:, None])
    pb = poisson.pmf(goals[None, :], mu[:, None])
    b_below = np.cumsum(pb, axis=1) - pb                 # P(away scores fewer than i)
    home = (pa * b_below).sum(axis=1)
    draw = (pa * pb).sum(axis=1)
    away = pa.sum(axis=1) * pb.sum(axis=1) - home - draw
    # Dixon-Coles: scale 0-0 by (1 - lam*mu*rho), 0-1 by (1 + lam*rho),
    # 1-0 by (1 + mu*rho) and 1-1 by (1 - rho); the four changes cancel out.
    r = _bounded_rho(rho, lam, mu)
    home = home + r * mu * pa[:, 1] * pb[:, 0]
    away = away + r * lam * pa[:, 0] * pb[:, 1]
    draw = draw - r * lam * mu * pa[:, 0] * pb[:, 0] - r * pa[:, 1] * pb[:, 1]
    P = np.clip(np.column_stack([home, draw, away]), 1e-9, None)
    return P / P.sum(axis=1, keepdims=True)


def fit_rho(lam: np.ndarray, mu: np.ndarray, goals_a: np.ndarray, goals_b: np.ndarray) -> float:
    """Maximum-likelihood ``rho`` given the scoring rates; only 0-0, 0-1, 1-0
    and 1-1 results carry information about it."""
    cells = [
        ((goals_a == 0) & (goals_b == 0), lambda r, m: 1.0 - lam[m] * mu[m] * r[m]),
        ((goals_a == 0) & (goals_b == 1), lambda r, m: 1.0 + lam[m] * r[m]),
        ((goals_a == 1) & (goals_b == 0), lambda r, m: 1.0 + mu[m] * r[m]),
        ((goals_a == 1) & (goals_b == 1), lambda r, m: 1.0 - r[m]),
    ]

    def negative_log_likelihood(rho: float) -> float:
        r = _bounded_rho(rho, lam, mu)
        return -float(sum(np.log(factor(r, mask)).sum() for mask, factor in cells))

    return float(minimize_scalar(negative_log_likelihood, bounds=_RHO_RANGE, method="bounded").x)


class DixonColesCandidate(Candidate):
    """Goals model on the frame's attack/defence ratings (see module docstring).

    Needs the outcome columns ``goals_a`` / ``goals_b`` on the training rows
    and the inputs in :data:`~.features.GOAL_MODEL_INPUTS`; a side without a
    rating yet is treated as average.
    """

    calibration = "platt"

    def __init__(self, name: str = "poisson_dc", *, description: str = "", complexity: int = 2,
                 alpha: float = 1e-4) -> None:
        self.name = name
        self.description = description
        self.complexity = complexity
        self.alpha = alpha            # ridge penalty of the Poisson regression
        self.glm_: Optional[PoissonRegressor] = None
        self.rho_: float = 0.0
        self.order_: List[int] = [0, 1, 2]

    def clone(self) -> "DixonColesCandidate":
        return DixonColesCandidate(self.name, description=self.description, complexity=self.complexity,
                                   alpha=self.alpha)

    def feature_names(self) -> List[str]:
        return list(GOAL_MODEL_INPUTS)

    @staticmethod
    def _design(rows: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Regression inputs for side A's goals and for side B's goals:
        playing at home, playing away, own attack, opposing defence, and the
        recent-form deviations of both."""
        x = {c: rows[c].fillna(0.0).to_numpy(dtype=float) for c in GOAL_MODEL_INPUTS}
        zero = np.zeros(len(rows))
        side_a = np.column_stack([x["home_edge"], zero, x["att_a"], x["def_b"], x["att_form_a"], x["def_form_b"]])
        side_b = np.column_stack([zero, x["home_edge"], x["att_b"], x["def_a"], x["att_form_b"], x["def_form_a"]])
        return side_a, side_b

    def _rates(self, rows: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if self.glm_ is None:
            raise RuntimeError("candidate is not fitted")
        side_a, side_b = self._design(rows)
        return (np.clip(self.glm_.predict(side_a), *_RATE_RANGE), np.clip(self.glm_.predict(side_b), *_RATE_RANGE))

    def fit(self, train: pd.DataFrame, classes: Sequence[str]) -> "DixonColesCandidate":
        self.order_ = [list(classes).index(c) for c in ("home", "draw", "away")]
        goals_a = train["goals_a"].to_numpy(dtype=float)
        goals_b = train["goals_b"].to_numpy(dtype=float)
        if not (np.isfinite(goals_a).all() and np.isfinite(goals_b).all()):
            raise ValueError(f"candidate {self.name!r}: training rows without a score")
        side_a, side_b = self._design(train)
        target = np.minimum(np.concatenate([goals_a, goals_b]), GOAL_CAP)
        self.glm_ = PoissonRegressor(alpha=self.alpha, max_iter=500).fit(np.vstack([side_a, side_b]), target)
        lam, mu = self._rates(train)
        self.rho_ = fit_rho(lam, mu, goals_a, goals_b)
        return self

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        lam, mu = self._rates(rows)
        P = outcome_probabilities(lam, mu, self.rho_)
        out = np.empty_like(P)
        out[:, self.order_] = P
        return out
