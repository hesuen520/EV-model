"""Dependencies between combo legs.

Every pair of legs is classified.  A combination is only priced when every
pair is either independent by construction, or dependent in a way that is
actually modelled.  Anything else is rejected - marginal probabilities are
never blindly multiplied.

Pair classes
------------
``exclusive``        same event and market, different selections: both cannot win.
``same_event``       same event, different markets (winner + total, team + player
                     prop, two props in one game): strongly dependent.  No joint
                     model exists for these yet, so the pair is **rejected**.
``same_participant`` the same team/player/competitor in two different events
                     (back-to-back games, successive rounds): the later leg's
                     probability depends on the earlier result.  **Rejected.**
``same_slate``       different events, no shared participant, same league and
                     market: linked only through shared model error and common
                     news.  Priced with the validated same-slate residual
                     correlation (independence when it is indistinguishable
                     from zero and inside the configured bound; a Gaussian
                     copula otherwise).  Rejected when no validated estimate
                     exists.
``unrelated``        different competitions, no shared participant or event:
                     treated as independent (stated as an assumption).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from scipy import optimize, stats

__all__ = ["Leg", "PairDependency", "classify_pair", "joint_probability", "JointResult", "latent_correlation"]


@dataclass(frozen=True)
class Leg:
    prediction_id: str
    event_id: str
    league: str
    sport: str
    market: str
    line: Optional[float]
    selection: str
    label: str
    prob: float
    prob_low: Optional[float]
    rec_status: str
    validation_status: str
    start_utc: str
    participants: frozenset          # every team/competitor id in the event
    entities: frozenset              # ids this leg's outcome is about (team, player)
    event_label: str = ""
    event_status: str = "scheduled"  # the event's status now, not when the prediction was made
    current_start_utc: Optional[str] = None      # the event's start now
    is_synthetic: bool = False
    date_only: bool = False          # the event has only a date: it is closed 14 hours before the stored instant

    @property
    def market_family(self) -> str:
        return self.market.split(":", 1)[0]


@dataclass
class PairDependency:
    a: str
    b: str
    kind: str
    treatment: str                   # independent | copula | rejected
    rho: float = 0.0
    note: str = ""


def classify_pair(x: Leg, y: Leg, residual_corr: Mapping[Tuple[str, str], Mapping[str, Any]],
                  max_abs_rho: float) -> PairDependency:
    """Classify one pair of legs.  ``residual_corr`` maps (league, market
    family) to the validated same-slate residual correlation
    ``{"rho", "low", "high", "n_pairs"}``."""
    if x.event_id == y.event_id:
        if x.market == y.market and x.line == y.line and x.selection != y.selection:
            return PairDependency(x.prediction_id, y.prediction_id, "exclusive", "rejected",
                                  note="two different outcomes of one market cannot both win")
        return PairDependency(x.prediction_id, y.prediction_id, "same_event", "rejected",
                              note="legs from the same event are strongly dependent and no joint model exists")
    shared = (x.participants & y.participants) | (x.entities & y.entities)
    if shared:
        return PairDependency(x.prediction_id, y.prediction_id, "same_participant", "rejected",
                              note="the same competitor appears in both events; the later result depends on the earlier one")
    if x.league == y.league and x.market_family == y.market_family:
        est = residual_corr.get((x.league, x.market_family))
        if not est or est.get("rho") is None:
            return PairDependency(x.prediction_id, y.prediction_id, "same_slate", "rejected",
                                  note=f"no validated same-slate dependence estimate for {x.league} {x.market_family}")
        rho, lo, hi = float(est["rho"]), float(est["low"]), float(est["high"])
        # Independent only when the data rule dependence out: the whole interval,
        # not just the point estimate, must lie inside the bound.  A wide interval
        # that merely contains zero says the dependence is unknown, not absent.
        if max(abs(lo), abs(hi)) <= max_abs_rho:
            return PairDependency(
                x.prediction_id, y.prediction_id, "same_slate", "independent", 0.0,
                note=(f"independence approximation: validated same-slate residual correlation {rho:+.4f} "
                      f"(95% interval {lo:+.4f} to {hi:+.4f}, {est.get('n_pairs', '?')} event pairs)"))
        # Otherwise model it.  The central figure uses the point estimate; the
        # conservative figure (see joint_probability) uses the lower end of the
        # interval, where the chance that every leg wins is smallest.
        return PairDependency(
            x.prediction_id, y.prediction_id, "same_slate", "copula", rho,
            note=(f"dependence modelled with a Gaussian copula at the validated same-slate residual correlation "
                  f"{rho:+.4f} (95% interval {lo:+.4f} to {hi:+.4f}; the conservative figure uses {lo:+.4f})"))
    if x.league == y.league:
        return PairDependency(x.prediction_id, y.prediction_id, "same_slate", "rejected",
                              note="different markets of one league on one slate: cross-market dependence is not validated")
    return PairDependency(x.prediction_id, y.prediction_id, "unrelated", "independent", 0.0,
                          note="different competitions with no shared event or competitor: assumed independent")


def latent_correlation(p1: float, p2: float, rho_binary: float) -> float:
    """Latent normal correlation that reproduces a correlation ``rho_binary``
    between two Bernoulli outcomes with success probabilities p1 and p2."""
    if abs(rho_binary) < 1e-12:
        return 0.0
    target = p1 * p2 + rho_binary * np.sqrt(p1 * (1 - p1) * p2 * (1 - p2))
    target = float(np.clip(target, max(0.0, p1 + p2 - 1.0) + 1e-9, min(p1, p2) - 1e-9))
    z1, z2 = stats.norm.ppf(p1), stats.norm.ppf(p2)

    def gap(r: float) -> float:
        return float(stats.multivariate_normal.cdf([z1, z2], mean=[0, 0], cov=[[1, r], [r, 1]])) - target

    try:
        return float(optimize.brentq(gap, -0.99, 0.99, xtol=1e-6))
    except ValueError:
        return float(np.clip(rho_binary, -0.99, 0.99))


@dataclass
class JointResult:
    valid: bool
    prob: Optional[float]
    prob_low: Optional[float]
    method: str
    pairs: List[PairDependency] = field(default_factory=list)
    rejected: List[PairDependency] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid, "method": self.method,
            "pairs": [p.__dict__ for p in self.pairs],
            "rejected": [p.__dict__ for p in self.rejected],
        }


def _copula_all(probs: Sequence[float], corr: np.ndarray) -> float:
    z = stats.norm.ppf(np.clip(probs, 1e-9, 1 - 1e-9))
    # Nearest positive semi-definite correlation matrix (pairwise solves can
    # leave tiny negative eigenvalues).
    w, v = np.linalg.eigh(corr)
    corr = (v * np.clip(w, 1e-8, None)) @ v.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    return float(stats.multivariate_normal.cdf(z, mean=np.zeros(len(z)), cov=corr, allow_singular=True))


def joint_probability(legs: Sequence[Leg], residual_corr: Mapping[Tuple[str, str], Mapping[str, Any]],
                      max_abs_rho: float) -> JointResult:
    """Probability that every leg wins, or an invalid result naming the pairs
    whose dependence is not modelled."""
    n = len(legs)
    if n == 0:
        return JointResult(False, None, None, "empty")
    if n == 1:
        leg = legs[0]
        return JointResult(True, leg.prob, leg.prob_low if leg.prob_low is not None else leg.prob, "single")
    pairs = [classify_pair(legs[i], legs[j], residual_corr, max_abs_rho)
             for i in range(n) for j in range(i + 1, n)]
    rejected = [p for p in pairs if p.treatment == "rejected"]
    if rejected:
        return JointResult(False, None, None, "rejected", pairs, rejected)
    probs = [leg.prob for leg in legs]
    lows = [leg.prob_low if leg.prob_low is not None else leg.prob for leg in legs]
    if all(p.treatment == "independent" for p in pairs):
        return JointResult(True, float(np.prod(probs)), float(np.prod(lows)),
                           "product of leg probabilities (independence justified pair by pair)", pairs)
    index = {leg.prediction_id: i for i, leg in enumerate(legs)}
    corr = np.eye(n)
    corr_low = np.eye(n)
    for p in pairs:
        if p.treatment == "copula":
            i, j = index[p.a], index[p.b]
            corr[i, j] = corr[j, i] = latent_correlation(probs[i], probs[j], p.rho)
            # For the conservative bound use the lower end of the correlation
            # interval: less positive dependence means a lower all-win probability.
            est = residual_corr[(legs[i].league, legs[i].market_family)]
            r_low = latent_correlation(lows[i], lows[j], float(est["low"]))
            corr_low[i, j] = corr_low[j, i] = r_low
    return JointResult(True, _copula_all(probs, corr), _copula_all(lows, corr_low),
                       "Gaussian copula with validated same-slate residual correlation", pairs)
