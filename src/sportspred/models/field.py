"""A field model for races: a finish distribution for every starter.

Yes-or-no questions about one starter ("does he win", "is he in the top
three") can be answered row by row, but the answers of one race are not
independent: exactly one driver wins.  This candidate models the race as a
whole:

* a **strength** per starter, linear in the declared inputs, fitted by the
  rank-ordered (Plackett-Luce) likelihood of the classified finishing orders
  of the training races - the order among finishers is what it explains;
* a **retirement probability** per starter (logistic regression), because
  whether a car is classified at all is a different question from how fast
  it is;
* a **simulation** of the race from those two: in each draw some starters
  retire, the rest are ordered by strength plus Gumbel noise (which is exactly
  the Plackett-Luce model), and each starter's finishing position is counted.

``predict_proba`` returns, for every row, the probability of a classified
finish at or inside ``top_k``.  Rows are grouped into races by the part of
``event_id`` before the first ``|``; all starters of a race must be passed
together (``predicts_field`` tells the prediction engine so).  The simulation
is seeded from the race's id, so the same inputs give the same numbers.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..sports.base import Candidate

__all__ = ["FieldRankCandidate", "race_of", "rank_log_likelihood", "simulate_field"]


def race_of(event_ids: Sequence[Any]) -> np.ndarray:
    """The race a row belongs to: the part of its id before the first ``|``."""
    return np.array([str(e).split("|", 1)[0] for e in event_ids], dtype=object)


def rank_log_likelihood(strength: np.ndarray, mask: np.ndarray) -> float:
    """Plackett-Luce log-likelihood of finishing orders.

    ``strength`` is (races, places) with each race's classified finishers in
    finishing order and padding after them; ``mask`` marks the real entries.
    The winner is chosen from everyone, the runner-up from the rest, and so on.
    """
    s = np.where(mask, strength, -np.inf)
    # log-sum-exp over each suffix: the starters still "in the race" for that place
    suffix = np.logaddexp.accumulate(s[:, ::-1], axis=1)[:, ::-1]
    terms = np.where(mask, strength, 0.0) - np.where(mask, suffix, 0.0)      # padding contributes nothing
    return float(terms.sum())


def simulate_field(strength: np.ndarray, p_retire: np.ndarray, *, n_sims: int, seed: int) -> np.ndarray:
    """Simulated finishing positions: an ``(n_sims, n)`` array of classified
    positions (1 = winner), with 0 where the starter retired in that draw."""
    rng = np.random.default_rng(seed)
    n = len(strength)
    noise = rng.gumbel(size=(n_sims, n))
    out = rng.random((n_sims, n)) < p_retire[None, :]
    score = np.where(out, -np.inf, strength[None, :] + noise)
    order = np.argsort(-score, axis=1, kind="stable")
    position = np.empty((n_sims, n), dtype=np.int32)
    position[np.arange(n_sims)[:, None], order] = np.arange(1, n + 1, dtype=np.int32)[None, :]
    return np.where(out, 0, position)


class FieldRankCandidate(Candidate):
    """Plackett-Luce strengths, a retirement model and a simulated race."""

    predicts_field = True

    def __init__(self, name: str, strength_features: Sequence[str], retire_features: Sequence[str], *,
                 top_k: int, description: str = "", complexity: int = 5, n_sims: int = 20000, l2: float = 0.05,
                 position_col: str = "finish_position", calibration: str = "platt",
                 uses_retrospective: bool = False, is_baseline: bool = False,
                 context: Optional[pd.DataFrame] = None) -> None:
        self.name, self.description, self.complexity = name, description, complexity
        self.strength_features, self.retire_features = list(strength_features), list(retire_features)
        self.top_k, self.n_sims, self.l2 = int(top_k), int(n_sims), float(l2)
        self.position_col, self.calibration = position_col, calibration
        self.uses_retrospective, self.is_baseline = uses_retrospective, is_baseline
        # Starters of past races who were on the grid but have no row of their own (they did
        # not take the start): pregame inputs only.  They are simulated with their race and
        # never scored - the others raced a field that had them in it.
        self.context = context if context is not None and len(context) else None
        self.weights_: Optional[np.ndarray] = None

    def clone(self) -> "FieldRankCandidate":
        return FieldRankCandidate(self.name, self.strength_features, self.retire_features, top_k=self.top_k,
                                  description=self.description, complexity=self.complexity, n_sims=self.n_sims,
                                  l2=self.l2, position_col=self.position_col, calibration=self.calibration,
                                  uses_retrospective=self.uses_retrospective, is_baseline=self.is_baseline,
                                  context=self.context)

    def feature_names(self) -> List[str]:
        return list(dict.fromkeys(self.strength_features + self.retire_features))

    # ------------------------------------------------------------------- fit
    def fit(self, train: pd.DataFrame, classes: Sequence[str]) -> "FieldRankCandidate":
        if tuple(classes) != ("no", "yes"):
            raise ValueError("a field candidate serves yes/no markets with classes ('no', 'yes')")
        self.kept_ = [c for c in self.strength_features if train[c].notna().any()]
        self.prep_ = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        X = self.prep_.fit_transform(train[self.kept_].to_numpy(dtype=float))
        position = train[self.position_col].to_numpy(dtype=float)
        races = race_of(train["event_id"])
        classified = np.isfinite(position)
        if not classified.any():
            raise ValueError(f"candidate {self.name!r}: no classified finisher in the training rows")
        # Pad every race's classified finishers, in finishing order, into one array.
        frame = pd.DataFrame({"race": races[classified], "pos": position[classified], "row": np.flatnonzero(classified)})
        frame = frame.sort_values(["race", "pos"])
        frame["slot"] = frame.groupby("race").cumcount()
        codes, _ = pd.factorize(frame["race"])
        width = int(frame["slot"].max()) + 1
        index = np.full((codes.max() + 1, width), -1, dtype=int)
        index[codes, frame["slot"].to_numpy()] = frame["row"].to_numpy()
        mask = index >= 0
        padded = X[np.where(mask, index, 0)]                           # (races, places, features)

        def loss(w: np.ndarray) -> float:
            return -rank_log_likelihood(padded @ w, mask) / mask.sum() + self.l2 * float(w @ w) / len(w)

        res = optimize.minimize(loss, np.zeros(X.shape[1]), method="L-BFGS-B")
        self.weights_ = res.x
        retired = (~classified).astype(int)
        self.retire_kept_ = [c for c in self.retire_features if train[c].notna().any()]
        self.retire_ = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                                 ("model", LogisticRegression(C=1.0, max_iter=1000))])
        if retired.min() == retired.max() or not self.retire_kept_:   # nothing to learn from: a constant rate
            self.retire_ = None
            self.retire_rate_ = float(retired.mean())
        else:
            self.retire_.fit(train[self.retire_kept_].to_numpy(dtype=float), retired)
        return self

    # --------------------------------------------------------------- predict
    def _parts(self, rows: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if self.weights_ is None:
            raise RuntimeError("candidate is not fitted")
        strength = self.prep_.transform(rows[self.kept_].to_numpy(dtype=float)) @ self.weights_
        if self.retire_ is None:
            retire = np.full(len(rows), self.retire_rate_)
        else:
            retire = self.retire_.predict_proba(rows[self.retire_kept_].to_numpy(dtype=float))[:, 1]
        return strength, np.clip(retire, 1e-4, 0.95)

    def _simulate(self, rows: pd.DataFrame) -> Dict[str, np.ndarray]:
        n = len(rows)
        ids = rows["event_id"].astype(str).to_numpy()
        full = rows
        if self.context is not None:
            extra = self.context[self.context["event_id"].map(lambda e: str(e).split("|", 1)[0]).isin(set(race_of(ids)))
                                 & ~self.context["event_id"].astype(str).isin(set(ids))]
            if len(extra):
                cols = list(dict.fromkeys(["event_id"] + self.kept_ + self.retire_kept_))
                full = pd.concat([rows[cols], extra[cols]], ignore_index=True)
        strength, retire = self._parts(full)
        all_ids = full["event_id"].astype(str).to_numpy()
        races = race_of(all_ids)
        out = {"p_top": np.zeros(n), "p_win": np.zeros(n), "p_podium": np.zeros(n), "p_points": np.zeros(n),
               "p_retire": retire[:n], "median_finish": np.full(n, np.nan)}
        for race in pd.unique(races):
            idx = np.flatnonzero(races == race)
            idx = idx[np.argsort(all_ids[idx], kind="stable")]         # the order rows arrive in must not matter
            seed = int(hashlib.sha1(str(race).encode()).hexdigest()[:8], 16)
            pos = simulate_field(strength[idx], retire[idx], n_sims=self.n_sims, seed=seed)
            done = pos > 0
            own = idx < n                                               # context drivers are simulated, not reported
            for key, k in (("p_top", self.top_k), ("p_win", 1), ("p_podium", 3), ("p_points", 10)):
                out[key][idx[own]] = (done & (pos <= k)).mean(axis=0)[own]
            with np.errstate(invalid="ignore"):
                out["median_finish"][idx[own]] = np.nanmedian(np.where(done, pos, np.nan), axis=0)[own]
        return out

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        p = np.clip(self._simulate(rows)["p_top"], 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - p, p])

    def projection(self, rows: pd.DataFrame) -> pd.DataFrame:
        """The simulated finish distribution of each starter, before
        calibration: win, podium, points, retirement, and the median finish
        when classified.  Needs the whole field of each race."""
        sim = self._simulate(rows)
        return pd.DataFrame({"win": sim["p_win"], "podium": sim["p_podium"], "points": sim["p_points"],
                             "not_classified": sim["p_retire"], "median_finish": sim["median_finish"]},
                            index=rows.index)
