"""Spread and total candidates: a score regression plus a discrete distribution.

A bet against a line is a statement about a score quantity ``q`` - the home
margin for a spread, total points for a total.  ``ScoreDistCandidate``

1. regresses ``q`` on **sports features only** (the line is never a regressor);
2. turns the prediction into a probability for every whole number ``k``: a
   normal distribution around the prediction with the residual spread of the
   training rows, re-weighted so that scores which football's scoring makes
   common (margins of 3 and 7, say) get their observed share.  Both the spread
   and the weights are fitted on the training rows of the fold;
3. reads the bet off that distribution: ``P(q > threshold)``,
   ``P(q < threshold)`` and ``P(q == threshold)`` - the push.

``predict_proba`` returns the two-way probabilities **given that the bet does
not push**, which is what the binary target (pushes removed) measures;
``push_proba`` returns the push probability itself.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone as sk_clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..base import Candidate

__all__ = ["ScoreDistCandidate", "ridge_score_candidate", "gbm_score_candidate", "EXPLAIN_KEYS"]

#: Extra columns ``contributions`` returns next to the per-feature ones.
EXPLAIN_KEYS = ("_expected", "_baseline", "_sd", "_p_push")


class ScoreDistCandidate(Candidate):
    """Regression of ``target_col`` with an in-fold discrete predictive distribution.

    ``threshold_sign`` turns the frame's ``line`` into the number the quantity
    is compared with: ``-1`` for a spread (the line is the home handicap, so
    the home side covers when ``margin > -line``) and ``+1`` for a total.
    ``extra_rows`` are completed events that are missing from the training
    frame only because their bet pushed; those inside the training blocks are
    added back when fitting, so the distribution is not starved of exactly the
    scores that land on the line.
    """

    def __init__(self, name: str, features: Sequence[str], estimator: Any, *, target_col: str,
                 threshold_sign: float, support: Tuple[int, int], symmetric: bool, line_col: str = "line",
                 extra_rows: Optional[pd.DataFrame] = None, description: str = "", complexity: int = 1,
                 is_baseline: bool = False, uses_retrospective: bool = False, linear: bool = True,
                 pseudo_count: float = 5.0) -> None:
        self.name = name
        self.features = list(features)
        self.estimator = estimator
        self.target_col = target_col
        self.threshold_sign = float(threshold_sign)
        self.support = (int(support[0]), int(support[1]))
        self.symmetric = symmetric
        self.line_col = line_col
        self.extra_rows = extra_rows
        self.description = description
        self.complexity = complexity
        self.is_baseline = is_baseline
        self.uses_retrospective = uses_retrospective
        self.linear = linear
        self.pseudo_count = float(pseudo_count)
        self.calibration = "platt"
        self.pipe_: Optional[Pipeline] = None
        self.kept_: List[str] = []
        self.sd_: float = float("nan")
        self.weights_: Optional[np.ndarray] = None

    def clone(self) -> "ScoreDistCandidate":
        return ScoreDistCandidate(
            self.name, self.features, self.estimator, target_col=self.target_col,
            threshold_sign=self.threshold_sign, support=self.support, symmetric=self.symmetric,
            line_col=self.line_col, extra_rows=self.extra_rows, description=self.description,
            complexity=self.complexity, is_baseline=self.is_baseline,
            uses_retrospective=self.uses_retrospective, linear=self.linear, pseudo_count=self.pseudo_count,
        )

    def feature_names(self) -> List[str]:
        return list(self.features)

    # ------------------------------------------------------------------ fit
    @property
    def _grid(self) -> np.ndarray:
        return np.arange(self.support[0], self.support[1] + 1)

    def _base_pmf(self, expected: np.ndarray) -> np.ndarray:
        """Normal distribution around each prediction, cut into whole numbers."""
        edges = np.arange(self.support[0], self.support[1] + 2) - 0.5
        cdf = norm.cdf((edges[None, :] - expected[:, None]) / self.sd_)
        cdf[:, 0], cdf[:, -1] = 0.0, 1.0           # the tails go to the end points
        return np.diff(cdf, axis=1)

    def fit(self, train: pd.DataFrame, classes: Sequence[str]) -> "ScoreDistCandidate":
        if self.extra_rows is not None and len(self.extra_rows):
            extra = self.extra_rows[self.extra_rows["block"].isin(set(train["block"].unique()))]
            train = pd.concat([train, extra], ignore_index=True)
        self.kept_ = [c for c in self.features if train[c].notna().any()]
        if not self.kept_:
            raise ValueError(f"candidate {self.name!r}: every feature is missing in the training rows")
        steps: List[Tuple[str, Any]] = []
        if self.linear:
            steps += [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        steps.append(("model", sk_clone(self.estimator)))
        self.pipe_ = Pipeline(steps)
        X = train[self.kept_].to_numpy(dtype=float)
        q = train[self.target_col].to_numpy(dtype=float)
        self.pipe_.fit(X, q)
        expected = self.pipe_.predict(X)
        self.sd_ = float(np.sqrt(np.mean((q - expected) ** 2)))
        # How much more (or less) often each whole number occurs than a smooth
        # distribution says, shrunk towards "no difference" by a pseudo-count.
        observed = np.bincount(np.clip(np.rint(q), *self.support).astype(int) - self.support[0],
                               minlength=len(self._grid)).astype(float)
        smooth = self._base_pmf(expected).sum(axis=0)
        if self.symmetric:
            observed, smooth = self._fold(observed), self._fold(smooth)
        self.weights_ = (observed + self.pseudo_count) / (smooth + self.pseudo_count)
        return self

    def _fold(self, values: np.ndarray) -> np.ndarray:
        """Pool ``k`` with ``-k`` (a 3-point margin is as common either way)."""
        out = values.copy()
        for i, k in enumerate(self._grid):
            j = -k - self.support[0]
            if 0 <= j < len(values):
                out[i] = values[i] + values[j] if k != 0 else values[i]
        return out

    # -------------------------------------------------------------- predict
    def expected(self, rows: pd.DataFrame) -> np.ndarray:
        if self.pipe_ is None:
            raise RuntimeError("candidate is not fitted")
        return self.pipe_.predict(rows[self.kept_].to_numpy(dtype=float))

    def distribution(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Probability of every whole-number value of the quantity, one row per event."""
        pmf = self._base_pmf(self.expected(rows)) * self.weights_[None, :]
        pmf /= pmf.sum(axis=1, keepdims=True)
        return pd.DataFrame(pmf, index=rows.index, columns=self._grid)

    def outcome_proba(self, rows: pd.DataFrame) -> pd.DataFrame:
        """``p_low``, ``p_push`` and ``p_high`` of the quantity against each row's threshold."""
        pmf = self.distribution(rows).to_numpy()
        threshold = self.threshold_sign * rows[self.line_col].to_numpy(dtype=float)
        grid = self._grid[None, :]
        with np.errstate(invalid="ignore"):
            high = np.where(grid > threshold[:, None], pmf, 0.0).sum(axis=1)
            push = np.where(grid == threshold[:, None], pmf, 0.0).sum(axis=1)
        out = pd.DataFrame({"p_low": 1.0 - high - push, "p_push": push, "p_high": high}, index=rows.index)
        out[np.isnan(threshold)] = np.nan
        return out

    def push_proba(self, rows: pd.DataFrame) -> np.ndarray:
        return self.outcome_proba(rows)["p_push"].to_numpy()

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        o = self.outcome_proba(rows)
        live = (o["p_low"] + o["p_high"]).to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            high = np.where(live > 0, o["p_high"].to_numpy() / live, 0.5)
        high = np.clip(np.where(np.isnan(high), 0.5, high), 1e-9, 1.0 - 1e-9)   # no line: no opinion
        return np.column_stack([1.0 - high, high])

    def contributions(self, rows: pd.DataFrame) -> Optional[pd.DataFrame]:
        """What moves the expected score, in **points** (not log-odds): one
        column per feature for a linear model, plus the expected value, the
        intercept (an average game), the residual spread and the push
        probability under the keys in :data:`EXPLAIN_KEYS`."""
        if self.pipe_ is None:
            return None
        out = pd.DataFrame(index=rows.index)
        model = self.pipe_.named_steps["model"]
        if self.linear and hasattr(model, "coef_"):
            X = rows[self.kept_].to_numpy(dtype=float)
            for _, step in self.pipe_.steps[:-1]:
                X = step.transform(X)
            out = pd.DataFrame(X * np.ravel(model.coef_), columns=self.kept_, index=rows.index)
            out["_baseline"] = float(np.ravel(model.intercept_)[0])
        out["_expected"] = self.expected(rows)
        out["_sd"] = self.sd_
        out["_p_push"] = self.push_proba(rows)
        return out


def ridge_score_candidate(name: str, features: Sequence[str], *, alpha: float = 10.0, **kwargs: Any
                          ) -> ScoreDistCandidate:
    """Ridge regression of the score quantity on standardised features."""
    return ScoreDistCandidate(name, features, Ridge(alpha=alpha), linear=True, **kwargs)


def gbm_score_candidate(name: str, features: Sequence[str], *, seed: int = 20261112, **kwargs: Any
                        ) -> ScoreDistCandidate:
    """Shallow gradient-boosted regression; handles missing values natively."""
    est = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.03, max_iter=200, min_samples_leaf=60,
                                        l2_regularization=1.0, random_state=seed)
    return ScoreDistCandidate(name, features, est, linear=False, **kwargs)
