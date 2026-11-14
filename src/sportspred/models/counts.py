"""Count-distribution candidates for statistic over/under markets.

A threshold market on a count (a player's points, rebounds, ...) needs more
than a projected average: it needs the probability of landing above, below or
exactly on the threshold.  Every candidate here therefore has two parts:

* a **mean model** - the expected count for the row;
* a **count distribution** around that mean (Poisson, or negative binomial
  with a dispersion learned from the training rows), from which
  ``P(count > line)``, ``P(count < line)`` and ``P(count == line)`` follow
  for any line, whole or half.

``predict_proba`` returns ``[P(under), P(over)]`` conditional on no push, the
shape the shared validation protocol expects for an over/under market;
``push_prob`` returns the probability of an exact tie with the line (zero for
half-point lines), which the prediction engine records separately.

The frame supplies two outcome columns the mean models learn from -
``count_col`` (the statistic) and, for the two-stage model, ``exposure_col``
(minutes actually played).  Neither is a declared input, so validation blanks
them on every row a candidate is scored on.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, special
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..sports.base import Candidate

__all__ = ["CountCandidate", "ColumnMean", "ProductMean", "TwoStageMean", "count_probabilities",
           "threshold_probabilities", "fit_dispersion", "nb_alpha"]

_MU_FLOOR = 0.05


# ----------------------------------------------------------------- distribution
def nb_alpha(mu: np.ndarray, params: Optional[Tuple[float, float]]) -> np.ndarray:
    """Negative-binomial dispersion ``alpha`` (variance = mu + alpha * mu^2) as
    a function of the mean: ``alpha = exp(a + b * log(mu))``.  ``None`` means
    Poisson (alpha = 0)."""
    if params is None:
        return np.zeros_like(mu, dtype=float)
    a, b = params
    return np.exp(np.clip(a + b * np.log(np.clip(mu, _MU_FLOOR, None)), -12.0, 3.0))


def _cdf(k: np.ndarray, mu: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """P(count <= k) for a negative binomial (Poisson where alpha is ~0)."""
    k = np.asarray(k, dtype=float)
    out = np.zeros_like(mu, dtype=float)
    ok = k >= 0
    pois = alpha < 1e-6
    use = ok & pois
    if use.any():
        out[use] = special.pdtr(np.floor(k[use]), mu[use])
    use = ok & ~pois
    if use.any():
        n = 1.0 / alpha[use]
        p = n / (n + mu[use])
        out[use] = special.betainc(n, np.floor(k[use]) + 1.0, p)
    return out


def count_probabilities(mu: np.ndarray, line: np.ndarray, params: Optional[Tuple[float, float]] = None
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(P(under), P(push), P(over))`` of a count with mean ``mu`` against
    ``line``.  A whole-number line can be tied; a half-point line cannot."""
    mu = np.clip(np.asarray(mu, dtype=float), _MU_FLOOR, None)
    line = np.asarray(line, dtype=float)
    if np.isnan(line).any():
        raise ValueError("a count candidate cannot price a row without a line")
    alpha = nb_alpha(mu, params)
    at_or_below = _cdf(np.floor(line), mu, alpha)               # P(count <= floor(line))
    below = _cdf(np.ceil(line) - 1.0, mu, alpha)                # P(count < line)
    over = np.clip(1.0 - at_or_below, 0.0, 1.0)
    under = np.clip(below, 0.0, 1.0)
    whole = line == np.floor(line)
    # Off a whole number there is no tie at all - not even a rounding remainder.
    push = np.where(whole, np.clip(1.0 - over - under, 0.0, 1.0), 0.0)
    under = np.where(whole, under, 1.0 - over)
    return under, push, over


def threshold_probabilities(over_at: Any, line: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(P(under), P(push), P(over))`` from one function of half-point
    thresholds.

    ``over_at(h)`` gives the (calibrated) probability of finishing above each
    half-point threshold ``h``.  A whole-number threshold ``L`` is read at the
    half points either side: over ``L`` is over ``L + 0.5``, under ``L`` is
    not over ``L - 0.5``, and the tie is what remains.  One event then has one
    probability whichever threshold it is asked through - "over 23" and "over
    23.5" are the same event - and the three always sum to one."""
    line = np.asarray(line, dtype=float)
    whole = line == np.floor(line)
    over = np.asarray(over_at(np.where(whole, line + 0.5, line)), dtype=float)
    not_under = np.asarray(over_at(np.where(whole, np.maximum(line - 0.5, 0.5), line)), dtype=float)
    under = np.where(whole & (line < 1.0), 0.0, 1.0 - not_under)          # nothing is under zero
    under = np.minimum(under, 1.0 - over)
    push = np.where(whole, np.clip(1.0 - over - under, 0.0, 1.0), 0.0)
    return under, push, over


def _nb_nll(params: Tuple[float, float], mu: np.ndarray, y: np.ndarray) -> float:
    alpha = nb_alpha(mu, params)
    n = 1.0 / alpha
    ll = (special.gammaln(y + n) - special.gammaln(n) - special.gammaln(y + 1.0)
          + n * np.log(n / (n + mu)) + y * np.log(mu / (n + mu)))
    return float(-ll.mean())


def fit_dispersion(mu: np.ndarray, y: np.ndarray, *, max_rows: int = 60000, seed: int = 20261112
                   ) -> Tuple[float, float]:
    """Maximum-likelihood ``(a, b)`` of the dispersion curve on (mean, count)
    pairs; at most ``max_rows`` rows (a fixed random subset) are used."""
    mu = np.clip(np.asarray(mu, dtype=float), _MU_FLOOR, None)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(mu) & np.isfinite(y) & (y >= 0)
    mu, y = mu[keep], y[keep]
    if len(mu) < 50:
        return (-3.0, 0.0)
    if len(mu) > max_rows:
        pick = np.random.default_rng(seed).choice(len(mu), size=max_rows, replace=False)
        mu, y = mu[pick], y[pick]
    res = optimize.minimize(_nb_nll, x0=np.array([-2.0, -0.5]), args=(mu, y), method="Nelder-Mead",
                            options={"xatol": 1e-3, "fatol": 1e-7, "maxiter": 400})
    return float(res.x[0]), float(res.x[1])


# ------------------------------------------------------------------ mean models
class ColumnMean:
    """The mean is a column of the frame (first non-missing of ``cols``)."""

    def __init__(self, cols: Sequence[str]) -> None:
        self.cols = list(cols)

    def features(self) -> List[str]:
        return list(self.cols)

    def clone(self) -> "ColumnMean":
        return ColumnMean(self.cols)

    def fit(self, train: pd.DataFrame, count_col: str) -> "ColumnMean":
        self.fallback_ = float(np.nanmean(train[count_col].to_numpy(dtype=float)))
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        out = np.full(len(rows), np.nan)
        for col in self.cols:
            values = rows[col].to_numpy(dtype=float)
            out = np.where(np.isnan(out), values, out)
        return np.where(np.isnan(out), self.fallback_, out)

    def parts(self, rows: pd.DataFrame) -> Dict[str, np.ndarray]:
        return {"mean": self.predict(rows)}


class ProductMean:
    """Expected exposure times expected rate, both taken from columns."""

    def __init__(self, exposure: str, rate: str) -> None:
        self.exposure, self.rate = exposure, rate

    def features(self) -> List[str]:
        return [self.exposure, self.rate]

    def clone(self) -> "ProductMean":
        return ProductMean(self.exposure, self.rate)

    def fit(self, train: pd.DataFrame, count_col: str) -> "ProductMean":
        self.fallback_ = float(np.nanmean(train[count_col].to_numpy(dtype=float)))
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        mu = rows[self.exposure].to_numpy(dtype=float) * rows[self.rate].to_numpy(dtype=float)
        return np.where(np.isnan(mu), self.fallback_, mu)

    def parts(self, rows: pd.DataFrame) -> Dict[str, np.ndarray]:
        return {"exposure": rows[self.exposure].to_numpy(dtype=float),
                "rate": rows[self.rate].to_numpy(dtype=float), "mean": self.predict(rows)}


class TwoStageMean:
    """Playing time, then production per minute.

    * exposure model: expected minutes from ``exposure_features``;
    * rate model: the statistic per minute from ``rate_features``, fitted as a
      Poisson model with the minutes actually played as exposure (each
      training row weighted by its minutes);
    * mean = expected minutes x expected rate.

    ``kind="glm"``: ridge regression for minutes and a log-linear Poisson
    regression for the rate (median imputation and scaling inside the
    pipeline).  ``kind="gbm"``: shallow gradient-boosted trees for both.
    """

    def __init__(self, exposure_features: Sequence[str], rate_features: Sequence[str], *, kind: str = "glm",
                 exposure_col: str = "minutes_played", seed: int = 20261112) -> None:
        if kind not in ("glm", "gbm"):
            raise ValueError(f"unknown two-stage kind {kind!r}")
        self.exposure_features, self.rate_features = list(exposure_features), list(rate_features)
        self.kind, self.exposure_col, self.seed = kind, exposure_col, seed

    def features(self) -> List[str]:
        return list(dict.fromkeys(self.exposure_features + self.rate_features))

    def clone(self) -> "TwoStageMean":
        return TwoStageMean(self.exposure_features, self.rate_features, kind=self.kind,
                            exposure_col=self.exposure_col, seed=self.seed)

    def _models(self) -> Tuple[Any, Any]:
        if self.kind == "glm":
            minutes = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                                ("model", Ridge(alpha=1.0))])
            rate = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                             ("model", PoissonRegressor(alpha=1e-4, max_iter=300))])
            return minutes, rate
        common = dict(max_depth=3, learning_rate=0.06, max_iter=250, min_samples_leaf=200,
                      l2_regularization=1.0, random_state=self.seed)
        return (HistGradientBoostingRegressor(loss="squared_error", **common),
                HistGradientBoostingRegressor(loss="poisson", **common))

    def fit(self, train: pd.DataFrame, count_col: str) -> "TwoStageMean":
        minutes = train[self.exposure_col].to_numpy(dtype=float)
        count = train[count_col].to_numpy(dtype=float)
        ok = np.isfinite(minutes) & np.isfinite(count) & (minutes > 0)
        part = train[ok]
        minutes, count = minutes[ok], count[ok]
        self.exp_kept_ = [c for c in self.exposure_features if part[c].notna().any()]
        self.rate_kept_ = [c for c in self.rate_features if part[c].notna().any()]
        self.minutes_model_, self.rate_model_ = self._models()
        self.minutes_model_.fit(part[self.exp_kept_].to_numpy(dtype=float), minutes)
        weight_key = "model__sample_weight" if self.kind == "glm" else "sample_weight"
        self.rate_model_.fit(part[self.rate_kept_].to_numpy(dtype=float), count / minutes, **{weight_key: minutes})
        self.minutes_range_ = (float(np.quantile(minutes, 0.001)), float(np.quantile(minutes, 0.999)))
        return self

    def parts(self, rows: pd.DataFrame) -> Dict[str, np.ndarray]:
        minutes = self.minutes_model_.predict(rows[self.exp_kept_].to_numpy(dtype=float))
        minutes = np.clip(minutes, max(self.minutes_range_[0], 1.0), self.minutes_range_[1])
        rate = np.clip(self.rate_model_.predict(rows[self.rate_kept_].to_numpy(dtype=float)), 1e-4, None)
        return {"exposure": minutes, "rate": rate, "mean": minutes * rate}

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return self.parts(rows)["mean"]


# -------------------------------------------------------------------- candidate
class CountCandidate(Candidate):
    """A mean model plus a count distribution, read at each row's ``line``."""

    def __init__(self, name: str, mean: Any, *, dist: str = "nb", description: str = "", complexity: int = 1,
                 is_baseline: bool = False, uses_retrospective: bool = False, calibration: str = "platt",
                 count_col: str = "stat_value", line_col: str = "line") -> None:
        if dist not in ("poisson", "nb"):
            raise ValueError(f"unknown count distribution {dist!r}")
        self.name, self.mean, self.dist = name, mean, dist
        self.description, self.complexity = description, complexity
        self.is_baseline, self.uses_retrospective, self.calibration = is_baseline, uses_retrospective, calibration
        self.count_col, self.line_col = count_col, line_col
        self.dispersion_: Optional[Tuple[float, float]] = None
        self.fitted_ = False

    def clone(self) -> "CountCandidate":
        return CountCandidate(self.name, self.mean.clone(), dist=self.dist, description=self.description,
                              complexity=self.complexity, is_baseline=self.is_baseline,
                              uses_retrospective=self.uses_retrospective, calibration=self.calibration,
                              count_col=self.count_col, line_col=self.line_col)

    def feature_names(self) -> List[str]:
        return self.mean.features()

    def fit(self, train: pd.DataFrame, classes: Sequence[str]) -> "CountCandidate":
        if tuple(classes) != ("under", "over"):
            raise ValueError("count candidates serve over/under markets with classes ('under', 'over')")
        self.mean.fit(train, self.count_col)
        if self.dist == "nb":
            self.dispersion_ = fit_dispersion(self.mean.predict(train), train[self.count_col].to_numpy(dtype=float))
        self.fitted_ = True
        return self

    def _parts(self, rows: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted_:
            raise RuntimeError("candidate is not fitted")
        return count_probabilities(self.mean.predict(rows), rows[self.line_col].to_numpy(dtype=float),
                                   self.dispersion_)

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        under, push, over = self._parts(rows)
        decided = np.clip(under + over, 1e-12, None)
        P = np.column_stack([under / decided, over / decided])
        P = np.clip(P, 1e-9, 1.0)
        return P / P.sum(axis=1, keepdims=True)

    def push_prob(self, rows: pd.DataFrame) -> np.ndarray:
        """Probability of landing exactly on the line (zero for half-point lines)."""
        return self._parts(rows)[1]

    def projection(self, rows: pd.DataFrame) -> pd.DataFrame:
        """The pieces behind the probability: expected exposure and rate where
        the mean model has them, the mean, and the 10th/50th/90th percentiles
        of the count distribution."""
        parts = self.mean.parts(rows)
        mu = np.clip(parts["mean"], _MU_FLOOR, None)
        alpha = nb_alpha(mu, self.dispersion_)
        out = pd.DataFrame({k: v for k, v in parts.items()}, index=rows.index)
        grid = np.arange(0, int(np.ceil(mu.max() * 4 + 20)))
        cdf = np.stack([_cdf(np.full(len(mu), k), mu, alpha) for k in grid], axis=1)
        for q in (0.1, 0.5, 0.9):
            out[f"q{int(q * 100)}"] = (cdf < q).sum(axis=1).astype(float)
        return out
