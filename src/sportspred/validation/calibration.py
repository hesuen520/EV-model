"""Probability calibrators, fitted only on a calibration block.

A calibrator maps a model's raw class probabilities to calibrated ones.  It is
always fitted on rows that are later than the model's training rows and
earlier than the rows it is applied to.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

from .metrics import EPS

__all__ = ["Calibrator", "IdentityCalibrator", "PlattCalibrator", "IsotonicCalibrator", "fit_calibrator"]


class Calibrator:
    method = "none"

    def fit(self, P: np.ndarray, y: np.ndarray) -> "Calibrator":
        return self

    def transform(self, P: np.ndarray) -> np.ndarray:
        return np.asarray(P, dtype=float)

    def describe(self) -> dict:
        return {"method": self.method}


class IdentityCalibrator(Calibrator):
    """No calibration (also the fallback when a calibration block is tiny)."""


class PlattCalibrator(Calibrator):
    """Logistic recalibration on the log-probability scale.

    Binary: ``logit(p') = a + b * logit(p)``.  Multi-class: ``log p'_c
    proportional to b * log p_c + a_c`` (temperature plus per-class bias).
    ``b`` is constrained positive, so the ranking of outcomes is preserved,
    and a light ridge penalty pulls towards the identity map (a=0, b=1) so a
    small calibration block cannot produce a wild correction.
    """

    method = "platt"

    def __init__(self, ridge: float = 1e-3) -> None:
        self.ridge = ridge
        self.a_: Optional[np.ndarray] = None
        self.b_: float = 1.0

    def fit(self, P: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        P = np.clip(np.asarray(P, dtype=float), EPS, 1.0)
        y = np.asarray(y).astype(int)
        k = P.shape[1]
        logp = np.log(P)
        n = len(y)

        def unpack(theta: np.ndarray):
            a = np.concatenate([[0.0], theta[: k - 1]])  # first class bias fixed at 0
            b = np.exp(theta[k - 1])
            return a, b

        def loss(theta: np.ndarray) -> float:
            a, b = unpack(theta)
            z = b * logp + a
            z = z - z.max(axis=1, keepdims=True)
            lse = np.log(np.exp(z).sum(axis=1))
            nll = -(z[np.arange(n), y] - lse).mean()
            return float(nll + self.ridge * (np.sum(theta[: k - 1] ** 2) + theta[k - 1] ** 2))

        res = minimize(loss, np.zeros(k), method="L-BFGS-B")
        a, b = unpack(res.x)
        self.a_, self.b_ = a, float(b)
        return self

    def transform(self, P: np.ndarray) -> np.ndarray:
        if self.a_ is None:
            raise RuntimeError("calibrator is not fitted")
        P = np.clip(np.asarray(P, dtype=float), EPS, 1.0)
        z = self.b_ * np.log(P) + self.a_
        z = z - z.max(axis=1, keepdims=True)
        out = np.exp(z)
        return out / out.sum(axis=1, keepdims=True)

    def describe(self) -> dict:
        return {"method": self.method, "slope": self.b_, "bias": None if self.a_ is None else self.a_.tolist()}


class IsotonicCalibrator(Calibrator):
    """Monotone, non-parametric recalibration of side A's probability (binary only)."""

    method = "isotonic"

    def __init__(self) -> None:
        self.iso_: Optional[IsotonicRegression] = None

    def fit(self, P: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        P = np.asarray(P, dtype=float)
        if P.shape[1] != 2:
            raise ValueError("isotonic calibration is implemented for binary markets only")
        self.iso_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.iso_.fit(P[:, 1], np.asarray(y).astype(float))
        return self

    def transform(self, P: np.ndarray) -> np.ndarray:
        if self.iso_ is None:
            raise RuntimeError("calibrator is not fitted")
        p = np.clip(self.iso_.predict(np.asarray(P, dtype=float)[:, 1]), 1e-4, 1 - 1e-4)
        return np.column_stack([1.0 - p, p])


def fit_calibrator(method: str, P: np.ndarray, y: np.ndarray, *, min_rows: int = 100) -> Calibrator:
    """Fit the named calibrator; fall back to identity on a tiny block or when
    a class is absent from it (a calibrator cannot be learned from that)."""
    y = np.asarray(y).astype(int)
    if method == "none" or len(y) < min_rows or len(np.unique(y)) < np.asarray(P).shape[1]:
        return IdentityCalibrator()
    if method == "platt":
        return PlattCalibrator().fit(P, y)
    if method == "isotonic":
        return IsotonicCalibrator().fit(P, y)
    raise ValueError(f"unknown calibration method {method!r}")
