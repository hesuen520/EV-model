"""Reusable candidates: a base-rate baseline and scikit-learn pipelines.

Every preprocessing step (imputation, scaling) lives inside the pipeline and
is therefore fitted on the training rows of each fold only.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone as sk_clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..sports.base import Candidate

__all__ = ["PriorCandidate", "SkCandidate", "logit_candidate", "gbm_candidate"]


class PriorCandidate(Candidate):
    """Class frequencies of the training rows (e.g. the home-win rate).

    The naive reference every model must beat.  With ``by`` it conditions on
    one categorical column (for instance neutral-site games).
    """

    is_baseline = True
    complexity = 0
    calibration = "none"

    def __init__(self, name: str = "base_rate", by: Optional[str] = None,
                 description: str = "historical outcome frequencies; knows nothing about the teams") -> None:
        self.name = name
        self.by = by
        self.description = description
        self._overall: Optional[np.ndarray] = None
        self._by: dict = {}

    def clone(self) -> "PriorCandidate":
        return PriorCandidate(self.name, self.by, self.description)

    def feature_names(self) -> List[str]:
        # The column the frequencies are split by is an input like any other: validation
        # hands a candidate only what it declares.
        return [self.by] if self.by is not None else []

    @staticmethod
    def _freq(y: np.ndarray, k: int) -> np.ndarray:
        counts = np.bincount(y.astype(int), minlength=k).astype(float) + 1.0  # Laplace smoothing
        return counts / counts.sum()

    def fit(self, train: pd.DataFrame, classes: Sequence[str]) -> "PriorCandidate":
        k = len(classes)
        self._overall = self._freq(train["y"].to_numpy(), k)
        self._by = {}
        if self.by is not None and self.by in train.columns:
            for key, part in train.groupby(self.by):
                if len(part) >= 200:
                    self._by[key] = self._freq(part["y"].to_numpy(), k)
        return self

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        if self._overall is None:
            raise RuntimeError("candidate is not fitted")
        out = np.tile(self._overall, (len(rows), 1))
        if self._by and self.by in rows.columns:
            keys = rows[self.by].to_numpy()
            for key, freq in self._by.items():
                out[keys == key] = freq
        return out


class SkCandidate(Candidate):
    """A scikit-learn classifier on a fixed list of point-in-time columns."""

    def __init__(
        self,
        name: str,
        features: Sequence[str],
        estimator: Any,
        *,
        description: str = "",
        complexity: int = 1,
        is_baseline: bool = False,
        uses_retrospective: bool = False,
        calibration: str = "platt",
        scale: bool = True,
        impute: bool = True,
    ) -> None:
        self.name = name
        self.features = list(features)
        self.estimator = estimator   # unfitted prototype; cloned for every fit
        self.description = description
        self.complexity = complexity
        self.is_baseline = is_baseline
        self.uses_retrospective = uses_retrospective
        self.calibration = calibration
        self.scale = scale
        self.impute = impute
        self.pipe_: Optional[Pipeline] = None
        self.n_classes_: int = 0
        self.kept_: List[str] = []

    def clone(self) -> "SkCandidate":
        return SkCandidate(
            self.name, self.features, self.estimator, description=self.description,
            complexity=self.complexity, is_baseline=self.is_baseline,
            uses_retrospective=self.uses_retrospective, calibration=self.calibration,
            scale=self.scale, impute=self.impute,
        )

    def feature_names(self) -> List[str]:
        return list(self.features)

    def fit(self, train: pd.DataFrame, classes: Sequence[str]) -> "SkCandidate":
        self.n_classes_ = len(classes)
        # A column that is entirely missing in the training rows cannot be
        # imputed; drop it for this fit rather than fail.
        self.kept_ = [c for c in self.features if train[c].notna().any()]
        if not self.kept_:
            raise ValueError(f"candidate {self.name!r}: every feature is missing in the training rows")
        steps = []
        if self.impute:
            steps.append(("impute", SimpleImputer(strategy="median")))
        if self.scale:
            steps.append(("scale", StandardScaler()))
        steps.append(("model", sk_clone(self.estimator)))
        self.pipe_ = Pipeline(steps)
        self.pipe_.fit(train[self.kept_].to_numpy(dtype=float), train["y"].to_numpy().astype(int))
        return self

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        if self.pipe_ is None:
            raise RuntimeError("candidate is not fitted")
        raw = self.pipe_.predict_proba(rows[self.kept_].to_numpy(dtype=float))
        out = np.zeros((len(rows), self.n_classes_))
        out[:, self.pipe_.named_steps["model"].classes_.astype(int)] = raw
        out = np.clip(out, 1e-9, 1.0)
        return out / out.sum(axis=1, keepdims=True)

    def contributions(self, rows: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Log-odds contributions towards side A for a binary linear model:
        coefficient times the standardised (imputed) feature value."""
        if self.pipe_ is None or self.n_classes_ != 2:
            return None
        model = self.pipe_.named_steps["model"]
        if not hasattr(model, "coef_"):
            return None
        X = rows[self.kept_].to_numpy(dtype=float)
        for step_name, step in self.pipe_.steps[:-1]:
            X = step.transform(X)
        return pd.DataFrame(X * model.coef_[0], columns=self.kept_, index=rows.index)


def logit_candidate(name: str, features: Sequence[str], *, C: float = 1.0, description: str = "",
                    complexity: int = 1, is_baseline: bool = False, uses_retrospective: bool = False,
                    calibration: str = "platt") -> SkCandidate:
    """L2-regularised logistic regression (multinomial when there are 3 classes)."""
    return SkCandidate(
        name, features, LogisticRegression(C=C, max_iter=2000),
        description=description, complexity=complexity, is_baseline=is_baseline,
        uses_retrospective=uses_retrospective, calibration=calibration,
    )


def gbm_candidate(name: str, features: Sequence[str], *, description: str = "", complexity: int = 5,
                  max_depth: int = 3, learning_rate: float = 0.05, max_iter: int = 300,
                  min_samples_leaf: int = 80, l2: float = 1.0, uses_retrospective: bool = False,
                  seed: int = 20261112) -> SkCandidate:
    """Shallow histogram gradient boosting; handles missing values natively."""
    return SkCandidate(
        name, features,
        HistGradientBoostingClassifier(
            max_depth=max_depth, learning_rate=learning_rate, max_iter=max_iter,
            min_samples_leaf=min_samples_leaf, l2_regularization=l2, random_state=seed,
        ),
        description=description, complexity=complexity, uses_retrospective=uses_retrospective,
        calibration="platt", scale=False, impute=False,
    )
