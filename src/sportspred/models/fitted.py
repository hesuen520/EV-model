"""Production models: a fitted candidate, its calibrator and bootstrap replicas.

The replicas quantify *model* uncertainty - how much the estimated probability
moves when the model is refitted on resampled history.  That is separate from
event uncertainty (a 65% favourite loses 35% of the time even with a perfect
model) and from data quality (missing or stale inputs).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from ..core.timeutil import iso_z, utcnow
from ..sports.base import Candidate, FeatureFrame, SplitPlan
from ..validation.backtest import fit_with_calibration, slate_dates
from ..validation.calibration import Calibrator
from ..validation.metrics import check_probabilities
from ..validation.splits import ordered_blocks

__all__ = ["FittedModel", "train_production", "production_split"]


@dataclass
class FittedModel:
    league: str
    market: str
    model_name: str
    model_version: str
    classes: Tuple[str, ...]
    candidate: Candidate
    calibrator: Calibrator
    replicas: List[Tuple[Candidate, Calibrator]] = field(default_factory=list)
    trained_at_utc: str = ""
    train_start_utc: Optional[str] = None
    train_end_utc: Optional[str] = None
    n_train: int = 0
    n_calib: int = 0
    interval_level: float = 0.90
    info: Dict[str, Any] = field(default_factory=dict)

    def predict(self, rows: pd.DataFrame) -> Dict[str, np.ndarray]:
        """``P`` (n, k) calibrated probabilities and the ``low``/``high``
        bootstrap interval per class (equal to ``P`` without replicas)."""
        P = check_probabilities(self.calibrator.transform(self.candidate.predict_proba(rows)))
        if not self.replicas or not len(rows):
            return {"P": P, "low": P.copy(), "high": P.copy()}
        stack = np.stack([cal.transform(model.predict_proba(rows)) for model, cal in self.replicas])
        alpha = (1.0 - self.interval_level) / 2.0
        low = np.minimum(np.quantile(stack, alpha, axis=0), P)
        high = np.maximum(np.quantile(stack, 1.0 - alpha, axis=0), P)
        return {"P": P, "low": low, "high": high}

    def contributions(self, rows: pd.DataFrame) -> Optional[pd.DataFrame]:
        return self.candidate.contributions(rows)

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        return path

    @staticmethod
    def load(path: Union[str, Path]) -> "FittedModel":
        obj = joblib.load(path)
        if not isinstance(obj, FittedModel):
            raise TypeError(f"{path} does not contain a FittedModel")
        return obj


def production_split(completed: pd.DataFrame, plan: SplitPlan) -> Tuple[List[Any], List[Any]]:
    """Blocks used to fit the production model and (later) blocks used to
    calibrate it: the most recent ``n_calib_blocks`` full blocks plus any
    trailing partial block are held out for calibration."""
    g = ordered_blocks(completed, plan.block_col)
    blocks = g[plan.block_col].tolist()
    sizes = dict(zip(blocks, g["size"]))
    calib: List[Any] = []
    full = 0
    for b in reversed(blocks):
        calib.insert(0, b)
        if sizes[b] >= plan.min_block_events:
            full += 1
        if full >= plan.n_calib_blocks:
            break
    train = [b for b in blocks if b not in calib][plan.burn_in_blocks:]
    if not train:
        raise ValueError("not enough history to train a production model")
    return train, calib


def _resample(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Cluster bootstrap by slate date, keeping chronological order."""
    days = slate_dates(frame["start_time_utc"])
    uniq, inverse = np.unique(days, return_inverse=True)
    pick = rng.integers(0, len(uniq), len(uniq))
    counts = np.bincount(pick, minlength=len(uniq))
    reps = counts[inverse]
    return frame.loc[frame.index.repeat(reps)]


def train_production(
    frame: FeatureFrame,
    candidate: Candidate,
    plan: SplitPlan,
    *,
    n_replicas: int = 30,
    seed: int = 20261112,
    calib_min_rows: int = 100,
    validation_run_id: Optional[str] = None,
) -> FittedModel:
    """Fit ``candidate`` on all completed history (calibrated on the most
    recent block) and draw bootstrap replicas for model-uncertainty intervals."""
    frame.validate()
    df = frame.completed.sort_values(["start_time_utc", "event_id"]).reset_index(drop=True)
    train_blocks, calib_blocks = production_split(df, plan)
    train = df[df[plan.block_col].isin(train_blocks)]
    calib = df[df[plan.block_col].isin(calib_blocks)]
    model, cal = fit_with_calibration(candidate, train, calib, frame.classes, calib_min_rows=calib_min_rows)
    rng = np.random.default_rng(seed)
    replicas = []
    for _ in range(n_replicas):
        replicas.append(fit_with_calibration(candidate, _resample(train, rng), _resample(calib, rng),
                                             frame.classes, calib_min_rows=calib_min_rows))
    trained = utcnow()
    digest = hashlib.sha1(
        ("|".join(candidate.feature_names()) + iso_z(df["start_time_utc"].max()) + str(len(df))).encode()
    ).hexdigest()[:8]
    version = f"{frame.league}.{frame.market}.{candidate.name}.{trained.strftime('%Y%m%dT%H%M')}.{digest}"
    return FittedModel(
        league=frame.league, market=frame.market, model_name=candidate.name, model_version=version,
        classes=tuple(frame.classes), candidate=model, calibrator=cal, replicas=replicas,
        trained_at_utc=iso_z(trained), train_start_utc=iso_z(train["start_time_utc"].min()),
        train_end_utc=iso_z(calib["start_time_utc"].max() if len(calib) else train["start_time_utc"].max()),
        n_train=int(len(train)), n_calib=int(len(calib)),
        info={
            "train_blocks": [str(b) for b in train_blocks], "calib_blocks": [str(b) for b in calib_blocks],
            "calibrator": cal.describe(), "n_replicas": n_replicas, "features": candidate.feature_names(),
            "validation_run_id": validation_run_id, "feature_frame": frame.extra,
        },
    )
