"""Probability-forecast metrics, reliability tables and cluster bootstrap.

All functions take ``P`` as an ``(n, k)`` array of class probabilities and
``y`` as integer class indices.  For binary markets ``k == 2`` and column 1
is side A.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "EPS",
    "check_probabilities",
    "accuracy",
    "log_loss",
    "brier",
    "rps",
    "ece",
    "reliability_table",
    "wilson_interval",
    "summary",
    "cluster_bootstrap",
    "paired_bootstrap_diff",
]

EPS = 1e-12


def check_probabilities(P: np.ndarray, atol: float = 1e-6) -> np.ndarray:
    """Validate a probability matrix and return it as float64."""
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[1] < 2:
        raise ValueError(f"probabilities must be (n, k>=2), got shape {P.shape}")
    if not np.isfinite(P).all():
        raise ValueError("probabilities contain NaN or infinity")
    if (P < -atol).any() or (P > 1 + atol).any():
        raise ValueError("probabilities outside [0, 1]")
    if not np.allclose(P.sum(axis=1), 1.0, atol=atol):
        raise ValueError("probability rows do not sum to 1")
    return P


def _prep(P: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    P = np.asarray(P, dtype=float)
    y = np.asarray(y).astype(int)
    if len(P) != len(y):
        raise ValueError("P and y differ in length")
    return P, y


def accuracy(P: np.ndarray, y: np.ndarray) -> float:
    P, y = _prep(P, y)
    return float((P.argmax(axis=1) == y).mean()) if len(y) else float("nan")


def log_loss(P: np.ndarray, y: np.ndarray) -> float:
    P, y = _prep(P, y)
    if not len(y):
        return float("nan")
    p = np.clip(P[np.arange(len(y)), y], EPS, 1.0)
    return float(-np.log(p).mean())


def brier(P: np.ndarray, y: np.ndarray) -> float:
    """Binary: mean (p_A - y)^2 (the usual 0..1 scale).  Multi-class: mean of
    the squared error summed over classes (0..2 scale)."""
    P, y = _prep(P, y)
    if not len(y):
        return float("nan")
    if P.shape[1] == 2:
        return float(((P[:, 1] - y) ** 2).mean())
    onehot = np.eye(P.shape[1])[y]
    return float(((P - onehot) ** 2).sum(axis=1).mean())


def rps(P: np.ndarray, y: np.ndarray) -> float:
    """Ranked probability score for ordered outcomes (e.g. home/draw/away)."""
    P, y = _prep(P, y)
    if not len(y):
        return float("nan")
    k = P.shape[1]
    onehot = np.eye(k)[y]
    diff = np.cumsum(P, axis=1)[:, :-1] - np.cumsum(onehot, axis=1)[:, :-1]
    return float((diff ** 2).sum(axis=1).mean() / (k - 1))


def wilson_interval(successes: float, n: float, z: float = 1.959964) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def reliability_table(P: np.ndarray, y: np.ndarray, n_bins: int = 10, *, top_label: bool = False) -> List[Dict]:
    """Predicted versus observed frequency by probability bin.

    Default: every (event, class) pair is pooled ("is a 70% call right 70% of
    the time, whichever outcome it is about?").  For binary markets that makes
    the table symmetric around 0.5 and uses every forecast twice; the reported
    ``n`` counts pairs.  ``top_label=True`` bins only each event's most likely
    class, which is the view that matters for selections.
    """
    P, y = _prep(P, y)
    if not len(y):
        return []
    if top_label:
        probs = P.max(axis=1)
        hits = (P.argmax(axis=1) == y).astype(float)
    else:
        onehot = np.eye(P.shape[1])[y]
        probs, hits = P.ravel(), onehot.ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        obs = float(hits[mask].mean())
        lo, hi = wilson_interval(hits[mask].sum(), n)
        rows.append({
            "bin_low": float(edges[b]), "bin_high": float(edges[b + 1]), "n": n,
            "mean_predicted": float(probs[mask].mean()), "observed": obs,
            "observed_low": lo, "observed_high": hi,
        })
    return rows


def ece(P: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error of the selected (most likely) class."""
    rows = reliability_table(P, y, n_bins, top_label=True)
    total = sum(r["n"] for r in rows)
    if not total:
        return float("nan")
    return float(sum(r["n"] * abs(r["observed"] - r["mean_predicted"]) for r in rows) / total)


def summary(P: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    out = {
        "n": int(len(y)),
        "accuracy": accuracy(P, y),
        "log_loss": log_loss(P, y),
        "brier": brier(P, y),
        "ece": ece(P, y),
    }
    if np.asarray(P).shape[1] == 3:
        out["rps"] = rps(P, y)
    return out


def _cluster_index(groups: Sequence) -> List[np.ndarray]:
    groups = np.asarray(groups)
    order = np.argsort(groups, kind="stable")
    sorted_groups = groups[order]
    bounds = np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]) + 1
    return np.split(order, bounds)


def cluster_bootstrap(
    metric: Callable[[np.ndarray, np.ndarray], float],
    P: np.ndarray,
    y: np.ndarray,
    groups: Sequence,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> Tuple[float, float]:
    """Percentile interval for ``metric`` resampling whole clusters.

    Events on the same day share news, schedules and officials, so resampling
    individual events would understate uncertainty; ``groups`` (usually the
    event date) are resampled instead.
    """
    P, y = _prep(P, y)
    clusters = _cluster_index(groups)
    if len(clusters) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(clusters), len(clusters))
        idx = np.concatenate([clusters[i] for i in pick])
        stats[b] = metric(P[idx], y[idx])
    alpha = (1.0 - level) / 2.0
    return (float(np.nanquantile(stats, alpha)), float(np.nanquantile(stats, 1 - alpha)))


def paired_bootstrap_diff(
    metric: Callable[[np.ndarray, np.ndarray], float],
    P_model: np.ndarray,
    P_reference: np.ndarray,
    y: np.ndarray,
    groups: Sequence,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> Dict[str, float]:
    """``metric(reference) - metric(model)`` with a cluster-bootstrap interval.

    For loss metrics a positive value means the model is better than the
    reference.  The same resampled events are scored for both, so the interval
    reflects the paired difference rather than two independent uncertainties.
    """
    P_model, y = _prep(P_model, y)
    P_reference = np.asarray(P_reference, dtype=float)
    point = metric(P_reference, y) - metric(P_model, y)
    clusters = _cluster_index(groups)
    if len(clusters) < 2:
        return {"diff": float(point), "low": float("nan"), "high": float("nan"), "se": float("nan")}
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(clusters), len(clusters))
        idx = np.concatenate([clusters[i] for i in pick])
        stats[b] = metric(P_reference[idx], y[idx]) - metric(P_model[idx], y[idx])
    alpha = (1.0 - level) / 2.0
    return {
        "diff": float(point),
        "low": float(np.nanquantile(stats, alpha)),
        "high": float(np.nanquantile(stats, 1 - alpha)),
        "se": float(np.nanstd(stats, ddof=1)),
    }
