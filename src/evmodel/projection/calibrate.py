"""Fit and check the shape of each market's predictive distribution.

The walk-forward features give a projected mean for every historical game.
Comparing those means with what actually happened tells us

* how spread out results are around the mean - the variance function
  ``var = c + phi*mean + alpha*mean**2`` of a
  :class:`~evmodel.distributions.FamilySpec`, fitted by maximum likelihood
  (:func:`fit_spec`, :func:`select_spec`);
* whether the resulting over/under probabilities are **calibrated**: among
  all props the model prices at 60%, do about 60% go over?
  (:func:`calibration_report`).

There are no free historical prop lines, so calibration is checked against
synthetic lines placed where books place them: at the half-point next to the
model's median, and a little either side of it.

Everything here is vectorised: one call scores hundreds of thousands of
player-games.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import special
from scipy.optimize import minimize

from ..distributions import FamilySpec

__all__ = [
    "cdf_matrix",
    "log_likelihood",
    "fit_spec",
    "select_spec",
    "over_probabilities",
    "calibration_report",
    "CalibrationResult",
    "reliability_table",
]

_TINY = 1e-300


# ---------------------------------------------------------------- vector maths
def _variance(spec: FamilySpec, mean: np.ndarray) -> np.ndarray:
    """Vectorised :meth:`FamilySpec.variance`."""
    mean = np.asarray(mean, dtype=float)
    m = np.maximum(mean, 0.0)
    if spec.family == "poisson":
        return m
    if spec.family == "normal" and spec.lower is None:
        return spec.c + spec.phi * m + spec.alpha * m * m
    var = spec.c * m / (m + 1.0) + spec.phi * m + spec.alpha * m * m
    if spec.family == "negbin":
        var = np.maximum(var, spec.scale * m)
    return var


def _clamped_latent_mean(target: np.ndarray, sd: np.ndarray, lower: int) -> np.ndarray:
    """Vectorised latent mean of a clamped, rounded normal with overall mean ``target``.

    Bisection on the same increasing function the scalar implementation in
    :mod:`evmodel.distributions` solves.
    """
    threshold = lower + 0.5
    target, sd = np.broadcast_arrays(np.asarray(target, float), np.asarray(sd, float))
    lo = threshold - 40.0 * sd
    hi = target + 1.0
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        t = (threshold - mid) / sd
        tail = special.ndtr(-t)
        value = lower * (1.0 - tail) + mid * tail + sd * np.exp(-0.5 * t * t) / math.sqrt(2.0 * math.pi)
        too_high = value > target
        hi = np.where(too_high, mid, hi)
        lo = np.where(too_high, lo, mid)
    return 0.5 * (lo + hi)


def _cdf(spec: FamilySpec, mean: np.ndarray, k: np.ndarray) -> np.ndarray:
    """``P(X <= k)`` elementwise for arrays of means and integer points.

    Mirrors ``spec.dist(mean).cdf(k)``.
    """
    mean = np.asarray(mean, dtype=float)
    k = np.asarray(k, dtype=float)
    pi = spec.zero_inflation
    base_mean = mean / (1.0 - pi) if pi else mean
    if spec.family == "normal":
        sd = np.sqrt(np.maximum(_variance(spec, base_mean), 1e-9))
        if spec.lower is None:
            latent = base_mean
        else:
            latent = _clamped_latent_mean(np.maximum(base_mean, spec.lower + 1e-9), sd, spec.lower)
        out = special.ndtr((k + 0.5 - latent) / sd)
        if spec.lower is not None:
            out = np.where(k < spec.lower, 0.0, out)
    else:
        m = np.maximum(base_mean, 1e-9)
        kk = np.maximum(k, 0.0)
        if spec.family == "poisson":
            out = special.gammaincc(kk + 1.0, m)
        elif spec.family == "negbin":
            # Event count N = X / scale (scale 1 is the ordinary negative binomial).
            scale = spec.scale
            event_mean = m / scale
            event_var = _variance(spec, m) / (scale * scale)
            over = event_var > event_mean * (1.0 + 1e-7)
            r = np.where(over, event_mean**2 / np.maximum(event_var - event_mean, 1e-300), 1.0)
            p = r / (r + event_mean)
            x = (kk + 0.5) / scale + 0.5 if scale != 1.0 else kk + 1.0
            out = np.where(over, special.betainc(r, x, p), special.gammaincc(x, event_mean))
        elif spec.family == "gamma":
            var = np.maximum(_variance(spec, m), 1e-9)
            out = special.gammainc(m * m / var, (kk + 0.5) * m / var)
        else:  # pragma: no cover - FamilySpec validates the family
            raise ValueError(spec.family)
        out = np.where(k < 0, 0.0, out)
    if pi:
        out = pi * (k >= 0) + (1.0 - pi) * out
    return out


def cdf_matrix(spec: FamilySpec, mean: np.ndarray, k_min: int, k_max: int) -> np.ndarray:
    """Matrix ``C[i, j] = P(X_i <= k_min + j)`` for a vector of means."""
    grid = np.arange(k_min, k_max + 1, dtype=float)
    return _cdf(spec, np.asarray(mean, float)[:, None], grid[None, :])


def log_likelihood(spec: FamilySpec, mean: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Log-probability of each observed integer outcome under ``spec``."""
    y = np.rint(np.asarray(y, dtype=float))
    mean = np.asarray(mean, dtype=float)
    if spec.family in ("poisson", "negbin") and not spec.zero_inflation and spec.scale == 1.0:
        # Closed-form pmf: much faster than differencing the cdf.
        m = np.maximum(mean, 1e-9)
        yy = np.maximum(y, 0.0)
        poisson = yy * np.log(m) - m - special.gammaln(yy + 1.0)
        if spec.family == "poisson":
            out = poisson
        else:
            var = _variance(spec, m)
            over = var > m * (1.0 + 1e-7)
            r = np.where(over, m * m / np.maximum(var - m, 1e-300), 1.0)
            negbin = (
                special.gammaln(yy + r) - special.gammaln(r) - special.gammaln(yy + 1.0)
                + r * np.log(r / (r + m)) + yy * np.log(m / (r + m))
            )
            out = np.where(over & (r < 1e5), negbin, poisson)
        return np.where(y < 0, math.log(_TINY), out)
    prob = _cdf(spec, mean, y) - _cdf(spec, mean, y - 1.0)
    return np.log(np.maximum(prob, _TINY))


def over_probabilities(spec: FamilySpec, mean: np.ndarray, line: np.ndarray) -> np.ndarray:
    """``P(X > line)`` elementwise (whole-number lines: strictly greater)."""
    line = np.asarray(line, dtype=float)
    return 1.0 - _cdf(spec, np.asarray(mean, float), np.floor(line + 1e-9))


def _median(spec: FamilySpec, mean: np.ndarray, y_max: float) -> np.ndarray:
    """Vectorised median: smallest ``k`` with ``P(X <= k) >= 0.5``."""
    mean = np.asarray(mean, dtype=float)
    lo = int(min(0, math.floor(np.min(mean) - 4 * math.sqrt(max(np.max(_variance(spec, np.abs(mean))), 1.0)))))
    hi = int(max(y_max, np.max(mean) * 2.0 + 20.0))
    out = np.empty(len(mean))
    for start in range(0, len(mean), 20_000):  # bound the size of the cdf matrix
        chunk = mean[start:start + 20_000]
        matrix = cdf_matrix(spec, chunk, lo, hi)
        out[start:start + 20_000] = lo + np.argmax(matrix >= 0.5, axis=1)
    return out


# ------------------------------------------------------------------- fitting
def _moment_start(mean: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Rough ``(c, phi, alpha)`` from a non-negative regression of squared errors."""
    from scipy.optimize import nnls

    design = np.column_stack([np.ones_like(mean), mean, mean * mean])
    coef, _ = nnls(design, (y - mean) ** 2)
    return float(coef[0]), float(coef[1]), float(coef[2])


def _compress(mean: np.ndarray, y: np.ndarray, n_bins: int = 300) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse observations into weighted ``(mean, outcome)`` cells.

    The likelihood only depends on the projected mean and the integer
    outcome, so games are grouped into narrow quantile bins of the mean
    (each represented by its average) and identical outcomes within a bin are
    counted once.  This uses every observation while evaluating the
    likelihood on a few thousand cells instead of hundreds of thousands of
    rows.
    """
    order = np.argsort(mean, kind="stable")
    mean, y = mean[order], np.rint(y[order])
    n = len(mean)
    bins = min(n_bins, n)
    bin_id = (np.arange(n) * bins) // n
    sums = np.bincount(bin_id, weights=mean, minlength=bins)
    counts = np.bincount(bin_id, minlength=bins)
    bin_mean = sums / np.maximum(counts, 1)
    frame = pd.DataFrame({"bin": bin_id, "y": y})
    cells = frame.groupby(["bin", "y"], sort=False).size().reset_index(name="w")
    return bin_mean[cells["bin"].to_numpy()], cells["y"].to_numpy(float), cells["w"].to_numpy(float)


def fit_spec(
    mean: np.ndarray,
    y: np.ndarray,
    family: str = "negbin",
    lower: Optional[int] = 0,
) -> Tuple[FamilySpec, float]:
    """Maximum-likelihood variance function for one family.

    ``family`` is one of the :class:`FamilySpec` families, or
    ``"negbin_scaled"`` to also fit the negative binomial's ``scale``.
    Returns ``(spec, mean_log_likelihood)`` on the fitting data.
    """
    mean = np.asarray(mean, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(mean) & np.isfinite(y) & (mean > 0)
    mean, y = mean[ok], y[ok]
    if len(mean) < 50:
        raise ValueError("need at least 50 observations to fit a variance function")
    raw_mean, raw_y = mean, y
    mean, y, weight = _compress(mean, y)
    total_weight = float(weight.sum())

    if family == "poisson":
        spec = FamilySpec("poisson", lower=lower)
        return spec, float(log_likelihood(spec, raw_mean, raw_y).mean())

    def build(theta: np.ndarray) -> FamilySpec:
        theta = np.clip(theta, -16.0, 12.0)
        if family == "negbin":
            return FamilySpec(
                "negbin", c=math.exp(theta[2]), phi=1.0 + math.exp(theta[0]), alpha=math.exp(theta[1]), lower=lower
            )
        if family == "negbin_scaled":
            return FamilySpec(
                "negbin", c=math.exp(theta[2]), phi=math.exp(theta[0]), alpha=math.exp(theta[1]),
                scale=1.0 + math.exp(theta[3]), lower=lower,
            )
        return FamilySpec(
            family, c=math.exp(theta[0]), phi=math.exp(theta[1]), alpha=math.exp(theta[2]), lower=lower
        )

    def loss(theta: np.ndarray) -> float:
        return -float((weight * log_likelihood(build(theta), mean, y)).sum()) / total_weight

    def safe_log(value: float) -> float:
        return math.log(max(value, 1e-6))

    c0, phi0, alpha0 = _moment_start(raw_mean, raw_y)
    if family == "negbin":
        starts = [
            np.array([safe_log(phi0 - 1.0), safe_log(alpha0), safe_log(c0)]),
            np.array([-6.0, safe_log(alpha0 + 0.02), -8.0]),
        ]
    elif family == "negbin_scaled":
        starts = [
            np.array([safe_log(phi0), safe_log(alpha0), safe_log(c0), 0.3]),
            np.array([0.0, safe_log(alpha0 + 0.01), 1.0, -1.0]),
        ]
    else:
        starts = [np.array([safe_log(c0), safe_log(phi0), safe_log(alpha0)]), np.array([-4.0, 0.0, -4.0])]
    best = None
    for start in starts:
        result = minimize(loss, start, method="Nelder-Mead",
                          options={"xatol": 1e-3, "fatol": 1e-7, "maxiter": 500})
        if best is None or result.fun < best.fun:
            best = result
    spec = build(best.x)
    # Tidy negligible terms so the stored spec is easy to read.
    spec = FamilySpec(
        spec.family,
        phi=(round(spec.phi, 4) if spec.phi > 1.0005 else 1.0) if family == "negbin"
        else (round(spec.phi, 4) if spec.phi > 1e-4 else 0.0),
        alpha=round(spec.alpha, 5) if spec.alpha > 1e-5 else 0.0,
        c=round(spec.c, 3) if spec.c > 1e-3 else 0.0,
        lower=lower,
        scale=round(spec.scale, 3) if spec.scale > 1.005 else 1.0,
    )
    return spec, float(log_likelihood(spec, raw_mean, raw_y).mean())


def select_spec(
    mean_fit: np.ndarray,
    y_fit: np.ndarray,
    mean_val: np.ndarray,
    y_val: np.ndarray,
    families: Sequence[str] = ("poisson", "negbin", "negbin_scaled", "normal"),
    lower: Optional[int] = 0,
) -> Tuple[FamilySpec, pd.DataFrame]:
    """Fit each family on the fitting set and pick the best on validation data.

    Returns the winning spec and a table of validation log-likelihoods.
    """
    rows = []
    best: Optional[Tuple[float, FamilySpec]] = None
    for family in families:
        if family == "gamma" and lower is None:
            continue
        spec, fit_ll = fit_spec(mean_fit, y_fit, family, lower=lower)
        ok = np.isfinite(mean_val) & np.isfinite(y_val) & (mean_val > 0)
        val_ll = float(log_likelihood(spec, mean_val[ok], y_val[ok]).mean())
        rows.append({"candidate": family, "fit_loglik": fit_ll, "val_loglik": val_ll, **spec.to_dict()})
        if best is None or val_ll > best[0]:
            best = (val_ll, spec)
    table = pd.DataFrame(rows).sort_values("val_loglik", ascending=False).reset_index(drop=True)
    assert best is not None
    return best[1], table


# -------------------------------------------------------------- calibration
def reliability_table(pred: np.ndarray, outcome: np.ndarray, bins: Sequence[float]) -> pd.DataFrame:
    """Predicted probability bucket vs observed frequency."""
    pred = np.asarray(pred, float)
    outcome = np.asarray(outcome, float)
    edges = np.asarray(bins, float)
    which = np.clip(np.digitize(pred, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        mask = which == b
        n = int(mask.sum())
        if n == 0:
            continue
        observed = float(outcome[mask].mean())
        rows.append(
            {
                "bin_low": float(edges[b]),
                "bin_high": float(edges[b + 1]),
                "n": n,
                "predicted": float(pred[mask].mean()),
                "observed": observed,
                "se": math.sqrt(max(observed * (1.0 - observed), 1e-12) / n),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class CalibrationResult:
    """Calibration of one market's over probabilities on held-out games."""

    market: str
    n_games: int
    n_props: int
    mean_loglik: float
    brier: float
    brier_baseline: float  # always predicting the observed base rate
    log_loss: float
    ece: float  # expected calibration error (probability points)
    max_gap: float  # worst |predicted - observed| over well-populated bins
    bias: float  # mean(projection - actual)
    mae: float
    reliability: pd.DataFrame
    by_offset: pd.DataFrame

    def to_row(self) -> Dict[str, float]:
        return {
            "market": self.market,
            "n_games": self.n_games,
            "n_props": self.n_props,
            "mean_loglik": self.mean_loglik,
            "brier": self.brier,
            "brier_baseline": self.brier_baseline,
            "log_loss": self.log_loss,
            "ece": self.ece,
            "max_gap": self.max_gap,
            "bias": self.bias,
            "mae": self.mae,
        }


def calibration_report(
    market: str,
    spec: FamilySpec,
    mean: np.ndarray,
    y: np.ndarray,
    offsets_sd: Sequence[float] = (-0.5, -0.25, 0.0, 0.25, 0.5),
    bins: Sequence[float] = (0.0, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 1.0),
    fixed_line: Optional[float] = None,
) -> CalibrationResult:
    """Score a market's predictive distribution on out-of-sample games.

    For every game a main line is set at the half-point next to the model's
    median, plus lines shifted by the given multiples of the predictive
    standard deviation (rounded to whole units, at least one).  The model's
    ``P(over)`` at each line is compared with what happened.  ``fixed_line``
    replaces all of that with a single line (``0.5`` for yes/no markets).
    """
    mean = np.asarray(mean, dtype=float)
    y = np.rint(np.asarray(y, dtype=float))
    ok = np.isfinite(mean) & np.isfinite(y) & (mean > 0)
    mean, y = mean[ok], y[ok]
    if len(mean) == 0:
        raise ValueError(f"no usable observations for {market}")

    if fixed_line is not None:
        offsets_sd = (0.0,)
        median = np.zeros(len(mean))
    else:
        median = _median(spec, mean, float(np.max(y)))
    sd = np.sqrt(np.maximum(_variance(spec, mean), 1e-9))
    preds: List[np.ndarray] = []
    outcomes: List[np.ndarray] = []
    offset_rows = []
    for off in offsets_sd:
        shift = np.sign(off) * np.maximum(np.rint(abs(off) * sd), 1.0) if off != 0 else np.zeros(len(mean))
        # Books hang the half-point just below the median when the mean is
        # below it, otherwise just above.
        line = np.where(mean >= median, median + 0.5, median - 0.5) + shift
        line = np.maximum(line, 0.5) if spec.lower is not None else line
        if fixed_line is not None:
            line = np.full(len(mean), float(fixed_line))
        p_over = np.clip(over_probabilities(spec, mean, line), 1e-6, 1.0 - 1e-6)
        went_over = (y > line).astype(float)
        preds.append(p_over)
        outcomes.append(went_over)
        offset_rows.append(
            {
                "offset_sd": off,
                "n": len(p_over),
                "predicted": float(p_over.mean()),
                "observed": float(went_over.mean()),
                "gap": float(p_over.mean() - went_over.mean()),
            }
        )
    pred = np.concatenate(preds)
    outcome = np.concatenate(outcomes)
    table = reliability_table(pred, outcome, bins)
    weights = table["n"] / table["n"].sum()
    gaps = (table["predicted"] - table["observed"]).abs()
    populated = table["n"] >= max(200, 0.01 * table["n"].sum())
    base_rate = outcome.mean()
    return CalibrationResult(
        market=market,
        n_games=int(len(mean)),
        n_props=int(len(pred)),
        mean_loglik=float(log_likelihood(spec, mean, y).mean()),
        brier=float(np.mean((pred - outcome) ** 2)),
        brier_baseline=float(base_rate * (1.0 - base_rate)),
        log_loss=float(-np.mean(outcome * np.log(pred) + (1.0 - outcome) * np.log(1.0 - pred))),
        ece=float((weights * gaps).sum()),
        max_gap=float(gaps[populated].max()) if populated.any() else float("nan"),
        bias=float(np.mean(mean - y)),
        mae=float(np.mean(np.abs(mean - y))),
        reliability=table,
        by_offset=pd.DataFrame(offset_rows),
    )
