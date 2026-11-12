"""Fit every market's distribution on real game logs and test it out of sample.

The workflow behind the numbers shipped in ``evmodel/data/market_specs.json``:

1. run the walk-forward estimator over many seasons (each projection uses
   only earlier games);
2. on the **fit** seasons, fit each candidate distribution family's variance
   function by maximum likelihood, and a mean correction factor;
3. on the **validation** seasons, pick the family with the best likelihood;
4. on the **test** seasons - never used for fitting or selection - measure
   how well calibrated the over/under probabilities are.

``evmodel calibrate`` runs this and can write the result back to the package.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..distributions import FamilySpec
from ..markets import MarketDef, markets_for_sport
from .calibrate import CalibrationResult, calibration_report, fit_spec, select_spec
from .features import FeatureParams
from .model import eligible_rows, feature_params, market_actual, market_mean, walk_forward

__all__ = [
    "MarketFit",
    "DEFAULT_SPLITS",
    "fit_markets",
    "evaluate_markets",
    "summary_frame",
    "specs_payload",
    "save_specs",
    "packaged_specs_path",
]

#: Seasons (by starting year) used to fit, select and test each sport's specs.
DEFAULT_SPLITS: Dict[str, Dict[str, Sequence[int]]] = {
    "nba": {"fit": range(2017, 2022), "val": range(2022, 2024), "test": range(2024, 2027)},
    "nhl": {"fit": range(2023, 2024), "val": range(2024, 2025), "test": range(2025, 2027)},
    "nfl": {"fit": range(2019, 2022), "val": range(2022, 2024), "test": range(2024, 2027)},
}

_CANDIDATES: Dict[str, Sequence[str]] = {
    "nba": ("poisson", "negbin", "negbin_scaled", "normal"),
    "nhl": ("poisson", "negbin", "normal"),
    "nfl": ("poisson", "negbin", "negbin_scaled", "normal", "gamma"),
}


@dataclass
class MarketFit:
    """Result of fitting one market."""

    market: MarketDef
    spec: FamilySpec
    mean_scale: float
    candidates: pd.DataFrame
    n_fit: int
    test: Optional[CalibrationResult] = None


#: Shapes are fitted on a wider range of projections than books post lines
#: for, so the variance function is not extrapolated for low-usage players.
_FIT_MEAN_FLOOR = 0.3


def _arrays(
    features: pd.DataFrame, market: MarketDef, seasons: Iterable[int], min_games: int, mean_floor: float = 1.0
):
    rows = eligible_rows(features, market, min_games, mean_floor) & features["season"].isin(list(seasons))
    rows &= ~features["_placeholder"]
    mean = market_mean(features, market)[rows].to_numpy(float)
    actual = market_actual(features, market)[rows].to_numpy(float)
    ok = np.isfinite(mean) & np.isfinite(actual)
    return mean[ok], actual[ok]


def fit_markets(
    sport: str,
    logs: pd.DataFrame,
    fit_seasons: Optional[Iterable[int]] = None,
    val_seasons: Optional[Iterable[int]] = None,
    test_seasons: Optional[Iterable[int]] = None,
    markets: Optional[Iterable[str]] = None,
    min_games: int = 8,
    params: Optional[FeatureParams] = None,
    candidates: Optional[Sequence[str]] = None,
    progress: Optional[callable] = None,
) -> Dict[str, MarketFit]:
    """Fit, select and (optionally) test the distribution of every market.

    Returns ``{market_key: MarketFit}``.  The registry is not modified; pass
    the result to :func:`save_specs` and reload to adopt it.
    """
    sport = sport.lower()
    splits = DEFAULT_SPLITS.get(sport, {})
    fit_seasons = list(fit_seasons if fit_seasons is not None else splits.get("fit", []))
    val_seasons = list(val_seasons if val_seasons is not None else splits.get("val", []))
    test_seasons = list(test_seasons if test_seasons is not None else splits.get("test", []))
    if not fit_seasons or not val_seasons:
        raise ValueError("fit_seasons and val_seasons are required")
    overlap = (set(fit_seasons) | set(val_seasons)) & set(test_seasons)
    if overlap:
        raise ValueError(f"test seasons overlap the fitting seasons: {sorted(overlap)}")
    params = params or feature_params(sport)
    families = tuple(candidates or _CANDIDATES.get(sport, ("poisson", "negbin", "normal")))
    features = walk_forward(logs, sport, params)

    wanted = set(markets) if markets is not None else None
    out: Dict[str, MarketFit] = {}
    for market in markets_for_sport(sport):
        if wanted is not None and market.key not in wanted:
            continue
        if not all(s in params.stats for s in market.stats):
            continue
        # Fit on unscaled means: the correction factor is estimated here.
        raw = MarketDef(**{**market.__dict__, "mean_scale": 1.0})
        mu_fit, y_fit = _arrays(features, raw, fit_seasons, min_games, _FIT_MEAN_FLOOR)
        mu_val, y_val = _arrays(features, raw, val_seasons, min_games, _FIT_MEAN_FLOOR)
        if len(mu_fit) < 500 or len(mu_val) < 200:
            if progress:
                progress(f"{market.key}: skipped (only {len(mu_fit)} fit / {len(mu_val)} validation games)")
            continue
        both_mu = np.concatenate([mu_fit, mu_val])
        both_y = np.concatenate([y_fit, y_val])
        # The mean correction is estimated on the range books actually post.
        posted = both_mu >= market.min_mean
        scale = float(np.clip(both_y[posted].sum() / both_mu[posted].sum(), 0.85, 1.15))
        # Yardage can be slightly negative.  Prop lines never are, so a
        # negative result is the same as zero for every over/under and the
        # stat is modelled on [0, inf).
        fams = list(families)
        if market.yes_no:
            fams = [f for f in fams if f in ("poisson", "negbin")] or ["poisson"]
        clip = lambda a: np.maximum(a, 0.0)  # noqa: E731
        spec, table = select_spec(mu_fit * scale, clip(y_fit), mu_val * scale, clip(y_val), fams, lower=0)
        winner = str(table.iloc[0]["candidate"])
        if winner != "poisson":  # refit the winning family on all non-test data
            spec, _ = fit_spec(both_mu * scale, clip(both_y), winner, lower=0)
        fit = MarketFit(market=market, spec=spec, mean_scale=round(scale, 4), candidates=table, n_fit=len(both_mu))
        if test_seasons:
            mu_test, y_test = _arrays(features, raw, test_seasons, min_games)
            if len(mu_test) >= 200:
                fit.test = calibration_report(
                    market.key, spec, mu_test * scale, clip(y_test),
                    fixed_line=0.5 if market.yes_no else None,
                )
        out[market.key] = fit
        if progress:
            note = f" test ECE {fit.test.ece:.4f} (n={fit.test.n_games})" if fit.test else ""
            progress(f"{market.key}: {winner} scale={spec.scale} mean_scale={fit.mean_scale}{note}")
    return out


def evaluate_markets(
    sport: str,
    logs: pd.DataFrame,
    seasons: Iterable[int],
    markets: Optional[Iterable[str]] = None,
    min_games: int = 8,
    params: Optional[FeatureParams] = None,
) -> Dict[str, CalibrationResult]:
    """Calibration of the *registered* specs on the given seasons.

    Use seasons the registered specs were not fitted on (see the ``meta``
    block of ``market_specs.json``) for an honest out-of-sample check.
    """
    sport = sport.lower()
    params = params or feature_params(sport)
    features = walk_forward(logs, sport, params)
    wanted = set(markets) if markets is not None else None
    out: Dict[str, CalibrationResult] = {}
    for market in markets_for_sport(sport):
        if wanted is not None and market.key not in wanted:
            continue
        if not all(s in params.stats for s in market.stats):
            continue
        mean, actual = _arrays(features, market, seasons, min_games)
        if len(mean) < 200:
            continue
        actual = np.maximum(actual, 0.0)  # see fit_markets: negatives count as zero
        out[market.key] = calibration_report(
            market.key, market.spec, mean, actual, fixed_line=0.5 if market.yes_no else None
        )
    return out


def summary_frame(results: Mapping[str, Union[MarketFit, CalibrationResult]]) -> pd.DataFrame:
    """One row of headline numbers per market."""
    rows: List[Dict[str, object]] = []
    for key, item in results.items():
        if isinstance(item, MarketFit):
            row: Dict[str, object] = {
                "market": key,
                "family": item.spec.family + ("*" if item.spec.scale != 1.0 else ""),
                "mean_scale": item.mean_scale,
                "n_fit": item.n_fit,
            }
            if item.test is not None:
                row.update({k: v for k, v in item.test.to_row().items() if k != "market"})
        else:
            row = item.to_row()
        rows.append(row)
    return pd.DataFrame(rows)


def specs_payload(fits: Mapping[str, MarketFit], sport: str, meta: Optional[Mapping[str, object]] = None) -> Dict[str, object]:
    """JSON-ready document of fitted specs for one sport."""
    specs = {}
    for key, fit in fits.items():
        entry = fit.spec.to_dict()
        entry["mean_scale"] = fit.mean_scale
        if fit.test is not None:
            entry["test_ece"] = round(fit.test.ece, 4)
            entry["test_games"] = fit.test.n_games
        specs[f"{sport}/{key}"] = entry
    return {"meta": {sport: dict(meta or {})}, "specs": specs}


def packaged_specs_path() -> Path:
    """Location of the specs file shipped inside the package."""
    return Path(__file__).resolve().parent.parent / "data" / "market_specs.json"


def save_specs(payload: Mapping[str, object], path: Union[str, Path, None] = None) -> Path:
    """Merge fitted specs into a specs file (other sports' entries are kept)."""
    path = Path(path) if path is not None else packaged_specs_path()
    existing: Dict[str, object] = {"meta": {}, "specs": {}}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing.setdefault("meta", {})
        existing.setdefault("specs", {})
    sports = set(payload.get("meta", {}))  # type: ignore[arg-type]
    existing["specs"] = {
        k: v for k, v in existing["specs"].items() if k.split("/", 1)[0] not in sports  # type: ignore[union-attr]
    }
    existing["specs"].update(payload.get("specs", {}))  # type: ignore[union-attr]
    existing["meta"].update(payload.get("meta", {}))  # type: ignore[union-attr]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path
