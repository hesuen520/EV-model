"""High-level steps shared by the CLI, the dashboard and tests:
ingest -> validate -> train -> predict."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import pandas as pd

from .adapters.http import Fetcher
from .config import Settings
from .models.fitted import FittedModel, train_production
from .sports import get_league
from .storage.store import Store, is_demo_league
from .validation.backtest import ValidationConfig, run_validation

__all__ = ["open_store", "ingest_league", "validate_league", "train_league"]

log = logging.getLogger(__name__)


def open_store(settings: Optional[Settings] = None) -> Store:
    settings = settings or Settings()
    return Store(settings.paths.store)


def ingest_league(store: Store, league: str, *, settings: Optional[Settings] = None,
                  seasons: Optional[Sequence[int]] = None, refresh: Optional[bool] = None):
    settings = settings or Settings()
    plugin, _ = get_league(league)
    return plugin.ingest(store, league, seasons=seasons, refresh=refresh, fetcher=Fetcher(settings.paths.raw))


def _validation_config(settings: Settings) -> ValidationConfig:
    v = settings.validation
    return ValidationConfig(n_boot=v.n_boot, seed=v.seed, min_final_events=v.min_final_events,
                            min_dev_folds=v.min_dev_folds, max_ece=v.max_ece)


def validate_league(store: Store, league: str, market: str = "winner", *,
                    settings: Optional[Settings] = None, now: Any = None) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Run walk-forward validation, persist the report and the out-of-sample
    predictions (table ``oos_<market>``), and return both."""
    settings = settings or Settings()
    plugin, spec = get_league(league)
    frame = plugin.feature_frame(store, league, market, now=now)
    candidates = plugin.candidates(league, market, frame)
    plan = plugin.split_plan(league, market, frame)
    benchmark = plugin.market_benchmark(store, league, market)
    report, oos = run_validation(
        frame, candidates, plan, sport=spec.sport, config=_validation_config(settings),
        benchmark=benchmark, is_synthetic=is_demo_league(league),
    )
    selected = next(c for c in candidates if c.name == report["selected_model"])
    supplement = plugin.validation_supplement(league, market, frame, selected, plan)
    if supplement is not None:
        report["supplement"] = supplement
    report["feature_frame"] = {k: v for k, v in frame.extra.items() if k not in ("rosters", "names", "rotation")}
    report["run_id"] = store.save_validation_run(report)
    store.write_table(league, f"oos_{market}", oos, is_synthetic=is_demo_league(league), mode="replace")
    return report, oos


def train_league(store: Store, league: str, market: str = "winner", *,
                 settings: Optional[Settings] = None, now: Any = None,
                 model_name: Optional[str] = None) -> FittedModel:
    """Fit the production model.  The model is the one selected by the latest
    validation run (never chosen here), unless ``model_name`` names another
    candidate explicitly - in which case it is registered but not promoted."""
    settings = settings or Settings()
    plugin, spec = get_league(league)
    report = store.latest_validation(league, market)
    if report is None:
        raise RuntimeError(
            f"no validation run for {league} {market}; run `sportspred validate` first so the model "
            "is chosen by out-of-sample evidence")
    chosen = model_name or report["selected_model"]
    frame = plugin.feature_frame(store, league, market, now=now)
    by_name = {c.name: c for c in plugin.candidates(league, market, frame)}
    if chosen not in by_name:
        raise RuntimeError(f"candidate {chosen!r} is not defined for {league} {market}")
    if by_name[chosen].uses_retrospective:
        raise RuntimeError(f"{chosen!r} uses information not available pregame and cannot be a production model")
    plan = plugin.split_plan(league, market, frame)
    model = train_production(frame, by_name[chosen], plan, n_replicas=settings.validation.n_replicas,
                             seed=settings.validation.seed, validation_run_id=report.get("run_id"))
    # The version names the market, and a player-statistic market has a colon in its key;
    # a colon is not a legal file-name character everywhere.
    path = store.models_dir / f"{model.model_version.replace(':', '_')}.joblib"
    model.save(path)
    store.register_model(
        model_version=model.model_version, sport=spec.sport, league=league, market=market,
        model_name=model.model_name, artifact_path=path, trained_at=model.trained_at_utc,
        train_start=model.train_start_utc, train_end=model.train_end_utc, n_train=model.n_train,
        validation_run_id=report.get("run_id"), promote=(chosen == report["selected_model"]),
        is_synthetic=is_demo_league(league),
        info={k: v for k, v in model.info.items() if k != "feature_frame"},
    )
    return model
