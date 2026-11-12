"""Walk-forward validation with a reserved final test period.

Protocol (the same for every sport and market):

1. Completed events are cut into chronological blocks (seasons).  The most
   recent block(s) are the **final test period**.
2. **Development folds** walk forward through the remaining blocks.  In each
   fold a candidate is fitted on the training blocks, its calibrator is fitted
   on the following calibration block, and both are scored on the next block.
   Nothing is ever fitted on a block it is scored on.
3. **Model selection** uses pooled development-fold log loss only, with a
   one-standard-error rule that prefers the simpler candidate unless the more
   complex one is clearly better.  Candidates that use retrospective inputs can
   never be selected.
4. The selected candidate (and the baselines, for comparison) are refitted on
   all pre-final blocks and scored **once** on the final test period.
5. Gates decide whether the result may be called ``validated``.

The final test period is read exactly once, after selection is fixed; the
function has no code path that feeds final-period scores back into selection.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..core.timeutil import DISPLAY_TZ_NAME, iso_z, utcnow
from ..sports.base import Candidate, FeatureFrame, SplitPlan
from . import metrics as M
from .calibration import Calibrator, fit_calibrator
from .splits import Fold, LeakageError, assert_chronological, make_folds

__all__ = ["ValidationConfig", "run_validation", "fit_with_calibration", "slate_dates", "same_day_residual_correlation"]

log = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    n_boot: int = 500
    seed: int = 20261112
    one_se_rule: bool = True
    calib_min_rows: int = 100
    #: Gates for the "validated" coverage status.
    min_final_events: int = 300
    min_dev_folds: int = 3
    max_ece: float = 0.03
    #: Probability thresholds at which PASS-rule performance is reported.
    pass_grid: Tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
    confidence_bin_width: float = 0.05


def slate_dates(start_time_utc: pd.Series) -> np.ndarray:
    """Pacific/Honolulu calendar date of each event (the slate it belongs to)."""
    return start_time_utc.dt.tz_convert(DISPLAY_TZ_NAME).dt.strftime("%Y-%m-%d").to_numpy()


def _prob_cols(classes: Sequence[str]) -> List[str]:
    return [f"p_{c}" for c in classes]


def fit_with_calibration(
    cand: Candidate,
    train: pd.DataFrame,
    calib: pd.DataFrame,
    classes: Sequence[str],
    *,
    calib_min_rows: int = 100,
) -> Tuple[Candidate, Calibrator]:
    """Fit ``cand`` on ``train`` and its calibrator on ``calib`` (later rows)."""
    if len(train) and len(calib) and not train["start_time_utc"].max() < calib["start_time_utc"].min():
        raise LeakageError("calibration rows must be strictly later than training rows")
    model = cand.clone().fit(train, classes)
    if len(calib):
        raw = M.check_probabilities(model.predict_proba(calib))
        cal = fit_calibrator(cand.calibration, raw, calib["y"].to_numpy(), min_rows=calib_min_rows)
    else:
        cal = fit_calibrator("none", np.zeros((0, len(classes))), np.zeros(0))
    return model, cal


def _run_fold(cand: Candidate, df: pd.DataFrame, fold: Fold, plan: SplitPlan, classes: Sequence[str],
              cfg: ValidationConfig) -> pd.DataFrame:
    assert_chronological(df, fold, plan.block_col)
    train = df[df[plan.block_col].isin(fold.train_blocks)]
    calib = df[df[plan.block_col].isin(fold.calib_blocks)]
    test = df[df[plan.block_col].isin(fold.test_blocks)]
    model, cal = fit_with_calibration(cand, train, calib, classes, calib_min_rows=cfg.calib_min_rows)
    P = M.check_probabilities(cal.transform(M.check_probabilities(model.predict_proba(test))))
    out = test[["event_id", "start_time_utc", plan.block_col, "y"]].copy()
    out = out.rename(columns={plan.block_col: "block"})
    out["fold"] = fold.name
    out["model"] = cand.name
    for j, col in enumerate(_prob_cols(classes)):
        out[col] = P[:, j]
    return out


def _P(frame: pd.DataFrame, classes: Sequence[str]) -> np.ndarray:
    return frame[_prob_cols(classes)].to_numpy(dtype=float)


def _y(frame: pd.DataFrame) -> np.ndarray:
    return frame["y"].to_numpy().astype(int)


def _metrics_with_ci(frame: pd.DataFrame, classes: Sequence[str], cfg: ValidationConfig) -> Dict[str, Any]:
    P, y = _P(frame, classes), _y(frame)
    out: Dict[str, Any] = M.summary(P, y)
    groups = slate_dates(frame["start_time_utc"])
    for key, fn in (("accuracy", M.accuracy), ("log_loss", M.log_loss), ("brier", M.brier)):
        lo, hi = M.cluster_bootstrap(fn, P, y, groups, n_boot=cfg.n_boot, seed=cfg.seed)
        out[f"{key}_ci"] = [lo, hi]
    out["first_event"] = iso_z(frame["start_time_utc"].min())
    out["last_event"] = iso_z(frame["start_time_utc"].max())
    return out


def _by_block(frame: pd.DataFrame, classes: Sequence[str]) -> List[Dict[str, Any]]:
    rows = []
    for block, part in frame.groupby("block", sort=False):
        row = {"block": str(block), **M.summary(_P(part, classes), _y(part))}
        rows.append(row)
    return rows


def _stability(selected: pd.DataFrame, reference: pd.DataFrame, classes: Sequence[str]) -> Dict[str, Any]:
    """Block-by-block log loss of the selected model against a reference on
    the same events: does the improvement hold in every season, or only on
    average?"""
    ref = reference.set_index("event_id").loc[selected["event_id"]].reset_index()
    rows = []
    for block, part in selected.groupby("block", sort=False):
        other = ref[ref["block"] == block]
        ll_sel = M.log_loss(_P(part, classes), _y(part))
        ll_ref = M.log_loss(_P(other, classes), _y(other))
        rows.append({"block": str(block), "n": int(len(part)), "log_loss_selected": ll_sel,
                     "log_loss_reference": ll_ref, "improvement": ll_ref - ll_sel})
    better = sum(1 for r in rows if r["improvement"] > 0)
    return {"blocks": len(rows), "blocks_improved": better, "by_block": rows}


def _pass_rule(frame: pd.DataFrame, classes: Sequence[str], grid: Sequence[float]) -> List[Dict[str, Any]]:
    """Performance and coverage when events whose top probability is below a
    threshold are passed.  Reported, never tuned on the final period."""
    P, y = _P(frame, classes), _y(frame)
    top = P.max(axis=1)
    hit = (P.argmax(axis=1) == y).astype(float)
    rows = []
    for thr in grid:
        mask = top >= thr
        n = int(mask.sum())
        row: Dict[str, Any] = {"min_prob": float(thr), "n_selected": n,
                               "coverage": float(mask.mean()) if len(mask) else float("nan")}
        if n:
            lo, hi = M.wilson_interval(hit[mask].sum(), n)
            row.update({
                "accuracy": float(hit[mask].mean()), "accuracy_low": lo, "accuracy_high": hi,
                "mean_predicted": float(top[mask].mean()),
                "log_loss": M.log_loss(P[mask], y[mask]), "brier": M.brier(P[mask], y[mask]),
            })
        rows.append(row)
    return rows


def _confidence_table(frame: pd.DataFrame, classes: Sequence[str], width: float) -> List[Dict[str, Any]]:
    """Out-of-sample hit rate of the selected outcome by predicted probability
    band - the evidence behind every confidence label shown to a user."""
    P, y = _P(frame, classes), _y(frame)
    top = P.max(axis=1)
    hit = (P.argmax(axis=1) == y).astype(float)
    start = np.floor((1.0 / len(classes)) / width) * width
    edges = np.arange(start, 1.0 + 1e-9, width)
    rows = []
    for lo_edge, hi_edge in zip(edges[:-1], edges[1:]):
        mask = (top >= lo_edge - 1e-12) & (top < hi_edge - 1e-12 if hi_edge < 1.0 else top <= 1.0)
        n = int(mask.sum())
        if not n:
            continue
        lo, hi = M.wilson_interval(hit[mask].sum(), n)
        rows.append({
            "prob_low": float(round(lo_edge, 4)), "prob_high": float(round(hi_edge, 4)), "n": n,
            "mean_predicted": float(top[mask].mean()), "observed": float(hit[mask].mean()),
            "observed_low": lo, "observed_high": hi,
        })
    return rows


def same_day_residual_correlation(frame: pd.DataFrame, classes: Sequence[str], *, n_boot: int = 500,
                                  seed: int = 0) -> Dict[str, Any]:
    """Correlation between the selection residuals of different events on the
    same Pacific/Honolulu slate.

    For each event the residual is ``hit - p`` of its most likely outcome.  If
    different events were independent given the model, the cross-products of
    residuals within a slate would average zero.  The estimate (with a
    bootstrap interval over slates) is what justifies - or forbids - treating
    legs from different events as independent when building combos.
    """
    P, y = _P(frame, classes), _y(frame)
    top = P.max(axis=1)
    resid = (P.argmax(axis=1) == y).astype(float) - top
    scale = np.sqrt(np.clip(top * (1 - top), 1e-9, None))
    days = slate_dates(frame["start_time_utc"])
    num_by_day: Dict[str, float] = {}
    den_by_day: Dict[str, float] = {}
    order = np.argsort(days, kind="stable")
    bounds = np.flatnonzero(days[order][1:] != days[order][:-1]) + 1
    for idx in np.split(order, bounds):
        if len(idx) < 2:
            continue
        r, s = resid[idx], scale[idx]
        # sum_{i<j} r_i r_j = ((sum r)^2 - sum r^2) / 2, and likewise for the scale.
        num_by_day[days[idx[0]]] = float((r.sum() ** 2 - (r ** 2).sum()) / 2.0)
        den_by_day[days[idx[0]]] = float((s.sum() ** 2 - (s ** 2).sum()) / 2.0)
    if not num_by_day:
        return {"rho": None, "low": None, "high": None, "n_slates": 0, "n_pairs": 0}
    keys = list(num_by_day)
    num = np.array([num_by_day[k] for k in keys])
    den = np.array([den_by_day[k] for k in keys])
    rho = float(num.sum() / den.sum())
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        boots[b] = num[pick].sum() / den[pick].sum()
    counts = pd.Series(days).value_counts()
    n_pairs = int((counts * (counts - 1) // 2).sum())
    return {"rho": rho, "low": float(np.quantile(boots, 0.025)), "high": float(np.quantile(boots, 0.975)),
            "n_slates": len(keys), "n_pairs": n_pairs}


def _select(cands: Sequence[Candidate], dev_oos: Dict[str, pd.DataFrame], classes: Sequence[str],
            cfg: ValidationConfig) -> Tuple[str, str]:
    eligible = [c for c in cands if not c.uses_retrospective and c.name in dev_oos]
    if not eligible:
        raise ValueError("no candidate is eligible for selection")
    ll = {c.name: M.log_loss(_P(dev_oos[c.name], classes), _y(dev_oos[c.name])) for c in eligible}
    best = min(eligible, key=lambda c: (ll[c.name], c.complexity))
    chosen, note = best, f"lowest pooled development log loss ({ll[best.name]:.5f})"
    if cfg.one_se_rule:
        ref = dev_oos[best.name]
        groups = slate_dates(ref["start_time_utc"])
        simpler = sorted((c for c in eligible if c.complexity < best.complexity),
                         key=lambda c: (c.complexity, ll[c.name]))
        for c in simpler:
            other = dev_oos[c.name]
            if not other["event_id"].equals(ref["event_id"]):
                other = other.set_index("event_id").loc[ref["event_id"]].reset_index()
            d = M.paired_bootstrap_diff(M.log_loss, _P(ref, classes), _P(other, classes), _y(ref), groups,
                                        n_boot=cfg.n_boot, seed=cfg.seed)
            # d["diff"] = loss(simpler) - loss(best) >= 0
            if np.isfinite(d["se"]) and d["diff"] <= d["se"]:
                chosen = c
                note = (
                    f"one-standard-error rule: {c.name} (log loss {ll[c.name]:.5f}) is simpler than "
                    f"{best.name} ({ll[best.name]:.5f}) and within one SE ({d['se']:.5f}) of it"
                )
                break
    return chosen.name, note


def run_validation(
    frame: FeatureFrame,
    candidates: Sequence[Candidate],
    plan: SplitPlan,
    *,
    sport: str,
    config: Optional[ValidationConfig] = None,
    benchmark: Optional[pd.DataFrame] = None,
    is_synthetic: bool = False,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Run the full protocol; returns ``(report, out_of_sample_predictions)``."""
    cfg = config or ValidationConfig()
    frame.validate()
    classes = list(frame.classes)
    names = [c.name for c in candidates]
    if len(set(names)) != len(names):
        raise ValueError("candidate names must be unique")
    for c in candidates:
        bad = set(c.feature_names()) & set(frame.retrospective_cols)
        if bad and not c.uses_retrospective:
            raise LeakageError(
                f"candidate {c.name!r} uses retrospective columns {sorted(bad)} but is not flagged as such"
            )
    df = frame.completed.sort_values(["start_time_utc", "event_id"]).reset_index(drop=True)
    folds = make_folds(df, plan)

    # ------------------------------------------------ development folds
    dev_oos: Dict[str, pd.DataFrame] = {}
    for cand in candidates:
        parts = [_run_fold(cand, df, fold, plan, classes, cfg) for fold in folds.dev]
        if parts:
            dev_oos[cand.name] = pd.concat(parts, ignore_index=True)
    if not dev_oos:
        raise ValueError("no development folds could be formed; more history is needed")

    cand_reports = []
    for cand in candidates:
        oos = dev_oos[cand.name]
        cand_reports.append({
            "name": cand.name,
            "description": cand.description,
            "is_baseline": cand.is_baseline,
            "complexity": cand.complexity,
            "uses_retrospective": cand.uses_retrospective,
            "calibration": cand.calibration,
            "features": cand.feature_names(),
            "dev": M.summary(_P(oos, classes), _y(oos)),
            "dev_by_block": _by_block(oos, classes),
        })
    selected, selection_note = _select(candidates, dev_oos, classes, cfg)
    by_name = {c.name: c for c in candidates}

    # ------------------------------------- final test: read once, after selection
    report_names = [selected] + [c.name for c in candidates if c.is_baseline and c.name != selected]
    experiments = [c.name for c in candidates if c.uses_retrospective]
    final_oos: Dict[str, pd.DataFrame] = {
        name: _run_fold(by_name[name], df, folds.final, plan, classes, cfg)
        for name in report_names + experiments
    }
    sel_final = final_oos[selected]
    final_groups = slate_dates(sel_final["start_time_utc"])
    final_block: Dict[str, Any] = {
        "blocks": [str(b) for b in folds.final.test_blocks],
        "selected": _metrics_with_ci(sel_final, classes, cfg),
        "baselines": {n: M.summary(_P(final_oos[n], classes), _y(final_oos[n]))
                      for n in report_names if n != selected},
        "retrospective_experiments": {
            n: {**M.summary(_P(final_oos[n], classes), _y(final_oos[n])),
                "note": "uses information not available pregame; an upper bound, not a realistic backtest"}
            for n in experiments
        },
        "by_block": _by_block(sel_final, classes),
        "reliability": M.reliability_table(_P(sel_final, classes), _y(sel_final)),
        "reliability_selected_outcome": M.reliability_table(_P(sel_final, classes), _y(sel_final), top_label=True),
        "pass_rule": _pass_rule(sel_final, classes, cfg.pass_grid),
    }
    # Skill versus the simplest baseline, paired on the same events.
    baseline_names = [c.name for c in sorted(candidates, key=lambda c: c.complexity) if c.is_baseline]
    skill = None
    if baseline_names and baseline_names[0] != selected:
        ref = final_oos[baseline_names[0]]
        d = M.paired_bootstrap_diff(M.log_loss, _P(sel_final, classes), _P(ref, classes), _y(sel_final),
                                    final_groups, n_boot=cfg.n_boot, seed=cfg.seed)
        skill = {"reference": baseline_names[0], "metric": "log_loss", **d}
    final_block["skill_vs_baseline"] = skill

    sel_dev = dev_oos[selected]
    dev_block = {
        "selected": _metrics_with_ci(sel_dev, classes, cfg),
        "by_block": _by_block(sel_dev, classes),
        "reliability": M.reliability_table(_P(sel_dev, classes), _y(sel_dev)),
        "reliability_selected_outcome": M.reliability_table(_P(sel_dev, classes), _y(sel_dev), top_label=True),
        "pass_rule": _pass_rule(sel_dev, classes, cfg.pass_grid),
        "stability_vs_baselines": {
            c.name: _stability(sel_dev, dev_oos[c.name], classes)
            for c in candidates if c.is_baseline and c.name != selected
        },
    }
    final_block["stability_vs_baselines"] = {
        n: _stability(sel_final, final_oos[n], classes) for n in report_names if n != selected
    }

    # ----------------------------------------------------- market benchmark
    bench_block = None
    if benchmark is not None and len(benchmark):
        cols = _prob_cols(classes)
        if not set(cols) <= set(benchmark.columns):
            raise ValueError(f"benchmark needs probability columns {cols}")
        merged = sel_final.merge(benchmark[["event_id", *cols]].dropna(), on="event_id", suffixes=("", "_mkt"))
        if len(merged):
            Pm = merged[[f"{c}_mkt" for c in cols]].to_numpy(dtype=float)
            Pm = Pm / Pm.sum(axis=1, keepdims=True)
            ym, groups = _y(merged), slate_dates(merged["start_time_utc"])
            d = M.paired_bootstrap_diff(M.log_loss, _P(merged, classes), Pm, ym, groups,
                                        n_boot=cfg.n_boot, seed=cfg.seed)
            bench_block = {
                "label": str(benchmark["benchmark_label"].iloc[0]) if "benchmark_label" in benchmark else "market",
                "timing": str(benchmark["benchmark_timing"].iloc[0]) if "benchmark_timing" in benchmark else "unknown",
                "n": int(len(merged)),
                "coverage_of_final_test": float(len(merged) / len(sel_final)),
                "market": M.summary(Pm, ym),
                "model_same_events": M.summary(_P(merged, classes), ym),
                "log_loss_market_minus_model": d,
                "note": (
                    "Comparison only. The primary model never sees prices. A closing line contains "
                    "information published after any realistic prediction time."
                ),
            }

    # ------------------------------------------------------------------ gates
    all_sel = pd.concat([sel_dev, sel_final], ignore_index=True)
    fm = final_block["selected"]
    gates = [
        {"gate": "real_data", "passed": not is_synthetic,
         "detail": "synthetic demo data can never be validated" if is_synthetic else "real events"},
        {"gate": "point_in_time_features", "passed": bool(frame.point_in_time),
         "detail": "features use only information available before each event"
         if frame.point_in_time else "features are a retrospective reconstruction"},
        {"gate": "final_test_sample", "passed": fm["n"] >= cfg.min_final_events,
         "detail": f"{fm['n']} final-test events (minimum {cfg.min_final_events})"},
        {"gate": "development_folds", "passed": len(folds.dev) >= cfg.min_dev_folds,
         "detail": f"{len(folds.dev)} walk-forward folds (minimum {cfg.min_dev_folds})"},
        {"gate": "skill_over_baseline",
         "passed": bool(skill is not None and np.isfinite(skill["low"]) and skill["low"] > 0),
         "detail": ("selected model is the baseline itself" if skill is None else
                    f"final-test log-loss improvement over {skill['reference']}: {skill['diff']:.5f} "
                    f"(95% interval {skill['low']:.5f} to {skill['high']:.5f})")},
        {"gate": "calibration", "passed": bool(fm["ece"] <= cfg.max_ece),
         "detail": f"final-test expected calibration error {fm['ece']:.4f} (maximum {cfg.max_ece})"},
    ]
    report: Dict[str, Any] = {
        "sport": sport,
        "league": frame.league,
        "market": frame.market,
        "created_at_utc": iso_z(utcnow()),
        "is_synthetic": bool(is_synthetic),
        "point_in_time": bool(frame.point_in_time),
        "outcome_type": str(frame.outcome),
        "classes": classes,
        "protocol": {
            "plan": asdict(plan),
            "dev_folds": [f.describe() for f in folds.dev],
            "final_fold": folds.final.describe(),
            "selection": "pooled development-fold log loss" + ("; one-standard-error rule" if cfg.one_se_rule else ""),
            "bootstrap": {"n_boot": cfg.n_boot, "cluster": "Pacific/Honolulu slate date", "seed": cfg.seed},
            "final_test_used_for_selection": False,
        },
        "data": {
            "n_completed": int(len(df)),
            "n_upcoming": int(len(frame.upcoming)),
            "first_event": iso_z(df["start_time_utc"].min()),
            "last_event": iso_z(df["start_time_utc"].max()),
            "blocks": folds.block_ranges,
            "feature_cols": list(frame.feature_cols),
            "retrospective_cols": list(frame.retrospective_cols),
            "feature_notes": dict(frame.feature_notes),
        },
        "candidates": cand_reports,
        "selected_model": selected,
        "selection_note": selection_note,
        "dev": dev_block,
        "final_test": final_block,
        "market_benchmark": bench_block,
        "confidence_table": _confidence_table(all_sel, classes, cfg.confidence_bin_width),
        "same_day_residual_correlation": same_day_residual_correlation(
            all_sel, classes, n_boot=cfg.n_boot, seed=cfg.seed),
        "gates": gates,
        "gates_passed": all(g["passed"] for g in gates),
        "limitations": list(frame.limitations),
        "config": asdict(cfg),
    }
    oos = pd.concat(
        [dev_oos[n] for n in report_names if n in dev_oos] + [final_oos[n] for n in report_names],
        ignore_index=True,
    )
    oos["is_final_test"] = oos["fold"] == "final"
    oos["selected"] = oos["model"] == selected
    return report, oos
