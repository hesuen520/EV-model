"""Combo selection for one Pacific/Honolulu slate.

Priority: the strongest supported **three-leg** combination by estimated
probability that every leg wins.  Also returned: strong singles, the best
two-leg alternative, and four-leg and larger combinations for as long as they
stay supported.  Candidates are compared within each size.

Adding a leg multiplies the all-win probability by a conditional probability
that is at most 1, so a larger combination can never be more likely to win
than the combination it extends; the output states the probability lost at
each step.

When fewer than three legs qualify the result contains fewer picks and the
message ``PASS — insufficient support for a three-leg combo.``  A combination
is never forced, and legs with unresolved conditions are never presented as
finalized.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from itertools import combinations
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from ..config import ComboConfig, Settings
from ..core.enums import RecStatus
from ..core.timeutil import hst_date, hst_day_bounds_utc, iso_z, parse_ts, utcnow
from .dependency import JointResult, Leg, joint_probability

__all__ = ["PASS_MESSAGE", "MONOTONE_NOTE", "ComboCandidate", "SlateCombos", "candidate_status", "legs_from_ledger",
           "residual_correlations", "select_combos", "build_slate_combos"]

PASS_MESSAGE = "PASS — insufficient support for a three-leg combo."
MONOTONE_NOTE = (
    "Adding a leg can never raise the probability that every selection wins: each added leg multiplies "
    "it by a probability no greater than 1."
)


@dataclass
class ComboCandidate:
    legs: List[Leg]
    joint: JointResult

    @property
    def size(self) -> int:
        return len(self.legs)

    @property
    def prob(self) -> float:
        return float(self.joint.prob)


@dataclass
class SlateCombos:
    slate_date_hst: str
    status: RecStatus                              # status of the primary recommendation
    message: str
    eligible: List[Leg] = field(default_factory=list)
    excluded: List[Dict[str, str]] = field(default_factory=list)    # leg label -> reason it is not eligible
    singles: List[ComboCandidate] = field(default_factory=list)
    two_leg: List[ComboCandidate] = field(default_factory=list)
    three_leg: List[ComboCandidate] = field(default_factory=list)   # best first
    larger: Dict[int, List[ComboCandidate]] = field(default_factory=dict)
    rejected_examples: List[Dict[str, Any]] = field(default_factory=list)
    preview: Optional["SlateCombos"] = None        # PROVISIONAL preview, never a recommendation
    rules: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def residual_correlations(store: Any, leagues: Sequence[str]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Validated same-slate residual correlation per (league, market family),
    from each league's latest validation run."""
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    runs = store.validation_runs()
    for _, row in runs.iterrows():
        if row["league"] not in leagues or int(row["is_synthetic"]) and not str(row["league"]).startswith("demo_"):
            continue
        est = json.loads(row["report_json"]).get("same_day_residual_correlation")
        if est:
            out[(row["league"], str(row["market"]).split(":", 1)[0])] = est
    return out


def legs_from_ledger(store: Any, predictions: pd.DataFrame) -> List[Leg]:
    """Turn ledger rows into legs, attaching each event's participants."""
    if predictions.empty:
        return []
    events = store.events(event_ids=predictions["event_id"].unique().tolist(), with_results=False,
                          include_synthetic=True).set_index("event_id")
    legs: List[Leg] = []
    for _, p in predictions.iterrows():
        ev = events.loc[p["event_id"]]
        participants = {x for x in (ev["home_id"], ev["away_id"]) if isinstance(x, str)}
        if isinstance(ev["participants_json"], str):
            participants |= set(json.loads(ev["participants_json"]))
        display = json.loads(p["display_json"])
        features = json.loads(p["features_json"])
        entities = {p["selection"]} if p["selection"] in participants else set()
        for key in ("player_id", "team_id"):
            for src in (display, features):
                if isinstance(src.get(key), str):
                    entities.add(src[key])
        a, b = display.get("side_a"), display.get("side_b")
        label = (f"{b} at {a}" if display.get("side_a_role") == "home" else f"{a} v {b}") if a and b else p["event_id"]
        legs.append(Leg(
            prediction_id=p["prediction_id"], event_id=p["event_id"], league=p["league"], sport=p["sport"],
            market=p["market"], line=None if pd.isna(p["line"]) else float(p["line"]),
            selection=p["selection"], label=p["selection_label"],
            prob=float(p["prob"]) if not pd.isna(p["prob"]) else float("nan"),
            prob_low=None if pd.isna(p["prob_low"]) else float(p["prob_low"]),
            rec_status=p["rec_status"], validation_status=p["validation_status"],
            start_utc=p["event_start_utc"], participants=frozenset(participants), entities=frozenset(entities),
            event_label=label,
        ))
    return legs


def candidate_status(cand: ComboCandidate) -> RecStatus:
    """A combination is only as final as its least final leg."""
    statuses = {l.rec_status for l in cand.legs}
    if statuses == {RecStatus.FINAL.value}:
        return RecStatus.FINAL
    if RecStatus.CONDITIONAL.value in statuses:
        return RecStatus.CONDITIONAL
    return RecStatus.PROVISIONAL


def _eligibility(leg: Leg, cfg: ComboConfig, statuses: Sequence[str], now_iso: str) -> Optional[str]:
    """``None`` when the leg may be used, otherwise the reason it may not."""
    if leg.rec_status not in statuses:
        return f"status {leg.rec_status} (needs {' or '.join(statuses)})"
    if leg.validation_status not in cfg.allowed_coverage:
        return f"coverage {leg.validation_status} (needs {' or '.join(cfg.allowed_coverage)})"
    if leg.prob != leg.prob or leg.prob < cfg.min_leg_prob:
        return f"probability {leg.prob:.3f} below the leg minimum {cfg.min_leg_prob:.2f}"
    if leg.start_utc <= now_iso:
        return "event has started"
    return None


def select_combos(legs: Sequence[Leg], cfg: ComboConfig,
                  residual_corr: Mapping[Tuple[str, str], Mapping[str, Any]],
                  *, pool_size: int = 14) -> Dict[str, Any]:
    """Enumerate valid combinations of every size among the strongest legs.

    Returns ``{"by_size": {k: [ComboCandidate best first]}, "rejected": [...]}``.
    Only the ``pool_size`` most probable legs are enumerated; the result notes
    when that bound applied.
    """
    ranked = sorted(legs, key=lambda l: -l.prob)
    pool = ranked[:pool_size]
    by_size: Dict[int, List[ComboCandidate]] = {}
    rejected: List[Dict[str, Any]] = []
    for k in range(1, min(cfg.max_size, len(pool)) + 1):
        found: List[ComboCandidate] = []
        for combo in combinations(pool, k):
            joint = joint_probability(combo, residual_corr, cfg.max_abs_residual_correlation)
            if joint.valid:
                found.append(ComboCandidate(list(combo), joint))
            elif len(rejected) < 12:
                rejected.append({"legs": [l.label for l in combo],
                                 "reasons": sorted({r.note for r in joint.rejected})})
        found.sort(key=lambda c: -c.prob)
        if found:
            by_size[k] = found[: max(cfg.alternatives_per_size, 1)]
    return {"by_size": by_size, "rejected": rejected, "pool_truncated": len(ranked) > len(pool)}


def _assemble(slate: str, legs: Sequence[Leg], cfg: ComboConfig, statuses: Sequence[str], now_iso: str,
              residual_corr: Mapping[Tuple[str, str], Mapping[str, Any]]) -> SlateCombos:
    eligible, excluded = [], []
    for leg in legs:
        reason = _eligibility(leg, cfg, statuses, now_iso)
        if reason is None:
            eligible.append(leg)
        else:
            excluded.append({"leg": f"{leg.label} ({leg.event_label}, {leg.market})", "reason": reason})
    picked = select_combos(eligible, cfg, residual_corr)
    by_size = picked["by_size"]
    out = SlateCombos(slate_date_hst=slate, status=RecStatus.PASS, message=PASS_MESSAGE,
                      eligible=sorted(eligible, key=lambda l: -l.prob), excluded=excluded,
                      rejected_examples=picked["rejected"], notes=[MONOTONE_NOTE])
    out.singles = by_size.get(1, [])
    out.two_leg = by_size.get(2, [])
    out.three_leg = by_size.get(cfg.primary_size, []) if cfg.primary_size != 2 else out.two_leg
    if picked["pool_truncated"]:
        out.notes.append("Only the 14 most probable eligible legs were combined.")
    if out.three_leg:
        best = out.three_leg[0]
        out.status = candidate_status(best)
        out.message = (f"Strongest supported {cfg.primary_size}-leg combo: estimated probability that all legs win "
                       f"{best.prob:.3f} (conservative {best.joint.prob_low:.3f}).")
        prev = best.prob
        for k in sorted(s for s in by_size if s > cfg.primary_size):
            cands = [c for c in by_size[k] if c.prob >= cfg.min_joint_prob_larger]
            if not cands:
                out.notes.append(
                    f"{k}-leg combinations are not supported: the best one has all-win probability "
                    f"{by_size[k][0].prob:.3f}, below the minimum {cfg.min_joint_prob_larger:.2f}.")
                break
            out.larger[k] = cands
            out.notes.append(f"Going from {k - 1} to {k} legs lowers the best all-win probability "
                             f"from {prev:.3f} to {cands[0].prob:.3f}.")
            prev = cands[0].prob
    elif len(eligible) >= cfg.primary_size:
        out.message = (PASS_MESSAGE + " Enough legs qualify individually, but every three-leg combination "
                       "contains a dependency that is not modelled.")
    return out


def build_slate_combos(store: Any, day: Optional[date] = None, *, settings: Optional[Settings] = None,
                       now: Any = None, leagues: Optional[Sequence[str]] = None, write: bool = True,
                       include_synthetic: bool = False) -> SlateCombos:
    """Select combos for one HST slate from the latest ledger predictions and
    (optionally) record them in the immutable ``combos`` table."""
    settings = settings or Settings()
    cfg = settings.combos
    now_dt = utcnow() if now is None else parse_ts(now)
    day = day or hst_date(now_dt)
    lo, hi = hst_day_bounds_utc(day)
    preds = store.predictions(league=list(leagues) if leagues else None, start_from=lo, start_before=hi,
                              latest_only=True, include_synthetic=include_synthetic)
    preds = preds[preds["rec_status"] != RecStatus.PASS.value]
    legs = legs_from_ledger(store, preds)
    corr = residual_correlations(store, sorted({l.league for l in legs}))
    now_iso = iso_z(now_dt)
    result = _assemble(day.isoformat(), legs, cfg, cfg.leg_statuses, now_iso, corr)
    result.rules = {
        "primary_size": cfg.primary_size, "max_size": cfg.max_size, "leg_statuses": list(cfg.leg_statuses),
        "min_leg_prob": cfg.min_leg_prob, "min_joint_prob_larger": cfg.min_joint_prob_larger,
        "allowed_coverage": list(cfg.allowed_coverage),
        "max_abs_residual_correlation": cfg.max_abs_residual_correlation,
    }
    if cfg.show_provisional_preview and not result.three_leg:
        statuses = sorted(set(cfg.leg_statuses) | {RecStatus.FINAL.value, RecStatus.PROVISIONAL.value})
        preview = _assemble(day.isoformat(), legs, cfg, statuses, now_iso, corr)
        if preview.three_leg or preview.singles:
            preview.status = RecStatus.PROVISIONAL
            preview.message = (
                "PROVISIONAL preview only - not a recommendation. These legs still have required pregame "
                "checks outstanding; re-run after the checks are on file."
                + ("" if preview.three_leg else " " + PASS_MESSAGE))
            result.preview = preview
    if write:
        _record(store, result, now_iso, include_synthetic)
    return result


def _record(store: Any, result: SlateCombos, now_iso: str, synthetic: bool) -> None:
    def put(role: str, cand: Optional[ComboCandidate], status: RecStatus, message: Optional[str]) -> None:
        store.insert_combo({
            "created_at_utc": now_iso, "slate_date_hst": result.slate_date_hst, "role": role,
            "leg_ids": [] if cand is None else [l.prediction_id for l in cand.legs],
            "joint_prob": None if cand is None else cand.prob,
            "joint_prob_low": None if cand is None else cand.joint.prob_low,
            "joint_method": "none" if cand is None else cand.joint.method,
            "dependency": {} if cand is None else cand.joint.as_dict(),
            "rec_status": status.value, "message": message, "rules": result.rules,
            "is_synthetic": synthetic,
        })

    if result.three_leg:
        put("primary_three_leg", result.three_leg[0], result.status, result.message)
    else:
        put("none", None, RecStatus.PASS, result.message)
    if result.two_leg:
        put("two_leg_alternative", result.two_leg[0], candidate_status(result.two_leg[0]), None)
    for cand in result.singles:
        put("single", cand, candidate_status(cand), None)
    for k, cands in result.larger.items():
        put(f"larger_{k}", cands[0], candidate_status(cands[0]), None)
    if result.preview is not None and result.preview.three_leg:
        put("provisional_preview", result.preview.three_leg[0], RecStatus.PROVISIONAL, result.preview.message)
