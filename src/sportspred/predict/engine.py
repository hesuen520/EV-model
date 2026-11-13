"""Pregame prediction engine.

For each upcoming event of a league and market it produces one ledger record:
the model's outcome probabilities, three separate kinds of uncertainty, the
state of every configured pregame check, an explanation, and a recommendation
status.

Status rules (first match wins):

1. **PASS** - coverage status not allowed to select, required inputs missing,
   the selection's probability is below the configured minimum, or its
   model-uncertainty interval reaches a coin flip.
2. **PROVISIONAL** - a required pregame check is missing, stale or due a
   re-check.
3. **CONDITIONAL** - checks are complete, but the selection would fail the
   rules if named players whose availability is uncertain do not play.
4. **FINAL** - every configured check is complete.  The result is still
   uncertain; FINAL describes the *information*, not the outcome.

Nothing here reads prices: the engine has no access to the EV module.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import Settings
from ..core.enums import REC_STATUS_DEFINITIONS, CheckState, OutcomeType, RecStatus
from ..core.timeutil import fmt_hst, iso_z, parse_ts, utcnow
from ..coverage import coverage_status
from ..models.fitted import FittedModel
from ..sports import get_league
from ..sports.base import FeatureFrame
from ..storage.store import LedgerError
from .checks import checks_complete, evaluate_checks
from .confidence import confidence_for

__all__ = ["predict_league", "load_promoted", "PredictionError"]

log = logging.getLogger(__name__)


class PredictionError(RuntimeError):
    pass


def load_promoted(store: Any, league: str, market: str) -> FittedModel:
    meta = store.promoted_model(league, market)
    if meta is None:
        raise PredictionError(
            f"no trained model for {league} {market}; run `sportspred train --league {league} --market {market}`")
    return FittedModel.load(meta["artifact_path"])


def _class_labels(frame: FeatureFrame, row: Mapping[str, Any], names: Mapping[str, str],
                  line: Optional[float]) -> Dict[str, Tuple[str, str]]:
    """class name -> (selection id, display label).

    Conventions: for spread markets ``line`` is the handicap applied to side A
    (the home side), negative when side A is favoured, so side A covers when
    ``margin_A + line > 0`` and side B's handicap is ``-line``.  For totals
    the classes are ``under`` / ``over`` at ``line``.
    """
    name_a = names.get(row["side_a_id"], row["side_a_id"])
    name_b = names.get(row["side_b_id"], row["side_b_id"])
    out: Dict[str, Tuple[str, str]] = {}
    for cls in frame.classes:
        if cls in ("home", "a"):
            out[cls] = (row["side_a_id"], name_a)
        elif cls in ("away", "b"):
            out[cls] = (row["side_b_id"], name_b)
        elif cls in ("home_cover", "a_cover"):
            out[cls] = (row["side_a_id"], name_a if line is None else f"{name_a} {line:+g}")
        elif cls in ("away_cover", "b_cover"):
            out[cls] = (row["side_b_id"], name_b if line is None else f"{name_b} {-line:+g}")
        elif cls in ("over", "under"):
            out[cls] = (cls, cls.capitalize() if line is None else f"{cls.capitalize()} {line:g}")
        else:
            out[cls] = (cls, cls.replace("_", " ").capitalize())
    return out


def _unresolved_past_events(store: Any, league: str, now_ts: pd.Timestamp) -> Tuple[Dict[str, int], str]:
    """Per participant: events that should have been played by ``now`` but
    have no result on file (postponed and cancelled events excluded), and the
    date of the latest result in the store."""
    ev = store.events(league, with_results=True, include_synthetic=True)
    past = ev[(ev["start_time_utc"] < now_ts) & ev["status"].isin(["scheduled", "live", "unknown"])]
    # A date-only event stored at 00:00Z of today may simply not have started yet.
    past = past[(past["time_precision"] == "datetime") | (past["start_time_utc"] < now_ts - pd.Timedelta(hours=36))]
    counts: Dict[str, int] = {}
    for col in ("home_id", "away_id"):
        for team, n in past[col].dropna().value_counts().items():
            counts[team] = counts.get(team, 0) + int(n)
    done = ev[ev["status"] == "final"]
    through = "none" if done.empty else done["start_time_utc"].max().strftime("%Y-%m-%d")
    return counts, through


def _entropy(p: Sequence[float]) -> float:
    return float(-sum(x * math.log(x) for x in p if x > 0) / math.log(len(p)))


def _data_quality(row: Mapping[str, Any], frame: FeatureFrame, checks: List[Dict[str, Any]],
                  missing_features: List[str]) -> Dict[str, Any]:
    incomplete = [c for c in checks if c["state"] != CheckState.COMPLETE.value]
    flags = [k for k in row.keys() if str(k).startswith("dq_") and row[k] == 1.0]
    if missing_features and len(missing_features) > max(2, len(frame.feature_cols) // 3):
        grade = "poor"
    elif incomplete or missing_features or flags:
        grade = "partial"
    else:
        grade = "complete"
    return {
        "grade": grade,
        "missing_features": missing_features,
        "flags": flags,
        "incomplete_checks": [f"{c['label']} ({c['subject']}): {c['state']}" for c in incomplete],
    }


def predict_league(
    store: Any,
    league: str,
    market: str = "winner",
    *,
    settings: Optional[Settings] = None,
    now: Any = None,
    horizon_hours: Optional[float] = None,
    write: bool = True,
) -> List[Dict[str, Any]]:
    """Predict every upcoming event of ``league`` within the horizon.

    Returns the prediction records (as written to the ledger when ``write``).
    An unchanged prediction (same model, status, selection and probability to
    three decimals) is not written again; a changed one is appended and linked
    to its predecessor through ``supersedes_id`` - nothing is ever rewritten.
    """
    settings = settings or Settings()
    now_dt = utcnow() if now is None else parse_ts(now)
    now_ts = pd.Timestamp(now_dt)
    horizon = settings.selection.horizon_hours if horizon_hours is None else horizon_hours
    plugin, spec = get_league(league)
    decl = spec.market(market)
    model = load_promoted(store, league, market)
    frame = plugin.feature_frame(store, league, market, now=now_dt)
    names = store.participant_names()
    frame.extra["names"] = names
    rows = frame.upcoming
    rows = rows[(rows["start_time_utc"] > now_ts)
                & (rows["start_time_utc"] <= now_ts + pd.Timedelta(hours=horizon))]
    if rows.empty:
        return []
    coverage = coverage_status(store, league, market)
    report = store.latest_validation(league, market)
    min_prob = settings.min_prob(league, market, str(frame.outcome))
    check_log = store.checks_as_of(league, now_dt, include_synthetic=True)
    specs = plugin.required_checks(league, market)
    unresolved, results_through = _unresolved_past_events(store, league, now_ts)
    events = store.events(event_ids=rows["event_id"].tolist(), with_results=False, include_synthetic=True)
    events = events.set_index("event_id")
    feature_names = model.candidate.feature_names() or frame.feature_cols
    prior = store.predictions(event_ids=rows["event_id"].tolist(), market=market, latest_only=True,
                              include_synthetic=True)
    prior_by_event = {r["event_id"]: r for _, r in prior.iterrows()}

    out: List[Dict[str, Any]] = []
    for _, base_row in rows.iterrows():
        ev = events.loc[base_row["event_id"]]
        team_ids = [base_row["side_a_id"], base_row["side_b_id"]]
        avail = store.availability_as_of(league, now_dt, team_ids=team_ids, event_id=base_row["event_id"],
                                         include_synthetic=True)
        effect = plugin.apply_availability(league, market, base_row, frame, avail)
        row = effect.row
        pred = model.predict(pd.DataFrame([row]))
        P, low, high = pred["P"][0], pred["low"][0], pred["high"][0]
        k = int(np.argmax(P))
        line = None
        if frame.outcome == OutcomeType.OVER_UNDER and "line" in row.index and row["line"] == row["line"]:
            line = float(row["line"])
        labels = _class_labels(frame, row, names, line)
        cls = frame.classes[k]
        selection, selection_label = labels[cls]
        # A plugin may supply the probability of an exact tie against the line
        # (``p_push``); the class probabilities are then conditional on no push
        # and are scaled so that the recorded distribution includes the push.
        p_push = 0.0
        if "p_push" in row.index and row["p_push"] == row["p_push"]:
            p_push = float(min(max(row["p_push"], 0.0), 0.5))
        scale = 1.0 - p_push
        prob, prob_low, prob_high = float(P[k]) * scale, float(low[k]) * scale, float(high[k]) * scale
        probs = {labels[c][0]: float(P[i]) * scale for i, c in enumerate(frame.classes)}
        if p_push > 0.0:
            probs["push"] = p_push

        checks = evaluate_checks(specs, row, check_log, now=now_ts, settings=settings, names=names)
        # Built in for every league: ratings are only current if every earlier
        # event of both sides has a result on file.
        behind = unresolved.get(row["side_a_id"], 0) + unresolved.get(row["side_b_id"], 0)
        checks.append({
            "key": "results_current", "label": "results on file are up to date", "subject": "event",
            "state": (CheckState.COMPLETE if behind == 0 else CheckState.MISSING).value,
            "detail": ("every earlier event of both sides has a result on file" if behind == 0 else
                       f"{behind} earlier event(s) of these sides have no result on file (latest result in the "
                       f"store: {results_through}); ratings are out of date until results are refreshed"),
            "checked_at_utc": iso_z(now_dt), "published_at_utc": None, "source_id": None,
        })
        if ev["time_precision"] != "datetime":
            # The stored instant of a date-only event is a placeholder, so no
            # freshness window can honestly be judged against it.
            checks.append({
                "key": "start_time_published", "label": "start time published", "subject": "event",
                "state": CheckState.MISSING.value,
                "detail": "the source gives only a date for this event; a published start time is needed before "
                          "pregame checks can be judged against it",
                "checked_at_utc": iso_z(now_dt), "published_at_utc": None, "source_id": None,
            })
        missing = [c for c in feature_names if c in row.index and row[c] != row[c]]
        quality = _data_quality(row, frame, checks, missing)

        reasons: List[str] = []
        status = RecStatus.FINAL
        # ---- 1. PASS rules
        pass_reasons: List[str] = []
        if coverage.value not in settings.selection.allowed_coverage:
            pass_reasons.append(f"coverage status is {coverage.value}; selections need one of "
                                f"{settings.selection.allowed_coverage}")
        if quality["grade"] == "poor":
            pass_reasons.append(f"too many model inputs are missing ({len(missing)} of {len(feature_names)})")
        if prob < min_prob:
            pass_reasons.append(f"model probability {prob:.3f} is below the selection minimum {min_prob:.2f}")
        elif (settings.selection.require_interval_above_even and frame.outcome == OutcomeType.BINARY
              and prob_low <= 0.5):
            pass_reasons.append(f"model-uncertainty interval {prob_low:.3f}-{prob_high:.3f} reaches a coin flip")
        if pass_reasons:
            status, reasons = RecStatus.PASS, pass_reasons
        # ---- 2. PROVISIONAL: checks
        elif not checks_complete(checks):
            status = RecStatus.PROVISIONAL
            for c in checks:
                if c["state"] != CheckState.COMPLETE.value:
                    reasons.append(f"{c['label']}: {c['detail']}")
            reasons.append("Re-run the prediction once these checks are on file; no confirmation time is assumed.")
        else:
            # ---- 3. CONDITIONAL: does the selection survive its own side's doubts sitting out?
            pick_side = ("a" if cls in ("home", "a", "home_cover", "a_cover") else
                         "b" if cls in ("away", "b", "away_cover", "b_cover") else None)
            doubts = effect.uncertain_a if pick_side == "a" else effect.uncertain_b if pick_side == "b" else []
            if doubts:
                worst = effect.worst_case_rows.get(pick_side) if pick_side else None
                if worst is None:
                    status = RecStatus.CONDITIONAL
                    reasons.append(
                        f"holds only if these {selection_label} players play (effect not quantified): "
                        + "; ".join(doubts))
                else:
                    Pw = model.predict(pd.DataFrame([worst]))["P"][0]
                    if int(np.argmax(Pw)) != k or float(Pw[k]) < min_prob:
                        status = RecStatus.CONDITIONAL
                        reasons.append(
                            f"condition: {'; '.join(doubts)} must play. If they sit, the model probability of "
                            f"{selection_label} becomes {float(Pw[k]):.3f}, which fails the selection rules.")
                    else:
                        reasons.append(
                            f"selection holds even if {'; '.join(doubts)} sit "
                            f"(probability would be {float(Pw[k]):.3f}).")
            if status == RecStatus.FINAL:
                reasons.insert(0, "all configured pregame checks are complete and fresh")

        contributions = None
        contrib = model.contributions(pd.DataFrame([row]))
        if contrib is not None:
            contributions = {c: float(contrib.iloc[0][c]) for c in contrib.columns}
        row_for_explain = row.copy()
        row_for_explain["_pick_is_a"] = cls in ("home", "a", "home_cover", "a_cover")
        row_for_explain["_pick_class"] = cls
        explanation = plugin.explain(league, market, row_for_explain, frame, contributions)
        explanation.availability_assumptions += effect.applied + effect.unmatched
        for side_name, items in ((names.get(row["side_a_id"]), effect.uncertain_a),
                                 (names.get(row["side_b_id"]), effect.uncertain_b)):
            if items:
                explanation.availability_assumptions.append(
                    f"uncertain for {side_name}: " + "; ".join(items) + " (assumed to play as recently)")
        if avail.empty:
            explanation.missing_data.append("no availability report on file for either side")

        confidence = confidence_for(prob, report, settings.confidence)
        confidence["definitions"] = {
            "event_uncertainty": "chance the selection loses even if the probability is exactly right",
            "model_uncertainty": "how far the probability moves when the model is refitted on resampled history",
            "data_quality": "missing or stale inputs and incomplete pregame checks",
        }
        confidence["event_uncertainty"] = {"prob_selection_loses": 1.0 - prob, "normalised_entropy": _entropy(P)}
        confidence["model_uncertainty"] = {
            "interval": [prob_low, prob_high], "level": model.interval_level,
            "method": f"{len(model.replicas)} bootstrap refits (resampled by slate)",
        }
        precision_date = ev["time_precision"] != "datetime"
        display = {
            "competition": spec.name, "league": league, "sport": spec.sport, "season": ev["season"],
            "stage": ev["stage"],
            "side_a": names.get(row["side_a_id"], row["side_a_id"]),
            "side_b": names.get(row["side_b_id"], row["side_b_id"]),
            "side_a_role": "home" if ev["home_id"] == row["side_a_id"] and ev["neutral_site"] != 1 else "a",
            "start_hst": fmt_hst(ev["start_time_utc"], date_only=precision_date),
            "start_utc": iso_z(ev["start_time_utc"]),
            "venue": ev["venue_name"], "venue_city": ev["venue_city"],
            "neutral_site": None if ev["neutral_site"] != ev["neutral_site"] else bool(ev["neutral_site"]),
            "market": market, "market_note": decl.note, "line": line,
            "status_definition": REC_STATUS_DEFINITIONS[status],
            "last_check_hst": fmt_hst(now_dt),
        }
        source_ids = [s for s in {ev["source_id"], *[c.get("source_id") for c in checks],
                                  *avail.get("source_id", pd.Series(dtype=object)).tolist()} if s]
        src = store.sources(source_ids)
        sources = [{"source_id": r["source_id"], "provider": r["provider"], "url": r["url"],
                    "collected_at_utc": r["collected_at_utc"], "published_at_utc": r["published_at_utc"]}
                   for _, r in src.iterrows()]
        snapshot = {c: (None if row[c] != row[c] else float(row[c])) for c in feature_names if c in row.index}
        record: Dict[str, Any] = {
            "created_at_utc": iso_z(now_dt), "event_id": row["event_id"], "market": market, "line": line,
            "selection": selection,
            "selection_label": selection_label, "prob": prob, "prob_low": prob_low, "prob_high": prob_high,
            "probs": probs, "model_name": model.model_name, "model_version": model.model_version,
            "validation_status": coverage.value, "rec_status": status.value, "reasons": reasons,
            "checks": checks, "confidence": confidence, "factors": explanation.as_dict(),
            "features": snapshot, "data_quality": quality, "sources": sources, "display": display,
        }
        previous = prior_by_event.get(row["event_id"])
        if previous is not None:
            same = (
                previous["model_version"] == record["model_version"]
                and previous["rec_status"] == record["rec_status"]
                and previous["selection"] == record["selection"]
                and previous["prob"] is not None and abs(float(previous["prob"]) - prob) < 5e-4
                and [c["state"] for c in json.loads(previous["checks_json"])] == [c["state"] for c in checks]
                and json.loads(previous["reasons_json"]) == reasons
            )
            if same:
                record["prediction_id"] = previous["prediction_id"]
                record["unchanged"] = True
                out.append(record)
                continue
            record["supersedes_id"] = previous["prediction_id"]
        if write:
            try:
                record["prediction_id"] = store.insert_prediction(record)
            except LedgerError as exc:
                log.warning("prediction for %s not written: %s", row["event_id"], exc)
                record["ledger_error"] = str(exc)
        out.append(record)
    return out
