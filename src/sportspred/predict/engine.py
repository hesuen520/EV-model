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
from ..core.markets import model_market
from ..core.timeutil import fmt_hst, iso_z, parse_ts, utcnow
from ..coverage import coverage_status
from ..models.fitted import FittedModel
from ..sports import get_league
from ..sports.base import FeatureFrame
from ..storage.store import LedgerError
from .checks import checks_complete, evaluate_checks
from .confidence import confidence_for

__all__ = ["predict_league", "load_promoted", "PredictionError", "results_current"]

from ..storage.store import DATE_ONLY_LEAD_HOURS  # noqa: E402  (the ledger applies the same rule)

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
                  line: Optional[float], subject: Optional[str] = None,
                  unit: Optional[str] = None) -> Dict[str, Tuple[str, str]]:
    """class name -> (selection id, display label).

    Conventions: for spread markets ``line`` is the handicap applied to side A
    (the home side), negative when side A is favoured, so side A covers when
    ``margin_A + line > 0`` and side B's handicap is ``-line``.  For totals
    the classes are ``under`` / ``over`` at ``line``.  A market about one
    player (``subject``) names him and the statistic (``unit``) in the label.
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
        elif cls in ("over", "under") and subject:
            out[cls] = (cls, f"{subject} {cls}" + ("" if line is None else f" {line:g}") + (f" {unit}" if unit else ""))
        elif cls in ("over", "under"):
            out[cls] = (cls, cls.capitalize() if line is None else f"{cls.capitalize()} {line:g}")
        else:
            out[cls] = (cls, cls.replace("_", " ").capitalize())
    return out


def _unresolved_past_events(store: Any, league: str, now_ts: pd.Timestamp, window_days: float = 45.0
                            ) -> Tuple[Dict[str, int], str, Optional[pd.Timestamp], Optional[float]]:
    """Per participant: events of the last ``window_days`` that should have been
    played by ``now`` but have no result on file (postponed and cancelled events
    excluded); the date of the latest result in the store; that latest result's
    instant; and the league's typical longest yearly break between results, in days.

    "No result" includes an event the source marks final without a score: the
    rating builders cannot use it either.  The window keeps a years-old gap in
    a source - a game that never got a score - from blocking a league for ever;
    a result that old is a permanent hole in the data, not a refresh away.

    The break is the median, over the complete calendar years on file, of each
    year's longest gap between consecutive result days (``None`` with fewer
    than three such years: the first and the last year of a history are partial
    and would understate it).  It is what lets a source that has simply stopped
    - with no unplayed fixtures listed - be told apart from an ordinary
    off-season."""
    ev = store.events(league, with_results=True, include_synthetic=True)
    decided = ev["winner_id"].notna() | (ev["is_draw"].fillna(0).astype(int) == 1)
    open_status = ev["status"].isin(["scheduled", "live", "unknown"]) | ((ev["status"] == "final") & ~decided)
    past = ev[(ev["start_time_utc"] < now_ts) & open_status
              & (ev["start_time_utc"] >= now_ts - pd.Timedelta(days=window_days))]
    # A date-only event stored at 00:00Z of today may simply not have started yet.
    past = past[(past["time_precision"] == "datetime") | (past["start_time_utc"] < now_ts - pd.Timedelta(hours=36))]
    counts: Dict[str, int] = {}
    for col in ("home_id", "away_id"):
        for team, n in past[col].dropna().value_counts().items():
            counts[team] = counts.get(team, 0) + int(n)
    done = ev[(ev["status"] == "final") & decided & (ev["start_time_utc"] <= now_ts)]
    if done.empty:
        return counts, "none", None, None
    latest = done["start_time_utc"].max()
    days = pd.Series(done["start_time_utc"].dt.normalize().unique()).sort_values().reset_index(drop=True)
    gaps = days.diff().dt.total_seconds().div(86400.0)
    yearly = gaps.groupby(days.dt.year).max().dropna()
    yearly = yearly[(yearly.index > days.dt.year.min()) & (yearly.index < days.dt.year.max())]
    usual_break = float(yearly.median()) if len(yearly) >= 3 else None
    return counts, latest.strftime("%Y-%m-%d"), latest, usual_break


def results_current(behind: int, days_since: Optional[float], usual_break: Optional[float], margin: float,
                    results_through: str) -> Tuple[bool, str]:
    """Whether the results behind a prediction are up to date, and why.

    ``behind`` counts earlier events of the two sides with no result on file.
    With none listed, the results still count as out of date when the newest
    one is more than ``margin`` typical off-seasons before the event: the
    source has stopped, not the league."""
    if behind:
        return False, (f"{behind} earlier event(s) of these sides have no result on file (latest result in the store: "
                       f"{results_through}); ratings are out of date until results are refreshed")
    if days_since is not None and usual_break is not None and days_since > margin * usual_break:
        return False, (f"the newest result on file ({results_through}) is {days_since:.0f} days before this event, "
                       f"more than {margin:g} times this league's typical longest yearly break ({usual_break:.0f} "
                       "days); results are very likely missing and ratings out of date")
    return True, "every earlier event of both sides has a result on file"


def _same_line(a: Any, b: Any) -> bool:
    missing_a, missing_b = a is None or a != a, b is None or b != b
    if missing_a or missing_b:
        return missing_a and missing_b
    return float(a) == float(b)


def _last_source_check(checks: Sequence[Mapping[str, Any]]) -> Optional[str]:
    """Newest time a source behind any check was read (``None`` when no source
    has been read at all - the built-in checks carry no source)."""
    times = [c["checked_at_utc"] for c in checks
             if isinstance(c.get("source_id"), str) and isinstance(c.get("checked_at_utc"), str)]
    return fmt_hst(max(times)) if times else None


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


def _outcome_probabilities(model: FittedModel, row: pd.Series, line: Optional[float], count_model: bool
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """``(P, low, high, push)`` for one row: the probability of each class as it
    will settle (a push is none of them), the model-uncertainty interval of
    each, and the probability of a push.

    * A count model at a whole-number threshold ``L`` is read at the half
      points either side - over ``L`` is over ``L + 0.5``, under ``L`` is under
      ``L - 0.5`` - so both come from the calibrated model and the push is
      what remains.  "Over 23" and "over 23.5" are one event and get one
      probability.
    * Any other model may carry a push probability in the row (``p_push``);
      its class probabilities are conditional on a decision and are scaled.
    """
    if count_model and line is not None and float(line) == math.floor(float(line)):
        upper, lower = row.copy(), row.copy()
        upper["line"], lower["line"] = float(line) + 0.5, max(float(line) - 0.5, 0.5)
        hi, lo = model.predict(pd.DataFrame([upper])), model.predict(pd.DataFrame([lower]))
        under = 0.0 if line < 1.0 else float(lo["P"][0][0])
        over = float(hi["P"][0][1])
        under = min(under, 1.0 - over)
        P = np.array([under, over])
        low = np.array([min(float(lo["low"][0][0]), under), float(hi["low"][0][1])])
        high = np.array([min(float(lo["high"][0][0]), 1.0 - over), float(hi["high"][0][1])])
        return P, low, high, float(max(0.0, 1.0 - over - under))
    pred = model.predict(pd.DataFrame([row]))
    P, low, high = pred["P"][0].astype(float), pred["low"][0].astype(float), pred["high"][0].astype(float)
    push = 0.0
    if not count_model and "p_push" in row.index and row["p_push"] == row["p_push"]:
        push = float(min(max(row["p_push"], 0.0), 0.5))
    scale = 1.0 - push
    return P * scale, low * scale, high * scale, push


_JSON_FIELDS = ("probs", "reasons", "checks", "confidence", "factors", "features", "data_quality", "sources", "display")


def _withdraw_stale(store: Any, league: str, market: str, kept: set, *, now_dt: Any, horizon: float,
                    write: bool) -> List[Dict[str, Any]]:
    """Withdraw selections about a subject the model no longer has a row for.

    A market about one player is recorded under its own key.  If that player
    drops out of the frame before the game - traded, or no longer in the
    modelled rotation - no new prediction would ever replace his old one and it
    would stay the current ledger row.  Each such selection is superseded by a
    PASS record that says so.  Only timed, scheduled events inside the horizon
    are touched, and only keys of this model market."""
    upcoming = store.upcoming(league, now=now_dt, horizon_hours=horizon, include_synthetic=True)
    upcoming = upcoming[upcoming["time_precision"] == "datetime"]
    if upcoming.empty:
        return []
    prior = store.predictions(event_ids=upcoming["event_id"].tolist(), latest_only=True, include_synthetic=True)
    out: List[Dict[str, Any]] = []
    for _, prev in prior.iterrows():
        key = str(prev["market"])
        if key == market or model_market(key) != market or prev["rec_status"] == RecStatus.PASS.value:
            continue
        if (prev["event_id"], key) in kept:
            continue
        record: Dict[str, Any] = {name: json.loads(prev[f"{name}_json"]) for name in _JSON_FIELDS}
        record["display"] = {**record["display"], "status_definition": REC_STATUS_DEFINITIONS[RecStatus.PASS],
                             "predicted_at_hst": fmt_hst(now_dt)}
        record.update({
            "created_at_utc": iso_z(now_dt), "event_id": prev["event_id"], "market": key,
            "line": None if prev["line"] != prev["line"] else prev["line"],
            "selection": prev["selection"], "selection_label": prev["selection_label"], "prob": prev["prob"],
            "prob_low": prev["prob_low"], "prob_high": prev["prob_high"], "model_name": prev["model_name"],
            "model_version": prev["model_version"], "validation_status": prev["validation_status"],
            "rec_status": RecStatus.PASS.value, "supersedes_id": prev["prediction_id"], "withdrawn": True,
            "reasons": ["withdrawn: the model no longer has a row for this player in this game (his team or his "
                        "recent minutes changed after the earlier prediction, or the threshold no longer applies); "
                        "the earlier selection does not stand"],
        })
        if write:
            try:
                record["prediction_id"] = store.insert_prediction(record)
            except LedgerError as exc:
                log.warning("withdrawal for %s %s not written: %s", prev["event_id"], key, exc)
                record["ledger_error"] = str(exc)
        out.append(record)
    return out


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
    A selection about a player the model no longer has a row for is withdrawn
    by a superseding PASS record.
    """
    settings = settings or Settings()
    out = _predict_league(store, league, market, settings=settings, now=now, horizon_hours=horizon_hours,
                          write=write)
    now_dt = utcnow() if now is None else parse_ts(now)
    horizon = settings.selection.horizon_hours if horizon_hours is None else horizon_hours
    kept = {(p["event_id"], p["market"]) for p in out}
    return out + _withdraw_stale(store, league, market, kept, now_dt=now_dt, horizon=horizon, write=write)


def _predict_league(store: Any, league: str, market: str, *, settings: Settings, now: Any,
                    horizon_hours: Optional[float], write: bool) -> List[Dict[str, Any]]:
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
    if frame.extra.get("requires_line"):
        # A market defined by a threshold cannot be priced without one: no threshold, no selection.
        rows = rows[rows["line"].notna()]
    if rows.empty:
        return []
    # A frame with several rows per game (one per player) names the game in ``game_id``
    # and the ledger key of each row's market in ``ledger_market``.
    rows = rows.assign(_event=rows["game_id"] if "game_id" in rows.columns else rows["event_id"],
                       _market=rows["ledger_market"] if "ledger_market" in rows.columns else market)
    # An event with only a date is stored at 00:00Z of that date, but the date began up
    # to 14 hours earlier somewhere east of Greenwich: it may already be under way.
    current = store.events(event_ids=rows["_event"].unique().tolist(), with_results=False,
                           include_synthetic=True).set_index("event_id")
    rows = rows[rows["_event"].map(current["status"]).eq("scheduled")]     # never a live or finished event
    if rows.empty:
        return []
    precision = current["time_precision"]
    date_only = rows["_event"].map(precision).ne("datetime").to_numpy()
    earliest = rows["start_time_utc"] - pd.to_timedelta(np.where(date_only, DATE_ONLY_LEAD_HOURS, 0.0), unit="h")
    rows = rows[earliest > now_ts]
    if rows.empty:
        return []
    coverage = coverage_status(store, league, market)
    report = store.latest_validation(league, market)
    # The coverage label and the confidence bands describe the candidate the newest
    # validation run selected.  They may only be attached to that candidate.
    model_mismatch: Optional[str] = None
    if report is not None and report.get("selected_model") and report["selected_model"] != model.model_name:
        model_mismatch = (
            f"the trained model in use ({model.model_name}) is not the candidate the newest validation run "
            f"selected ({report['selected_model']}); run `sportspred train --league {league} --market {market}`")
    min_prob = settings.min_prob(league, market, str(frame.outcome))
    check_log = store.checks_as_of(league, now_dt, include_synthetic=True)
    specs = plugin.required_checks(league, market)
    unresolved, results_through, latest_result, usual_break = _unresolved_past_events(
        store, league, now_ts, settings.selection.unresolved_window_days)
    break_margin = settings.selection.results_break_margin
    events = store.events(event_ids=rows["_event"].unique().tolist(), with_results=False, include_synthetic=True)
    events = events.set_index("event_id")
    feature_names = model.candidate.feature_names() or frame.feature_cols
    prior = store.predictions(event_ids=rows["_event"].unique().tolist(), latest_only=True, include_synthetic=True)
    prior_by_event = {(r["event_id"], r["market"]): r for _, r in prior.iterrows()}
    count_model = callable(getattr(model.candidate, "push_prob", None))
    projection_fn = getattr(model.candidate, "projection", None)
    # What a plugin needs to fold availability in honestly: which inputs the trained model
    # reads (exactly what it declares - none is none), and, set per row below, for which
    # teams a team-level report has been read for that game.
    frame.extra["model_features"] = list(model.candidate.feature_names())
    subject_col = "player_name" if "player_name" in rows.columns else None
    unit = frame.extra.get("stat_label")

    out: List[Dict[str, Any]] = []
    for _, base_row in rows.iterrows():
        event_id, ledger_market = base_row["_event"], base_row["_market"]
        ev = events.loc[event_id]
        team_ids = [base_row["side_a_id"], base_row["side_b_id"]]
        avail = store.availability_as_of(league, now_dt, team_ids=team_ids, event_id=event_id,
                                         include_synthetic=True)
        # Checks first: whether a team's report has been read *for this game* (on file for
        # this event or for no event, and inside its freshness window) decides what its
        # statements may be used for.  A check from another game or another week does not.
        check_row = base_row
        if base_row["event_id"] != event_id:            # checks are logged against the game, not the player row
            check_row = base_row.copy()
            check_row["event_id"] = event_id
        checks = evaluate_checks(specs, check_row, check_log, now=now_ts, settings=settings, names=names)
        report_time = {c["team_id"]: (c.get("published_at_utc") or c.get("checked_at_utc")) for c in checks
                       if c.get("team_id") and c["state"] in (CheckState.COMPLETE.value, CheckState.RECHECK.value)}
        frame.extra["report_checked_teams"] = set(report_time)
        frame.extra["report_info_time"] = report_time
        effect = plugin.apply_availability(league, market, base_row, frame, avail)
        row = effect.row
        line = None
        if frame.outcome == OutcomeType.OVER_UNDER and "line" in row.index and row["line"] == row["line"]:
            line = float(row["line"])
        # P: each class as it will settle; a push (exact tie with the line) is none of them.
        P, low, high, p_push = _outcome_probabilities(model, row, line, count_model)
        k = int(np.argmax(P))
        subject = str(row[subject_col]) if subject_col else None
        labels = _class_labels(frame, row, names, line, subject, unit if subject else None)
        cls = frame.classes[k]
        selection, selection_label = labels[cls]
        # Recorded: the probability that the selection wins, with the push as an outcome of
        # its own.  Judged by the rules: the probability given a decision - a push returns the
        # stake and is graded as neither hit nor miss, so it is what validated hit rates,
        # the selection minimum and the coin-flip test are about.
        prob, prob_low, prob_high = float(P[k]), float(low[k]), float(high[k])
        probs = {labels[c][0]: float(P[i]) for i, c in enumerate(frame.classes)}
        if p_push > 1e-9:
            probs["push"] = p_push
        else:
            p_push = 0.0
        decided = 1.0 - p_push
        prob_decided, low_decided, high_decided = prob / decided, prob_low / decided, prob_high / decided
        given = "" if p_push == 0.0 else (f" given a decision ({prob:.3f} with a {p_push:.3f} chance of a push)")

        # Built in for every league: ratings are only current if every earlier
        # event of both sides has a result on file.
        behind = unresolved.get(row["side_a_id"], 0) + unresolved.get(row["side_b_id"], 0)
        days_since = None if latest_result is None else (row["start_time_utc"] - latest_result).total_seconds() / 86400.0
        current, detail = results_current(behind, days_since, usual_break, break_margin, results_through)
        checks.append({
            "key": "results_current", "label": "results on file are up to date", "subject": "event",
            "state": (CheckState.COMPLETE if current else CheckState.MISSING).value, "detail": detail,
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
        if model_mismatch:
            pass_reasons.append(model_mismatch)
        pass_reasons += list(effect.blockers)
        interval_rule = (settings.selection.require_interval_above_even
                         and frame.outcome in (OutcomeType.BINARY, OutcomeType.OVER_UNDER))
        if prob_decided < min_prob:
            pass_reasons.append(f"model probability {prob_decided:.3f}{given} is below the selection minimum "
                                f"{min_prob:.2f}")
        elif interval_rule and low_decided <= 0.5:
            pass_reasons.append(f"model-uncertainty interval {low_decided:.3f}-{high_decided:.3f}{given} reaches "
                                "a coin flip")
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
            # ---- 3. CONDITIONAL: does the selection survive the doubtful players sitting out?
            # A side's own doubts threaten a pick on that side.  A pick with no side
            # (over, under, draw) can be moved by either side's, so both are tested.
            pick_side = ("a" if cls in ("home", "a", "home_cover", "a_cover") else
                         "b" if cls in ("away", "b", "away_cover", "b_cover") else None)
            sides = [pick_side] if pick_side else ["a", "b"]
            for side in sides:
                doubts = effect.uncertain_a if side == "a" else effect.uncertain_b
                if not doubts:
                    continue
                whose = names.get(row[f"side_{side}_id"], f"side {side.upper()}")
                worst = effect.worst_case_rows.get(side)
                if worst is None:
                    status = RecStatus.CONDITIONAL
                    reasons.append(f"holds only if these {whose} players play (effect not quantified): "
                                   + "; ".join(doubts))
                    continue
                # The same rules as the selection itself: the minimum and the interval test.
                Pw, low_w, _, push_w = _outcome_probabilities(model, worst, line, count_model)
                worst_decided = float(Pw[k]) / (1.0 - push_w)
                worst_low = float(low_w[k]) / (1.0 - push_w)
                if int(np.argmax(Pw)) != k or worst_decided < min_prob or (interval_rule and worst_low <= 0.5):
                    status = RecStatus.CONDITIONAL
                    reasons.append(
                        f"condition: {'; '.join(doubts)} ({whose}) must play. If they sit, the model probability of "
                        f"{selection_label} becomes {worst_decided:.3f}, which fails the selection rules.")
                else:
                    reasons.append(f"selection holds even if {'; '.join(doubts)} ({whose}) sit "
                                   f"(probability would be {worst_decided:.3f}).")
            # Conditions the plugin states in full: what the probability was computed as if,
            # against what is listed.  Recorded word for word.
            if effect.conditions:
                status = RecStatus.CONDITIONAL
                reasons.extend(effect.conditions)
            if status == RecStatus.FINAL:
                reasons.insert(0, "all configured pregame checks are complete and fresh")

        contributions = None
        contrib = model.contributions(pd.DataFrame([row]))
        if contrib is not None:
            contributions = {c: float(contrib.iloc[0][c]) for c in contrib.columns}
        row_for_explain = row.copy()
        row_for_explain["_pick_is_a"] = cls in ("home", "a", "home_cover", "a_cover")
        row_for_explain["_pick_class"] = cls
        projection = None
        if callable(projection_fn):
            projection = {k: float(v) for k, v in projection_fn(pd.DataFrame([row])).iloc[0].items()}
            row_for_explain["_projection"] = projection
        explanation = plugin.explain(league, market, row_for_explain, frame, contributions)
        explanation.availability_assumptions += effect.applied + effect.unmatched + effect.conditions
        for side_name, items in ((names.get(row["side_a_id"]), effect.uncertain_a),
                                 (names.get(row["side_b_id"]), effect.uncertain_b)):
            if items:
                explanation.availability_assumptions.append(
                    f"uncertain for {side_name}: " + "; ".join(items) + " (assumed to play as recently)")
        if avail.empty:
            explanation.missing_data.append("no availability report on file for either side")

        confidence = confidence_for(prob_decided, None if model_mismatch else report, settings.confidence)
        caveat = frame.extra.get("confidence_caveat")
        if caveat:
            # The observed hit rates were not measured on the kind of selection being made
            # (reference thresholds, not market thresholds): the label is withheld, the
            # measurement is still shown for what it is.
            if confidence.get("label") != "Unrated":
                confidence["measured_label"] = confidence["label"]
            confidence["label"] = "Unrated"
            confidence["basis"] = f"{caveat} ({confidence['basis']})"
        confidence["definitions"] = {
            "event_uncertainty": "chance the selection loses even if the probability is exactly right",
            "model_uncertainty": "how far the probability moves when the model is refitted on resampled history",
            "data_quality": "missing or stale inputs and incomplete pregame checks",
        }
        confidence["event_uncertainty"] = {"prob_selection_loses": max(0.0, 1.0 - prob - p_push),
                                           "normalised_entropy": _entropy(list(probs.values()))}
        if p_push > 0.0:
            confidence["event_uncertainty"]["prob_push"] = p_push
            confidence["prob_if_decided"] = prob_decided
        confidence["model_uncertainty"] = {
            "interval": [prob_low, prob_high], "level": model.interval_level,
            "method": f"{len(model.replicas)} bootstrap refits (resampled by slate)",
        }
        precision_date = ev["time_precision"] != "datetime"
        display = {
            "competition": spec.name, "league": league, "sport": spec.sport, "season": ev["season"],
            "stage": ev["stage"],
            # A row about one player has his team as side A; the event is shown as it is played.
            "side_a": names.get(ev["home_id"] if subject else row["side_a_id"], row["side_a_id"]),
            "side_b": names.get(ev["away_id"] if subject else row["side_b_id"], row["side_b_id"]),
            "side_a_role": "home" if (subject or ev["home_id"] == row["side_a_id"]) and ev["neutral_site"] != 1
            else "a",
            "start_hst": fmt_hst(ev["start_time_utc"], date_only=precision_date),
            "start_utc": iso_z(ev["start_time_utc"]),
            "venue": ev["venue_name"], "venue_city": ev["venue_city"],
            "neutral_site": None if ev["neutral_site"] != ev["neutral_site"] else bool(ev["neutral_site"]),
            "market": market, "market_note": decl.note, "line": line,
            **({"player": subject, "player_id": str(row["player_id"]), "stat": frame.extra.get("stat"),
                "player_team": names.get(row["side_a_id"], row["side_a_id"]), "projection": projection}
               if subject else {}),
            "status_definition": REC_STATUS_DEFINITIONS[status],
            # When this prediction was made - not when any source was last read; each
            # check carries its own time.
            "predicted_at_hst": fmt_hst(now_dt),
            "last_source_check_hst": _last_source_check(checks),
            "date_only": bool(precision_date),
        }
        source_ids = [s for s in {ev["source_id"], *[c.get("source_id") for c in checks],
                                  *avail.get("source_id", pd.Series(dtype=object)).tolist()} if s]
        src = store.sources(source_ids)
        sources = [{"source_id": r["source_id"], "provider": r["provider"], "url": r["url"],
                    "collected_at_utc": r["collected_at_utc"], "published_at_utc": r["published_at_utc"]}
                   for _, r in src.iterrows()]
        snapshot = {c: (None if row[c] != row[c] else float(row[c])) for c in feature_names if c in row.index}
        record: Dict[str, Any] = {
            "created_at_utc": iso_z(now_dt), "event_id": event_id, "market": ledger_market, "line": line,
            "selection": selection,
            "selection_label": selection_label, "prob": prob, "prob_low": prob_low, "prob_high": prob_high,
            "probs": probs, "model_name": model.model_name, "model_version": model.model_version,
            "validation_status": coverage.value, "rec_status": status.value, "reasons": reasons,
            "checks": checks, "confidence": confidence, "factors": explanation.as_dict(),
            "features": snapshot, "data_quality": quality, "sources": sources, "display": display,
        }
        previous = prior_by_event.get((event_id, ledger_market))
        if previous is not None:
            same = (
                previous["model_version"] == record["model_version"]
                and previous["rec_status"] == record["rec_status"]
                and previous["selection"] == record["selection"]
                and previous["validation_status"] == record["validation_status"]
                and previous["event_start_utc"] == iso_z(ev["start_time_utc"])
                and _same_line(previous["line"], line)
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
                log.warning("prediction for %s %s not written: %s", event_id, ledger_market, exc)
                record["ledger_error"] = str(exc)
        out.append(record)
    return out
