"""Post-event reviews, lessons and drift monitoring.

A review classifies each graded prediction using what was known when it was
made and what demonstrably changed afterwards.  The categories are kept apart
on purpose:

``availability_change``  news published between the prediction and the start
                         changed who was available;
``data_issue``           the prediction was made with missing or stale inputs,
                         or the event's data was later corrected;
``model_error``          the probability band this prediction sits in has been
                         systematically wrong on recent graded predictions;
``variance``             none of the above: the selection lost at a rate its
                         own probability allowed for.

A loss on its own is evidence of nothing: a 70% selection is supposed to lose
three times in ten.  ``model_error`` therefore needs a statistically
significant shortfall across many predictions, never a single result.

Lessons turn a hypothesis into a testable change.  They move
``proposed -> testing -> supported | rejected -> promoted`` and can only be
promoted with a recorded validation run whose development-fold evidence
supports them.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..core.ids import stable_id
from ..core.timeutil import iso_z, utcnow

__all__ = ["graded_frame", "review_predictions", "review_summary", "add_lesson", "set_lesson_status",
           "lessons", "propose_lessons", "calibration_monitor", "feature_drift"]

OPPOSITION_KEYS = ("elo_diff", "net_slow_diff", "net_fast_diff", "form_diff", "roster_q_diff", "absence_diff")


def graded_frame(store: Any, *, include_synthetic: bool = False) -> pd.DataFrame:
    """The current prediction of each (event, market) joined with its latest
    grade.  Only ``correct`` / ``incorrect`` rows carry a ``hit`` value."""
    preds = store.predictions(latest_only=True, include_synthetic=include_synthetic)
    grades = store.grades(latest_only=True)
    if preds.empty or grades.empty:
        return pd.DataFrame()
    df = preds.merge(grades[["prediction_id", "outcome", "graded_at_utc", "result_json"]], on="prediction_id")
    df["hit"] = df["outcome"].map({"correct": 1.0, "incorrect": 0.0})
    return df


def _latest_reviews(store: Any) -> pd.DataFrame:
    """The newest review of each prediction, with the grade it was written for."""
    reviews = store.reviews()
    if reviews.empty:
        return reviews.assign(reviewed_outcome=pd.Series(dtype=object))
    latest = reviews.loc[~reviews["prediction_id"].duplicated(keep="last")].copy()
    latest["reviewed_outcome"] = [json.loads(e).get("outcome") for e in latest["evidence_json"]]
    return latest.reset_index(drop=True)


def _band_shortfall(df: pd.DataFrame, row: pd.Series, *, width: float = 0.05, min_n: int = 100,
                    alpha: float = 0.01) -> Optional[Dict[str, Any]]:
    """Evidence that this prediction's probability band is systematically
    over-confident for its league, market and model (one-sided binomial test)."""
    lo = np.floor(row["prob"] / width) * width
    band = df[(df["league"] == row["league"]) & (df["market"] == row["market"])
              & (df["model_name"] == row["model_name"]) & df["hit"].notna()
              & (df["prob"] >= lo) & (df["prob"] < lo + width)]
    n = len(band)
    if n < min_n:
        return None
    hits, expected = int(band["hit"].sum()), float(band["prob"].mean())
    p_value = float(stats.binomtest(hits, n, expected, alternative="less").pvalue)
    if p_value >= alpha:
        return None
    return {"band": [float(lo), float(lo + width)], "n": n, "observed": hits / n, "expected": expected,
            "p_value": p_value}


def review_predictions(store: Any, *, now: Any = None, include_synthetic: bool = False) -> pd.DataFrame:
    """Write a review for every graded prediction that has none, and a new
    one when a prediction's grade has changed since it was reviewed (a result
    was corrected); returns the reviews written.  Earlier reviews are kept."""
    df = graded_frame(store, include_synthetic=include_synthetic)
    if df.empty:
        return pd.DataFrame()
    reviewed = _latest_reviews(store)
    seen = dict(zip(reviewed["prediction_id"], reviewed["reviewed_outcome"]))
    stale = df["prediction_id"].map(seen) != df["outcome"]          # never reviewed, or reviewed under another grade
    todo = df[stale & df["outcome"].isin(["correct", "incorrect", "void", "push"])]
    out: List[Dict[str, Any]] = []
    issues = store.issues()
    for _, row in todo.iterrows():
        quality = json.loads(row["data_quality_json"])
        features = json.loads(row["features_json"])
        evidence: Dict[str, Any] = {
            "probability": row["prob"], "outcome": row["outcome"],
            "expected_loss_rate": None if pd.isna(row["prob"]) else 1.0 - float(row["prob"]),
            "result": json.loads(row["result_json"]),
            "opposition_at_prediction_time": {k: features[k] for k in OPPOSITION_KEYS if k in features},
            "status_at_prediction_time": row["rec_status"],
        }
        # News collected after the prediction and before the start.
        avail = store.query(
            "SELECT player_name, status, team_id, collected_at_utc FROM availability "
            "WHERE league=? AND collected_at_utc > ? AND collected_at_utc < ? AND (event_id IS NULL OR event_id=?)",
            [row["league"], row["created_at_utc"], row["event_start_utc"], row["event_id"]])
        ev = store.events(event_ids=[row["event_id"]], with_results=False, include_synthetic=True)
        teams = set(ev[["home_id", "away_id"]].iloc[0].dropna()) if len(ev) else set()
        late = avail[avail["team_id"].isin(teams) & avail["status"].isin(["out", "doubtful"])]
        event_issues = issues[issues["event_id"] == row["event_id"]]
        shortfall = _band_shortfall(df, row) if row["outcome"] == "incorrect" else None

        if row["outcome"] in ("void", "push"):
            category = "unreviewable"
            note = f"graded {row['outcome']}: no right-or-wrong result to review"
        elif len(late):
            category = "availability_change"
            evidence["late_availability"] = late.to_dict("records")
            note = ("availability changed after the prediction was made: "
                    + "; ".join(f"{r['player_name']} {r['status']}" for _, r in late.iterrows()))
        elif quality.get("missing_features") or len(event_issues):
            category = "data_issue"
            evidence["missing_features"] = quality.get("missing_features", [])
            evidence["event_issues"] = event_issues["kind"].tolist()
            note = "the prediction was made with missing inputs or the event's data was later corrected"
        elif shortfall is not None:
            category = "model_error"
            evidence["band_shortfall"] = shortfall
            note = (f"selections in the {shortfall['band'][0]:.2f}-{shortfall['band'][1]:.2f} band won "
                    f"{100 * shortfall['observed']:.1f}% of {shortfall['n']} graded predictions against "
                    f"{100 * shortfall['expected']:.1f}% expected (p={shortfall['p_value']:.4f})")
        else:
            category = "variance"
            if row["outcome"] == "correct":
                note = "selection won; nothing indicates the probability was wrong"
            else:
                note = (f"selection lost at a stated probability of {row['prob']:.3f}; such selections are "
                        f"expected to lose {100 * (1 - row['prob']):.0f}% of the time and no evidence points "
                        "to an error in the analysis")
        if quality.get("incomplete_checks") and category in ("variance", "model_error"):
            evidence["incomplete_checks_at_prediction_time"] = quality["incomplete_checks"]
        rid = store.add_review(row["prediction_id"], category, evidence, note, created_at=now)
        out.append({"review_id": rid, "prediction_id": row["prediction_id"], "league": row["league"],
                    "selection_label": row["selection_label"], "prob": row["prob"], "outcome": row["outcome"],
                    "category": category, "note": note})
    return pd.DataFrame(out)


def review_summary(store: Any, *, include_synthetic: bool = False) -> Dict[str, Any]:
    df = graded_frame(store, include_synthetic=include_synthetic)
    reviews = _latest_reviews(store)          # one per prediction: the review of its current grade
    if df.empty:
        return {"n_graded": 0}
    scored = df[df["hit"].notna()]
    out: Dict[str, Any] = {
        "n_graded": int(len(df)), "n_scored": int(len(scored)),
        "outcomes": df["outcome"].value_counts().to_dict(),
        "categories": reviews[reviews["prediction_id"].isin(df["prediction_id"])]["category"].value_counts().to_dict(),
    }
    if len(scored):
        out["accuracy"] = float(scored["hit"].mean())
        out["mean_probability"] = float(scored["prob"].mean())
        p = np.clip(scored["prob"].to_numpy(dtype=float), 1e-9, 1 - 1e-9)
        out["log_loss_of_selections"] = float(-np.mean(np.where(scored["hit"] == 1.0, np.log(p), np.log(1 - p))))
        out["by_status"] = scored.groupby("rec_status")["hit"].agg(["size", "mean"]).round(4).to_dict("index")
    return out


# ------------------------------------------------------------------- lessons
_TRANSITIONS = {
    "proposed": {"testing", "rejected"},
    "testing": {"supported", "rejected"},
    "supported": {"promoted", "rejected"},
    "rejected": set(),
    "promoted": set(),
}


def add_lesson(store: Any, *, title: str, hypothesis: str, proposed_change: str, test_plan: str,
               league: Optional[str] = None, market: Optional[str] = None,
               evidence: Optional[Mapping[str, Any]] = None) -> str:
    """Record a lesson as a testable proposal (status ``proposed``)."""
    now = iso_z(utcnow())
    lesson_id = stable_id("lsn", title, hypothesis, now)
    with store.conn() as con:
        con.execute(
            """INSERT INTO lessons (lesson_id, created_at_utc, league, market, title, hypothesis, proposed_change,
                                    test_plan, status, evidence_json, validation_run_id, updated_at_utc)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (lesson_id, now, league, market, title, hypothesis, proposed_change, test_plan, "proposed",
             json.dumps(dict(evidence or {}), default=str), None, now))
    return lesson_id


def set_lesson_status(store: Any, lesson_id: str, status: str, *, validation_run_id: Optional[str] = None,
                      evidence: Optional[Mapping[str, Any]] = None) -> None:
    """Move a lesson along its life-cycle.

    ``supported`` and ``promoted`` require a recorded validation run.  A
    lesson is ``promoted`` only when that run exists in the store - a change
    is never promoted on the strength of individual results.
    """
    with store.conn() as con:
        row = con.execute("SELECT * FROM lessons WHERE lesson_id=?", (lesson_id,)).fetchone()
        if row is None:
            raise ValueError(f"unknown lesson {lesson_id}")
        if status not in _TRANSITIONS[row["status"]]:
            raise ValueError(f"a {row['status']} lesson cannot become {status}; "
                             f"allowed: {sorted(_TRANSITIONS[row['status']]) or 'nothing'}")
        run_id = validation_run_id or row["validation_run_id"]
        if status in ("supported", "promoted"):
            if not run_id:
                raise ValueError(f"a lesson can only be {status} with a validation run that tested it")
            if con.execute("SELECT 1 FROM validation_runs WHERE run_id=?", (run_id,)).fetchone() is None:
                raise ValueError(f"validation run {run_id} is not on file")
        merged = {**json.loads(row["evidence_json"] or "{}"), **dict(evidence or {})}
        con.execute(
            "UPDATE lessons SET status=?, validation_run_id=?, evidence_json=?, updated_at_utc=? WHERE lesson_id=?",
            (status, run_id, json.dumps(merged, default=str), iso_z(utcnow()), lesson_id))


def lessons(store: Any) -> pd.DataFrame:
    return store.query("SELECT * FROM lessons ORDER BY created_at_utc")


def propose_lessons(store: Any, *, min_reviews: int = 20) -> List[str]:
    """Draft lessons from patterns in the reviews.  Each is only a proposal
    with a test plan; nothing changes until it has been validated."""
    reviews = _latest_reviews(store)
    if reviews.empty:
        return []
    preds = store.predictions(include_synthetic=True)[["prediction_id", "league", "market"]]
    df = reviews.merge(preds, on="prediction_id")
    existing = set(lessons(store)["title"])
    created: List[str] = []
    for (league, market), part in df.groupby(["league", "market"]):
        scored = part[part["category"] != "unreviewable"]
        if len(scored) < min_reviews:
            continue
        share = scored["category"].value_counts(normalize=True)
        drafts = []
        if share.get("availability_change", 0) >= 0.15:
            drafts.append((
                f"{league} {market}: late availability changes",
                "A material share of reviewed selections were made before availability news that changed them.",
                "Tighten the injury-report re-check window for this league, or require the check closer to the start.",
                "Re-run the predictions of the reviewed slates with the tighter window and compare the hit rate of "
                "FINAL selections; adopt only if the comparison over development slates supports it."))
        if share.get("model_error", 0) >= 0.10:
            drafts.append((
                f"{league} {market}: over-confident probability band",
                "Graded selections in at least one probability band won significantly less often than predicted.",
                "Refit the calibrator on more recent blocks or add the candidate feature the misses share.",
                "Add the change as a new candidate, run walk-forward validation, and promote only if pooled "
                "development log loss improves beyond one standard error; the final test stays untouched."))
        if share.get("data_issue", 0) >= 0.15:
            drafts.append((
                f"{league} {market}: predictions made with missing inputs",
                "A material share of reviewed selections had missing features or corrected event data.",
                "PASS when the affected inputs are missing, or fix the ingestion gap that causes them.",
                "Measure accuracy with and without the affected selections on development slates."))
        for title, hypothesis, change, plan in drafts:
            if title not in existing:
                created.append(add_lesson(store, title=title, hypothesis=hypothesis, proposed_change=change,
                                          test_plan=plan, league=league, market=market,
                                          evidence={"category_share": share.round(3).to_dict(),
                                                    "n_reviews": int(len(scored))}))
    return created


# ---------------------------------------------------------------- monitoring
def calibration_monitor(store: Any, league: str, market: str = "winner", *, window: int = 200,
                        include_synthetic: bool = False) -> Dict[str, Any]:
    """Compare live graded selections with what validation says to expect."""
    df = graded_frame(store, include_synthetic=include_synthetic)
    report = store.latest_validation(league, market)
    if df.empty:
        return {"league": league, "market": market, "n": 0, "status": "no graded predictions yet"}
    live = df[(df["league"] == league) & (df["market"] == market) & df["hit"].notna()].tail(window)
    n = len(live)
    if n == 0:
        return {"league": league, "market": market, "n": 0, "status": "no graded predictions yet"}
    hits, expected = float(live["hit"].sum()), float(live["prob"].sum())
    var = float((live["prob"] * (1 - live["prob"])).sum())
    z = (hits - expected) / np.sqrt(var) if var > 0 else float("nan")
    p = np.clip(live["prob"].to_numpy(dtype=float), 1e-9, 1 - 1e-9)
    out: Dict[str, Any] = {
        "league": league, "market": market, "n": int(n), "window": window,
        "observed_accuracy": hits / n, "mean_predicted": expected / n, "z_score": float(z),
        "log_loss_of_selections": float(-np.mean(np.where(live["hit"] == 1.0, np.log(p), np.log(1 - p)))),
        "first": live["event_start_utc"].min(), "last": live["event_start_utc"].max(),
    }
    if n < 30:
        out["status"] = f"too few graded predictions ({n}) to judge drift"
    elif abs(z) > 3:
        out["status"] = "ALERT: selections are winning at a rate inconsistent with their probabilities"
    elif abs(z) > 2:
        out["status"] = "watch: hit rate is drifting from the stated probabilities"
    else:
        out["status"] = "within expectation"
    if report:
        out["validation_final_test"] = {k: report["final_test"]["selected"].get(k) for k in
                                        ("n", "accuracy", "log_loss", "brier", "ece")}
    return out


def feature_drift(reference: pd.DataFrame, recent: pd.DataFrame, columns: Sequence[str], *,
                  n_bins: int = 10) -> pd.DataFrame:
    """Population stability index of each feature: recent rows versus the
    reference (training) rows.  Above 0.25 is a material shift."""
    rows = []
    for col in columns:
        if col not in reference or col not in recent:
            continue
        ref = pd.to_numeric(reference[col], errors="coerce").dropna().to_numpy()
        new = pd.to_numeric(recent[col], errors="coerce").dropna().to_numpy()
        if len(ref) < 50 or len(new) < 10:
            continue
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 3:
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        a = np.clip(np.histogram(ref, edges)[0] / len(ref), 1e-4, None)
        b = np.clip(np.histogram(new, edges)[0] / len(new), 1e-4, None)
        psi = float(np.sum((b - a) * np.log(b / a)))
        rows.append({"feature": col, "psi": psi, "n_reference": len(ref), "n_recent": len(new),
                     "reference_mean": float(ref.mean()), "recent_mean": float(new.mean()),
                     "flag": "shift" if psi > 0.25 else ("watch" if psi > 0.10 else "stable")})
    return pd.DataFrame(rows).sort_values("psi", ascending=False) if rows else pd.DataFrame()
