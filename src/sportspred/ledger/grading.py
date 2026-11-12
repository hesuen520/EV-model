"""Grading predictions against verified sporting results.

Grading answers one question - was the selection right? - from the result in
the store.  It involves no prices; settlement of paper positions under a
venue's rules is a separate step in :mod:`sportspred.ev`.

Outcomes: ``correct``, ``incorrect``, ``push`` (an exact tie against a line,
or a drawn game in a market with no draw outcome), ``void`` (event postponed
or cancelled, or rescheduled after the prediction was made) and ``no_result``
(the market cannot be graded from the data on file).  When a result is
corrected a new grade is appended and linked to the one it replaces.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import pandas as pd

from ..core.timeutil import iso_z, utcnow

__all__ = ["grade_predictions", "grade_one", "register_grader", "PROP_STATS"]

Grader = Callable[[Any, Mapping[str, Any], Mapping[str, Any]], Tuple[str, Dict[str, Any]]]
_GRADERS: Dict[str, Grader] = {}

#: Statistic names accepted in ``player_prop:<stat>`` markets -> box-score columns.
PROP_STATS: Dict[str, Tuple[str, ...]] = {
    "points": ("pts",), "pts": ("pts",), "rebounds": ("reb",), "reb": ("reb",), "assists": ("ast",),
    "ast": ("ast",), "threes": ("fg3m",), "fg3m": ("fg3m",), "steals": ("stl",), "stl": ("stl",),
    "blocks": ("blk",), "blk": ("blk",), "turnovers": ("tov",), "tov": ("tov",),
    "pra": ("pts", "reb", "ast"), "points_rebounds_assists": ("pts", "reb", "ast"),
    "pr": ("pts", "reb"), "pa": ("pts", "ast"), "ra": ("reb", "ast"),
}


def register_grader(market_family: str, grader: Grader) -> None:
    """Register a grader for markets named ``<family>`` or ``<family>:<detail>``."""
    _GRADERS[market_family] = grader


def _scores(ev: Mapping[str, Any], regulation: bool = False) -> Optional[Tuple[float, float]]:
    if regulation and ev.get("reg_home_score") == ev.get("reg_home_score") and ev.get("reg_home_score") is not None:
        return float(ev["reg_home_score"]), float(ev["reg_away_score"])
    if ev.get("home_score") is None or ev["home_score"] != ev["home_score"]:
        return None
    return float(ev["home_score"]), float(ev["away_score"])


def _winner(store: Any, pred: Mapping[str, Any], ev: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    detail = {"winner_id": ev.get("winner_id"), "home_score": ev.get("home_score"), "away_score": ev.get("away_score")}
    if int(ev.get("is_draw") or 0):
        return "push", {**detail, "note": "drawn game in a market without a draw outcome"}
    if not ev.get("winner_id"):
        return "no_result", detail
    return ("correct" if pred["selection"] == ev["winner_id"] else "incorrect"), detail


def _winner_3way(store: Any, pred: Mapping[str, Any], ev: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    went_extra = bool(ev.get("went_overtime") or ev.get("went_shootout"))
    scores = _scores(ev, regulation=True)
    if scores is None:
        return "no_result", {"note": "no score on file"}
    has_reg = ev.get("reg_home_score") is not None and ev.get("reg_home_score") == ev.get("reg_home_score")
    if went_extra and not has_reg:
        actual = "draw"      # a game that needed overtime was level after regulation
    else:
        actual = "draw" if scores[0] == scores[1] else (ev["home_id"] if scores[0] > scores[1] else ev["away_id"])
    detail = {"regulation_result": actual, "scores_used": list(scores), "went_overtime": went_extra}
    return ("correct" if pred["selection"] == actual else "incorrect"), detail


def _spread(store: Any, pred: Mapping[str, Any], ev: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    scores = _scores(ev)
    if scores is None or pred.get("line") is None or pred["line"] != pred["line"]:
        return "no_result", {"note": "score or line missing"}
    line = float(pred["line"])                      # the line is the home side's handicap
    margin = scores[0] - scores[1] + line
    detail = {"home_margin": scores[0] - scores[1], "line": line}
    if margin == 0:
        return "push", detail
    home_covers = margin > 0
    picked_home = pred["selection"] in (ev["home_id"], "home_cover", "home")
    return ("correct" if picked_home == home_covers else "incorrect"), detail


def _total(store: Any, pred: Mapping[str, Any], ev: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    scores = _scores(ev)
    if scores is None or pred.get("line") is None or pred["line"] != pred["line"]:
        return "no_result", {"note": "score or line missing"}
    total, line = scores[0] + scores[1], float(pred["line"])
    detail = {"total": total, "line": line}
    if total == line:
        return "push", detail
    return ("correct" if (pred["selection"] == "over") == (total > line) else "incorrect"), detail


def _player_prop(store: Any, pred: Mapping[str, Any], ev: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    stat = pred["market"].split(":", 1)[1] if ":" in pred["market"] else ""
    cols = PROP_STATS.get(stat)
    meta = {**json.loads(pred.get("features_json") or "{}"), **json.loads(pred.get("display_json") or "{}")}
    player_id = meta.get("player_id")
    if cols is None or player_id is None or not store.has_table(ev["league"], "player_box"):
        return "no_result", {"note": f"cannot grade statistic {stat!r} from the data on file"}
    box = store.read_table(ev["league"], "player_box")
    row = box[(box["event_id"] == ev["event_id"]) & (box["player_id"].astype(str) == str(player_id))]
    if row.empty or float(row.iloc[0]["minutes"] or 0) <= 0:
        return "void", {"note": "player did not play"}
    value = float(sum(float(row.iloc[0][c] or 0) for c in cols))
    line = float(pred["line"])
    detail = {"stat": stat, "value": value, "line": line}
    if value == line:
        return "push", detail
    return ("correct" if (pred["selection"] == "over") == (value > line) else "incorrect"), detail


for _family, _grader in (("winner", _winner), ("winner_3way", _winner_3way), ("spread", _spread),
                         ("total", _total), ("player_prop", _player_prop)):
    register_grader(_family, _grader)


def grade_one(store: Any, pred: Mapping[str, Any], ev: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Grade one ledger row against one event row (joined with its result)."""
    if ev["status"] in ("postponed", "canceled"):
        return "void", {"note": f"event {ev['status']}"}
    if iso_z(ev["start_time_utc"]) != pred["event_start_utc"] and ev["time_precision"] == "datetime":
        # The event moved after the prediction was made: the pregame context
        # the prediction described no longer applies.
        return "void", {"note": "event was rescheduled after the prediction",
                        "predicted_start": pred["event_start_utc"], "actual_start": iso_z(ev["start_time_utc"])}
    if ev["status"] != "final":
        return "no_result", {"note": f"event is {ev['status']}"}
    grader = _GRADERS.get(pred["market"].split(":", 1)[0])
    if grader is None:
        return "no_result", {"note": f"no grader for market {pred['market']!r}"}
    return grader(store, pred, ev)


def grade_predictions(store: Any, *, now: Any = None, include_synthetic: bool = False) -> pd.DataFrame:
    """Grade every ungraded prediction whose event is decided, and re-grade
    predictions whose result changed since they were graded.

    Returns the grades written by this call.
    """
    now_iso = iso_z(now or utcnow())
    preds = store.predictions(include_synthetic=include_synthetic)
    preds = preds[preds["rec_status"] != "PASS"]
    if preds.empty:
        return pd.DataFrame()
    events = store.events(event_ids=preds["event_id"].unique().tolist(), with_results=True,
                          include_synthetic=True).set_index("event_id")
    existing = store.grades(latest_only=True)
    last = {r["prediction_id"]: r for _, r in existing.iterrows()}
    written: List[Dict[str, Any]] = []
    for _, pred in preds.iterrows():
        ev = events.loc[pred["event_id"]].to_dict()
        ev["event_id"] = pred["event_id"]
        outcome, detail = grade_one(store, pred, ev)
        if outcome == "no_result":
            continue
        prev = last.get(pred["prediction_id"])
        if prev is not None and prev["outcome"] == outcome and json.loads(prev["result_json"]) == json.loads(
                json.dumps(detail, default=str)):
            continue
        gid = store.add_grade(pred["prediction_id"], outcome, detail, graded_at=now_iso,
                              result_source_id=ev.get("result_source_id"),
                              supersedes_grade_id=None if prev is None else int(prev["grade_id"]))
        written.append({"grade_id": gid, "prediction_id": pred["prediction_id"], "league": pred["league"],
                        "market": pred["market"], "selection_label": pred["selection_label"],
                        "prob": pred["prob"], "outcome": outcome, "regrade": prev is not None})
    return pd.DataFrame(written)
