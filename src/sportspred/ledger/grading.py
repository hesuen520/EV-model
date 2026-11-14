"""Grading predictions against verified sporting results.

Grading answers one question - was the selection right? - from the result in
the store.  It involves no prices; settlement of paper positions under a
venue's rules is a separate step in :mod:`sportspred.ev`.

Outcomes: ``correct``, ``incorrect``, ``push`` (an exact tie against a line,
or a drawn game in a market with no draw outcome), ``void`` (event postponed
or cancelled, or rescheduled after the prediction was made) and ``no_result``
(the market cannot be graded from the data on file).  When a result is
corrected a new grade is appended and linked to the one it replaces.

Rows are read through :mod:`sportspred.core.results`: a frame holding several
leagues has ``NaN`` where a league has no value, and ``NaN`` must never be
read as "true".  One prediction that cannot be graded never stops the others.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import pandas as pd

from ..core.results import clean_row, flag, regulation_result
from ..core.slates import predicted_date_only
from ..core.timeutil import iso_z, utcnow

__all__ = ["grade_predictions", "grade_one", "register_grader", "rescheduled", "PROP_STATS"]

#: A timed event that starts within this many hours of the time the prediction
#: recorded is the same game (a delay or a small schedule change).
SAME_GAME_HOURS = 12.0
#: A date-only event is stored at 00:00Z of its calendar date.  A start on
#: that date, in any time zone on Earth, lies in this window around it.
_DATE_WINDOW_HOURS = (-14.0, 36.0)

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


def _scores(ev: Mapping[str, Any]) -> Optional[Tuple[float, float]]:
    if ev.get("home_score") is None or ev.get("away_score") is None:
        return None
    return float(ev["home_score"]), float(ev["away_score"])


def _winner(store: Any, pred: Mapping[str, Any], ev: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    detail = {"winner_id": ev.get("winner_id"), "home_score": ev.get("home_score"), "away_score": ev.get("away_score")}
    if flag(ev.get("is_draw")):
        return "push", {**detail, "note": "drawn game in a market without a draw outcome"}
    if not ev.get("winner_id"):
        return "no_result", {**detail, "note": "no winner on file"}
    return ("correct" if pred["selection"] == ev["winner_id"] else "incorrect"), detail


def _winner_3way(store: Any, pred: Mapping[str, Any], ev: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    actual, basis = regulation_result(ev)
    if actual is None:
        return "no_result", {"note": basis}
    detail = {"regulation_result": actual, "basis": basis, "home_score": ev.get("home_score"),
              "away_score": ev.get("away_score"), "reg_home_score": ev.get("reg_home_score"),
              "reg_away_score": ev.get("reg_away_score"), "went_overtime": flag(ev.get("went_overtime")),
              "went_shootout": flag(ev.get("went_shootout"))}
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


_BOX_CACHE: Dict[Tuple[str, str], Tuple[Any, Any]] = {}


def _box_score(store: Any, league: str, event_id: str) -> pd.DataFrame:
    """Box-score rows of one game.  The league's table is read once per file
    version and grouped by game, not once per graded prediction."""
    path = store._table_path(league, "player_box")
    info = path.stat()
    # A rewrite replaces the file, so the inode changes even if the time and size do not.
    stamp = (info.st_mtime_ns, info.st_size, info.st_ino)
    key = (str(store.root), league)
    cached = _BOX_CACHE.get(key)
    if cached is None or cached[0] != stamp:
        box = store.read_table(league, "player_box")
        cached = (stamp, {k: g for k, g in box.groupby("event_id", sort=False)})
        while len(_BOX_CACHE) >= 4 and key not in _BOX_CACHE:        # a few leagues, never unbounded
            _BOX_CACHE.pop(next(iter(_BOX_CACHE)))
        _BOX_CACHE[key] = cached
    game = cached[1].get(event_id)
    return game if game is not None else pd.DataFrame(columns=["team_id", "player_id", "minutes"])


def _player_prop(store: Any, pred: Mapping[str, Any], ev: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    # ``player_prop:<stat>:<player id>``; older records carry the player in their display data.
    parts = pred["market"].split(":")
    stat = parts[1] if len(parts) > 1 else ""
    cols = PROP_STATS.get(stat)
    meta = {**json.loads(pred.get("features_json") or "{}"), **json.loads(pred.get("display_json") or "{}")}
    player_id = parts[2] if len(parts) > 2 else meta.get("player_id")
    if cols is None or player_id is None or not store.has_table(ev["league"], "player_box"):
        return "no_result", {"note": f"cannot grade statistic {stat!r} from the data on file"}
    game = _box_score(store, ev["league"], ev["event_id"])
    # "He did not play" can only be read from a box score that is there.  A result can be
    # on file before its box score; until both teams' players are listed nothing is graded.
    appeared = game[pd.to_numeric(game["minutes"], errors="coerce") > 0]
    if appeared["team_id"].nunique() < 2:
        return "no_result", {"note": "the box score of this game is not on file yet"}
    row = game[game["player_id"].astype(str) == str(player_id)]
    # If he is listed more than once, the row with minutes is the game he played.
    with_minutes = row[pd.to_numeric(row["minutes"], errors="coerce") > 0]
    if with_minutes.empty:
        has_stats = len(row) and row[list(cols)].notna().any().any()
        if has_stats:
            return "no_result", {"note": "statistics on file without minutes; cannot tell whether he played"}
        return "void", {"note": "player did not play"}
    row = with_minutes.iloc[[-1]]
    stats = [row.iloc[0][c] for c in cols]
    if any(v is None or v != v for v in stats) or pred.get("line") is None or pred["line"] != pred["line"]:
        return "no_result", {"note": f"statistic {stat!r} or the line is missing; not graded"}
    value = float(sum(float(v) for v in stats))
    line = float(pred["line"])
    detail = {"stat": stat, "value": value, "line": line}
    if value == line:
        return "push", detail
    return ("correct" if (pred["selection"] == "over") == (value > line) else "incorrect"), detail


for _family, _grader in (("winner", _winner), ("winner_3way", _winner_3way), ("spread", _spread),
                         ("total", _total), ("player_prop", _player_prop)):
    register_grader(_family, _grader)


def rescheduled(pred: Mapping[str, Any], ev: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Details of a real reschedule since the prediction, or ``None``.

    A prediction describes one game in its pregame context.  It still applies
    when a date-only event gets its start time published, or when a timed
    event moves by a few hours; it does not when the game moves to another day.
    """
    predicted, actual = pd.Timestamp(pred["event_start_utc"]), pd.Timestamp(ev["start_time_utc"])
    if predicted.tzinfo is None:
        predicted = predicted.tz_localize("UTC")
    if actual.tzinfo is None:
        actual = actual.tz_localize("UTC")
    shift = (actual - predicted).total_seconds() / 3600.0
    if shift == 0.0:
        return None
    if predicted_date_only(pred):
        if ev.get("time_precision") == "datetime":
            same = _DATE_WINDOW_HOURS[0] <= shift < _DATE_WINDOW_HOURS[1]
        else:
            same = False                  # still date-only, and the date itself changed
    else:
        same = abs(shift) < SAME_GAME_HOURS
    if same:
        return None
    return {"note": "event was rescheduled after the prediction", "predicted_start": pred["event_start_utc"],
            "actual_start": iso_z(actual), "moved_hours": round(shift, 2)}


def grade_one(store: Any, pred: Mapping[str, Any], ev: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Grade one ledger row against one event row (joined with its result)."""
    ev = clean_row(ev)
    if ev["status"] in ("postponed", "canceled"):
        return "void", {"note": f"event {ev['status']}"}
    moved = rescheduled(pred, ev)
    if moved is not None:
        # The event moved after the prediction was made: the pregame context
        # the prediction described no longer applies.
        return "void", moved
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
        if pred["event_id"] not in events.index:
            continue
        ev = events.loc[pred["event_id"]].to_dict()
        ev["event_id"] = pred["event_id"]
        try:
            outcome, detail = grade_one(store, pred, ev)
        except Exception as exc:  # noqa: BLE001 - one bad row must not stop the run
            store.log_issue("grading_failed", f"{pred['prediction_id']}: {type(exc).__name__}: {exc}",
                            league=pred["league"], event_id=pred["event_id"])
            continue
        prev = last.get(pred["prediction_id"])
        if outcome == "no_result" and (prev is None or prev["outcome"] == "no_result"):
            continue
        # A grade that the data on file no longer supports (the result was withdrawn or
        # lost a field the grade relied on) is withdrawn too, by a linked "no_result" grade.
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
