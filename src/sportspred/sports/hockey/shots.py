"""Per-game shot statistics, including five-on-five expected goals.

Expected goals here are this module's own shot-location model: a logistic
regression of "was this unblocked five-on-five attempt a goal" on distance,
angle and shot type, fitted **only on the burn-in seasons** and then applied
unchanged to every later shot.  The model therefore never sees a goal from the
seasons it is used on.  (The source files also carry an ``xg`` column from an
upstream model whose training period is not documented; it is not used.)

Blocked shots are left out: the source locates them where they were blocked,
not where they were taken.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

__all__ = ["SHOT_TYPES", "fit_shot_model", "expected_goals", "game_shot_stats"]

#: Shot types recorded in every season; anything else is the reference level.
SHOT_TYPES = ("wrist", "slap", "snap", "backhand", "tip-in", "deflected", "wrap-around")


def _design(shots: pd.DataFrame) -> np.ndarray:
    dist = np.clip(shots["distance"].to_numpy(dtype=float), 0.0, 100.0) / 10.0
    angle = np.clip(shots["angle"].to_numpy(dtype=float), 0.0, 90.0) / 10.0
    kind = shots["shot_type"].to_numpy()
    cols = [dist, dist ** 2, angle, angle ** 2] + [(kind == t).astype(float) for t in SHOT_TYPES]
    return np.column_stack(cols)


def fit_shot_model(shots: pd.DataFrame, min_rows: int = 20000) -> Optional[LogisticRegression]:
    """Goal probability of an unblocked five-on-five attempt; ``None`` when
    fewer than ``min_rows`` located attempts are available to learn from."""
    located = shots[shots["distance"].notna() & shots["angle"].notna()]
    if len(located) < min_rows or located["is_goal"].nunique() < 2:
        return None
    return LogisticRegression(C=10.0, max_iter=1000).fit(_design(located), located["is_goal"].to_numpy(dtype=int))


def expected_goals(model: LogisticRegression, shots: pd.DataFrame) -> np.ndarray:
    """Goal probability per attempt (NaN where the location is missing)."""
    out = np.full(len(shots), np.nan)
    located = (shots["distance"].notna() & shots["angle"].notna()).to_numpy()
    if located.any():
        out[located] = model.predict_proba(_design(shots[located]))[:, 1]
    return out


def game_shot_stats(event_ids: pd.Series, shots: pd.DataFrame, usable: np.ndarray, fit_events: np.ndarray, *,
                    min_fit_rows: int = 20000) -> pd.DataFrame:
    """One row per event of ``event_ids`` with, for the home (``a``) and away
    (``b``) side: ``sog_*`` shots on goal in all situations, ``ppg_*``
    power-play goals, ``xg5_*`` five-on-five expected goals.

    ``usable`` and ``fit_events`` are boolean arrays parallel to
    ``event_ids``: shots of events outside ``usable`` are ignored entirely,
    and only shots of ``fit_events`` train the expected-goals model.  Rows are
    NaN for events without usable play-by-play; ``xg5_*`` is NaN when no
    model could be fitted."""
    n = len(event_ids)
    row = pd.Index(event_ids).get_indexer(shots["event_id"])
    keep = row >= 0
    keep[keep] = usable[row[keep]]
    shots, row = shots[keep], row[keep]
    home = shots["is_home"].to_numpy() == 1
    five = (shots["situation"] == "5v5").to_numpy()
    model = fit_shot_model(shots[five & fit_events[row]], min_rows=min_fit_rows)
    xg = np.zeros(len(shots))
    if model is not None:
        xg[five] = np.nan_to_num(expected_goals(model, shots[five]))
    on_goal = shots["on_goal"].to_numpy(dtype=float)
    pp_goal = ((shots["situation"] == "pp") & (shots["is_goal"] == 1)).to_numpy(dtype=float)
    has_shots = np.bincount(row, minlength=n) > 0
    out = {}
    for side, mask in (("a", home), ("b", ~home)):
        for name, weight in (("sog", on_goal), ("ppg", pp_goal), ("xg5", xg)):
            total = np.bincount(row[mask], weights=weight[mask], minlength=n)
            out[f"{name}_{side}"] = np.where(has_shots, total, np.nan)
    frame = pd.DataFrame(out)
    if model is None:
        frame[["xg5_a", "xg5_b"]] = np.nan
    return frame
