"""Point-in-time goalie quality.

Two numbers are tracked from goalie box scores of **earlier slates** only:

* each goalie's save percentage, shrunk towards the league rate (a goalie with
  few shots faced is rated close to average) and expressed in percentage
  points above the league rate;
* for each team, how its recent starts were shared between goalies.  The
  goalie with the largest recent share is the team's *primary* goalie.

``goalie_sv_*`` is the primary goalie's rating and is knowable before the
game.  ``goalie_actual_*`` is the rating of the goalie who really started the
game being predicted: his identity is only known at puck drop, so that column
is retrospective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = ["GoalieConfig", "goalie_pregame", "starter_rating"]


@dataclass(frozen=True)
class GoalieConfig:
    #: Shots of league-average goaltending mixed into every goalie's record.
    #: Save percentage is mostly noise over one season, so the prior is heavy.
    prior_shots: float = 3000.0
    decay: float = 0.997           # per appearance; old games fade
    start_lr: float = 0.25         # memory of who has been starting (about the last 7 games)
    league_decay: float = 0.995    # per slate, for the running league save rate
    min_share: float = 0.02        # a goalie below this share of recent starts is forgotten


def goalie_pregame(ev: pd.DataFrame, goalie_box: Optional[pd.DataFrame], batch: np.ndarray,
                   completed: np.ndarray, cfg: GoalieConfig) -> pd.DataFrame:
    """Per event (rows of ``ev``, sorted by time): ``goalie_sv_a/b`` and the
    retrospective ``goalie_actual_a/b``.

    ``attrs["goalies"]`` maps each team to its goalies after the last
    completed slate (``player_id, name, start_share, rating, shots``) and
    ``attrs["league_sv"]`` is the league save rate at that point."""
    n = len(ev)
    out = {c: np.full(n, np.nan) for c in ("goalie_sv_a", "goalie_sv_b", "goalie_actual_a", "goalie_actual_b")}
    frame = pd.DataFrame(out)
    frame.attrs["goalies"], frame.attrs["league_sv"] = {}, None
    if goalie_box is None or goalie_box.empty:
        return frame
    gb = goalie_box.reset_index(drop=True)
    gidx, _ = pd.factorize(gb["player_id"])
    faced = gb["shots_against"].to_numpy(dtype=float)
    saved = gb["saves"].to_numpy(dtype=float)
    started = gb["starter"].to_numpy(dtype=int) == 1
    groups = gb.groupby(["event_id", "team_id"], sort=False).indices
    shots = np.zeros(gidx.max() + 1)
    saves = np.zeros(gidx.max() + 1)
    league = [0.0, 0.0]                      # decayed saves, shots faced
    shares: Dict[str, Dict[int, float]] = {}
    pending: List[Tuple[str, np.ndarray]] = []

    def rating(g: int) -> float:
        rate = league[0] / league[1]
        return 100.0 * (saves[g] - shots[g] * rate) / (shots[g] + cfg.prior_shots)

    def flush() -> None:
        """Apply a finished slate.  A goalie's record is shared by every team
        that still lists him, so nothing is applied while the slate is open."""
        league[0] *= cfg.league_decay
        league[1] *= cfg.league_decay
        for team, rows in pending:
            starter = next((int(gidx[r]) for r in rows if started[r]), None)
            share = shares.setdefault(team, {})
            if starter is not None:
                share.setdefault(starter, 0.0)
                for g in list(share):
                    share[g] += cfg.start_lr * ((g == starter) - share[g])
                    if share[g] < cfg.min_share:
                        del share[g]
            for r in rows:
                if faced[r] > 0:
                    g = gidx[r]
                    shots[g] = cfg.decay * shots[g] + faced[r]
                    saves[g] = cfg.decay * saves[g] + saved[r]
                    league[0] += saved[r]
                    league[1] += faced[r]
        pending.clear()

    eids = ev["event_id"].to_numpy()
    sides = (("a", ev["home_id"].to_numpy()), ("b", ev["away_id"].to_numpy()))
    for i in range(n):
        if i and batch[i] != batch[i - 1]:
            flush()
        for side, teams in sides:
            share = shares.get(teams[i])
            if share and league[1] > 0:
                out[f"goalie_sv_{side}"][i] = rating(max(share, key=share.get))
            rows = groups.get((eids[i], teams[i]))
            if rows is None or not completed[i]:
                continue
            starter = next((int(gidx[r]) for r in rows if started[r]), None)
            if starter is not None and league[1] > 0:
                out[f"goalie_actual_{side}"][i] = rating(starter)
            pending.append((teams[i], rows))
    flush()

    frame = pd.DataFrame(out)
    names = dict(zip(gidx.tolist(), gb["player_name"].tolist()))
    ids = dict(zip(gidx.tolist(), gb["player_id"].astype(str).tolist()))
    frame.attrs["league_sv"] = float(league[0] / league[1]) if league[1] > 0 else None
    frame.attrs["goalies"] = {
        team: [{"player_id": ids[g], "name": names[g], "start_share": round(float(s), 4),
                "rating": round(float(rating(g)), 4), "shots": round(float(shots[g]), 1)}
               for g, s in sorted(share.items(), key=lambda kv: -kv[1])]
        for team, share in shares.items() if share and league[1] > 0
    }
    return frame


def starter_rating(goalies: Sequence[Dict[str, Any]], *, confirmed: Optional[str] = None,
                   ruled_out: Sequence[str] = ()) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
    """Rating and record of the goalie expected to start for a team.

    ``goalies`` is the team's list from :func:`goalie_pregame` (largest recent
    share first).  A ``confirmed`` starter is used when the team has a record
    of him; otherwise the first goalie not ``ruled_out``.  Returns ``(None,
    None)`` when no listed goalie qualifies - never a guess."""
    by_id = {g["player_id"]: g for g in goalies}
    if confirmed is not None:
        pick = by_id.get(confirmed)
    else:
        pick = next((g for g in goalies if g["player_id"] not in set(ruled_out)), None)
    return (None, None) if pick is None else (float(pick["rating"]), pick)
