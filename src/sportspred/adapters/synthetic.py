"""SYNTHETIC demo data.  Nothing produced here describes a real event.

The generator exists so the whole pipeline (ingest -> validate -> train ->
predict -> combos -> grade) can be exercised offline and tested end to end.
Every row is flagged ``is_synthetic`` and can only be written to a league
whose key starts with ``demo_``; the store refuses anything else, and queries
hide synthetic rows unless asked for them.

Results are drawn from a known model (latent strengths that drift between
seasons, a home edge, normal scoring noise), so the true win probability of
every event is known and is stored in the ``truth`` table for comparison.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..core.timeutil import parse_ts, utcnow
from ..sports.base import IngestReport
from ..storage.store import DEMO_PREFIX

__all__ = ["generate_demo_league", "SYNTHETIC_BANNER"]

SYNTHETIC_BANNER = "SYNTHETIC DEMO DATA - not real events, teams, results or probabilities"


def generate_demo_league(store: Any, league: str, *, now: Any = None, home_adv: bool = True, n_teams: int = 12,
                         n_seasons: int = 9, rounds_per_season: int = 3, upcoming_slates: int = 2,
                         seed: int = 20261112) -> IngestReport:
    """Write a synthetic league: ``n_seasons`` completed seasons ending before
    ``now`` and a few scheduled slates after it."""
    if not league.startswith(DEMO_PREFIX):
        raise ValueError(f"synthetic data may only be generated for a {DEMO_PREFIX}* league, not {league!r}")
    now_ts = pd.Timestamp(utcnow() if now is None else parse_ts(now))
    rng = np.random.default_rng(seed)
    sport = "demo"
    teams = pd.DataFrame({"name": [f"Demo {'Club' if home_adv else 'Player'} {i + 1:02d}" for i in range(n_teams)]})
    ids = list(store.resolve_participants(teams, sport=sport, scope=league, kind="team" if home_adv else "player",
                                          is_synthetic=True))
    strength = rng.normal(0.0, 6.0, n_teams)
    edge, sd = (3.0 if home_adv else 0.0), 11.0
    source = store.add_source("synthetic", f"sportspred://synthetic/{league}", notes=SYNTHETIC_BANNER,
                              sha256=f"seed-{seed}", is_synthetic=True)
    slates_per_season = rounds_per_season * (n_teams - 1)
    first_day = (now_ts.normalize() - pd.Timedelta(days=365 * n_seasons + 30)).replace(hour=3)
    rows, truth = [], []
    for s in range(n_seasons + 1):
        future = s == n_seasons
        day = (now_ts.normalize() + pd.Timedelta(days=1)).replace(hour=3) if future else \
            first_day + pd.Timedelta(days=365 * s)
        for r in range(upcoming_slates if future else slates_per_season):
            order = np.r_[0, np.roll(np.arange(1, n_teams), r)]
            for j in range(n_teams // 2):
                h, a = int(order[j]), int(order[-1 - j])
                if (r + j) % 2:
                    h, a = a, h
                p_home = float(norm.cdf((strength[h] - strength[a] + edge) / sd))
                gid = f"{league}-{s}-{r}-{j}"
                row = {"league": league, "season": str(first_day.year + s), "start_time_utc": day,
                       "status": "scheduled" if future else "final", "provider": "synthetic",
                       "provider_event_id": gid, "home_id": ids[h], "away_id": ids[a],
                       "neutral_site": 0 if home_adv else 1, "is_synthetic": True, "source_id": source,
                       "venue_name": "Synthetic Arena"}
                if not future:
                    margin = strength[h] - strength[a] + edge + rng.normal(0.0, sd)
                    hs = int(round(100 + margin / 2 + rng.normal(0.0, 5.0)))
                    aw = int(round(hs - margin))
                    aw = aw - 1 if aw == hs else aw
                    row.update(home_score=hs, away_score=aw)
                rows.append(row)
                truth.append({"provider_event_id": gid, "home_id": ids[h], "p_home_true": p_home})
            day += pd.Timedelta(days=1 if future else 2)
        strength = 0.85 * strength + rng.normal(0.0, 2.2, n_teams)
    events, _ = store.upsert_events(pd.DataFrame(rows), sport=sport)
    done = events[events["status"] == "final"]
    store.upsert_results(done[["event_id", "home_score", "away_score"]].assign(is_synthetic=True), source_id=source)
    truth_df = pd.DataFrame(truth).merge(events[["provider_event_id", "event_id"]], on="provider_event_id")
    store.write_table(league, "truth", truth_df[["event_id", "home_id", "p_home_true"]], source_ids=[source],
                      is_synthetic=True)
    return IngestReport(
        league=league, n_events=len(events), n_results=len(done), n_upcoming=int((events["status"] == "scheduled").sum()),
        first_event=events["start_time_utc"].min().strftime("%Y-%m-%d"),
        last_event=events["start_time_utc"].max().strftime("%Y-%m-%d"), sources=[source],
        tables={"truth": len(truth_df)}, warnings=[SYNTHETIC_BANNER])
