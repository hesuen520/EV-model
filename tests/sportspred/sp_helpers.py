"""Helpers shared by sportspred tests (importable without package context)."""

from __future__ import annotations


def prediction(event_id, team_ids, **over):
    rec = dict(
        created_at_utc="2030-01-01T00:00:00Z", event_id=event_id, market="winner", selection=team_ids[0],
        selection_label="Boston Celtics", probs={team_ids[0]: 0.6, team_ids[1]: 0.4}, prob=0.6,
        model_name="m", model_version="v1", validation_status="experimental", rec_status="PROVISIONAL",
    )
    rec.update(over)
    return rec


def synthetic_two_sided(store, league="demo_league", sport="basketball", *, n_teams=8, n_seasons=7,
                        rounds_per_season=4, seed=7, first_year=2015, mean_score=100.0, score_sd=10.0,
                        home_adv=2.5, upcoming_rounds=1):
    """Write a small synthetic league (flagged synthetic, ``demo_`` key) into
    ``store``: double round-robins with persistent team strengths, one slate
    every two days, plus a few scheduled games after the last result.

    Returns ``(strengths, events_frame)``.  Only schedules and scores are
    written, which is enough for rating-based feature builders.
    """
    import numpy as np
    import pandas as pd

    assert league.startswith("demo_")
    rng = np.random.default_rng(seed)
    teams = pd.DataFrame({"name": [f"Demo Team {i:02d}" for i in range(n_teams)]})
    ids = list(store.resolve_participants(teams, sport=sport, scope=league, is_synthetic=True))
    strength = dict(zip(ids, rng.normal(0.0, 8.0, n_teams)))
    rows, results = [], []
    for s in range(n_seasons + 1):
        final_season = s == n_seasons
        day = pd.Timestamp(f"{first_year + s}-11-01T03:00:00Z")
        n_rounds = upcoming_rounds if final_season else rounds_per_season
        for r in range(n_rounds):
            order = rng.permutation(n_teams)
            for k in range(n_teams - 1):
                rot = np.r_[order[0], np.roll(order[1:], k)]
                for j in range(n_teams // 2):
                    h, a = ids[rot[j]], ids[rot[-1 - j]]
                    if (k + r) % 2:
                        h, a = a, h
                    gid = f"{league}-{s}-{r}-{k}-{j}"
                    rows.append({
                        "league": league, "season": str(first_year + s + 1), "start_time_utc": day,
                        "status": "scheduled" if final_season else "final", "provider": "synthetic",
                        "provider_event_id": gid, "home_id": h, "away_id": a, "is_synthetic": True,
                        "venue_city": None, "neutral_site": 0,
                    })
                    if not final_season:
                        margin = strength[h] - strength[a] + home_adv + rng.normal(0.0, score_sd)
                        hs = int(round(mean_score + margin / 2 + rng.normal(0.0, 4.0)))
                        aw = int(round(hs - margin))
                        if aw == hs:
                            hs += 1
                        results.append((gid, hs, aw))
                day += pd.Timedelta(days=2)
        for t in ids:   # strengths drift between seasons
            strength[t] = 0.8 * strength[t] + rng.normal(0.0, 2.0)
    events, _ = store.upsert_events(pd.DataFrame(rows), sport=sport)
    eid = dict(zip(events["provider_event_id"], events["event_id"]))
    store.upsert_results(pd.DataFrame(
        [{"event_id": eid[g], "home_score": hs, "away_score": aw, "is_synthetic": True} for g, hs, aw in results]))
    return strength, events
