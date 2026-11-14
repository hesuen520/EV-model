"""As-of leakage test on a REAL store, for every league-market with a recorded
validation run.  Opt-in, because it needs ingested data and takes minutes:

    SPORTSPRED_REAL_STORE=data/store python -m pytest tests/sportspred/test_real_store_leakage.py

Without the variable every case is skipped.  (The same check for one league:
``sportspred check-leakage --league nfl``.)
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from sportspred.sports import get_league
from sportspred.storage import Store
from sportspred.validation.leakage import assert_asof_invariance

_STORE = os.environ.get("SPORTSPRED_REAL_STORE")


def _cases():
    if not _STORE or not Path(_STORE).exists():
        return []
    runs = Store(_STORE).validation_runs(include_synthetic=False)
    return sorted({(r["league"], r["market"]) for _, r in runs.iterrows()})


@pytest.mark.skipif(not _cases(), reason="set SPORTSPRED_REAL_STORE to a store with recorded validation runs")
@pytest.mark.parametrize("league,market", _cases() or [("none", "none")])
def test_features_of_real_events_use_only_earlier_information(league, market):
    plugin, _ = get_league(league)
    result = assert_asof_invariance(plugin, Store(_STORE), league, market, n_cutoffs=2)
    assert result["ok"].all() and result["n_events"].min() > 0


@pytest.mark.skipif(not _cases(), reason="set SPORTSPRED_REAL_STORE to a store with recorded validation runs")
@pytest.mark.parametrize("league,market", _cases() or [("none", "none")])
def test_every_real_candidate_predicts_the_same_on_blinded_rows(league, market):
    """Validation blanks every column a candidate has not declared before it
    predicts.  A candidate that silently relied on an undeclared column - as
    the base-rate baseline once did - would predict differently here."""
    import numpy as np

    from sportspred.validation.backtest import _blind

    plugin, _ = get_league(league)
    frame = plugin.feature_frame(Store(_STORE), league, market)
    done = frame.completed.sort_values(["start_time_utc", "event_id"])
    train, rows = done.iloc[-4500:-500], done.iloc[-500:]
    plan = plugin.split_plan(league, market, frame)
    for cand in plugin.candidates(league, market, frame):
        model = cand.clone().fit(train, frame.classes)
        plain = np.asarray(model.predict_proba(rows), dtype=float)
        blind = np.asarray(model.predict_proba(_blind(rows, cand, plan.block_col)), dtype=float)
        assert np.allclose(plain, blind, equal_nan=True), f"{cand.name} reads a column it does not declare"
