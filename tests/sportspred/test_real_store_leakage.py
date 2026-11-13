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
