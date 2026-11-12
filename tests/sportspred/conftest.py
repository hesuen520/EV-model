"""Shared fixtures for the sportspred suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from sportspred.storage import Store  # noqa: E402


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(tmp_path / "store")


@pytest.fixture
def two_teams(store):
    teams = pd.DataFrame({"name": ["Boston Celtics", "Los Angeles Lakers"], "pid": [2, 13]})
    ids = store.resolve_participants(teams, sport="basketball", scope="nba", provider="espn",
                                     provider_id_col="pid")
    return list(ids)


@pytest.fixture
def scheduled_event(store, two_teams):
    frame = pd.DataFrame([{
        "league": "nba", "season": "2030", "start_time_utc": "2030-01-02T03:30:00Z", "status": "scheduled",
        "provider": "espn", "provider_event_id": "401", "home_id": two_teams[0], "away_id": two_teams[1],
        "venue_name": "TD Garden",
    }])
    out, _ = store.upsert_events(frame, sport="basketball")
    return out.iloc[0]["event_id"]
