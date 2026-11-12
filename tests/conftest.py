"""Shared pytest configuration for the evmodel core test-suite.

The suite is an independent verification of the core modules, so nothing here
re-uses implementation code: the only shared pieces are the import path and a
deterministic random generator.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    # Lets ``pytest`` run from a fresh checkout without ``PYTHONPATH=src``.
    sys.path.insert(0, str(_SRC))

#: Seed shared by every randomised test so that failures are reproducible.
SEED = 20251112


@pytest.fixture
def rng() -> np.random.Generator:
    """A freshly seeded generator: every test sees the same random stream."""
    return np.random.default_rng(SEED)
