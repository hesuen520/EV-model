"""Multi-leg combinations with explicit dependency handling."""

from .backtest import combo_backtest, tune_min_leg_prob
from .builder import (
    MONOTONE_NOTE,
    PASS_MESSAGE,
    ComboCandidate,
    SlateCombos,
    build_slate_combos,
    candidate_status,
    legs_from_ledger,
    residual_correlations,
    select_combos,
)
from .dependency import JointResult, Leg, PairDependency, classify_pair, joint_probability, latent_correlation

__all__ = [
    "combo_backtest", "tune_min_leg_prob", "MONOTONE_NOTE", "PASS_MESSAGE", "ComboCandidate", "SlateCombos",
    "build_slate_combos", "candidate_status", "legs_from_ledger", "residual_correlations", "select_combos",
    "JointResult", "Leg", "PairDependency", "classify_pair", "joint_probability", "latent_correlation",
]
