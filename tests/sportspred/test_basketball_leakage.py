"""Leakage and validation-protocol tests on a synthetic basketball league."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sp_helpers import synthetic_two_sided
from sportspred.models.candidates import PriorCandidate, logit_candidate
from sportspred.sports.base import FeatureFrame, SplitPlan
from sportspred.sports.basketball.features import BasketballConfig, build_winner_frame
from sportspred.validation import LeakageError, make_folds, run_validation
from sportspred.validation.backtest import ValidationConfig
from sportspred.validation.leakage import asof_invariance, assert_asof_invariance

LEAGUE = "demo_basketball"
CFG = BasketballConfig(burn_in_seasons=2)
#: The synthetic league's scheduled round starts on 2022-11-01; "now" sits just before it.
NOW = pd.Timestamp("2022-10-31T00:00:00Z").to_pydatetime()


class _Builder:
    """Duck-typed plugin so the generic leakage test can drive the builder."""

    def feature_frame(self, store, league, market, *, now=None):
        return build_winner_frame(_SyntheticView(store), league, now=now, cfg=CFG)


class _SyntheticView:
    """The feature builder reads real rows by default; tests read the demo league."""

    def __init__(self, store):
        self._store = store

    def events(self, league, **kw):
        return self._store.events(league, include_synthetic=True, **kw)

    def __getattr__(self, name):
        return getattr(self._store, name)


@pytest.fixture
def league(store):
    synthetic_two_sided(store, LEAGUE)
    return store


def test_features_do_not_change_when_the_future_is_hidden(league):
    result = assert_asof_invariance(_Builder(), league, LEAGUE, "winner", n_cutoffs=3)
    assert result["n_events"].min() >= 1 and result["max_abs_diff"].max() == 0.0


def test_the_leakage_test_detects_a_leaky_feature(league):
    class Leaky(_Builder):
        def feature_frame(self, store, lg, market, *, now=None):
            frame = super().feature_frame(store, lg, market, now=now)
            # A feature built from the game's own margin: knowable only afterwards.
            frame.df["leaky"] = frame.df["margin"].fillna(0.0)
            frame.feature_cols = frame.feature_cols + ["leaky"]
            return frame

    result = asof_invariance(Leaky(), league, LEAGUE, "winner", n_cutoffs=2)
    assert not result[result["feature"] == "leaky"]["ok"].all()
    assert result[result["feature"] != "leaky"]["ok"].all()


def test_frame_rejects_information_from_at_or_after_the_start(league):
    frame = _Builder().feature_frame(league, LEAGUE, "winner")
    frame.validate()
    bad = frame.df.copy()
    bad.loc[bad.index[50], "info_cutoff_utc"] = bad.loc[bad.index[50], "start_time_utc"]
    broken = FeatureFrame(frame.league, frame.market, frame.outcome, frame.classes, bad, frame.feature_cols)
    with pytest.raises(ValueError, match="at/after their own start"):
        broken.validate()


def test_folds_are_chronological_and_final_period_is_reserved(league):
    frame = _Builder().feature_frame(league, LEAGUE, "winner")
    plan = SplitPlan(n_final_blocks=1, min_train_blocks=2, n_calib_blocks=1, burn_in_blocks=2, min_block_events=20)
    done = frame.completed
    folds = make_folds(done, plan)
    final_blocks = set(folds.final.test_blocks)
    assert final_blocks == {done["block"].max()}
    for fold in folds.dev:
        used = set(fold.train_blocks) | set(fold.calib_blocks) | set(fold.test_blocks)
        assert not used & final_blocks, "a development fold touched the final test period"
        assert max(fold.train_blocks) < min(fold.calib_blocks) < min(fold.test_blocks)
    assert max(folds.final.calib_blocks) < min(folds.final.test_blocks)


def test_overlapping_blocks_are_refused(league):
    frame = _Builder().feature_frame(league, LEAGUE, "winner")
    done = frame.completed.copy()
    done["block"] = np.arange(len(done)) % 3            # blocks interleaved in time
    with pytest.raises(LeakageError, match="overlap"):
        make_folds(done, SplitPlan(min_block_events=1))


def test_retrospective_columns_cannot_feed_an_unflagged_candidate(league):
    frame = _Builder().feature_frame(league, LEAGUE, "winner")
    plan = SplitPlan(n_final_blocks=1, min_train_blocks=2, n_calib_blocks=1, burn_in_blocks=2, min_block_events=20)
    sneaky = logit_candidate("sneaky", ["elo_diff", "roster_q_actual_diff"])
    with pytest.raises(LeakageError, match="retrospective"):
        run_validation(frame, [PriorCandidate(), sneaky], plan, sport="basketball",
                       config=ValidationConfig(n_boot=20), is_synthetic=True)


def test_validation_selects_on_dev_folds_and_never_validates_synthetic_data(league):
    frame = _Builder().feature_frame(league, LEAGUE, "winner")
    plan = SplitPlan(n_final_blocks=1, min_train_blocks=2, n_calib_blocks=1, burn_in_blocks=2, min_block_events=20)
    cands = [PriorCandidate(), logit_candidate("elo_logit", ["home_edge", "elo_diff"], is_baseline=True)]
    report, oos = run_validation(frame, cands, plan, sport="basketball",
                                 config=ValidationConfig(n_boot=50, calib_min_rows=30, one_se_rule=False),
                                 is_synthetic=True)
    assert report["protocol"]["final_test_used_for_selection"] is False
    assert report["selected_model"] == "elo_logit"          # team strength is real in the synthetic league
    assert report["gates_passed"] is False                  # synthetic data can never be "validated"
    assert not next(g for g in report["gates"] if g["gate"] == "real_data")["passed"]
    probs = oos[["p_away", "p_home"]].to_numpy()
    assert np.allclose(probs.sum(axis=1), 1.0) and (probs >= 0).all() and (probs <= 1).all()
    final_ids = set(oos[oos["is_final_test"]]["event_id"])
    dev_ids = set(oos[~oos["is_final_test"]]["event_id"])
    assert final_ids and dev_ids and not final_ids & dev_ids


def test_side_a_is_home_and_target_matches_results(league):
    frame = _Builder().feature_frame(league, LEAGUE, "winner", now=NOW)
    ev = league.events(LEAGUE, include_synthetic=True).set_index("event_id")
    done = frame.completed.set_index("event_id")
    assert (done["side_a_id"] == ev.loc[done.index, "home_id"]).all()
    assert (done["y"] == (ev.loc[done.index, "home_score"] > ev.loc[done.index, "away_score"]).astype(float)).all()
    assert frame.upcoming["y"].isna().all() and len(frame.upcoming) > 0


def test_a_candidate_cannot_read_a_column_the_frame_does_not_declare_as_an_input(league):
    """Only retrospective columns used to be checked: a candidate built on an
    undeclared column (a copy of the outcome, say) was selected with a
    near-zero log loss."""
    frame = _Builder().feature_frame(league, LEAGUE, "winner")
    frame.df["y_copy"] = frame.df["y"]
    plan = SplitPlan(n_final_blocks=1, min_train_blocks=2, n_calib_blocks=1, burn_in_blocks=2, min_block_events=20)
    with pytest.raises(LeakageError, match="does not declare as model inputs"):
        run_validation(frame, [PriorCandidate(), logit_candidate("cheat", ["elo_diff", "y_copy"])], plan,
                       sport="basketball", config=ValidationConfig(n_boot=20), is_synthetic=True)


def test_scored_rows_are_blinded_so_a_candidate_that_peeks_at_the_outcome_fails_loudly(league):
    from sportspred.sports.base import Candidate

    class Peeker(Candidate):
        """Declares an honest input, then reads the outcome when it predicts."""
        name, complexity = "peeker", 1

        def feature_names(self):
            return ["elo_diff"]

        def clone(self):
            return Peeker()

        def fit(self, train, classes):
            return self

        def predict_proba(self, rows):
            p_home = np.where(rows["y"].to_numpy() == 1, 0.99, 0.01)
            return np.column_stack([1 - p_home, p_home])

    frame = _Builder().feature_frame(league, LEAGUE, "winner")
    plan = SplitPlan(n_final_blocks=1, min_train_blocks=2, n_calib_blocks=1, burn_in_blocks=2, min_block_events=20)
    report, oos = run_validation(frame, [PriorCandidate(), Peeker()], plan, sport="basketball",
                                 config=ValidationConfig(n_boot=20, calib_min_rows=30), is_synthetic=True)
    scored = oos[oos["model"] == "peeker"]
    # With the outcome blanked the peeker predicts the same thing for every game: no skill left to report.
    assert scored["p_home"].nunique() <= 2 and report["selected_model"] != "peeker"


def test_results_are_reported_per_segment_and_pass_thresholds_fit_the_market(league):
    frame = _Builder().feature_frame(league, LEAGUE, "winner")
    frame.df["segment"] = np.where(frame.df["elo_diff"].abs() > frame.df["elo_diff"].abs().median(),
                                   "mismatch", "close game")
    plan = SplitPlan(n_final_blocks=1, min_train_blocks=2, n_calib_blocks=1, burn_in_blocks=2, min_block_events=20)
    cands = [PriorCandidate(), logit_candidate("elo_logit", ["home_edge", "elo_diff"], is_baseline=True)]
    report, _ = run_validation(frame, cands, plan, sport="basketball",
                               config=ValidationConfig(n_boot=30, calib_min_rows=30, one_se_rule=False),
                               is_synthetic=True)
    seg = {r["segment"]: r for r in report["final_test"]["by_segment"]}
    assert set(seg) == {"mismatch", "close game"} and abs(sum(r["share"] for r in seg.values()) - 1) < 1e-9
    # The model is far surer of itself in the mismatches; the pooled figure hides the difference.
    assert seg["mismatch"]["mean_top_probability"] > seg["close game"]["mean_top_probability"]
    assert seg["mismatch"]["n"] + seg["close game"]["n"] == report["final_test"]["selected"]["n"]
    # A two-way market reports no threshold below 0.5 (every event clears it).
    assert min(r["min_prob"] for r in report["final_test"]["pass_rule"]) == 0.5
