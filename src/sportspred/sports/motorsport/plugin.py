"""Motorsport plugin: Formula 1 finishing markets."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ...core.enums import OutcomeType
from ...models.candidates import PriorCandidate, gbm_candidate, logit_candidate
from ...models.field import FieldRankCandidate, race_of
from ..base import (Candidate, CheckSpec, Explanation, FeatureFrame, IngestReport, LeagueSpec, MarketDecl, SplitPlan,
                    SportPlugin)
from .adapter import LEAGUE, ingest_f1
from .features import (DNF_FEATURES, FORM_FEATURES, FULL_FEATURES, GRID_FEATURES, KIND_LABELS, KINDS, F1Config,
                       build_finish_frame, finish_market, split_finish_market)

__all__ = ["MotorsportPlugin", "MIN_FINAL_RACES"]

#: A final test of fewer independent races than this cannot carry "validated":
#: rows are drivers, but a race is one event.
MIN_FINAL_RACES = 100
_FINISH = tuple(
    MarketDecl(finish_market(kind), OutcomeType.BINARY, True,
               note=f"each starter {KIND_LABELS[kind][0]}: yes or no; void if he does not take the start")
    for kind in KINDS)


class MotorsportPlugin(SportPlugin):
    sport = "motorsport"
    leagues = (
        LeagueSpec(LEAGUE, "motorsport", "Formula 1", gender="open", level="pro", markets=_FINISH,
                   real_data_source="F1DB release files on GitHub (results, qualifying, starting grids; CC BY 4.0)",
                   live_feeds="none reachable; the starting grid is imported by CSV",
                   notes="Predictions start from the official starting grid; before one is on file there are none."),
    )
    _CONFIG = F1Config()

    def participant_kind(self, league: str) -> str:
        return "person"

    # ------------------------------------------------------------------ data
    def ingest(self, store: Any, league: str, *, seasons: Optional[Sequence[int]] = None,
               refresh: Optional[bool] = None, fetcher: Any = None) -> IngestReport:
        self.league(league)
        return ingest_f1(store, seasons=seasons, refresh=refresh, fetcher=fetcher)

    # -------------------------------------------------------------- modelling
    def feature_frame(self, store: Any, league: str, market: str, *, now: Any = None) -> FeatureFrame:
        self.league(league).market(market)
        kind, _ = split_finish_market(market)
        if kind is None:
            raise NotImplementedError(f"{league} market {market!r} has no feature builder")
        frame = build_finish_frame(store, league, kind, now=now, cfg=self._CONFIG)
        # Rows are drivers; the independent events are races.  Say how many the final test has.
        plan = self.split_plan(league, market, frame)
        done = frame.completed
        seasons = sorted(done["block"].unique())
        final = done[done["block"].isin(seasons[-plan.n_final_blocks:])]
        n_races = int(final["game_id"].nunique())
        frame.extra["gates"] = [{
            "gate": "independent_races", "passed": n_races >= MIN_FINAL_RACES,
            "detail": f"{n_races} races in the final-test seasons (minimum {MIN_FINAL_RACES}); the row count "
                      "counts drivers, and the drivers of one race are not independent",
        }]
        return frame

    def candidates(self, league: str, market: str, frame: FeatureFrame) -> List[Candidate]:
        kind, _ = split_finish_market(market)
        what = KIND_LABELS[kind][0]
        return [
            PriorCandidate("base_rate", description=f"how often a starter is {what.replace('to ', '')}; knows "
                                                    "nothing about the driver, the car or the grid"),
            logit_candidate("grid_logit", GRID_FEATURES, is_baseline=True, complexity=1,
                            description="logistic regression on the starting grid position alone"),
            logit_candidate("form_logit", FORM_FEATURES, complexity=2,
                            description="driver form, car form, field strength and track fit - without the grid "
                                        "(what is known before qualifying)"),
            logit_candidate("full_logit", FULL_FEATURES, complexity=4,
                            description="grid and qualifying gap, driver and car form, track fit and reliability"),
            FieldRankCandidate("field_rank", FULL_FEATURES, DNF_FEATURES, top_k=KINDS[kind], complexity=5,
                               description="a model of the whole race: Plackett-Luce strengths from finishing "
                                           "orders, a retirement model, and a simulated finish distribution"),
            gbm_candidate("full_gbm", FULL_FEATURES, complexity=6, min_samples_leaf=40,
                          description="gradient-boosted trees on the same inputs"),
        ]

    def split_plan(self, league: str, market: str, frame: FeatureFrame) -> SplitPlan:
        per_season = frame.completed.groupby("block").size()
        typical = float(per_season.median()) if len(per_season) else 0.0
        return SplitPlan(n_final_blocks=5, min_train_blocks=6, n_calib_blocks=1,
                         burn_in_blocks=int(frame.extra.get("burn_in_seasons", 2)),
                         min_block_events=max(50, int(0.5 * typical)), max_dev_folds=10)

    def required_checks(self, league: str, market: str) -> List[CheckSpec]:
        return [
            CheckSpec("starting_grid", "official starting grid", scope="event", max_age_hours=36.0,
                      recheck_within_hours=4.0,
                      why="the grid, with penalties, is the strongest pregame fact and can change on race morning"),
            CheckSpec("prior_games_complete", "every earlier race of the season has a result on file", scope="event",
                      max_age_hours=1e9, why="driver and car form are only current once the previous race is in"),
        ]

    # ----------------------------------------------------------- supplement
    def validation_supplement(self, league: str, market: str, frame: FeatureFrame, candidate: Candidate,
                              plan: SplitPlan) -> Optional[Dict[str, Any]]:
        """Per-race figures: a race is the event, and "who does the model rank
        first" is how its picks would actually be read."""
        from ...validation.backtest import _blind, fit_with_calibration
        from ...validation.splits import make_folds

        kind, _ = split_finish_market(market)
        k = KINDS[kind]
        df = frame.completed.sort_values(["start_time_utc", "event_id"]).reset_index(drop=True)
        folds = make_folds(df, plan)
        out: Dict[str, Any] = {
            "kind": "per_race",
            "note": f"Per race: the {k} starter(s) the model rates most likely {KIND_LABELS[kind][0]}, against the "
                    f"first {k} on the grid. 'Hits' is how many of them did; for the win market it is whether the "
                    "top-rated driver won.",
            "folds": {},
        }
        targets = [("last_development_fold", folds.dev[-1])] if folds.dev else []
        targets.append(("final_test", folds.final))
        for label, fold in targets:
            train = df[df[plan.block_col].isin(fold.train_blocks)]
            calib = df[df[plan.block_col].isin(fold.calib_blocks)]
            test = df[df[plan.block_col].isin(fold.test_blocks)]
            model, cal = fit_with_calibration(candidate, train, calib, frame.classes)
            p = cal.transform(model.predict_proba(_blind(test, model, plan.block_col)))[:, 1]
            races = race_of(test["event_id"])
            y, grid = test["y"].to_numpy(dtype=float), test["grid_pos"].to_numpy(dtype=float)
            model_hits, grid_hits, sums = [], [], []
            for race in pd.unique(races):
                idx = np.flatnonzero(races == race)
                top_model = idx[np.argsort(-p[idx], kind="stable")[:k]]
                top_grid = idx[np.argsort(grid[idx], kind="stable")[:k]]
                model_hits.append(float(y[top_model].sum()))
                grid_hits.append(float(y[top_grid].sum()))
                sums.append(float(p[idx].sum()))
            out["folds"][label] = {
                "blocks": [str(b) for b in fold.test_blocks], "races": len(model_hits), "k": k,
                "model_hits_per_race": float(np.mean(model_hits)), "grid_hits_per_race": float(np.mean(grid_hits)),
                "mean_probability_sum_per_race": float(np.mean(sums)),
                "note_sum": f"the probabilities of one race should add up to about {k}",
            }
        return out

    # ------------------------------------------------------------ explanation
    def explain(self, league: str, market: str, row: pd.Series, frame: FeatureFrame,
                contributions: Optional[Mapping[str, float]] = None) -> Explanation:
        exp = Explanation()
        team = row.get("constructor_name") or "his team"
        n = int(row["n_field"]) if row.get("n_field") == row.get("n_field") else None
        grid = "from the pit lane" if bool(row.get("pit_lane")) else f"{int(row['grid_pos'])}"
        exp.supporting.append(f"starts {grid}" + (f" of {n}" if n else "") + f" for {team}"
                              + (f" ({row['grid_source']})" if row.get("grid_source") else ""))
        if row.get("quali_gap_pct") == row.get("quali_gap_pct"):
            exp.supporting.append(f"qualified {row['quali_gap_pct']:.2f}% off the pole time")
        proj = row.get("_projection") or {}
        if proj:
            exp.supporting.append(
                f"simulated race: wins {100 * proj.get('win', float('nan')):.1f}%, podium "
                f"{100 * proj.get('podium', float('nan')):.1f}%, points {100 * proj.get('points', float('nan')):.1f}%, "
                f"not classified {100 * proj.get('not_classified', float('nan')):.1f}% (before calibration)")

        def rel(value: Any) -> Optional[str]:
            return None if value != value else ("above" if value > 0 else "below")

        if rel(row.get("con_perf_rel")):
            exp.opponent_strength.append(
                f"{team}'s recent results are {rel(row['con_perf_rel'])} the average of this field "
                f"({row['con_perf_rel']:+.2f} on a 0-1 scale)")
        if rel(row.get("drv_perf_rel")):
            exp.opponent_strength.append(
                f"his own recent results are {rel(row['drv_perf_rel'])} the field's average "
                f"({row['drv_perf_rel']:+.2f})")
        if row.get("mate_gap_ew") == row.get("mate_gap_ew"):
            ahead = row["mate_gap_ew"] < 0
            exp.supporting.append(f"qualifies {abs(row['mate_gap_ew']):.1f} places "
                                  f"{'ahead of' if ahead else 'behind'} his team-mates on average")
        if row.get("drv_dnf_ew") == row.get("drv_dnf_ew") and row.get("con_dnf_ew") == row.get("con_dnf_ew"):
            worst = max(float(row["drv_dnf_ew"]), float(row["con_dnf_ew"]))
            if worst >= 0.2:
                exp.upset_risks.append(f"reliability: {100 * worst:.0f}% of recent starts (driver or car) were not "
                                       "classified")
        if row.get("drv_starts") == row.get("drv_starts") and row["drv_starts"] < 0.1:
            exp.upset_risks.append("fewer than ten race starts on file: his form rests on little")
        if row.get("drv_circuit_delta") == row.get("drv_circuit_delta") and abs(row["drv_circuit_delta"]) >= 0.05:
            exp.supporting.append("has gone " + ("better" if row["drv_circuit_delta"] > 0 else "worse")
                                  + " at this circuit than his form at the time suggested")
        exp.availability_assumptions.append("Assumes he takes the start; a market on a driver who does not is void.")
        exp.missing_data += ["weather and track conditions, tyres and strategy are not in the data",
                             "practice and sprint results are not used"]
        return exp
