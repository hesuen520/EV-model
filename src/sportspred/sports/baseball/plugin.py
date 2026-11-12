"""Baseball plugin: Major League Baseball game winners from Retrosheet data."""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

import pandas as pd

from ...core.enums import OutcomeType
from ...models.candidates import PriorCandidate, gbm_candidate, logit_candidate
from ..base import (
    Candidate,
    CheckSpec,
    Explanation,
    FeatureFrame,
    IngestReport,
    LeagueSpec,
    MarketDecl,
    SplitPlan,
    SportPlugin,
)
from .adapter import RETROSHEET_NOTICE, ingest_retrosheet
from .features import FEATURE_LABELS, STARTER_FEATURES, BaseballConfig, build_winner_frame

__all__ = ["BaseballPlugin"]

_WINNER = MarketDecl("winner", OutcomeType.BINARY, True, includes_overtime=True,
                     note="game winner including extra innings")
_TOTAL = MarketDecl("total", OutcomeType.OVER_UNDER, False, includes_overtime=True,
                    note="total runs at an exact line; the source has no line history")
_STARTER_NOTE = (
    "near-pregame reconstruction: uses the pitchers who actually started (their prior starts only); "
    "announced probable starters are not recorded and can change")


class BaseballPlugin(SportPlugin):
    sport = "baseball"
    leagues = (
        LeagueSpec("mlb", "baseball", "Major League Baseball", gender="men", level="pro",
                   markets=(_WINNER, _TOTAL),
                   real_data_source="Retrosheet game logs and day-by-day files (Chadwick Bureau GitHub mirror)",
                   notes=RETROSHEET_NOTICE),
    )
    _CONFIG = BaseballConfig()

    # ------------------------------------------------------------------ data
    def ingest(self, store: Any, league: str, *, seasons: Optional[Sequence[int]] = None,
               refresh: Optional[bool] = None, fetcher: Any = None) -> IngestReport:
        self.league(league)
        return ingest_retrosheet(store, league, seasons=seasons, refresh=refresh, fetcher=fetcher)

    # -------------------------------------------------------------- features
    def feature_frame(self, store: Any, league: str, market: str, *, now: Any = None) -> FeatureFrame:
        decl = self.league(league).market(market)
        if market != "winner" or not decl.implemented:
            raise NotImplementedError(f"{league} market {market!r} has no feature builder yet")
        return build_winner_frame(store, league, now=now, cfg=self._CONFIG)

    def candidates(self, league: str, market: str, frame: FeatureFrame) -> List[Candidate]:
        f = frame.feature_cols
        runs = ["home_edge", "elo_diff", "net_slow_diff"]
        core = runs + ["bsr_slow_diff", "net_fast_diff"]
        return [
            PriorCandidate("base_rate", by="home_edge",
                           description="home-win frequency (separately for neutral sites); knows nothing about the teams"),
            logit_candidate("elo_logit", ["home_edge", "elo_diff"], is_baseline=True, complexity=1,
                            description="logistic regression on the Elo difference and home field"),
            logit_candidate("runs_logit", runs, complexity=2,
                            description="adds the long-term opponent-adjusted run differential"),
            logit_candidate("core_logit", core, complexity=3,
                            description="adds BaseRuns-based strength and recent form"),
            logit_candidate("full_logit", f, complexity=4,
                            description="all point-in-time features: park, day/night, rest, travel, "
                                        "doubleheaders, bullpen workload"),
            gbm_candidate("full_gbm", f, complexity=6,
                          description="gradient-boosted trees on all point-in-time features"),
            logit_candidate("starter_logit", f + STARTER_FEATURES, complexity=5, uses_retrospective=True,
                            description="full_logit plus the starting pitchers; " + _STARTER_NOTE),
            gbm_candidate("starter_gbm", f + STARTER_FEATURES, complexity=7, uses_retrospective=True,
                          description="full_gbm plus the starting pitchers; " + _STARTER_NOTE),
        ]

    def split_plan(self, league: str, market: str, frame: FeatureFrame) -> SplitPlan:
        per_season = frame.completed.groupby("block").size()
        typical = float(per_season.median()) if len(per_season) else 0.0
        return SplitPlan(
            n_final_blocks=2, min_train_blocks=3, n_calib_blocks=1,
            burn_in_blocks=self._CONFIG.burn_in_seasons, min_block_events=max(50, int(0.5 * typical)),
        )

    def required_checks(self, league: str, market: str) -> List[CheckSpec]:
        return [
            CheckSpec("probable_pitcher", "probable starting pitcher", scope="team", max_age_hours=24.0,
                      recheck_within_hours=3.0,
                      why="the starter is the largest single-game factor and announced starters get scratched"),
            CheckSpec("lineup", "starting lineup", scope="team", max_age_hours=6.0, recheck_within_hours=2.0,
                      why="rest days and late scratches change the batting order on the day"),
            CheckSpec("weather", "weather and postponement risk", scope="event", max_age_hours=6.0,
                      why="rain delays, postponements and wind change whether and how the game is played"),
            CheckSpec("prior_games_complete", "both teams' earlier games are final", scope="event",
                      max_age_hours=1e9,
                      why="ratings, rest and bullpen workload are only current once each team's previous "
                          "game (including the first game of a doubleheader) has been played"),
        ]

    # ------------------------------------------------------------ explanation
    def explain(self, league: str, market: str, row: pd.Series, frame: FeatureFrame,
                contributions: Optional[Mapping[str, float]] = None) -> Explanation:
        exp = Explanation()
        names = frame.extra.get("names", {})
        home = names.get(row["side_a_id"], "home side")
        away = names.get(row["side_b_id"], "away side")
        pick_home = bool(row.get("_pick_is_a", True))
        pick, other = (home, away) if pick_home else (away, home)
        sign = 1.0 if pick_home else -1.0
        if contributions:
            for feat, value in sorted(contributions.items(), key=lambda kv: -abs(kv[1])):
                if abs(value) < 0.02:
                    continue
                toward = value * sign
                text = (f"{FEATURE_LABELS.get(feat, feat)} "
                        f"({'favours' if toward > 0 else 'against'} {pick}, {abs(value):.2f} log-odds)")
                (exp.supporting if toward > 0 else exp.upset_risks).append(text)
            exp.supporting = exp.supporting[:5]
            exp.upset_risks = exp.upset_risks[:5]

        def known(key: str) -> bool:
            return row.get(key) == row.get(key) and row.get(key) is not None

        own_elo, opp_elo = (row["elo_a"], row["elo_b"]) if pick_home else (row["elo_b"], row["elo_a"])
        exp.opponent_strength.append(
            f"{other} Elo {opp_elo:.0f} versus {pick} {own_elo:.0f} (league average 1500)")
        for key, label in (("net_slow_diff", "long-term opponent-adjusted run differential"),
                           ("net_fast_diff", "recent-form run differential (about the last 16 games)")):
            if known(key):
                gap = sign * row[key]
                exp.opponent_strength.append(
                    f"{label}: {abs(gap):.2f} runs per game in favour of {pick if gap > 0 else other}")
        exp.upset_risks.append(
            "a single baseball game is close to a coin flip: clear favourites still lose about four times in ten")
        if row.get("dh_game2") == 1.0:
            exp.upset_risks.append(
                "second game of a doubleheader: the first game's result and bullpen use are not in these numbers")
        own_games = row.get("season_games_a" if pick_home else "season_games_b", 99)
        if own_games < 20:
            exp.upset_risks.append(
                f"early season ({int(own_games)} games played): ratings lean on last season")
        exp.availability_assumptions.append(
            "The model does not know who pitches or bats: it assumes each club's usual pitching and lineup. "
            "The probable-pitcher and lineup checks must be on file before a selection is FINAL, but they do "
            "not change the probability; the validation report shows how much knowing the starter is worth.")
        if row.get("dq_no_team_stats") == 1.0:
            exp.missing_data.append("one side has no rated game yet: run-differential ratings are missing")
        if row.get("dq_no_bullpen") == 1.0:
            exp.missing_data.append("no pitching lines for the previous three days: bullpen workload is missing")
        for side, key in ((home, "travel_km_a"), (away, "travel_km_b")):
            if not known(key):
                exp.missing_data.append(f"travel distance unknown for {side}")
        return exp
