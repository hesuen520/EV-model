"""Hockey plugin: NHL game winner (overtime and shootout included) and the
regulation-time three-way result."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from ...core.enums import OutcomeType
from ...core.ids import norm_text
from ...models.candidates import PriorCandidate, gbm_candidate, logit_candidate
from ..base import (
    AvailabilityEffect,
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
from .adapter import ingest_nhl
from .features import FEATURE_LABELS, HockeyConfig, build_frame
from .goalies import starter_rating

__all__ = ["HockeyPlugin"]

_SOURCE = "sportsdataverse-data release files (NHL schedules, goalie box scores, play-by-play shots)"


class HockeyPlugin(SportPlugin):
    sport = "hockey"
    leagues = (
        LeagueSpec(
            "nhl", "hockey", "NHL", gender="men", level="pro",
            markets=(
                MarketDecl("winner", OutcomeType.BINARY, True, includes_overtime=True,
                           note="game winner including overtime and shootout"),
                MarketDecl("winner_3way", OutcomeType.THREE_WAY, True, includes_overtime=False,
                           note="result after 60 minutes of regulation: home / draw / away"),
                MarketDecl("total", OutcomeType.OVER_UNDER, False, note="total goals at an exact line"),
            ),
            real_data_source=_SOURCE, live_feeds="ESPN site API (scores); no official NHL injury list exists",
        ),
    )
    _CONFIG = HockeyConfig()

    def ingest(self, store: Any, league: str, *, seasons: Optional[Sequence[int]] = None,
               refresh: Optional[bool] = None, fetcher: Any = None) -> IngestReport:
        self.league(league)
        return ingest_nhl(store, league, seasons=seasons, refresh=refresh, fetcher=fetcher)

    def feature_frame(self, store: Any, league: str, market: str, *, now: Any = None) -> FeatureFrame:
        if not self.league(league).market(market).implemented:
            raise NotImplementedError(f"{league} market {market!r} has no feature builder yet")
        return build_frame(store, league, market, now=now, cfg=self._CONFIG)

    def candidates(self, league: str, market: str, frame: FeatureFrame) -> List[Candidate]:
        f = frame.feature_cols
        rating = ["home_edge", "elo_diff"]
        if market == "winner_3way":
            return [
                PriorCandidate("base_rate", by="home_edge",
                               description="regulation home / draw / away frequencies; knows nothing about the teams"),
                logit_candidate("elo_logit", rating + ["elo_gap_abs"], is_baseline=True, complexity=1,
                                description="multinomial logistic regression on the Elo difference and its size"),
                logit_candidate("goals_logit", rating + ["elo_gap_abs", "gd_slow_diff", "exp_total_goals"],
                                complexity=2, description="adds opponent-adjusted goals and the expected total"),
                logit_candidate("full_logit", f, complexity=4, description="all point-in-time features"),
                gbm_candidate("full_gbm", f, complexity=6, description="gradient-boosted trees on all features"),
            ]
        core = rating + ["gd_slow_diff", "gd_fast_diff", "xg5_diff", "goalie_diff", "rest_diff", "b2b_a", "b2b_b"]
        return [
            PriorCandidate("base_rate", by="home_edge",
                           description="home-win frequency; knows nothing about the teams"),
            logit_candidate("elo_logit", rating, is_baseline=True, complexity=1,
                            description="logistic regression on the Elo difference and home ice"),
            logit_candidate("goals_logit", rating + ["gd_slow_diff"], complexity=2,
                            description="adds long-term opponent-adjusted goal difference"),
            logit_candidate("core_logit", core, complexity=3,
                            description="adds recent form, five-on-five expected goals, primary goalie and rest"),
            logit_candidate("full_logit", f, complexity=4, description="all point-in-time features"),
            gbm_candidate("full_gbm", f, complexity=6, description="gradient-boosted trees on all features"),
            logit_candidate("actual_goalie_logit", f + ["goalie_actual_diff"], complexity=4, uses_retrospective=True,
                            description="full_logit plus the goalies who actually started (known only at puck drop)"),
        ]

    def split_plan(self, league: str, market: str, frame: FeatureFrame) -> SplitPlan:
        per_season = frame.completed.groupby("block").size()
        typical = float(per_season.median()) if len(per_season) else 0.0
        return SplitPlan(n_final_blocks=2, min_train_blocks=3, n_calib_blocks=1,
                         burn_in_blocks=self._CONFIG.burn_in_seasons,
                         min_block_events=max(50, int(0.5 * typical)))

    def required_checks(self, league: str, market: str) -> List[CheckSpec]:
        return [
            CheckSpec("starting_goalie", "confirmed starting goalie", scope="team", max_age_hours=6.0,
                      recheck_within_hours=2.0,
                      why="the starter is the largest single pregame swing and is confirmed only on game day"),
            CheckSpec("injury_report", "injuries and scratches", scope="team", max_age_hours=24.0,
                      why="skater absences are not in the model and must be checked by hand"),
            CheckSpec("prior_games_complete", "both teams' earlier games are final", scope="event",
                      max_age_hours=1e9, why="ratings and rest are only current once earlier games are played"),
        ]

    def apply_availability(self, league: str, market: str, row: pd.Series, frame: FeatureFrame,
                           availability: pd.DataFrame) -> AvailabilityEffect:
        """Use a confirmed starting goalie (role ``starting_goalie``, status
        ``available``) or a ruled-out goalie to replace the assumed starter.
        Skater news is listed but not quantified by this model."""
        effect = AvailabilityEffect(row=row.copy())
        goalies: Dict[str, List[Dict[str, Any]]] = frame.extra.get("goalies") or {}
        if availability is None or availability.empty:
            return effect
        effect.quantified = bool(goalies)
        new = row.copy()
        for side, team in (("a", row["side_a_id"]), ("b", row["side_b_id"])):
            roster = goalies.get(team, [])
            by_name = {norm_text(g["name"]): g for g in roster}
            confirmed, ruled_out = None, []
            for _, rep in availability[availability["team_id"] == team].iterrows():
                goalie = by_name.get(norm_text(rep["player_name"]))
                status = str(rep["status"])
                if goalie is None:
                    text = f"{rep['player_name']} ({status})"
                    if status in ("questionable", "uncertain", "doubtful", "probable"):
                        (effect.uncertain_a if side == "a" else effect.uncertain_b).append(
                            text + " - skater, effect not quantified")
                    else:
                        effect.unmatched.append(text + ": skater availability is not an input of this model")
                    continue
                if status == "available" and str(rep.get("role") or "") == "starting_goalie":
                    confirmed = goalie["player_id"]
                    effect.applied.append(f"{goalie['name']} confirmed as starting goalie")
                elif status in ("out", "doubtful"):
                    ruled_out.append(goalie["player_id"])
                    effect.applied.append(f"{goalie['name']} {status}: not expected to start")
            if confirmed is None and not ruled_out:
                continue
            rating, _ = starter_rating(roster, confirmed=confirmed, ruled_out=ruled_out)
            new[f"goalie_sv_{side}"] = float("nan") if rating is None else rating
        new["goalie_diff"] = new["goalie_sv_a"] - new["goalie_sv_b"]
        effect.row = new
        return effect

    def explain(self, league: str, market: str, row: pd.Series, frame: FeatureFrame,
                contributions: Optional[Mapping[str, float]] = None) -> Explanation:
        exp = Explanation()
        names = frame.extra.get("names", {})
        home, away = names.get(row["side_a_id"], "home side"), names.get(row["side_b_id"], "away side")
        pick_home = bool(row.get("_pick_is_a", True))
        pick, other = (home, away) if pick_home else (away, home)
        sign = 1.0 if pick_home else -1.0
        if contributions:
            for feat, value in sorted(contributions.items(), key=lambda kv: -abs(kv[1])):
                if abs(value) < 0.03:
                    continue
                toward = value * sign
                text = (f"{FEATURE_LABELS.get(feat, feat)} "
                        f"({'favours' if toward > 0 else 'against'} {pick}, {abs(value):.2f} log-odds)")
                (exp.supporting if toward > 0 else exp.upset_risks).append(text)
            exp.supporting, exp.upset_risks = exp.supporting[:5], exp.upset_risks[:5]

        def fmt(v: Any, digits: int = 2) -> str:
            return "n/a" if v is None or v != v else f"{v:.{digits}f}"

        opp_elo, own_elo = (row["elo_b"], row["elo_a"]) if pick_home else (row["elo_a"], row["elo_b"])
        exp.opponent_strength.append(f"{other} Elo {fmt(opp_elo, 0)} versus {pick} {fmt(own_elo, 0)} (average 1500)")
        if row.get("gd_slow_diff") == row.get("gd_slow_diff"):
            exp.opponent_strength.append(
                f"long-term opponent-adjusted goal difference gap: {fmt(sign * row['gd_slow_diff'])} goals per game "
                f"towards {pick if sign * row['gd_slow_diff'] > 0 else other}")
        if row.get("xg5_diff") == row.get("xg5_diff"):
            exp.opponent_strength.append(
                f"five-on-five expected-goal gap: {fmt(sign * row['xg5_diff'])} per game towards "
                f"{pick if sign * row['xg5_diff'] > 0 else other}")
        if (row.get("b2b_a") if pick_home else row.get("b2b_b")) == 1.0:
            exp.upset_risks.append(f"{pick} are on the second night of a back-to-back")
        exp.upset_risks.append("a different starting goalie than assumed can move this probability materially")
        exp.availability_assumptions.append(
            "Assumes each team starts the goalie who has started most of its recent games, and that skater "
            "line-ups match recent games; neither is confirmed unless a check below says so.")
        if row.get("dq_no_shot_data") == 1.0:
            exp.missing_data.append("no shot data yet for one side: expected-goal ratings are missing")
        if row.get("dq_no_goalie_history") == 1.0:
            exp.missing_data.append("no goalie history for one side: the goalie feature is missing")
        return exp
