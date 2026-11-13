"""Tennis plugin: ATP and WTA tour-level main draw and qualifying,
Challenger / WTA 125, and ITF singles, each as its own league."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from ...core.enums import OutcomeType
from ...models.candidates import PriorCandidate, gbm_candidate, logit_candidate
from ..base import Candidate, CheckSpec, FeatureFrame, IngestReport, LeagueSpec, MarketDecl, SplitPlan
from ..generic.features import GenericOptions
from ..generic.plugin import ResultsOnlyPlugin
from .adapter import TENNIS_LEAGUES, ingest_tennis
from .features import TennisConfig, build_tennis_frame

__all__ = ["TennisPlugin"]

_SOURCE = "mirror of Jeff Sackmann's match archive (CC BY-NC-SA 4.0, non-commercial; snapshot to mid-2026)"
_WINNER = MarketDecl("winner", OutcomeType.BINARY, True, note="match winner (a retirement counts for the player who advances)")
_NAMES = {
    "atp": ("ATP Tour main draw singles", "men", "tour"),
    "atp_qual": ("ATP Tour qualifying singles", "men", "tour-qualifying"),
    "atp_challenger": ("ATP Challenger main draw singles", "men", "challenger"),
    "itf_men": ("ITF men's singles", "men", "itf"),
    "wta": ("WTA Tour main draw singles", "women", "tour"),
    "wta_qual": ("WTA Tour qualifying singles", "women", "tour-qualifying"),
    "wta_125": ("WTA 125 main draw singles", "women", "wta125"),
    "itf_women": ("ITF women's singles", "women", "itf"),
}


class TennisPlugin(ResultsOnlyPlugin):
    sport = "tennis"
    leagues = tuple(
        LeagueSpec(key, "tennis", name, gender=gender, level=level, discipline="singles", markets=(_WINNER,),
                   real_data_source=_SOURCE, live_feeds="none (order of play and withdrawals must be imported)")
        for key, (name, gender, level) in _NAMES.items())
    _CONFIG = TennisConfig()

    def participant_scope(self, league: str) -> str:
        return f"tennis_{TENNIS_LEAGUES[league].tour}"      # one pool of players per tour, as in the adapter

    def participant_kind(self, league: str) -> str:
        return "player"

    def options(self, league: str) -> GenericOptions:
        return GenericOptions(home_adv=False, burn_in_blocks=self._CONFIG.burn_in_years)

    def ingest(self, store: Any, league: str, *, seasons: Optional[Sequence[int]] = None,
               refresh: Optional[bool] = None, fetcher: Any = None) -> IngestReport:
        self.league(league)
        return ingest_tennis(store, league, seasons=seasons, refresh=refresh, fetcher=fetcher)

    def feature_frame(self, store: Any, league: str, market: str, *, now: Any = None) -> FeatureFrame:
        self.league(league).market(market)
        return build_tennis_frame(store, league, market, now=now, cfg=self._CONFIG)

    def candidates(self, league: str, market: str, frame: FeatureFrame) -> List[Candidate]:
        f = frame.feature_cols
        return [
            PriorCandidate("base_rate", description="50/50 by construction: side A is assigned by ID, not by strength"),
            logit_candidate("elo_logit", ["elo_diff"], is_baseline=True, complexity=1,
                            description="logistic regression on the all-surface Elo difference"),
            logit_candidate("surface_logit", ["elo_diff", "surf_elo_diff"], complexity=2,
                            description="adds the surface-specific rating"),
            logit_candidate("full_logit", f, complexity=3,
                            description="adds serve/return strength, rest, recent load, experience and format"),
            gbm_candidate("full_gbm", f, complexity=6, description="gradient-boosted trees on all features"),
        ]

    def split_plan(self, league: str, market: str, frame: FeatureFrame) -> SplitPlan:
        sizes = frame.completed.groupby("block").size()
        typical = float(sizes.median()) if len(sizes) else 0.0
        return SplitPlan(n_final_blocks=2, min_train_blocks=3, n_calib_blocks=1,
                         burn_in_blocks=self._CONFIG.burn_in_years, min_block_events=max(50, int(0.4 * typical)))

    def required_checks(self, league: str, market: str) -> List[CheckSpec]:
        return [
            CheckSpec("injury_report", "withdrawal / fitness report", scope="team", max_age_hours=24.0,
                      recheck_within_hours=3.0, why="a player carrying an injury or about to withdraw is invisible to the model"),
            CheckSpec("order_of_play", "order of play published", scope="event", max_age_hours=24.0,
                      why="confirms the match, its court and that neither player has withdrawn"),
            CheckSpec("prior_games_complete", "both players' earlier matches are final", scope="event",
                      max_age_hours=1e9, why="ratings are only current once earlier matches are on file"),
        ]
