"""SYNTHETIC demo leagues.  Isolated from real data by the ``demo_`` key
prefix; their coverage status can never be better than ``demo-only``."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ...adapters.synthetic import SYNTHETIC_BANNER, generate_demo_league
from ...core.enums import OutcomeType
from ..base import IngestReport, LeagueSpec, MarketDecl
from ..generic.features import GenericOptions
from ..generic.plugin import ResultsOnlyPlugin

__all__ = ["PLUGIN", "DemoPlugin"]

_OPTIONS = {"demo_basketball": GenericOptions(home_adv=True), "demo_duel": GenericOptions(home_adv=False)}
_WINNER = MarketDecl("winner", OutcomeType.BINARY, True, note="winner (synthetic)")


class DemoPlugin(ResultsOnlyPlugin):
    sport = "demo"
    leagues = (
        LeagueSpec("demo_basketball", "demo", "SYNTHETIC demo league (home advantage)", gender="open",
                   level="synthetic", markets=(_WINNER,), real_data_source="", notes=SYNTHETIC_BANNER),
        LeagueSpec("demo_duel", "demo", "SYNTHETIC demo league (no home side)", gender="open",
                   level="synthetic", markets=(_WINNER,), real_data_source="", notes=SYNTHETIC_BANNER),
    )

    def options(self, league: str) -> GenericOptions:
        return _OPTIONS[league]

    def ingest(self, store: Any, league: str, *, seasons: Optional[Sequence[int]] = None,
               refresh: Optional[bool] = None, fetcher: Any = None, now: Any = None) -> IngestReport:
        self.league(league)
        return generate_demo_league(store, league, now=now, home_adv=_OPTIONS[league].home_adv)


PLUGIN = DemoPlugin()
