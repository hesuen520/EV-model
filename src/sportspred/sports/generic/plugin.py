"""Leagues without a sport-specific plugin: a results-only rating model that
runs on whatever is imported through the documented CSV import.

None of these leagues has a reachable real data feed in this installation, so
each is ``adapter-only`` until results are imported and validated.  Every key
is one competition level, gender and discipline - they are never mixed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from ...core.enums import OutcomeType
from ...models.candidates import PriorCandidate, logit_candidate
from .. import DataUnavailable
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
from .features import FEATURE_LABELS, GenericOptions, build_generic_frame

__all__ = ["GenericPlugin", "ResultsOnlyPlugin", "GENERIC_LEAGUES"]

_ESPORTS_NOTE = ("Map vetoes, drafts and starting sides are unknown before a match and are never used; "
                 "only roster-level results are rated. Roster changes are not tracked: re-key a team after a rebuild.")
_TENNIS_NOTE = ("Doubles has no reachable data: results-only rating of pairs via CSV import. Singles leagues "
                "are handled by the tennis plugin. Walkovers must not be imported as results.")
_MULTI_NOTE = "Field events need a finish-distribution model that is not implemented; results can be stored only."

#: key -> (sport, name, gender, level, discipline, options, notes)
GENERIC_LEAGUES: Dict[str, Tuple[str, str, str, str, Optional[str], GenericOptions, str]] = {
    "cricket_t20_men": ("cricket", "Men's T20 cricket", "men", "pro", "t20", GenericOptions(), ""),
    "cricket_t20_women": ("cricket", "Women's T20 cricket", "women", "pro", "t20", GenericOptions(), ""),
    "cricket_odi_men": ("cricket", "Men's one-day cricket", "men", "pro", "odi", GenericOptions(), ""),
    "cricket_test_men": ("cricket", "Men's Test cricket", "men", "pro", "test", GenericOptions(draws=True),
                         "Draws are a regular outcome: three-way market."),
    "rugby_union_men": ("rugby", "Men's rugby union", "men", "pro", "union", GenericOptions(), ""),
    "rugby_union_women": ("rugby", "Women's rugby union", "women", "pro", "union", GenericOptions(), ""),
    "rugby_league_men": ("rugby", "Men's rugby league", "men", "pro", "league", GenericOptions(), ""),
    "volleyball_men": ("volleyball", "Men's volleyball", "men", "pro", None, GenericOptions(), ""),
    "volleyball_women": ("volleyball", "Women's volleyball", "women", "pro", None, GenericOptions(), ""),
    "badminton_ms": ("badminton", "Badminton men's singles", "men", "pro", "singles", GenericOptions(home_adv=False), ""),
    "badminton_ws": ("badminton", "Badminton women's singles", "women", "pro", "singles", GenericOptions(home_adv=False), ""),
    "badminton_md": ("badminton", "Badminton men's doubles", "men", "pro", "doubles", GenericOptions(home_adv=False), ""),
    "badminton_wd": ("badminton", "Badminton women's doubles", "women", "pro", "doubles", GenericOptions(home_adv=False), ""),
    "badminton_xd": ("badminton", "Badminton mixed doubles", "mixed", "pro", "doubles", GenericOptions(home_adv=False), ""),
    "tabletennis_ms": ("table_tennis", "Table tennis men's singles", "men", "pro", "singles", GenericOptions(home_adv=False), ""),
    "tabletennis_ws": ("table_tennis", "Table tennis women's singles", "women", "pro", "singles", GenericOptions(home_adv=False), ""),
    "atp_doubles": ("tennis", "ATP Tour doubles", "men", "tour", "doubles", GenericOptions(home_adv=False), _TENNIS_NOTE),
    "wta_doubles": ("tennis", "WTA Tour doubles", "women", "tour", "doubles", GenericOptions(home_adv=False), _TENNIS_NOTE),
    "ufc": ("mma", "UFC", "open", "pro", None, GenericOptions(home_adv=False, season_carry=1.0),
            "Draws and no-contests have no winner and are not rows. Style, reach, age and weight class are not modelled yet."),
    "boxing_men": ("boxing", "Men's professional boxing", "men", "pro", None, GenericOptions(home_adv=False, season_carry=1.0), ""),
    "boxing_women": ("boxing", "Women's professional boxing", "women", "pro", None, GenericOptions(home_adv=False, season_carry=1.0), ""),
    "cs2": ("esports", "Counter-Strike 2", "open", "pro", None, GenericOptions(home_adv=False), _ESPORTS_NOTE),
    "valorant": ("esports", "Valorant", "open", "pro", None, GenericOptions(home_adv=False), _ESPORTS_NOTE),
    "lol": ("esports", "League of Legends", "open", "pro", None, GenericOptions(home_adv=False), _ESPORTS_NOTE),
    "dota2": ("esports", "Dota 2", "open", "pro", None, GenericOptions(home_adv=False), _ESPORTS_NOTE),
    "basketball_intl_men": ("basketball", "International men's basketball", "men", "international", None, GenericOptions(), ""),
    "basketball_intl_women": ("basketball", "International women's basketball", "women", "international", None, GenericOptions(), ""),
}
#: Multi-competitor competitions: declared so the coverage table is complete and honest.
FIELD_LEAGUES: Dict[str, Tuple[str, str]] = {
    "pga_tour": ("golf", "PGA Tour"), "nascar_cup": ("motorsport", "NASCAR Cup Series"),
    "indycar": ("motorsport", "IndyCar"), "motogp": ("motorsport", "MotoGP"),
}


class ResultsOnlyPlugin(SportPlugin):
    """Shared behaviour of plugins built on the results-only feature builder."""

    def options(self, league: str) -> GenericOptions:
        raise NotImplementedError

    def market_key(self, league: str) -> str:
        return "winner_3way" if self.options(league).draws else "winner"

    def feature_frame(self, store: Any, league: str, market: str, *, now: Any = None) -> FeatureFrame:
        if not self.league(league).market(market).implemented:
            raise NotImplementedError(f"{league} market {market!r} has no model")
        return build_generic_frame(store, league, market, now=now, opts=self.options(league))

    def candidates(self, league: str, market: str, frame: FeatureFrame) -> List[Candidate]:
        base = [c for c in ("home_edge", "elo_diff") if c in frame.feature_cols]
        out: List[Candidate] = [
            PriorCandidate("base_rate", by="home_edge" if "home_edge" in frame.feature_cols else None,
                           description="outcome frequencies; knows nothing about the competitors"),
            logit_candidate("elo_logit", base, is_baseline=True, complexity=1,
                            description="logistic regression on the Elo difference"),
        ]
        if len(frame.feature_cols) > len(base):
            out.append(logit_candidate("ratings_logit", frame.feature_cols, complexity=3,
                                       description="adds opponent-adjusted scoring, recent form, rest and experience"))
        return out

    def split_plan(self, league: str, market: str, frame: FeatureFrame) -> SplitPlan:
        sizes = frame.completed.groupby("block").size()
        typical = float(sizes.median()) if len(sizes) else 0.0
        return SplitPlan(n_final_blocks=1, min_train_blocks=2, n_calib_blocks=1,
                         burn_in_blocks=self.options(league).burn_in_blocks,
                         min_block_events=max(20, int(0.4 * typical)))

    def required_checks(self, league: str, market: str) -> List[CheckSpec]:
        return [
            CheckSpec("injury_report", "availability / line-up report", scope="team", max_age_hours=24.0,
                      recheck_within_hours=3.0, why="this model sees results only; who competes must be checked"),
            CheckSpec("prior_games_complete", "both sides' earlier events are final", scope="event",
                      max_age_hours=1e9, why="ratings are only current once earlier events are played"),
        ]

    def explain(self, league: str, market: str, row: pd.Series, frame: FeatureFrame,
                contributions: Optional[Mapping[str, float]] = None) -> Explanation:
        exp = Explanation()
        names = frame.extra.get("names", {})
        a, b = names.get(row["side_a_id"], row["side_a_id"]), names.get(row["side_b_id"], row["side_b_id"])
        pick_a = bool(row.get("_pick_is_a", True))
        pick, other, sign = (a, b, 1.0) if pick_a else (b, a, -1.0)
        for feat, value in sorted((contributions or {}).items(), key=lambda kv: -abs(kv[1])):
            if abs(value) < 0.03:
                continue
            toward = value * sign
            text = f"{FEATURE_LABELS.get(feat, feat)} ({'favours' if toward > 0 else 'against'} {pick}, {abs(value):.2f} log-odds)"
            (exp.supporting if toward > 0 else exp.upset_risks).append(text)
        own, opp = (row["elo_a"], row["elo_b"]) if pick_a else (row["elo_b"], row["elo_a"])
        exp.opponent_strength.append(f"{other} Elo {opp:.0f} versus {pick} {own:.0f} (average 1500)")
        if row.get("net_slow_diff") == row.get("net_slow_diff"):
            exp.opponent_strength.append(
                f"long-term opponent-adjusted scoring margin gap: {sign * row['net_slow_diff']:+.2f} for {pick}")
        if row.get("dq_few_events") == 1.0:
            exp.upset_risks.append("few rated events for one side: the rating is mostly prior")
        exp.availability_assumptions.append(
            "Assumes both sides field what they fielded in their recent results; nothing about line-ups is known.")
        exp.missing_data.append("results-only model: no sport-specific statistics, venue or line-up information")
        return exp


def _spec(key: str) -> LeagueSpec:
    sport, name, gender, level, discipline, opts, note = GENERIC_LEAGUES[key]
    market = (MarketDecl("winner_3way", OutcomeType.THREE_WAY, True, note="win / draw / loss")
              if opts.draws else MarketDecl("winner", OutcomeType.BINARY, True, note="match winner"))
    return LeagueSpec(key, sport, name, gender=gender, level=level, discipline=discipline, markets=(market,),
                      real_data_source="", notes=note)


class GenericPlugin(ResultsOnlyPlugin):
    sport = "generic"
    leagues = tuple(_spec(k) for k in GENERIC_LEAGUES) + tuple(
        LeagueSpec(key, sport, name, gender="open", level="pro",
                   markets=(MarketDecl("win", OutcomeType.MULTI, False, note="outright winner"),
                            MarketDecl("top3", OutcomeType.MULTI, False, note="podium / top-three finish")),
                   real_data_source="", notes=_MULTI_NOTE)
        for key, (sport, name) in FIELD_LEAGUES.items())

    def options(self, league: str) -> GenericOptions:
        return GENERIC_LEAGUES[league][5]

    def ingest(self, store: Any, league: str, *, seasons: Optional[Sequence[int]] = None,
               refresh: Optional[bool] = None, fetcher: Any = None) -> IngestReport:
        self.league(league)
        raise DataUnavailable(
            f"{league} has no reachable public feed in this installation. Import results with "
            f"`sportspred import events --league {league} --csv <file>` (see docs/CSV_IMPORT.md).")
