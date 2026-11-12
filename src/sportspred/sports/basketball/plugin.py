"""Basketball plugin: NBA, WNBA, NCAA men and women, plus CSV-only leagues."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ...core.enums import OutcomeType
from ...models.candidates import PriorCandidate, gbm_candidate, logit_candidate
from ...core.ids import norm_text
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
from .adapter import SDV_LEAGUES, ingest_sdv
from .features import FEATURE_LABELS, BasketballConfig, build_winner_frame, roster_quality

__all__ = ["BasketballPlugin"]

_SDV_SOURCE = "sportsdataverse-data release files (ESPN-derived schedules and box scores)"
_WINNER = MarketDecl("winner", OutcomeType.BINARY, True, includes_overtime=True,
                     note="game winner including overtime")
_SPREAD = MarketDecl("spread", OutcomeType.OVER_UNDER, False, includes_overtime=True,
                     note="point spread at an exact line")
_TOTAL = MarketDecl("total", OutcomeType.OVER_UNDER, False, includes_overtime=True,
                    note="total points at an exact line")
_PROP = MarketDecl("player_prop", OutcomeType.OVER_UNDER, False, note="player statistic over/under")


class BasketballPlugin(SportPlugin):
    sport = "basketball"
    leagues = (
        LeagueSpec("nba", "basketball", "NBA", gender="men", level="pro",
                   markets=(_WINNER, _SPREAD, _TOTAL, _PROP), real_data_source=_SDV_SOURCE,
                   live_feeds="ESPN site API; official NBA injury report"),
        LeagueSpec("wnba", "basketball", "WNBA", gender="women", level="pro",
                   markets=(_WINNER, _SPREAD, _TOTAL, _PROP), real_data_source=_SDV_SOURCE,
                   live_feeds="ESPN site API"),
        LeagueSpec("ncaam", "basketball", "NCAA Division I men", gender="men", level="college",
                   markets=(_WINNER, _SPREAD, _TOTAL), real_data_source=_SDV_SOURCE,
                   live_feeds="ESPN site API"),
        LeagueSpec("ncaaw", "basketball", "NCAA Division I women", gender="women", level="college",
                   markets=(_WINNER, _SPREAD, _TOTAL), real_data_source=_SDV_SOURCE,
                   live_feeds="ESPN site API"),
    )

    #: 40-minute games use 200 team minutes.
    _CONFIG: Dict[str, BasketballConfig] = {
        "nba": BasketballConfig(),
        "wnba": BasketballConfig(regulation_minutes=200.0),
        "ncaam": BasketballConfig(regulation_minutes=200.0),
        "ncaaw": BasketballConfig(regulation_minutes=200.0),
    }

    # ------------------------------------------------------------------ data
    def ingest(self, store: Any, league: str, *, seasons: Optional[Sequence[int]] = None,
               refresh: Optional[bool] = None, fetcher: Any = None) -> IngestReport:
        self.league(league)
        return ingest_sdv(store, league, seasons=seasons, refresh=refresh, fetcher=fetcher)

    # -------------------------------------------------------------- features
    def feature_frame(self, store: Any, league: str, market: str, *, now: Any = None) -> FeatureFrame:
        decl = self.league(league).market(market)
        if market != "winner" or not decl.implemented:
            raise NotImplementedError(f"{league} market {market!r} has no feature builder yet")
        return build_winner_frame(store, league, now=now, cfg=self._CONFIG.get(league))

    def candidates(self, league: str, market: str, frame: FeatureFrame) -> List[Candidate]:
        f = frame.feature_cols
        team = ["home_edge", "elo_diff", "net_slow_diff"]
        core = team + ["net_fast_diff", "rest_diff", "b2b_a", "b2b_b"]
        return [
            PriorCandidate("base_rate", by="home_edge",
                           description="home-win frequency (separately for neutral sites); knows nothing about the teams"),
            logit_candidate("elo_logit", ["home_edge", "elo_diff"], is_baseline=True, complexity=1,
                            description="logistic regression on the Elo difference and home court"),
            logit_candidate("efficiency_logit", team, complexity=2,
                            description="adds long-term opponent-adjusted efficiency"),
            logit_candidate("core_logit", core, complexity=3,
                            description="adds recent opponent-adjusted form and rest"),
            logit_candidate("full_logit", f, complexity=4,
                            description="all point-in-time features: four-factor matchups, expected rotation, schedule"),
            gbm_candidate("full_gbm", f, complexity=6,
                          description="gradient-boosted trees on all point-in-time features"),
            logit_candidate("oracle_rotation_logit", f + ["roster_q_actual_diff"], complexity=4,
                            uses_retrospective=True,
                            description="full_logit plus the rotation that actually played (not knowable pregame)"),
        ]

    def split_plan(self, league: str, market: str, frame: FeatureFrame) -> SplitPlan:
        cfg = self._CONFIG.get(league, BasketballConfig())
        per_season = frame.completed.groupby("block").size()
        typical = float(per_season.median()) if len(per_season) else 0.0
        return SplitPlan(
            n_final_blocks=2, min_train_blocks=3, n_calib_blocks=1,
            burn_in_blocks=cfg.burn_in_seasons, min_block_events=max(50, int(0.5 * typical)),
        )

    def required_checks(self, league: str, market: str) -> List[CheckSpec]:
        return [
            CheckSpec("injury_report", "injury / availability report", scope="team", max_age_hours=24.0,
                      recheck_within_hours=3.0,
                      why="who plays changes team strength more than any other pregame fact"),
            CheckSpec("prior_games_complete", "both teams' earlier games are final", scope="event",
                      max_age_hours=1e9,
                      why="ratings and rest are only current once each team's previous game has been played"),
        ]

    # ----------------------------------------------------------- availability
    #: Players below this many expected minutes are outside the rotation; an
    #: uncertain status for them does not make a selection conditional.
    ROTATION_MINUTES = 8.0

    def apply_availability(self, league: str, market: str, row: pd.Series, frame: FeatureFrame,
                           availability: pd.DataFrame) -> AvailabilityEffect:
        """Remove confirmed-out (and doubtful) players from the expected
        rotation, restore confirmed returns, and build the worst-case rows in
        which each side's questionable players also sit."""
        effect = AvailabilityEffect(row=row.copy())
        rosters = frame.extra.get("rosters") or {}
        rate = frame.extra.get("league_rate")
        reg = float(frame.extra.get("regulation_minutes") or 240.0)
        if rate is None or not rosters:
            return super().apply_availability(league, market, row, frame, availability)
        effect.quantified = True
        if availability is None or availability.empty:
            return effect
        quality: Dict[str, Dict[str, float]] = {}
        for side, team in (("a", row["side_a_id"]), ("b", row["side_b_id"])):
            players = rosters.get(team, [])
            by_id = {str(p["player_id"]): p for p in players}
            by_name = {norm_text(p["name"]): p for p in players}
            out, back, doubt = [], [], []
            for _, rep in availability[availability["team_id"] == team].iterrows():
                p = by_id.get(str(rep.get("player_id"))) or by_name.get(norm_text(rep["player_name"]))
                status = str(rep["status"])
                if p is None:
                    effect.unmatched.append(
                        f"{rep['player_name']} ({status}): not in the recent rotation, no effect on the model")
                    continue
                mins = p["exp_min"]
                if status in ("out", "doubtful"):
                    out.append(p["player_id"])
                    effect.applied.append(
                        f"{p['name']} {status}: treated as not playing ({mins:.0f} expected minutes removed)")
                elif status == "available":
                    if p["role_min"] > mins + 3.0:
                        back.append(p["player_id"])
                        effect.applied.append(
                            f"{p['name']} confirmed available: restored to a {p['role_min']:.0f}-minute role")
                elif status == "probable":
                    effect.applied.append(f"{p['name']} probable: treated as playing")
                elif mins >= self.ROTATION_MINUTES:
                    doubt.append(p["player_id"])
                    text = f"{p['name']} ({status}, {mins:.0f} expected minutes)"
                    (effect.uncertain_a if side == "a" else effect.uncertain_b).append(text)
            q, ab = roster_quality(players, rate, reg, out=out, back=back)
            wq, wab = roster_quality(players, rate, reg, out=out + doubt, back=back)
            quality[side] = {"q": q, "ab": ab, "wq": wq, "wab": wab}

        def with_values(qa: float, aa: float, qb: float, ab: float) -> pd.Series:
            new = row.copy()
            new["roster_q_a"], new["roster_q_b"] = qa, qb
            new["absence_a"], new["absence_b"] = aa, ab
            new["roster_q_diff"] = qa - qb
            new["absence_diff"] = aa - ab
            return new

        A, B = quality["a"], quality["b"]
        effect.row = with_values(A["q"], A["ab"], B["q"], B["ab"])
        effect.worst_case_rows = {
            "a": with_values(A["wq"], A["wab"], B["q"], B["ab"]),
            "b": with_values(A["q"], A["ab"], B["wq"], B["wab"]),
        }
        return effect

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
            ranked = sorted(contributions.items(), key=lambda kv: -abs(kv[1]))
            for feat, value in ranked:
                if abs(value) < 0.03:
                    continue
                label = FEATURE_LABELS.get(feat, feat)
                toward = value * sign
                text = f"{label} ({'favours' if toward > 0 else 'against'} {pick}, {abs(value):.2f} log-odds)"
                (exp.supporting if toward > 0 else exp.upset_risks).append(text)
            exp.supporting = exp.supporting[:5]
            exp.upset_risks = exp.upset_risks[:5]

        def fmt(v: Any, digits: int = 1) -> str:
            return "n/a" if v is None or v != v else f"{v:.{digits}f}"

        opp_elo = row["elo_b"] if pick_home else row["elo_a"]
        own_elo = row["elo_a"] if pick_home else row["elo_b"]
        exp.opponent_strength.append(
            f"{other} Elo {fmt(opp_elo, 0)} versus {pick} {fmt(own_elo, 0)} (league average 1500)")
        if row.get("net_slow_diff") == row.get("net_slow_diff"):
            exp.opponent_strength.append(
                f"long-term opponent-adjusted net efficiency gap: {fmt(sign * row['net_slow_diff'])} "
                f"points per 100 possessions in favour of {pick if sign * row['net_slow_diff'] > 0 else other}")
        if row.get("net_fast_diff") == row.get("net_fast_diff"):
            exp.opponent_strength.append(
                f"recent-form gap (about the last 7 games, opponent-adjusted): {fmt(sign * row['net_fast_diff'])}")
        own_b2b = row.get("b2b_a") if pick_home else row.get("b2b_b")
        if own_b2b == 1.0:
            exp.upset_risks.append(f"{pick} are on the second night of a back-to-back")
        if row.get("games_played_a", 99) < 10:
            exp.upset_risks.append("early season: ratings rest on few games and lean on last season")
        exp.availability_assumptions.append(
            "Assumes each team's rotation matches who has played in its recent games; "
            "no injury report was used unless a check below says otherwise.")
        if row.get("dq_no_roster") == 1.0:
            exp.missing_data.append("no recent player box scores: expected-rotation quality is missing")
        if row.get("dq_no_team_stats") == 1.0:
            exp.missing_data.append("no team box scores yet for one side: efficiency ratings are missing")
        for side, key in ((home, "travel_km_a"), (away, "travel_km_b")):
            if row.get(key) != row.get(key):
                exp.missing_data.append(f"travel distance unknown for {side}")
        return exp
