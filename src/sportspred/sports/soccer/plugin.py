"""Soccer plugin: international football (men, women) and club leagues,
three-way (home / draw / away) market."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from ...core.enums import OutcomeType
from ...models.candidates import PriorCandidate, gbm_candidate, logit_candidate
from ...sports import DataUnavailable
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
from .clubs import CLUB_LEAGUES, ingest_club
from .features import SoccerConfig, build_three_way_frame
from .intl import INTL_SOURCES, ingest_international
from .models import DixonColesCandidate

__all__ = ["SoccerPlugin", "MARKET"]

MARKET = "winner_3way"
_INTL_MARKET = MarketDecl(
    MARKET, OutcomeType.THREE_WAY, True, includes_overtime=True,
    note="home / draw / away on the score after extra time where it was played (the source does not give "
         "90-minute scores); a match that went to penalties is a draw")
_CLUB_MARKET = MarketDecl(
    MARKET, OutcomeType.THREE_WAY, True, includes_overtime=False,
    note="home / draw / away at full time (league matches have no extra time)")
_INTL_SOURCE = "{repo} results.csv + shootouts.csv (GitHub raw)"
_CLUB_SOURCE = ("xgabora/Club-Football-Match-Data-2000-2025 Matches.csv (Football-Data.co.uk results) + "
                "openfootball/football.json current-season fixtures (GitHub raw)")
_NO_SOURCE_NOTE = ("no reachable public file with trustworthy dates was found; import results with "
                   "`sportspred import events` (CSV)")


def _club(key: str, name: str) -> LeagueSpec:
    return LeagueSpec(key, "soccer", name, gender="men", level="pro", markets=(_CLUB_MARKET,),
                      real_data_source=_CLUB_SOURCE)


class SoccerPlugin(SportPlugin):
    sport = "soccer"
    leagues = (
        LeagueSpec("soccer_intl_men", "soccer", "Men's international football", gender="men",
                   level="international", markets=(_INTL_MARKET,),
                   real_data_source=_INTL_SOURCE.format(repo="martj42/international_results"),
                   notes="date-only; the source also lists non-FIFA representative sides and their tournaments, which "
                         "are rated on the same scale; fixtures appear upstream only occasionally"),
        LeagueSpec("soccer_intl_women", "soccer", "Women's international football", gender="women",
                   level="international", markets=(_INTL_MARKET,),
                   real_data_source=_INTL_SOURCE.format(repo="martj42/womens-international-results"),
                   notes="upstream is incomplete: major tournaments are covered, many friendlies are missing"),
        _club("epl", "English Premier League"),
        _club("laliga", "Spanish La Liga"),
        _club("bundesliga", "German Bundesliga"),
        _club("seriea", "Italian Serie A"),
        _club("ligue1", "French Ligue 1"),
        LeagueSpec("mls", "soccer", "Major League Soccer", gender="men", level="pro", markets=(_CLUB_MARKET,),
                   notes=_NO_SOURCE_NOTE),
        LeagueSpec("nwsl", "soccer", "National Women's Soccer League", gender="women", level="pro",
                   markets=(_CLUB_MARKET,), notes=_NO_SOURCE_NOTE),
    )

    #: Blocks are calendar years for internationals and seasons for clubs.
    #: Men's ratings start in 1980 and women's in 1995 (the women's file is
    #: thin before that); the years up to 1993 / 2003 are warm-up and tuning.
    _CONFIG: Dict[str, SoccerConfig] = {
        "soccer_intl_men": SoccerConfig(international=True, history_start=1980, burn_in_blocks=14,
                                        warmup_blocks=6, min_train_blocks=5, max_dev_folds=12,
                                        fast_lr=0.25, slow_lr_grid=(0.03, 0.06, 0.1)),
        "soccer_intl_women": SoccerConfig(international=True, history_start=1995, burn_in_blocks=9,
                                          warmup_blocks=3, min_train_blocks=5, max_dev_folds=12,
                                          fast_lr=0.25, slow_lr_grid=(0.03, 0.06, 0.1)),
    }

    def _config(self, league: str) -> SoccerConfig:
        return self._CONFIG.get(league, SoccerConfig())

    # ------------------------------------------------------------------ data
    def ingest(self, store: Any, league: str, *, seasons: Optional[Sequence[int]] = None,
               refresh: Optional[bool] = None, fetcher: Any = None) -> IngestReport:
        self.league(league)
        if league in INTL_SOURCES:
            return ingest_international(store, league, seasons=seasons, refresh=refresh, fetcher=fetcher)
        if league in CLUB_LEAGUES:
            return ingest_club(store, league, seasons=seasons, refresh=refresh, fetcher=fetcher)
        raise DataUnavailable(f"no public dataset adapter for soccer league {league!r}: {_NO_SOURCE_NOTE}")

    # -------------------------------------------------------------- features
    def feature_frame(self, store: Any, league: str, market: str, *, now: Any = None) -> FeatureFrame:
        self.league(league).market(market)
        return build_three_way_frame(store, league, now=now, cfg=self._config(league))

    def candidates(self, league: str, market: str, frame: FeatureFrame) -> List[Candidate]:
        features = frame.extra["model_features"]
        return [
            PriorCandidate("base_rate", by="home_edge",
                           description="home / draw / away frequencies (separately for neutral venues); "
                                       "knows nothing about the teams"),
            logit_candidate("elo_logit", ["home_edge", "elo_diff"], is_baseline=True, complexity=1,
                            description="multinomial logistic regression on the Elo difference and home advantage"),
            DixonColesCandidate("poisson_dc", complexity=2,
                                description="Poisson goals model on attack/defence ratings and recent form "
                                            "with a fitted Dixon-Coles draw correction"),
            logit_candidate("ratings_logit", ["home_edge", "elo_diff", "goal_diff_exp", "goal_sum_exp"],
                            complexity=3, description="adds opponent-adjusted goal ratings to Elo"),
            logit_candidate("full_logit", features, complexity=4,
                            description="all point-in-time features: ratings, recent form, rest, competition "
                                        "or division context"),
            gbm_candidate("full_gbm", features, complexity=6,
                          description="gradient-boosted trees on all point-in-time features"),
        ]

    def split_plan(self, league: str, market: str, frame: FeatureFrame) -> SplitPlan:
        cfg = self._config(league)
        per_block = frame.completed.groupby("block").size()
        typical = float(per_block.median()) if len(per_block) else 0.0
        return SplitPlan(
            n_final_blocks=2, min_train_blocks=cfg.min_train_blocks, n_calib_blocks=1,
            burn_in_blocks=cfg.burn_in_blocks, min_block_events=max(50, int(0.5 * typical)),
            max_dev_folds=cfg.max_dev_folds,
        )

    def required_checks(self, league: str, market: str) -> List[CheckSpec]:
        return [
            CheckSpec("lineup", "confirmed starting line-ups", scope="team", max_age_hours=3.0,
                      recheck_within_hours=1.5,
                      why="line-ups are announced about an hour before kick-off; rotation and late "
                          "withdrawals are invisible to the model until then"),
            CheckSpec("injury_report", "injuries, suspensions and call-ups", scope="team", max_age_hours=36.0,
                      why="the model has no player information at all, so known absences must be reviewed"),
            CheckSpec("prior_games_complete", "both sides' earlier matches have results on file", scope="event",
                      max_age_hours=1e9,
                      why="ratings and rest are only current once each side's previous match is in the data"),
        ]

    def market_benchmark(self, store: Any, league: str, market: str) -> Optional[pd.DataFrame]:
        """Bet365 home/draw/away odds from the club results file, with the
        bookmaker margin removed proportionally.  Comparison only."""
        if not store.has_table(league, "odds_1x2"):
            return None
        odds = store.read_table(league, "odds_1x2")
        odds = odds[(odds[["odd_home", "odd_draw", "odd_away"]] > 1.0).all(axis=1)]
        implied = 1.0 / odds[["odd_home", "odd_draw", "odd_away"]].to_numpy(dtype=float)
        implied /= implied.sum(axis=1, keepdims=True)
        return pd.DataFrame({
            "event_id": odds["event_id"].to_numpy(),
            "p_home": implied[:, 0], "p_draw": implied[:, 1], "p_away": implied[:, 2],
            "benchmark_label": "bookmaker 1X2 price in the Football-Data.co.uk file (via xgabora): Bet365 in most "
                               "seasons, apparently a multi-bookmaker average in 2019/20-2024/25; margin removed "
                               "proportionally",
            "benchmark_timing": "pre-match; the file does not record when the quote was taken (not marked closing)",
        })

    @staticmethod
    def _explain_draw(exp: Explanation, row: pd.Series, home: str, away: str) -> Explanation:
        """A draw selection is supported by the sides being close and the game
        being low scoring; anything one-sided is a risk to it."""
        elo_gap = float(row["elo_diff"])
        exp.opponent_strength.append(
            f"Elo: {home} {row['elo_a']:.0f}, {away} {row['elo_b']:.0f} (1500 is average; home advantage not included)")
        (exp.supporting if abs(elo_gap) < 60 else exp.upset_risks).append(
            f"Elo gap of {abs(elo_gap):.0f} points " + ("- the sides are closely matched" if abs(elo_gap) < 60
                                                        else f"favours {home if elo_gap > 0 else away}"))
        if row.get("goal_sum_exp") == row.get("goal_sum_exp"):
            low = float(row["goal_sum_exp"]) < 2.4
            (exp.supporting if low else exp.upset_risks).append(
                f"goal ratings expect about {float(row['goal_sum_exp']):.1f} goals in total "
                + ("(low-scoring games are drawn more often)" if low else "(open games are drawn less often)"))
        if row.get("home_edge") == 1.0:
            exp.upset_risks.append(f"home advantage favours {home}")
        exp.upset_risks.append("a draw is the least likely single outcome in most matches, even when it is the "
                               "model's most likely one here")
        exp.availability_assumptions.append(
            "No line-up, injury or suspension information is in the model: both sides are assumed to field "
            "the strength their recent results reflect.")
        exp.missing_data.append("expected goals (xG) and player-level data are not available to this model")
        return exp

    # ------------------------------------------------------------ explanation
    def explain(self, league: str, market: str, row: pd.Series, frame: FeatureFrame,
                contributions: Optional[Mapping[str, float]] = None) -> Explanation:
        exp = Explanation()
        names = frame.extra.get("names", {})
        home = names.get(row["side_a_id"], "home side")
        away = names.get(row["side_b_id"], "away side")
        pick_home = bool(row.get("_pick_is_a", True))
        pick = home if pick_home else away
        if row.get("_pick_class") == "draw":
            return self._explain_draw(exp, row, home, away)

        def known(key: str) -> bool:
            return row.get(key) == row.get(key) and row.get(key) is not None

        def side_with(value: float, text: str) -> None:
            """File a fact under the side it favours (positive = home side)."""
            favoured = home if value > 0 else away
            (exp.supporting if favoured == pick else exp.upset_risks).append(f"{text} favours {favoured}")

        elo_gap = float(row["elo_diff"])
        exp.opponent_strength.append(
            f"Elo: {home} {row['elo_a']:.0f}, {away} {row['elo_b']:.0f} (1500 is average; home advantage not included)")
        if abs(elo_gap) >= 25:
            side_with(elo_gap, f"results against comparable opposition (Elo gap {abs(elo_gap):.0f})")
        if known("goal_diff_exp"):
            exp.opponent_strength.append(
                f"goal ratings expect about {max(row['rating_goals_a'], 0):.1f} goals for {home} and "
                f"{max(row['rating_goals_b'], 0):.1f} for {away}")
            if abs(row["goal_diff_exp"]) >= 0.15:
                side_with(row["goal_diff_exp"], "long-term attack and defence, adjusted for opponents,")
            if abs(row["form_diff"]) >= 0.15:
                side_with(row["form_diff"], "recent form relative to long-term level")
        if row["home_edge"] == 1.0:
            side_with(1.0, "playing at home")
        else:
            exp.opponent_strength.append("neutral venue: no home advantage applied")
        if known("sot_edge") and abs(row["sot_edge"]) >= 0.5:
            side_with(row["sot_edge"], "shots on target for and against, adjusted for opponents,")
        if known("rest_diff") and abs(row["rest_diff"]) >= 2:
            side_with(row["rest_diff"], f"{abs(row['rest_diff']):.0f} more days of rest")
        for flag, side in (("promoted_a", home), ("promoted_b", away)):
            if row.get(flag) == 1.0:
                exp.upset_risks.append(
                    f"{side} are new to the division this season: their rating starts near average and is uncertain")
        if row.get("is_friendly") == 1.0:
            exp.upset_risks.append("friendly: sides often rotate and results are less predictable")
        exp.upset_risks.append("the draw is a separate outcome: a selection of either side loses if the match is drawn")
        if row.get("matches_a", 99) < 10:
            exp.upset_risks.append(f"{home} have few rated matches; ratings lean on the starting value")
        exp.availability_assumptions.append(
            "No line-up, injury or suspension information is in the model: both sides are assumed to field "
            "the strength their recent results reflect.")
        if row.get("dq_unrated_team") == 1.0:
            exp.missing_data.append("one side has no earlier match on file: attack/defence ratings are missing")
        if row.get("n_pending_prior", 0):
            exp.missing_data.append(
                f"{int(row['n_pending_prior'])} earlier match(es) of these sides have no result on file; "
                "ratings do not include them")
        exp.missing_data.append("expected goals (xG) and player-level data are not available to this model")
        return exp
