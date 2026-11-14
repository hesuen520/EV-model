"""Football plugin: NFL (winner, spread, total) and NCAA FBS (winner)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from ...core.enums import OutcomeType
from ...core.ids import norm_text
from ...models.candidates import PriorCandidate, gbm_candidate, logit_candidate
from ...sports import DataUnavailable
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
from .adapter import american_implied, ingest_nflverse
from .college import ingest_cfb
from .features import FEATURE_LABELS, MARKETS, FootballConfig, build_frame, expected_qb_values
from .models import EXPLAIN_KEYS, gbm_score_candidate, ridge_score_candidate

__all__ = ["FootballPlugin"]

_NFL_SOURCE = "nflverse-data release files (schedule, results, team and quarterback statistics)"
_CFB_SOURCE = "sportsdataverse-data cfb_schedules release files (schedules and results)"
_LINE_TIMING = ("undated - the source states neither the bookmaker nor whether the number is an opening "
                "or a closing one; a closing number would contain news from after any realistic prediction time")
_QB_ROLES = ("QB1", "QB", "STARTING_QB", "STARTER")


def _markets(with_lines: bool) -> tuple:
    why_not = "" if with_lines else "; no line history in the reachable source"
    return (
        MarketDecl("winner", OutcomeType.BINARY, True, includes_overtime=True,
                   note="game winner including overtime; a tied game has no winner"),
        MarketDecl("spread", OutcomeType.OVER_UNDER, with_lines, includes_overtime=True,
                   note="point spread at an exact line (the home side's handicap); a push is no bet" + why_not),
        MarketDecl("total", OutcomeType.OVER_UNDER, with_lines, includes_overtime=True,
                   note="total points at an exact line; a push is no bet" + why_not),
    )


class FootballPlugin(SportPlugin):
    sport = "football"
    leagues = (
        LeagueSpec("nfl", "football", "NFL", gender="men", level="pro", markets=_markets(True),
                   real_data_source=_NFL_SOURCE,
                   live_feeds="ESPN site API; official NFL injury report and inactives",
                   notes="1999 onwards; lines in the schedule file are undated"),
        LeagueSpec("ncaaf", "football", "NCAA Division I FBS", gender="men", level="college",
                   markets=_markets(False), real_data_source=_CFB_SOURCE, live_feeds="ESPN site API",
                   notes="games between two FBS teams; games against lower divisions are context only"),
    )
    #: An NFL season is about 270 games, so the update rates are tuned on four
    #: seasons (after one warm-up season) rather than two.
    _CONFIG: Dict[str, FootballConfig] = {"nfl": FootballConfig(burn_in_seasons=5), "ncaaf": FootballConfig()}

    # ------------------------------------------------------------------ data
    def ingest(self, store: Any, league: str, *, seasons: Optional[Sequence[int]] = None,
               refresh: Optional[bool] = None, fetcher: Any = None) -> IngestReport:
        self.league(league)
        if league == "nfl":
            return ingest_nflverse(store, league, seasons=seasons, refresh=refresh, fetcher=fetcher)
        if league == "ncaaf":
            return ingest_cfb(store, league, seasons=seasons, refresh=refresh, fetcher=fetcher)
        raise DataUnavailable(
            f"no public dataset adapter for football league {league!r}; import results with "
            "`sportspred import events`")

    # -------------------------------------------------------------- features
    def feature_frame(self, store: Any, league: str, market: str, *, now: Any = None) -> FeatureFrame:
        if not self.league(league).market(market).implemented:
            raise NotImplementedError(f"{league} market {market!r} has no feature builder yet")
        return build_frame(store, league, market, now=now, cfg=self._CONFIG.get(league))

    def candidates(self, league: str, market: str, frame: FeatureFrame) -> List[Candidate]:
        have = set(frame.feature_cols) | set(frame.retrospective_cols)
        pick = lambda *cols: [c for c in cols if c in have]  # noqa: E731
        full = list(frame.feature_cols)
        if market == "total":
            return self._total_candidates(frame, pick, full)
        ratings = pick("home_edge", "elo_diff", "pts_slow_diff", "epa_slow_diff")
        qb = pick("qb_inc_diff", "qb_inc_adj_diff")
        actual = pick("qb_act_diff", "qb_act_adj_diff")
        if market == "winner":
            out: List[Candidate] = [
                PriorCandidate("base_rate", by="home_edge",
                               description="home-win frequency (separately for neutral sites); knows nothing "
                                           "about the teams"),
                logit_candidate("elo_logit", ["home_edge", "elo_diff"], is_baseline=True, complexity=1,
                                description="logistic regression on the Elo difference and home field"),
                logit_candidate("ratings_logit", ratings, complexity=2,
                                description="adds long-term opponent-adjusted scoring and efficiency"),
            ]
            if qb:
                out.append(logit_candidate("qb_logit", ratings + qb, complexity=3,
                                           description="adds the incumbent quarterbacks' quality"))
            out += [
                logit_candidate("full_logit", full, complexity=4,
                                description="all point-in-time features: recent form, matchups, rest, travel"),
                gbm_candidate("full_gbm", full, complexity=6,
                              description="gradient-boosted trees on all point-in-time features"),
            ]
            if actual:
                out.append(logit_candidate(
                    "actual_qb_logit", ratings + actual, complexity=3, uses_retrospective=True,
                    description="ratings plus the quarterbacks who actually started (near-pregame "
                                "reconstruction: usually but not always known before kickoff)"))
            return out
        kw = self._score_kwargs(market, frame)
        out = [
            PriorCandidate("base_rate", description="how often the home side has covered; knows nothing "
                                                    "about the teams (the natural benchmark is 50%)"),
            ridge_score_candidate("elo_margin", ["home_edge", "elo_diff"], is_baseline=True, complexity=1,
                                  description="margin regression on the Elo difference and home field", **kw),
            ridge_score_candidate("ratings_margin", ratings, complexity=2,
                                  description="adds long-term opponent-adjusted scoring and efficiency", **kw),
        ]
        if qb:
            out.append(ridge_score_candidate("qb_margin", ratings + qb, complexity=3,
                                             description="adds the incumbent quarterbacks' quality", **kw))
        out += [
            ridge_score_candidate("full_margin", full, complexity=4,
                                  description="margin regression on all point-in-time features", **kw),
            gbm_score_candidate("gbm_margin", full, complexity=6,
                                description="gradient-boosted margin regression on all point-in-time features",
                                **kw),
        ]
        if actual:
            out.append(ridge_score_candidate(
                "actual_qb_margin", ratings + actual, complexity=3, uses_retrospective=True,
                description="ratings plus the quarterbacks who actually started (near-pregame reconstruction)",
                **kw))
        return out

    def _total_candidates(self, frame: FeatureFrame, pick: Any, full: List[str]) -> List[Candidate]:
        kw = self._score_kwargs("total", frame)
        ratings = pick("league_pts", "pts_total_exp")
        efficiency = ratings + pick("epa_slow_sum", "pace_sum", "qb_inc_sum")
        retro = list(frame.retrospective_cols)
        out: List[Candidate] = [
            PriorCandidate("base_rate", description="how often games have gone over; knows nothing about "
                                                    "the teams (the natural benchmark is 50%)"),
            ridge_score_candidate("ratings_total", ratings, is_baseline=True, complexity=1,
                                  description="total regression on league scoring and both sides' "
                                              "opponent-adjusted scoring", **kw),
        ]
        if len(efficiency) > len(ratings):
            out.append(ridge_score_candidate("efficiency_total", efficiency, complexity=2,
                                             description="adds efficiency, pace and the incumbent "
                                                         "quarterbacks", **kw))
        out += [
            ridge_score_candidate("full_total", full, complexity=4,
                                  description="total regression on all point-in-time features", **kw),
            gbm_score_candidate("gbm_total", full, complexity=6,
                                description="gradient-boosted total regression on all point-in-time features",
                                **kw),
        ]
        if retro:
            out.append(ridge_score_candidate(
                "weather_total", full + retro, complexity=4, uses_retrospective=True,
                description="full_total plus observed wind and cold, the roof as it was, and the actual "
                            "starting quarterbacks (not knowable at a realistic prediction time)", **kw))
        return out

    @staticmethod
    def _score_kwargs(market: str, frame: FeatureFrame) -> Dict[str, Any]:
        level = (frame.extra.get("rosters") or {}).get("level_rows")
        if market == "spread":
            return dict(target_col="margin", threshold_sign=-1.0, support=(-80, 80), symmetric=True,
                        extra_rows=level)
        return dict(target_col="total_points", threshold_sign=1.0, support=(0, 150), symmetric=False,
                    extra_rows=level)

    def split_plan(self, league: str, market: str, frame: FeatureFrame) -> SplitPlan:
        """Seasons are the blocks.  The last two full seasons plus the season
        in progress are the final test; a season counts as full once it has
        90% of a typical season's games."""
        cfg = self._CONFIG.get(league, FootballConfig())
        per_season = frame.completed.groupby("block").size()
        typical = float(per_season.median()) if len(per_season) else 0.0
        return SplitPlan(n_final_blocks=2, min_train_blocks=6, n_calib_blocks=2,
                         burn_in_blocks=cfg.burn_in_seasons, min_block_events=max(50, int(0.9 * typical)))

    def required_checks(self, league: str, market: str) -> List[CheckSpec]:
        qb_why = ("the quarterback is the largest single-player effect; the model assumes the incumbent starts "
                  "unless told otherwise" if league == "nfl" else
                  "the quarterback is the largest single-player effect and this league's model has no "
                  "quarterback information at all")
        checks = [
            CheckSpec("injury_report", "injury / availability report", scope="team", max_age_hours=24.0,
                      why="the final game-status report and the inactive list change who plays"),
            CheckSpec("starting_qb", "starting quarterback confirmed", scope="team", max_age_hours=24.0,
                      recheck_within_hours=2.0, why=qb_why),
        ]
        if league == "nfl":
            checks.append(CheckSpec(
                "weather", "weather forecast (log 'nothing_listed' for a fixed-roof stadium)", scope="event",
                max_age_hours=12.0, recheck_within_hours=6.0,
                why="wind and cold change scoring and passing in outdoor games; the model has no forecast"))
        checks.append(CheckSpec("prior_games_complete", "both teams' earlier games are final", scope="event",
                                max_age_hours=1e9,
                                why="ratings, rest and the incumbent quarterback are only current once each "
                                    "team's previous game has been played"))
        return checks

    # -------------------------------------------------------------- benchmark
    def market_benchmark(self, store: Any, league: str, market: str) -> Optional[pd.DataFrame]:
        """De-vigged prices from the schedule file, for comparison only.

        Winner: the two moneylines.  Spread and total: the two prices at the
        listed line, which sit close to 50% by construction - against a line
        the natural benchmark is a coin flip.
        """
        pairs = {"winner": ("away_moneyline", "home_moneyline"), "spread": ("away_spread_odds", "home_spread_odds"),
                 "total": ("under_odds", "over_odds")}
        if market not in pairs or not store.has_table(league, "games"):
            return None
        games = store.read_table(league, "games")
        low, high = pairs[market]
        if not {low, high} <= set(games.columns):
            return None
        p_low, p_high = american_implied(games[low]), american_implied(games[high])
        classes = MARKETS[market][1]
        out = pd.DataFrame({
            "event_id": games["event_id"],
            f"p_{classes[0]}": p_low / (p_low + p_high),          # proportional removal of the margin
            f"p_{classes[1]}": p_high / (p_low + p_high),
        }).dropna()
        out["benchmark_label"] = "de-vigged prices in the nflverse schedule file"
        out["benchmark_timing"] = _LINE_TIMING
        return out if len(out) else None

    # ----------------------------------------------------------- availability
    def apply_availability(self, league: str, market: str, row: pd.Series, frame: FeatureFrame,
                           availability: pd.DataFrame) -> AvailabilityEffect:
        """Replace the assumed starter (the incumbent) by a quarterback named
        as the starter, rate an unknown replacement as an inexperienced
        quarterback when the expected starter is out or doubtful, and build
        the worst-case rows in which a questionable starter sits."""
        table = (frame.extra.get("rosters") or {}).get("qb") or {}
        if not table or "qb_inc_a" not in row.index:
            return super().apply_availability(league, market, row, frame, availability)
        effect = AvailabilityEffect(row=row.copy(), quantified=True)
        if availability is None or availability.empty:
            return effect
        qbs, prior = table["qbs"], float(table["prior"])
        by_name = {norm_text(v["name"]): q for q, v in qbs.items()}

        def known(rep: Mapping[str, Any]) -> Optional[str]:
            pid = rep.get("player_id")
            return str(pid) if pid is not None and str(pid) in qbs else by_name.get(norm_text(rep["player_name"]))

        value: Dict[str, float] = {}
        worst: Dict[str, float] = {}
        for side, team in (("a", row["side_a_id"]), ("b", row["side_b_id"])):
            starter = table["teams"].get(team, {}).get("incumbent")
            starter_name = qbs.get(starter, {}).get("name", "the incumbent")
            value[side] = float(row[f"qb_inc_{side}"])
            reps = availability[availability["team_id"] == team].sort_values("collected_at_utc")
            named = reps[reps["role"].fillna("").str.upper().isin(_QB_ROLES)]
            if len(named):                      # the latest statement naming a starter
                rep = named.iloc[-1]
                starter, starter_name = known(rep), str(rep["player_name"])
                value[side] = qbs[starter]["rating"] if starter in qbs else prior
                note = "" if starter in qbs else " (no earlier games on file: rated as an inexperienced quarterback)"
                effect.applied.append(f"{starter_name} named as the starting quarterback ({rep['status']}){note}")
            # Statements are read in time order and the newest one about the
            # starter decides: a later confirmation clears an earlier "doubtful",
            # and a later "out" overrides an earlier confirmation.
            healthy = value[side]
            state, note = "plays", None
            for _, rep in reps.iterrows():
                text = f"{rep['player_name']} ({rep['status']})"
                if norm_text(rep["player_name"]) != norm_text(starter_name):
                    effect.unmatched.append(text + " - reported but not used by this model")
                elif rep["status"] in ("out", "doubtful"):
                    state, note = "out", f"{text}: an inexperienced replacement quarterback is assumed"
                elif rep["status"] in ("questionable", "uncertain"):
                    state, note = "doubt", f"{text}, quarterback"
                else:
                    state, note = "plays", None
            if state == "out":
                value[side] = prior
                effect.applied.append(note)
            elif state == "doubt":
                value[side], worst[side] = healthy, prior
                (effect.uncertain_a if side == "a" else effect.uncertain_b).append(note)
        effect.row = expected_qb_values(row, value["a"], value["b"])
        effect.worst_case_rows = {
            side: expected_qb_values(row, worst.get("a", value["a"]) if side == "a" else value["a"],
                                     worst.get("b", value["b"]) if side == "b" else value["b"])
            for side in worst
        }
        return effect

    # ------------------------------------------------------------ explanation
    def explain(self, league: str, market: str, row: pd.Series, frame: FeatureFrame,
                contributions: Optional[Mapping[str, float]] = None) -> Explanation:
        names = frame.extra.get("names", {})
        home = names.get(row["side_a_id"], "home side")
        away = names.get(row["side_b_id"], "away side")
        exp = Explanation()
        if market == "winner":
            self._explain_winner(exp, row, home, away, contributions)
        else:
            self._explain_line(exp, market, row, home, away, contributions)
        table = (frame.extra.get("rosters") or {}).get("qb") or {}
        for side, team in ((home, row["side_a_id"]), (away, row["side_b_id"])):
            inc = table.get("teams", {}).get(team, {}).get("incumbent")
            if inc:
                exp.availability_assumptions.append(
                    f"{side}: assumes {table['qbs'].get(inc, {}).get('name', inc)} (started the previous game) "
                    "starts at quarterback unless a statement below says otherwise")
        exp.availability_assumptions.append(
            "No injury report was used for any other position; the ratings reflect who has been playing."
            if table.get("teams") else
            "No player, quarterback or injury information is modelled; the ratings reflect results only.")
        if row.get("dq_no_qb_history") == 1.0:
            exp.missing_data.append("no earlier start on file for one side: quarterback features are missing")
        if row.get("dq_no_team_stats") == 1.0:
            exp.missing_data.append("no team statistics yet for one side: efficiency ratings are missing")
        if row.get("needs_weather") == 1.0:
            exp.missing_data.append("weather forecast is not a model input (the stadium is not a fixed dome)")
        if min(row.get("season_games_a", 99), row.get("season_games_b", 99)) < 4:
            exp.upset_risks.append("early season: ratings lean on last season and rosters have changed")
        return exp

    @staticmethod
    def _explain_winner(exp: Explanation, row: pd.Series, home: str, away: str,
                        contributions: Optional[Mapping[str, float]]) -> None:
        pick_home = bool(row.get("_pick_is_a", True))
        pick, other = (home, away) if pick_home else (away, home)
        sign = 1.0 if pick_home else -1.0
        for feat, value in sorted((contributions or {}).items(), key=lambda kv: -abs(kv[1])):
            if abs(value) < 0.03:
                continue
            toward = value * sign
            text = (f"{FEATURE_LABELS.get(feat, feat)} ({'favours' if toward > 0 else 'against'} {pick}, "
                    f"{abs(value):.2f} log-odds)")
            target = exp.supporting if toward > 0 else exp.upset_risks
            if len(target) < 5:
                target.append(text)
        own, opp = (row["elo_a"], row["elo_b"]) if pick_home else (row["elo_b"], row["elo_a"])
        exp.opponent_strength.append(f"{other} Elo {opp:.0f} versus {pick} {own:.0f} (league average 1500)")
        exp.opponent_strength.append(
            f"long-term opponent-adjusted scoring margin: {sign * row['pts_slow_diff']:+.1f} points per game "
            f"for {pick}; recent (about 7 games): {sign * row['pts_fast_diff']:+.1f}")
        if row.get("short_a" if pick_home else "short_b") == 1.0:
            exp.upset_risks.append(f"{pick} are on a short week")
        exp.upset_risks.append("a tied game (about 1 in 450) has no winner")

    @staticmethod
    def _explain_line(exp: Explanation, market: str, row: pd.Series, home: str, away: str,
                      contributions: Optional[Mapping[str, float]]) -> None:
        c = dict(contributions or {})
        unit = "home margin" if market == "spread" else "total"
        line = float(row["line"])
        if market == "spread":
            exp.supporting.append(f"the bet: {home} {line:+g} (home handicap) against {away}")
        else:
            exp.supporting.append(f"the bet: total points over or under {line:g}")
        if "_expected" in c:
            gap = c["_expected"] + line if market == "spread" else c["_expected"] - line
            side = (home if gap > 0 else away) if market == "spread" else ("over" if gap > 0 else "under")
            exp.supporting.append(
                f"model's expected {unit} {c['_expected']:+.1f}" if market == "spread"
                else f"model's expected {unit} {c['_expected']:.1f}")
            exp.supporting.append(f"that is {abs(gap):.1f} points on the {side} side of the line; a typical "
                                  f"game misses the model's number by about {c.get('_sd', float('nan')):.1f} points")
            exp.upset_risks.append(
                f"probability of a push (no bet) is about {100 * c.get('_p_push', 0.0):.1f}%; the stated "
                "probabilities are for the bet given that it does not push")
        for feat, value in sorted(((k, v) for k, v in c.items() if k not in EXPLAIN_KEYS),
                                  key=lambda kv: -abs(kv[1]))[:5]:
            if abs(value) >= 0.2:
                exp.supporting.append(f"{FEATURE_LABELS.get(feat, feat)}: moves the expected {unit} by "
                                      f"{value:+.1f} points")
        exp.opponent_strength.append(f"{home} Elo {row['elo_a']:.0f}, {away} Elo {row['elo_b']:.0f} "
                                     "(league average 1500)")
        exp.upset_risks.append("the model never sees the market; the line already contains information "
                               "(injuries, weather, starters) that this model does not have")
