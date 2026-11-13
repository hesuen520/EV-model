"""Player statistic over/under markets for basketball.

One row per player-game.  Three things are modelled, as a threshold market
needs: **playing time** (expected minutes), **opportunity and production per
minute** (the player's own rates, the opponent's tendency to allow the
statistic, and the production vacated by absent team-mates) and the **outcome
distribution** (a count distribution around the expected value, so that the
probability of finishing above, below or exactly on any threshold follows).

Point-in-time rule: every input of a row comes from games on earlier
Pacific/Honolulu days - a player and a team each play at most once a day, and
league averages are shifted by a day - so nothing from a row's own slate, let
alone its own game, can reach it.

**What the validation can and cannot say.**  No history of bookmakers'
thresholds is reachable, so completed rows are scored against a *reference
threshold* computed from the player's own earlier games (the half point just
above the median of his previous 15 appearances).  Skill and calibration
against that threshold show the distribution is informative and honest about
its spread.  They do not show an edge against a market threshold, which
embeds information this model does not have.  The validation report carries
a gate that says so, and the market cannot be labelled validated until real
thresholds have been collected and scored.

Rows are games the player appeared in: a statistic market is void when the
player does not play, so the model is conditional on playing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ...core.enums import OutcomeType
from ...core.ids import person_key
from ...core.timeutil import DISPLAY_TZ_NAME, parse_ts, utcnow
from ..base import FeatureFrame
from .features import EXCLUDED_STAGES, _pregame_reports

__all__ = ["STATS", "STAT_LABELS", "PropConfig", "prop_market", "split_prop_market", "is_prop_market",
           "build_prop_frame", "MINUTES_FEATURES", "rate_features", "report_swap", "candidate_players",
           "LINES_GATE", "prop_candidates", "apply_prop_availability", "explain_prop", "threshold_sweep",
           "SWEEP_SHIFTS"]

#: Statistic -> box-score columns summed to give it.
STATS: Dict[str, Tuple[str, ...]] = {
    "points": ("pts",), "rebounds": ("reb",), "assists": ("ast",), "threes": ("fg3m",),
    "pra": ("pts", "reb", "ast"),
}
STAT_LABELS = {"points": "points", "rebounds": "rebounds", "assists": "assists",
               "threes": "three-pointers made", "pra": "points + rebounds + assists"}
_RAW = ("minutes", "starter", "pts", "reb", "ast", "fg3m", "fga", "fta", "fg3a")
_EXPECTED_ABSENT = ("out", "doubtful")

LINES_GATE = {
    "gate": "market_thresholds", "passed": False,
    "detail": "scored against reference thresholds built from each player's earlier games; no history of "
              "bookmakers' thresholds is on file, so performance against a real market line is untested",
}


def prop_market(stat: str, player_id: Optional[str] = None) -> str:
    """``player_prop:<stat>`` (the modelled market) or, with a player, the
    ledger key of one player's market ``player_prop:<stat>:<player id>``."""
    return f"player_prop:{stat}" + (f":{player_id}" if player_id is not None else "")


def split_prop_market(market: str) -> Tuple[Optional[str], Optional[str]]:
    """``(stat, player id)`` of a prop market key; ``(None, None)`` otherwise."""
    parts = str(market).split(":")
    if parts[0] != "player_prop" or len(parts) < 2:
        return None, None
    return parts[1], (parts[2] if len(parts) > 2 else None)


def is_prop_market(market: str) -> bool:
    return split_prop_market(market)[0] is not None


@dataclass(frozen=True)
class PropConfig:
    first_season: int = 2008           # earlier seasons are not used
    regulation_minutes: float = 240.0  # team minutes in a regulation game
    min_prior_games: int = 8           # appearances needed before a player-game is modelled
    min_expected_minutes: float = 15.0 # only rotation players are modelled
    ref_window: int = 15               # appearances behind the reference threshold
    minutes_halflife: float = 6.0      # in appearances
    rate_halflife: float = 10.0
    long_halflife: float = 40.0
    opponent_halflife: float = 15.0    # in team games
    league_halflife: float = 45.0      # in game days
    rotation_window: int = 10          # team games a player stays in the rotation after his last appearance
    rotation_minutes: float = 12.0
    new_absence_games: int = 2         # an absence is "new" while he has missed at most this many
    rate_prior_minutes: float = 3.0    # league-rate minutes added to every per-minute rate (keeps it positive)
    box_check_games: int = 2           # a box score missing from this many latest games holds a prediction back
    horizon_days: float = 8.0          # upcoming games further ahead get no rows
    burn_in_seasons: int = 1
    starter_minutes: float = 28.0      # boundary of the two reported segments


MINUTES_FEATURES = ["min_ew", "min_long", "min_last3", "min_season", "starter_ew", "rest_days", "b2b",
                    "own_missed", "gp_season", "is_home", "abs_margin_gap", "mates_out_min", "mates_new_out_min"]
_OPPORTUNITY = {"points": ["log_fga_pm", "log_fta_pm", "log_fg3a_pm"], "threes": ["log_fg3a_pm"],
                "pra": ["log_fga_pm", "log_fta_pm"], "rebounds": [], "assists": []}


def rate_features(stat: str) -> List[str]:
    return (["log_rate_ew", "log_rate_long", "log_opp_ratio", "is_home", "b2b", "starter_ew",
             "mates_out_share", "mates_new_out_share"] + _OPPORTUNITY[stat])


#: Inputs that have a version built from the official injury report as published.
_REPORT_SWAP = {"mates_out_min": "mates_out_min_rep", "mates_new_out_min": "mates_new_out_min_rep",
                "mates_out_share": "mates_out_share_rep", "mates_new_out_share": "mates_new_out_share_rep"}


def report_swap(features: Sequence[str]) -> List[str]:
    return [_REPORT_SWAP.get(c, c) for c in features]


FEATURE_NOTES = {
    "min_ew": "minutes per appearance, recent (exponentially weighted)",
    "min_long": "minutes per appearance, longer run",
    "min_last3": "minutes over the last three appearances",
    "min_season": "minutes per appearance this season",
    "starter_ew": "share of recent appearances as a starter",
    "rest_days": "days since the player's last appearance (capped at 10)",
    "b2b": "second day of a back-to-back for the player",
    "own_missed": "team games the player missed immediately before this one",
    "gp_season": "appearances this season",
    "is_home": "playing at home",
    "abs_margin_gap": "gap between the two teams' recent scoring margins (lopsided games shorten minutes)",
    "mates_out_min": "minutes of rotation team-mates who missed the team's previous game",
    "mates_new_out_min": "the same, for team-mates whose absence began in the last two games",
    "log_rate_ew": "the statistic per minute, recent (log)",
    "log_rate_long": "the statistic per minute, longer run (log)",
    "log_opp_ratio": "how much of the statistic the opponent has allowed relative to the league (log)",
    "mates_out_share": "share of a typical team total produced by rotation team-mates who missed the previous game",
    "mates_new_out_share": "the same, for absences that began in the last two games",
    "log_fga_pm": "field-goal attempts per minute, recent (log)",
    "log_fta_pm": "free-throw attempts per minute, recent (log)",
    "log_fg3a_pm": "three-point attempts per minute, recent (log)",
    "mates_out_min_rep": "minutes of rotation team-mates the official report lists as out or doubtful",
    "mates_new_out_min_rep": "the same, for absences that are new",
    "mates_out_share_rep": "share of a typical team total produced by team-mates the report lists as out or doubtful",
    "mates_new_out_share_rep": "the same, for absences that are new",
    "avg_season": "the statistic per appearance this season",
    "pg_ew": "the statistic per appearance, recent",
    "rate_ew": "the statistic per minute, recent",
}


# --------------------------------------------------------------------- helpers
def _hold_within_day(frame: pd.DataFrame, key: str, values: Any) -> Any:
    """Pregame values of a second game on one day are those of the first.

    A player or team almost never plays twice on one Pacific/Honolulu day, but
    when one does, the later game must not see the earlier one: every input
    of a slate comes from earlier days.  Rows that are not the group's first
    of their day take that first row's values."""
    later = frame.duplicated([key, "day"], keep="first").to_numpy()
    if not later.any():
        return values
    first = frame.loc[~later, [key, "day"]].reset_index().rename(columns={"index": "_first"})
    again = frame.loc[later, [key, "day"]].reset_index().rename(columns={"index": "_later"})
    pairs = again.merge(first, on=[key, "day"], how="left")
    out = values.copy()
    out.loc[pairs["_later"].to_numpy()] = out.loc[pairs["_first"].to_numpy()].to_numpy()
    return out


def _ew(frame: pd.DataFrame, key: str, cols: Sequence[str], halflife: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Exponentially weighted means of ``cols`` within each ``key`` group, in
    row order.  Returns ``(prior, post)``: the mean over the group's rows on
    earlier days, and the mean including the row itself.  Rows with missing
    values (games not played yet) are skipped and carry the last value forward."""
    post = frame.groupby(key, sort=False)[list(cols)].ewm(halflife=halflife, ignore_na=True).mean()
    post.index = post.index.droplevel(0)
    post = post.reindex(frame.index)
    prior = post.groupby(frame[key], sort=False).shift(1)
    return _hold_within_day(frame, key, prior), post


def _rolling_prior(frame: pd.DataFrame, key: str, col: str, window: int, min_periods: int, how: str) -> pd.Series:
    """Rolling mean or median of ``col`` over the group's *played* rows on
    earlier days.  A row not played yet takes the value as it stood after the
    last game played."""
    done = frame[frame["played"]]
    roll = done.groupby(key, sort=False)[col].rolling(window, min_periods=min_periods)
    post = roll.median() if how == "median" else roll.mean()        # includes the row itself
    post.index = post.index.droplevel(0)
    post = post.reindex(done.index)
    out = pd.Series(np.nan, index=frame.index)
    out.loc[done.index] = post.groupby(done[key], sort=False).shift(1)
    last = post.groupby(done[key], sort=False).tail(1)
    latest = pd.Series(last.to_numpy(), index=done.loc[last.index, key].to_numpy())
    ahead = frame.index[~frame["played"].to_numpy()]
    out.loc[ahead] = frame.loc[ahead, key].map(latest).to_numpy()
    return _hold_within_day(frame, key, out)


_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}


def _cache_key(store: Any, league: str, ev: pd.DataFrame, now_ts: pd.Timestamp, cfg: PropConfig) -> Tuple[Any, ...]:
    """Identifies what a build depends on: the files, and how ``now`` splits
    the events into played and upcoming.  Two instants between the same two
    games give the same frame."""
    stamps = []
    for name in ("player_box", "pregame_report", "pregame_report_games"):
        path = store._table_path(league, name)
        stamps.append(path.stat().st_mtime_ns if path.exists() else None)
    done = ev[ev["done"]]
    ahead = ev[~ev["done"]]
    # Which report versions exist as of ``now``: counted, not rounded to a clock hour
    # (most versions are published at half past).
    reports_cut = None
    if stamps[1] is not None:
        now_iso = now_ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        listed = store.read_table(league, "pregame_report", columns=["published_at_utc"])
        reports_cut = int((listed["published_at_utc"] <= now_iso).sum())
        if stamps[2] is not None:
            games = store.read_table(league, "pregame_report_games", columns=["published_at_utc"])
            reports_cut = (reports_cut, int((games["published_at_utc"] <= now_iso).sum()))
    return (str(store.root), league, cfg, tuple(stamps), int(len(done)),
            str(done["start_time_utc"].max()) if len(done) else "", str(done["status"].count()),
            hash(tuple(ahead["event_id"])), str(pd.util.hash_pandas_object(
                ev[["event_id", "status", "home_score", "away_score"]].astype(str), index=False).sum()), reports_cut)


def _base(store: Any, league: str, now_ts: pd.Timestamp, cfg: PropConfig) -> Dict[str, Any]:
    """Everything that does not depend on the statistic, built once and shared
    by the five markets (cached for the latest ``now``)."""
    if not store.has_table(league, "player_box"):
        raise ValueError(f"no player box scores on file for {league}; run `sportspred ingest --league {league}`")
    ev = store.events(league, with_results=True)
    ev = ev[~ev["stage"].isin(EXCLUDED_STAGES)]
    ev = ev[pd.to_numeric(ev["season"], errors="coerce") >= cfg.first_season]
    done = (ev["status"] == "final") & ev["winner_id"].notna() & (ev["start_time_utc"] < now_ts)
    ahead = (ev["status"].isin(["scheduled", "final"]) & (ev["start_time_utc"] >= now_ts)
             & (ev["start_time_utc"] <= now_ts + pd.Timedelta(days=cfg.horizon_days)))
    ev = ev[done | ahead].sort_values(["start_time_utc", "event_id"]).reset_index(drop=True)
    if ev.empty:
        raise ValueError(f"no events on file for {league}; run `sportspred ingest --league {league}` first")
    ev["done"] = ((ev["status"] == "final") & ev["winner_id"].notna() & (ev["start_time_utc"] < now_ts)).to_numpy()
    ev["season"] = pd.to_numeric(ev["season"], errors="coerce").astype(int)
    local = ev["start_time_utc"].dt.tz_convert(DISPLAY_TZ_NAME).dt.tz_localize(None).dt.normalize()
    ev["day"] = (local - pd.Timestamp("2000-01-01")).dt.days.astype(int)
    key = _cache_key(store, league, ev, now_ts, cfg)
    if key in _CACHE:
        return _CACHE[key]
    neutral = ev["neutral_site"].fillna(0).astype(int).to_numpy() == 1

    # ---- team-games: two rows per event
    def side(team: str, opp: str, home: bool) -> pd.DataFrame:
        scored = ev["home_score" if home else "away_score"].astype(float)
        allowed = ev["away_score" if home else "home_score"].astype(float)
        return pd.DataFrame({
            "event_id": ev["event_id"], "start_time_utc": ev["start_time_utc"], "season": ev["season"],
            "day": ev["day"], "done": ev["done"], "team_id": ev[team], "opp_id": ev[opp],
            "is_home": np.where(neutral, 0.0, 1.0 if home else 0.0),
            "margin": np.where(ev["done"], scored - allowed, np.nan),
        })

    tg = pd.concat([side("home_id", "away_id", True), side("away_id", "home_id", False)], ignore_index=True)
    tg = tg.sort_values(["start_time_utc", "event_id", "team_id"]).reset_index(drop=True)

    # ---- player-games that were played, and candidate rows for upcoming games
    box = store.read_table(league, "player_box", columns=["event_id", "team_id", "player_id", "player_name", *_RAW])
    box = box[box["event_id"].isin(set(ev.loc[ev["done"], "event_id"]))]
    box = box[pd.to_numeric(box["minutes"], errors="coerce") > 0].copy()
    box["player_id"] = box["player_id"].astype(str)
    box = box.drop_duplicates(["event_id", "player_id"], keep="last")
    for col in _RAW:
        box[col] = pd.to_numeric(box[col], errors="coerce").astype(float)
    played = box.merge(tg[["event_id", "team_id", "opp_id", "is_home", "start_time_utc", "season", "day"]],
                       on=["event_id", "team_id"], how="inner")
    played["played"] = True
    last = (played.sort_values(["start_time_utc", "event_id"]).groupby("player_id", sort=False).tail(1)
            [["player_id", "player_name", "team_id", "season"]])
    upcoming = tg.loc[~tg["done"], ["event_id", "team_id", "opp_id", "is_home", "start_time_utc", "season", "day"]]
    cand = upcoming.merge(last, on=["team_id", "season"], how="inner")
    cand["played"] = False
    pg = pd.concat([played, cand], ignore_index=True)
    pg = pg.sort_values(["start_time_utc", "event_id", "player_id"]).reset_index(drop=True)
    pg["pra"] = pg["pts"] + pg["reb"] + pg["ast"]
    stat_cols = ["pts", "reb", "ast", "fg3m", "pra"]

    g = pg.groupby("player_id", sort=False)
    is_played = pg["played"].astype(float)
    pg["gp_prior"] = _hold_within_day(pg, "player_id", g["played"].cumsum() - is_played)
    season_key = pg["player_id"] + "|" + pg["season"].astype(str)
    gs = pg.groupby(season_key, sort=False)
    pg["gp_season"] = _hold_within_day(pg, "player_id", gs["played"].cumsum() - is_played)

    fast_prior, fast_post = _ew(pg, "player_id", ["minutes", "starter"], cfg.minutes_halflife)
    rate_prior, rate_post = _ew(pg, "player_id", ["minutes", "fga", "fta", "fg3a", *stat_cols], cfg.rate_halflife)
    long_prior, _ = _ew(pg, "player_id", ["minutes", *stat_cols], cfg.long_halflife)
    pg["min_ew"], pg["starter_ew"] = fast_prior["minutes"], fast_prior["starter"]
    pg["min_long"] = long_prior["minutes"]
    pg["min_last3"] = _rolling_prior(pg, "player_id", "minutes", 3, 2, "mean")
    filled = pg["minutes"].fillna(0.0)
    season_sum = _hold_within_day(pg, "player_id", filled.groupby(season_key, sort=False).cumsum() - filled)
    pg["min_season"] = (season_sum / pg["gp_season"]).where(pg["gp_season"] > 0)

    # Days since his previous game.  For a game two ahead that is the upcoming game before it:
    # he is assumed to play it, which is what the schedule says and all that can be known.
    rest = _hold_within_day(pg, "player_id", (pg["day"] - g["day"].shift(1)).clip(upper=10.0))
    pg["rest_days"] = rest
    pg["b2b"] = (rest == 1).astype(float).where(rest.notna())

    # ---- team and opponent context
    totals = played.assign(pra=played["pts"] + played["reb"] + played["ast"]).groupby(
        ["event_id", "team_id"], sort=False)[stat_cols].sum().reset_index()
    tg = tg.merge(totals, on=["event_id", "team_id"], how="left")
    allowed = totals.rename(columns={"team_id": "opp_id", **{c: f"allowed_{c}" for c in stat_cols}})
    tg = tg.merge(allowed, on=["event_id", "opp_id"], how="left")
    allowed_cols = [f"allowed_{c}" for c in stat_cols]
    # What a row's inputs rest on: earlier games of the season that are not final yet, and
    # recent finished games whose box score is not on file (a result can arrive before it).
    tg["has_box"] = tg[stat_cols[0]].notna()
    not_final = (~tg["done"]).astype(float)
    team_season = tg["team_id"] + "|" + tg["season"].astype(str)
    tg["pending_before"] = _hold_within_day(
        tg, "team_id", not_final.groupby(team_season, sort=False).cumsum() - not_final)
    no_box = (tg["done"] & ~tg["has_box"]).astype(float)
    recent = no_box.groupby(tg["team_id"], sort=False).rolling(cfg.box_check_games, min_periods=1).sum()
    recent.index = recent.index.droplevel(0)
    recent = recent.reindex(tg.index).groupby(tg["team_id"], sort=False).shift(1).fillna(0.0)
    tg["box_missing_before"] = _hold_within_day(tg, "team_id", recent)
    opp_prior, _ = _ew(tg, "team_id", ["margin", *allowed_cols], cfg.opponent_halflife)
    tg["margin_ew"] = opp_prior["margin"]
    by_day = tg.groupby("day")[stat_cols].mean()                      # NaN on days with no finished game
    league_prior = by_day.ewm(halflife=cfg.league_halflife, ignore_na=True).mean().shift(1)
    league_on = league_prior.reindex(tg["day"]).reset_index(drop=True)
    for c in stat_cols:
        tg[f"league_{c}"] = league_on[c].to_numpy()
        tg[f"opp_allow_ratio_{c}"] = (opp_prior[f"allowed_{c}"] / tg[f"league_{c}"]).to_numpy()

    own = tg[["event_id", "team_id", "margin_ew", "pending_before", "box_missing_before",
              *[f"league_{c}" for c in stat_cols]]]
    pg = pg.merge(own, on=["event_id", "team_id"], how="left")
    opp = tg[["event_id", "team_id", "margin_ew", "pending_before", "box_missing_before",
              *[f"opp_allow_ratio_{c}" for c in stat_cols]]].rename(
        columns={"team_id": "opp_id", "margin_ew": "opp_margin_ew", "pending_before": "opp_pending_before",
                 "box_missing_before": "opp_box_missing_before"})
    pg = pg.merge(opp, on=["event_id", "opp_id"], how="left")
    pg["abs_margin_gap"] = (pg["margin_ew"] - pg["opp_margin_ew"]).abs()
    pg["n_pending_prior"] = (pg["pending_before"].fillna(0) + pg["opp_pending_before"].fillna(0)).astype(int)
    pg["n_box_missing"] = (pg["box_missing_before"].fillna(0) + pg["opp_box_missing_before"].fillna(0)).astype(int)

    # ---- per-minute rates, kept positive by a few minutes at the league rate
    k = cfg.rate_prior_minutes
    den_fast, den_long = rate_prior["minutes"], long_prior["minutes"]
    add: Dict[str, Any] = {}
    for c in stat_cols:
        league_pm = pg[f"league_{c}"] / cfg.regulation_minutes
        add[f"rate_ew_{c}"] = rate_prior[c] / den_fast
        add[f"pg_ew_{c}"] = rate_prior[c]
        add[f"log_rate_ew_{c}"] = np.log((rate_prior[c] + league_pm * k) / (den_fast + k))
        add[f"log_rate_long_{c}"] = np.log((long_prior[c] + league_pm * k) / (den_long + k))
        add[f"log_opp_ratio_{c}"] = np.log(pg[f"opp_allow_ratio_{c}"].clip(lower=0.5, upper=2.0))
        cur = pg[c].fillna(0.0)
        season_total = _hold_within_day(pg, "player_id", cur.groupby(season_key, sort=False).cumsum() - cur)
        add[f"avg_season_{c}"] = (season_total / pg["gp_season"]).where(pg["gp_season"] > 0)
        add[f"ref_median_{c}"] = _rolling_prior(pg, "player_id", c, cfg.ref_window, cfg.min_prior_games, "median")
    for c in ("fga", "fta", "fg3a"):
        typical = 0.2                                                 # only sets the size of the floor
        add[f"log_{c}_pm"] = np.log((rate_prior[c] + typical * k) / (den_fast + k))

    # ---- team-mates expected to be absent (previous game; official report where one is on file)
    reports, covered = _pregame_reports(store, league, now_ts)
    out = _absences(pg, tg, fast_post["minutes"].to_numpy(), rate_post[stat_cols].to_numpy(), reports, covered, cfg)
    rotation = out.pop("_rotation")
    for name, values in out.items():
        if not name.startswith("_"):
            add[name] = values
    for c in stat_cols:
        typical_total = pg[f"league_{c}"].to_numpy()
        for tag in ("", "_rep"):
            add[f"mates_out_share_{c}{tag}"] = out[f"_out_{c}{tag}"] / typical_total
            add[f"mates_new_out_share_{c}{tag}"] = out[f"_new_out_{c}{tag}"] / typical_total
    pg = pd.concat([pg, pd.DataFrame(add, index=pg.index)], axis=1)

    # Whether the league has a report history at all - not whether one was published by
    # ``now``: a frame has the same columns whenever it is built (before the first report
    # the report-based inputs simply equal the previous-game ones).
    result = {"pg": pg, "has_reports": covered is not None, "rotation": rotation,
              "n_events": int(ev["done"].sum()), "covered": covered or set()}
    _CACHE.clear()
    _CACHE[key] = result
    return result


def _absences(pg: pd.DataFrame, tg: pd.DataFrame, min_post: np.ndarray, stat_post: np.ndarray,
              reports: Dict[Tuple[str, str], List[Tuple[str, str]]], covered: Optional[set],
              cfg: PropConfig) -> Dict[str, np.ndarray]:
    """Per player-game: what rotation team-mates expected to be absent usually
    contribute - minutes and each statistic per game - and how many team games
    the player himself has just missed.

    A team-mate is *expected absent* when he missed the team's previous game
    (all seasons) or, in the ``_rep`` version, when the official report in
    force lists him as out or doubtful (where a report covers the game;
    elsewhere the two versions are equal).  The player's own entry is never
    counted against himself.  State is updated when a day ends, so a row sees
    games on earlier days only - also in the rare case of two games on one day.
    """
    n, k = len(pg), stat_post.shape[1]
    names = ["mates_out_min", "mates_new_out_min"]
    res = {f"{name}{tag}": np.full(n, np.nan) for name in names for tag in ("", "_rep")}
    out_stat = {tag: np.full((n, k), np.nan) for tag in ("", "_rep")}
    new_stat = {tag: np.full((n, k), np.nan) for tag in ("", "_rep")}
    own_missed = np.full(n, np.nan)

    rows_of = pg.groupby(["event_id", "team_id"], sort=False).indices
    player = pg["player_id"].to_numpy()
    keys = np.array([person_key(v) for v in pg["player_name"].fillna("").to_numpy()], dtype=object)
    played = pg["played"].to_numpy()
    state: Dict[str, Dict[str, Any]] = {}
    team_of: Dict[str, str] = {}
    zero = np.zeros(k)
    waiting: List[Tuple[str, Any, str]] = []          # games of the current day, applied when the day ends

    def apply_day() -> None:
        for team, idx, start in waiting:
            st = state[team]
            games = st["games"]
            for i in idx:
                if not played[i]:
                    continue
                pid = player[i]
                old = team_of.get(pid)
                if old is not None and old != team and old in state:
                    state[old]["players"].pop(pid, None)
                team_of[pid] = team
                st["players"][pid] = (games, float(min_post[i]), stat_post[i].copy(), keys[i], start)
            st["games"] += 1
        waiting.clear()

    starts = tg["start_time_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    today = None
    for event_id, team, season, done, start, day in zip(tg["event_id"], tg["team_id"], tg["season"], tg["done"],
                                                        starts, tg["day"]):
        if day != today:
            apply_day()
            today = day
        st = state.get(team)
        if st is None or st["season"] != season:
            st = state[team] = {"season": season, "games": 0, "players": {}}
        idx = rows_of.get((event_id, team))
        if idx is None:
            # A finished game with no box score on file tells nothing about who played: it is
            # not counted as a game anyone missed (the row carries ``n_box_missing`` instead).
            continue
        games = st["games"]
        listed = dict(reports.get((event_id, team), ())) if covered is not None and event_id in covered else None
        sums = {"": [0.0, zero.copy(), 0.0, zero.copy()], "_rep": [0.0, zero.copy(), 0.0, zero.copy()]}
        counted: Dict[str, Dict[str, Tuple[float, np.ndarray, bool]]] = {"": {}, "_rep": {}}
        for pid, rec in st["players"].items():
            missed = games - 1 - rec[0]
            if missed > cfg.rotation_window or rec[1] < cfg.rotation_minutes:
                continue
            new = missed <= cfg.new_absence_games
            prev_absent = missed >= 1
            rep_absent = prev_absent if listed is None else listed.get(rec[3]) in _EXPECTED_ABSENT
            for tag, absent in (("", prev_absent), ("_rep", rep_absent)):
                if not absent:
                    continue
                s = sums[tag]
                s[0] += rec[1]
                s[1] += rec[2]
                if new:
                    s[2] += rec[1]
                    s[3] += rec[2]
                counted[tag][pid] = (rec[1], rec[2], new)
        for i in idx:
            pid = player[i]
            rec = st["players"].get(pid)
            if rec is not None:
                own_missed[i] = min(games - 1 - rec[0], cfg.rotation_window)
            for tag in ("", "_rep"):
                s, mine = sums[tag], counted[tag].get(pid)
                minutes, stats, new_minutes, new_stats = s[0], s[1], s[2], s[3]
                if mine is not None:
                    minutes, stats = minutes - mine[0], stats - mine[1]
                    if mine[2]:
                        new_minutes, new_stats = new_minutes - mine[0], new_stats - mine[1]
                res[f"mates_out_min{tag}"][i] = minutes
                res[f"mates_new_out_min{tag}"][i] = new_minutes
                out_stat[tag][i] = stats
                new_stat[tag][i] = new_stats
        if done:
            waiting.append((team, idx, start))
    apply_day()

    res["own_missed"] = own_missed
    # Each team's rotation as it stands after its latest game: what a statement published
    # from now on is applied to.
    names = pg["player_name"].to_numpy()
    name_of = {pid: nm for pid, nm in zip(player, names)}
    rotation: Dict[str, Dict[str, Any]] = {}
    for team, st in state.items():
        rotation[team] = {"season": int(st["season"]), "players": [
            {"player_id": pid, "key": rec[3], "name": name_of.get(pid, pid), "min": float(rec[1]),
             "stats": [float(v) for v in rec[2]], "missed": int(st["games"] - 1 - rec[0]),
             "last_played_utc": rec[4]}
            for pid, rec in st["players"].items()
            if st["games"] - 1 - rec[0] <= cfg.rotation_window and rec[1] >= cfg.rotation_minutes]}
    res["_rotation"] = rotation  # type: ignore[assignment]
    stat_cols = ["pts", "reb", "ast", "fg3m", "pra"]
    for tag in ("", "_rep"):
        for j, c in enumerate(stat_cols):
            res[f"_out_{c}{tag}"] = out_stat[tag][:, j]
            res[f"_new_out_{c}{tag}"] = new_stat[tag][:, j]
    return res


# ----------------------------------------------------------------------- frame
def _stat_column(stat: str) -> str:
    cols = STATS[stat]
    return cols[0] if len(cols) == 1 else "pra"


def build_prop_frame(store: Any, league: str, stat: str, *, now: Any = None,
                     cfg: Optional[PropConfig] = None) -> FeatureFrame:
    if stat not in STATS:
        raise KeyError(f"unknown player statistic {stat!r}; one of {sorted(STATS)}")
    cfg = cfg or PropConfig()
    now_ts = pd.Timestamp(utcnow() if now is None else parse_ts(now))
    base = _base(store, league, now_ts, cfg)
    pg, c = base["pg"], _stat_column(stat)

    keep = (pg["gp_prior"] >= cfg.min_prior_games) & (pg["min_ew"] >= cfg.min_expected_minutes) \
        & pg[f"ref_median_{c}"].notna()
    src = pg[keep]
    ref_line = np.floor(src[f"ref_median_{c}"].to_numpy()) + 0.5
    value = src[c].to_numpy(dtype=float)
    is_played = src["played"].to_numpy()
    df = pd.DataFrame({
        "event_id": (src["event_id"] + "|" + src["player_id"]).to_numpy(),
        "game_id": src["event_id"].to_numpy(),
        "start_time_utc": src["start_time_utc"].to_numpy(),
        "block": src["season"].to_numpy(),
        "y": np.where(is_played, (value > ref_line).astype(float), np.nan),
        "side_a_id": src["team_id"].to_numpy(), "side_b_id": src["opp_id"].to_numpy(),
        "player_id": src["player_id"].to_numpy(), "player_name": src["player_name"].to_numpy(),
        "ledger_market": [prop_market(stat, p) for p in src["player_id"]],
        "line": np.where(is_played, ref_line, np.nan), "ref_line": ref_line,
        "stat_value": np.where(is_played, value, np.nan),
        "minutes_played": np.where(is_played, src["minutes"].to_numpy(dtype=float), np.nan),
        "league_total": src[f"league_{c}"].to_numpy(dtype=float),
        "n_pending_prior": src["n_pending_prior"].to_numpy(dtype=int),
        "n_box_missing": src["n_box_missing"].to_numpy(dtype=int),
        "segment": np.where(src["min_ew"].to_numpy() >= cfg.starter_minutes,
                            f"expected {cfg.starter_minutes:g} minutes or more",
                            f"expected {cfg.min_expected_minutes:g} to {cfg.starter_minutes:g} minutes"),
    })
    df["start_time_utc"] = pd.to_datetime(df["start_time_utc"], utc=True)
    generic = list(dict.fromkeys(MINUTES_FEATURES + ["starter_ew", "is_home", "b2b"]))
    for name in generic + ["log_fga_pm", "log_fta_pm", "log_fg3a_pm"]:
        df[name] = src[name].to_numpy(dtype=float)
    rename = {f"log_rate_ew_{c}": "log_rate_ew", f"log_rate_long_{c}": "log_rate_long",
              f"log_opp_ratio_{c}": "log_opp_ratio", f"mates_out_share_{c}": "mates_out_share",
              f"mates_new_out_share_{c}": "mates_new_out_share", f"avg_season_{c}": "avg_season",
              f"pg_ew_{c}": "pg_ew", f"rate_ew_{c}": "rate_ew"}
    for old, new in rename.items():
        df[new] = src[old].to_numpy(dtype=float)
    feature_cols = list(dict.fromkeys(MINUTES_FEATURES + rate_features(stat) + ["avg_season", "pg_ew", "rate_ew"]))
    has_reports = bool(base["has_reports"])
    if has_reports:
        for name in ("mates_out_min_rep", "mates_new_out_min_rep"):
            df[name] = src[name].to_numpy(dtype=float)
        df["mates_out_share_rep"] = src[f"mates_out_share_{c}_rep"].to_numpy(dtype=float)
        df["mates_new_out_share_rep"] = src[f"mates_new_out_share_{c}_rep"].to_numpy(dtype=float)
        df["report_on_file"] = src["event_id"].isin(base["covered"]).to_numpy().astype(float)
        feature_cols += list(_REPORT_SWAP.values())

    # Upcoming rows are priced only at a threshold that has been imported for them.
    lines = store.prop_lines_as_of(league, now_ts, stat=stat) if hasattr(store, "prop_lines_as_of") else None
    if lines is not None and len(lines):
        on_file = dict(zip(lines["event_id"] + "|" + lines["player_id"].astype(str), lines["line"].astype(float)))
        imported = df["event_id"].map(on_file)
        df["line"] = np.where(df["y"].isna(), imported, df["line"])
    df = df.sort_values(["start_time_utc", "event_id"]).reset_index(drop=True)

    label = STAT_LABELS[stat]
    limitations = [
        f"No history of bookmakers' thresholds is on file. Completed games are scored against a reference "
        f"threshold: the half point just above the median {label} of the player's previous {cfg.ref_window} "
        "appearances. Skill and calibration against it show the distribution is informative; they do not show "
        "an edge against a market threshold, which reflects information this model lacks.",
        "The model is conditional on the player appearing (a statistic market is void when he does not play). "
        "A player who leaves a game early still counts, as he does when the market settles.",
        f"Only rotation players are modelled: at least {cfg.min_prior_games} earlier appearances and "
        f"{cfg.min_expected_minutes:g} recent minutes per appearance.",
        "Announced minutes limits, role changes and confirmed starting line-ups are not in the data.",
        "A new prediction needs a threshold imported for that player and game; without one there is no selection.",
    ]
    if has_reports:
        limitations.append(
            "Team-mates' absences come in two versions: who missed the team's previous game (all seasons) and "
            "the official injury report as it stood an hour before the day's first game (from 2018-12-17). "
            "Development folds decide which one the model uses.")
    return FeatureFrame(
        league=league, market=prop_market(stat), outcome=OutcomeType.OVER_UNDER, classes=("under", "over"),
        df=df, feature_cols=feature_cols, feature_notes={k: v for k, v in FEATURE_NOTES.items() if k in feature_cols},
        limitations=limitations,
        extra={"stat": stat, "stat_label": label, "requires_line": True, "gates": [dict(LINES_GATE)],
               "confidence_caveat": "not rated: hit rates were measured against reference thresholds built from "
                                    "players' earlier games, not against market thresholds",
               "config": {"first_season": cfg.first_season, "min_prior_games": cfg.min_prior_games,
                          "min_expected_minutes": cfg.min_expected_minutes, "ref_window": cfg.ref_window},
               "burn_in_seasons": cfg.burn_in_seasons, "has_reports": has_reports,
               "rotation": base["rotation"], "stat_index": ["pts", "reb", "ast", "fg3m", "pra"].index(c),
               "rotation_window": cfg.rotation_window, "new_absence_games": cfg.new_absence_games},
    )


def candidate_players(store: Any, league: str, *, now: Any = None, cfg: Optional[PropConfig] = None) -> pd.DataFrame:
    """Players a threshold can be imported for: the modelled rotation of every
    team with a game inside the horizon (``event_id, team_id, player_id,
    player_name, player_key``)."""
    cfg = cfg or PropConfig()
    now_ts = pd.Timestamp(utcnow() if now is None else parse_ts(now))
    pg = _base(store, league, now_ts, cfg)["pg"]
    up = pg[~pg["played"]]
    out = up[["event_id", "team_id", "player_id", "player_name", "start_time_utc", "min_ew", "gp_prior"]].copy()
    out["player_key"] = [person_key(v) for v in out["player_name"].fillna("")]
    out["modelled"] = (out["gp_prior"] >= cfg.min_prior_games) & (out["min_ew"] >= cfg.min_expected_minutes)
    return out.reset_index(drop=True)


# ------------------------------------------------------------------ candidates
def prop_candidates(stat: str, frame: FeatureFrame) -> List[Any]:
    """Baselines first, then models in increasing complexity."""
    from ...models.candidates import PriorCandidate
    from ...models.counts import ColumnMean, CountCandidate, ProductMean, TwoStageMean

    label = STAT_LABELS[stat]
    minutes, rate = list(MINUTES_FEATURES), rate_features(stat)
    out: List[Any] = [
        PriorCandidate("base_rate", description="how often players finish over the reference threshold; knows "
                                                "nothing about the player"),
        CountCandidate("season_average_poisson", ColumnMean(["avg_season", "pg_ew"]), dist="poisson",
                       is_baseline=True, complexity=1,
                       description=f"the player's {label} per game this season with a Poisson spread - a "
                                   "projected average and nothing else"),
        CountCandidate("recent_form_nb", ProductMean("min_ew", "rate_ew"), dist="nb", complexity=2,
                       description="recent minutes times recent production per minute, with a spread learned "
                                   "from history (negative binomial)"),
        CountCandidate("two_stage_glm", TwoStageMean(minutes, rate, kind="glm"), dist="nb", complexity=4,
                       description="a minutes model and an opponent-adjusted per-minute model (log-linear), "
                                   "negative-binomial spread"),
        CountCandidate("two_stage_gbm", TwoStageMean(minutes, rate, kind="gbm"), dist="nb", complexity=6,
                       description="the same two stages with gradient-boosted trees"),
    ]
    if frame.extra.get("has_reports"):
        out += [
            CountCandidate("report_glm", TwoStageMean(report_swap(minutes), report_swap(rate), kind="glm"),
                           dist="nb", complexity=5,
                           description="two_stage_glm with team-mates' absences taken from the official injury "
                                       "report as it stood an hour before the day's first game"),
            CountCandidate("report_gbm", TwoStageMean(report_swap(minutes), report_swap(rate), kind="gbm"),
                           dist="nb", complexity=7,
                           description="two_stage_gbm with the report-based absences"),
        ]
    return out


# ---------------------------------------------------------------- availability
_UNCERTAIN = ("questionable", "uncertain", "probable")
_DOUBT = ("questionable", "uncertain")


def apply_prop_availability(row: pd.Series, frame: FeatureFrame, availability: pd.DataFrame, effect: Any) -> Any:
    """Fold published availability into one player's row.

    * **The player himself.**  Listed out: no selection can be made (the
      market would be void).  Doubtful, questionable or probable: any selection
      is conditional on him playing, and limited minutes are not quantified.
    * **Team-mates, when the model reads the report-based inputs and the
      team's report has been read** (an injury-report check is on file for the
      team): rotation team-mates listed out or doubtful replace those inputs -
      the rule the history was built with, including "report read, nobody
      listed".  Questionable team-mates are assumed to play; the worst-case
      row assumes they sit.
    * **Team-mates, otherwise** (the model takes absences from who missed the
      previous game, or only stray statements are on file): nothing is
      changed, and every listed team-mate the inputs do not already reflect is
      a stated condition with an unquantified effect.  A statement never
      silently counts as applied.
    * A statement that matches nobody in the rotation is listed as such.
    * The opponent's absences are not an input of this model.

    The engine supplies ``frame.extra["model_features"]`` (what the trained
    model reads) and ``frame.extra["report_checked_teams"]`` (teams whose
    injury report has been read).
    """
    effect.quantified = True
    team = row["side_a_id"]
    checked = frame.extra.get("report_checked_teams")
    report_read = checked is not None and team in checked
    statements = availability if availability is not None else pd.DataFrame(columns=["team_id", "player_name", "status"])
    mine = statements[statements["team_id"] == team] if len(statements) else statements
    theirs = statements[statements["team_id"] == row["side_b_id"]] if len(statements) else statements
    if len(theirs):
        listed = "; ".join(f"{r['player_name']} ({r['status']})" for _, r in theirs.iterrows()
                           if str(r["status"]) != "available")
        if listed:
            effect.unmatched.append(f"opponent absences are not an input of this model: {listed}")
    if mine.empty and not report_read:
        return effect

    rot = (frame.extra.get("rotation") or {}).get(team)
    players = list((rot or {}).get("players", []))
    last_played = {p["key"]: p.get("last_played_utc") for p in players}
    own_key, own_id, name = person_key(row["player_name"]), str(row["player_id"]), row["player_name"]
    status_of: Dict[str, str] = {}
    own: Optional[str] = None
    for _, r in mine.iterrows():
        key, status = person_key(r["player_name"]), str(r["status"])
        is_own = key == own_key or str(r.get("player_id")) == own_id
        stated = r.get("published_at_utc")
        stated = stated if isinstance(stated, str) and stated else r.get("collected_at_utc")
        played = last_played.get(own_key if is_own else key)
        if status in ("out", "doubtful") and isinstance(stated, str) and played and stated < played:
            # He has played since he was listed: the statement is out of date.
            effect.unmatched.append(f"{r['player_name']} ({status}, stated {stated[:10]}): has played since "
                                    f"(last on {played[:10]}); the statement is out of date and was not applied")
            continue
        if is_own:
            own = status
        else:
            status_of[key] = status
    if own == "out":
        effect.blockers.append(f"{name} is listed out: a market on him would be void")
    elif own in ("doubtful",) + _UNCERTAIN:
        effect.uncertain_a.append(f"{name} ({own}) - the market is void if he does not play, and his minutes "
                                  "may be limited if he does")

    mates = [p for p in players if p["key"] != own_key and str(p["player_id"]) != own_id]
    in_rotation = {p["key"] for p in mates}
    for key, status in status_of.items():
        if key not in in_rotation and status != "available":
            effect.unmatched.append(f"{key} ({status}): not in the team's recent rotation, no effect on the model")

    block = row.get("block")
    wrong_season = rot is None or block != block or int(block) != int(rot["season"])
    features = frame.extra.get("model_features")
    has_rep = "mates_out_min_rep" in row.index
    reads_rep = has_rep and (features is None or any(c in features for c in _REPORT_SWAP.values()))

    def describe(p: Dict[str, Any], status: str) -> str:
        return f"{p['name']} ({status}, {p['min']:.0f} minutes)"

    if wrong_season or not reads_rep or not report_read:
        # Nothing can be applied.  What the inputs do not already reflect stays a condition.
        why = ("no rotation on file for this game" if wrong_season else
               "this model takes team-mates' absences from who missed the previous game" if not reads_rep else
               "no injury report has been read for this team, only single statements")
        for p in mates:
            status = status_of.get(p["key"])
            if status in ("out", "doubtful"):
                if not wrong_season and not reads_rep and p["missed"] >= 1:
                    effect.applied.append(f"{describe(p, status)}: already counted as absent (he missed the "
                                          "previous game)")
                else:
                    effect.uncertain_a.append(f"team-mate {describe(p, status)} - not reflected in the model "
                                              f"inputs ({why}); effect not quantified")
            elif status in _DOUBT:
                effect.uncertain_a.append(f"team-mate {describe(p, status)} - effect not quantified ({why})")
        return effect

    j = int(frame.extra["stat_index"])
    new_limit = int(frame.extra.get("new_absence_games", 2))
    typical = float(row["league_total"]) if row.get("league_total") == row.get("league_total") else float("nan")

    def inputs(absent_statuses: Tuple[str, ...]) -> Tuple[Dict[str, float], List[str]]:
        minutes = new_minutes = stat = new_stat = 0.0
        who: List[str] = []
        for p in mates:
            status = status_of.get(p["key"])
            if status not in absent_statuses:
                continue
            who.append(describe(p, status))
            minutes += p["min"]
            stat += p["stats"][j]
            if p["missed"] <= new_limit:
                new_minutes += p["min"]
                new_stat += p["stats"][j]
        return ({"mates_out_min_rep": minutes, "mates_new_out_min_rep": new_minutes,
                 "mates_out_share_rep": stat / typical, "mates_new_out_share_rep": new_stat / typical}, who)

    base, absent = inputs(("out", "doubtful"))
    worst, _ = inputs(("out", "doubtful") + _DOUBT)
    _, doubts = inputs(_DOUBT)
    new = row.copy()
    for col, value in base.items():
        new[col] = value
    effect.row = new
    if absent:
        effect.applied.append("team-mates treated as not playing: " + "; ".join(absent))
    else:
        effect.applied.append("the team's report lists no rotation team-mate as out or doubtful")
    if doubts:
        effect.uncertain_a += [f"team-mate {d}" for d in doubts]
        if own not in ("doubtful",) + _UNCERTAIN:       # his own doubt cannot be put into a row
            sat = row.copy()
            for col, value in worst.items():
                sat[col] = value
            effect.worst_case_rows["a"] = sat
    return effect


# ----------------------------------------------------------------- explanation
def explain_prop(row: pd.Series, frame: FeatureFrame, exp: Any) -> Any:
    names = frame.extra.get("names", {})
    label = frame.extra.get("stat_label", "statistic")
    player = row.get("player_name", "the player")
    opp = names.get(row["side_b_id"], "the opponent")
    over = row.get("_pick_class") == "over"
    proj = row.get("_projection") or {}

    def num(v: Any, digits: int = 1) -> str:
        return "n/a" if v is None or v != v else f"{v:.{digits}f}"

    if proj:
        if "exposure" in proj:
            exp.supporting.append(f"expected minutes {num(proj.get('exposure'))}; expected {label} per minute "
                                  f"{num(proj.get('rate'), 3)}")
        exp.supporting.append(f"projected {label}: {num(proj.get('mean'))} (middle 80% of outcomes "
                              f"{num(proj.get('q10'), 0)} to {num(proj.get('q90'), 0)}, median {num(proj.get('q50'), 0)})")
    recent, longer = row.get("log_rate_ew"), row.get("log_rate_long")
    if recent == recent and longer == longer:
        change = 100.0 * (float(np.exp(recent - longer)) - 1.0)
        text = (f"recent production per minute is {abs(change):.0f}% {'above' if change > 0 else 'below'} "
                "his longer-run rate")
        (exp.supporting if (change > 0) == over else exp.upset_risks).append(text)
    ratio = row.get("log_opp_ratio")
    if ratio == ratio:
        pct = 100.0 * (float(np.exp(ratio)) - 1.0)
        exp.opponent_strength.append(
            f"{opp} have allowed {abs(pct):.1f}% {'more' if pct > 0 else 'fewer'} {label} than the league "
            "average in their recent games")
    if row.get("abs_margin_gap") == row.get("abs_margin_gap") and row["abs_margin_gap"] >= 10:
        exp.upset_risks.append("the two teams' recent margins are far apart: a lopsided game can shorten minutes")
    if row.get("b2b") == 1.0:
        exp.upset_risks.append("second day of a back-to-back for the player")
    if row.get("own_missed") == row.get("own_missed") and row["own_missed"] >= 1:
        exp.upset_risks.append(f"he missed the team's last {int(row['own_missed'])} game(s): minutes on return "
                               "are uncertain")
    out_min = row.get("mates_out_min_rep", row.get("mates_out_min"))
    if out_min == out_min and out_min and out_min > 0:
        exp.availability_assumptions.append(
            f"rotation team-mates expected to be absent usually play {out_min:.0f} minutes between them")
    exp.availability_assumptions.append(
        "Conditional on the player appearing; a statistic market is void when he does not play.")
    exp.missing_data.append("confirmed starting line-ups and announced minutes limits are not available")
    exp.missing_data.append(
        "no history of market thresholds: this probability has been checked against reference thresholds built "
        "from players' earlier games, never against a bookmaker's line, which already reflects most of what the "
        "model knows and news it does not")
    return exp


# ------------------------------------------------------- thresholds elsewhere
#: How far from the reference threshold each statistic is also scored (in its own units).
SWEEP_SHIFTS: Dict[str, Tuple[float, ...]] = {
    "points": (-6.0, -3.0, -0.5, 0.5, 3.0, 6.0), "rebounds": (-3.0, -1.5, -0.5, 0.5, 1.5, 3.0),
    "assists": (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0), "threes": (-1.0, -0.5, 0.5, 1.0),
    "pra": (-8.0, -4.0, -0.5, 0.5, 4.0, 8.0),
}


def threshold_sweep(frame: FeatureFrame, candidate: Any, plan: Any) -> Dict[str, Any]:
    """Score the selected candidate at thresholds other than the reference one.

    A real market threshold need not sit at the player's recent median, and
    the calibration step was fitted at the reference threshold only.  So the
    candidate is refitted exactly as in validation and asked for probabilities
    at thresholds moved up and down by fixed amounts - including whole-number
    thresholds, where a tie (push) is possible - on the last development fold
    and on the final test.  Nothing here feeds back into model selection.
    """
    from ...models.counts import threshold_probabilities
    from ...validation import metrics as M
    from ...validation.backtest import _blind, fit_with_calibration
    from ...validation.splits import make_folds

    stat = frame.extra["stat"]
    df = frame.completed.sort_values(["start_time_utc", "event_id"]).reset_index(drop=True)
    folds = make_folds(df, plan)
    out: Dict[str, Any] = {
        "kind": "threshold_sweep",
        "note": "The selected model scored at thresholds shifted from the reference threshold. A shift of half a "
                "point gives a whole-number threshold, where an exact tie is a push: over and under are then read "
                "from the calibrated probabilities at the half points either side, the tie is what remains, ties "
                "are left out of the over/under score, and the predicted and observed tie rates are compared.",
        "folds": {},
    }
    targets = [("last_development_fold", folds.dev[-1])] if folds.dev else []
    targets.append(("final_test", folds.final))
    for label, fold in targets:
        train = df[df[plan.block_col].isin(fold.train_blocks)]
        calib = df[df[plan.block_col].isin(fold.calib_blocks)]
        test = df[df[plan.block_col].isin(fold.test_blocks)]
        model, cal = fit_with_calibration(candidate, train, calib, frame.classes)
        rows = []
        for shift in (0.0,) + tuple(SWEEP_SHIFTS[stat]):
            line = test["ref_line"].to_numpy(dtype=float) + shift
            ok = line >= 0.5 if shift != 0 else np.ones(len(test), dtype=bool)
            part = test[ok]
            if part.empty:
                continue
            blind = _blind(part, model, plan.block_col)

            def over_at(thresholds: np.ndarray) -> np.ndarray:
                rows_at = blind.copy()
                rows_at["line"] = thresholds
                return M.check_probabilities(cal.transform(M.check_probabilities(model.predict_proba(rows_at))))[:, 1]

            under, p_push, over = threshold_probabilities(over_at, line[ok])
            value = part["stat_value"].to_numpy(dtype=float)
            push = value == line[ok]
            decided = ~push
            y = (value > line[ok]).astype(float)
            share = np.clip(under + over, 1e-9, None)
            P = np.column_stack([under / share, over / share])
            rows.append({
                "shift": float(shift), "n": int(decided.sum()),
                "over_rate": float(y[decided].mean()), "mean_predicted_over": float(P[decided, 1].mean()),
                "log_loss": float(M.log_loss(P[decided], y[decided])), "accuracy": float(M.accuracy(P[decided], y[decided])),
                "ece": float(M.ece(P[decided], y[decided])),
                "push_rate": float(push.mean()), "mean_predicted_push": float(np.mean(p_push)),
            })
        out["folds"][label] = {"blocks": [str(b) for b in fold.test_blocks], "rows": rows}
    return out
