"""Point-in-time football features for the winner, spread and total markets.

Every number in a row is computed from games on **earlier slates** only (a
slate is a Pacific/Honolulu calendar day, so all of an NFL Sunday is one
slate) plus the public schedule.  Side A is the home team (the designated home
team at a neutral site).

Feature families - the weight of each is learned in validation, not set here:

* team strength, opponent-adjusted - Elo with margin of victory, and
  offence/defence ratings on points and (where team statistics are on file) on
  expected points added per play, with a slow memory and a fast one (about the
  last 7 games).  Seasons are short, so ratings carry over between seasons
  with regression to the mean and the update rates are tuned on the burn-in
  seasons only;
* matchup - passing and rushing offence against the opposing defence,
  turnovers, pace;
* quarterback - each quarterback's shrunk efficiency from his earlier games.
  The honest pregame version uses the **incumbent** (whoever started the
  team's previous game); the version that uses the quarterback who actually
  started is retrospective;
* schedule and site - rest days, byes, short weeks, travel distance and
  time-zone shift of the visiting side, division game, neutral site, roof
  type and playing surface.

The **line** of a spread or total bet is carried in the frame (column
``line``) because it defines the bet, but it is not a feature: no model input
is derived from any betting number.

Retrospective columns (never available to selectable candidates): everything
built from the actual starting quarterback, the observed temperature and
wind, and whether a retractable roof was open.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...core.enums import OutcomeType
from ...core.timeutil import DISPLAY_TZ_NAME, parse_ts, utcnow
from ...features.ratings import EloParams, elo_pregame, offdef_pregame, tune_elo
from ...features.schedule import haversine_km
from ..base import FeatureFrame
from .teams import site_coords, team_coords

__all__ = ["FootballConfig", "build_frame", "MARKETS", "FEATURES", "RETROSPECTIVE", "FEATURE_LABELS",
           "expected_qb_values"]

_LOCAL_TZ = "America/New_York"


@dataclass(frozen=True)
class FootballConfig:
    burn_in_seasons: int = 3              # earliest seasons: rating warm-up and update-rate tuning only
    fast_lr: float = 0.25                 # "recent form": roughly the last 7 games
    slow_lr_grid: Tuple[float, ...] = (0.03, 0.06, 0.10)
    season_carry: float = 0.7
    elo_margin_grid: Tuple[Optional[float], ...] = (None, 7.0, 14.0)
    min_tune_events: int = 300
    qb_prior_plays: float = 400.0         # shrinkage of a quarterback's rate (about ten starts)
    qb_decay: float = 0.97                # per game he plays; old games fade
    qb_prior_min_plays: float = 2000.0    # inexperienced-quarterback plays needed before their mean is the prior
    opener_rest_days: float = 7.0         # rest assigned to a team's first game of a season
    rest_cap_days: float = 14.0
    bye_rest_days: float = 13.0           # at least this much rest = coming off a bye
    short_rest_days: float = 5.0          # at most this much rest = short week
    pending_window_days: float = 8.0      # a started game without a result counts as pending this long


#: market -> (outcome type, classes, column holding the modelled score quantity)
MARKETS: Dict[str, Tuple[OutcomeType, Tuple[str, str], str]] = {
    "winner": (OutcomeType.BINARY, ("away", "home"), "margin"),
    "spread": (OutcomeType.OVER_UNDER, ("away_cover", "home_cover"), "margin"),
    "total": (OutcomeType.OVER_UNDER, ("under", "over"), "total_points"),
}

_SIDE_FEATURES: Dict[str, List[str]] = {
    "core": ["home_edge", "elo_diff", "pts_slow_diff", "pts_fast_diff", "rest_diff", "bye_a", "bye_b",
             "short_a", "short_b", "is_playoff"],
    "stats": ["epa_slow_diff", "epa_fast_diff", "pass_edge", "rush_edge", "to_edge"],
    "qb": ["qb_inc_diff", "qb_inc_adj_diff"],
    "site": ["travel_diff", "tz_shift_b"],
    "context": ["div_game"],
}
_TOTAL_FEATURES: Dict[str, List[str]] = {
    "core": ["league_pts", "pts_total_exp", "pts_total_fast", "abs_elo_diff", "home_edge", "is_playoff"],
    "stats": ["epa_slow_sum", "pace_sum"],
    "qb": ["qb_inc_sum"],
    "context": ["div_game"],
    "roof": ["dome", "retractable", "turf"],
}
#: market -> family -> point-in-time feature columns (a family is used when its data is on file)
FEATURES: Dict[str, Dict[str, List[str]]] = {
    "winner": _SIDE_FEATURES, "spread": _SIDE_FEATURES, "total": _TOTAL_FEATURES,
}
_SIDE_RETRO = {"qb": ["qb_act_diff", "qb_act_adj_diff", "qb_change_a", "qb_change_b"]}
RETROSPECTIVE: Dict[str, Dict[str, List[str]]] = {
    "winner": _SIDE_RETRO, "spread": _SIDE_RETRO,
    "total": {"qb": ["qb_act_sum"], "roof": ["wind_out", "cold_out", "indoor_actual"]},
}

FEATURE_LABELS: Dict[str, str] = {
    "home_edge": "home field",
    "elo_diff": "opponent-adjusted results (Elo)",
    "pts_slow_diff": "long-term opponent-adjusted scoring margin",
    "pts_fast_diff": "recent (about 7 games) opponent-adjusted scoring margin",
    "rest_diff": "rest days",
    "bye_a": "home side coming off a bye", "bye_b": "away side coming off a bye",
    "short_a": "home side on a short week", "short_b": "away side on a short week",
    "is_playoff": "playoff game",
    "epa_slow_diff": "long-term opponent-adjusted efficiency (EPA per play)",
    "epa_fast_diff": "recent opponent-adjusted efficiency (EPA per play)",
    "pass_edge": "passing matchup (offence vs opposing defence)",
    "rush_edge": "rushing matchup",
    "to_edge": "turnover matchup",
    "qb_inc_diff": "quality of the incumbent quarterbacks",
    "qb_inc_adj_diff": "incumbent quarterback versus the quarterbacks behind the team's ratings",
    "travel_diff": "travel distance", "tz_shift_b": "away side time-zone shift",
    "div_game": "division game",
    "league_pts": "league scoring level",
    "pts_total_exp": "long-term opponent-adjusted scoring of both sides",
    "pts_total_fast": "recent opponent-adjusted scoring of both sides",
    "abs_elo_diff": "mismatch size (Elo gap)",
    "epa_slow_sum": "combined offensive efficiency against these defences",
    "pace_sum": "expected number of plays",
    "qb_inc_sum": "combined quality of the incumbent quarterbacks",
    "dome": "fixed-roof stadium", "retractable": "retractable-roof stadium", "turf": "artificial surface",
    "qb_act_diff": "quality of the quarterbacks who actually started (retrospective)",
    "qb_act_adj_diff": "actual starters versus the quarterbacks behind the ratings (retrospective)",
    "qb_change_a": "home side started a different quarterback (retrospective)",
    "qb_change_b": "away side started a different quarterback (retrospective)",
    "qb_act_sum": "combined quality of the actual starters (retrospective)",
    "wind_out": "observed wind, outdoor games (retrospective)",
    "cold_out": "observed cold, outdoor games (retrospective)",
    "indoor_actual": "played under a roof (retrospective)",
}


def slate_batches(start_time_utc: pd.Series) -> np.ndarray:
    """Batch label = Pacific/Honolulu calendar date, so games on one slate
    never inform each other's features."""
    days = start_time_utc.dt.tz_convert(DISPLAY_TZ_NAME).dt.strftime("%Y-%m-%d")
    return pd.factorize(days, sort=False)[0]


def _load_events(store: Any, league: str, now_ts: pd.Timestamp) -> pd.DataFrame:
    """Scheduled and final events in time order with their as-of state.

    ``played``: finished before ``now`` with a score.  ``upcoming``: starts
    after ``now`` - whatever the store knows about it today is ignored, which
    makes a rebuild "as of" a past instant see exactly what was knowable then.
    Anything else has started but has no result yet.
    """
    ev = store.events(league, with_results=True)
    ev = ev[ev["status"].isin(["scheduled", "final"])].sort_values(["start_time_utc", "event_id"])
    ev = ev.reset_index(drop=True)
    if ev.empty:
        raise ValueError(f"no events on file for {league}; run `sportspred ingest --league {league}` first")
    scored = ev["home_score"].notna() & ev["away_score"].notna()
    ev["played"] = (ev["status"] == "final") & scored & (ev["start_time_utc"] < now_ts)
    ev["upcoming"] = ev["start_time_utc"] > now_ts
    return ev


def _local_days(ev: pd.DataFrame, extra: pd.DataFrame) -> np.ndarray:
    """Calendar day number of each game where it is played (US Eastern for
    timed games; the source's own date for date-only games)."""
    timed = ev["start_time_utc"].dt.tz_convert(_LOCAL_TZ).dt.strftime("%Y-%m-%d")
    dates = timed.where(ev["time_precision"] == "datetime", ev["start_time_utc"].dt.strftime("%Y-%m-%d"))
    if "local_date" in extra.columns:
        dates = extra["local_date"].astype("string").fillna(dates.astype("string"))
    stamp = pd.to_datetime(dates.astype(str), format="%Y-%m-%d")
    return ((stamp - pd.Timestamp("1970-01-01")) // pd.Timedelta(days=1)).to_numpy()


def _rest_days(ev: pd.DataFrame, day: np.ndarray, cfg: FootballConfig) -> Dict[str, np.ndarray]:
    """Days since each side's previous scheduled game of the season (schedule only)."""
    n = len(ev)
    out = {f"rest_{s}": np.full(n, cfg.opener_rest_days) for s in "ab"}
    out.update({f"season_games_{s}": np.zeros(n) for s in "ab"})
    last: Dict[Tuple[str, str], int] = {}
    count: Dict[Tuple[str, str], int] = {}
    homes, aways, seasons = ev["home_id"].to_numpy(), ev["away_id"].to_numpy(), ev["season"].to_numpy()
    for i in range(n):
        for side, team in (("a", homes[i]), ("b", aways[i])):
            key = (team, seasons[i])
            if key in last:
                out[f"rest_{side}"][i] = min(float(day[i] - last[key]), cfg.rest_cap_days)
            out[f"season_games_{side}"][i] = count.get(key, 0)
            last[key] = int(day[i])
            count[key] = count.get(key, 0) + 1
    return out


def _pending_prior(ev: pd.DataFrame, now_ts: pd.Timestamp, cfg: FootballConfig) -> np.ndarray:
    """Earlier games of either side that have no result yet.  A game that
    started long ago without a result is a gap in the data, not a pending game."""
    pending = np.zeros(len(ev), dtype=int)
    seen: Dict[str, int] = {}
    stale = ev["start_time_utc"] < now_ts - pd.Timedelta(days=cfg.pending_window_days)
    for i in np.flatnonzero((~ev["played"] & ~stale).to_numpy()):
        h, a = ev.at[i, "home_id"], ev.at[i, "away_id"]
        pending[i] = seen.get(h, 0) + seen.get(a, 0)
        seen[h] = seen.get(h, 0) + 1
        seen[a] = seen.get(a, 0) + 1
    return pending


def _site_features(ev: pd.DataFrame, extra: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Travel of each side to the game and the visiting side's time-zone shift."""
    home = np.array([team_coords(a) for a in extra["home_abbr"]])
    away = np.array([team_coords(a) for a in extra["away_abbr"]])
    neutral = ev["neutral_site"].fillna(0).astype(int).to_numpy() == 1
    site = np.where(neutral[:, None], np.array([site_coords(s) for s in extra["stadium_id"]]), home)
    km_a = haversine_km(home[:, 0], home[:, 1], site[:, 0], site[:, 1])
    km_b = haversine_km(away[:, 0], away[:, 1], site[:, 0], site[:, 1])
    shift = (site[:, 1] - away[:, 1]) / 15.0            # hours of longitude, eastward positive
    return {"travel_km_a": km_a, "travel_km_b": km_b, "travel_diff": np.log1p(km_b) - np.log1p(km_a),
            "tz_shift_b": (shift + 12.0) % 24.0 - 12.0}  # the short way round the clock


def _roof_features(extra: pd.DataFrame, played: np.ndarray) -> Dict[str, np.ndarray]:
    """Stadium type and surface (public in advance) and, retrospectively, the weather."""
    roof = extra["roof"].astype("string").str.lower()
    surface = extra["surface"].astype("string").str.lower()
    dome = (roof == "dome").fillna(False).to_numpy(dtype=bool)
    outdoors = (roof == "outdoors").fillna(False).to_numpy(dtype=bool)
    # ``open``/``closed`` is decided on the day and is blank for unplayed
    # games, so before kickoff all three mean the same thing: retractable.
    retractable = ~dome & ~outdoors
    turf = np.where(surface.isna(), np.nan, (~surface.str.contains("grass").fillna(False)).astype(float))
    indoor = dome | (roof == "closed").fillna(False).to_numpy(dtype=bool)
    wind = pd.to_numeric(extra["wind"], errors="coerce").to_numpy(dtype=float)
    temp = pd.to_numeric(extra["temp"], errors="coerce").to_numpy(dtype=float)
    return {
        "dome": dome.astype(float), "retractable": retractable.astype(float), "turf": turf.astype(float),
        "needs_weather": (~dome).astype(float),
        "indoor_actual": np.where(played, indoor.astype(float), np.nan),
        "wind_out": np.where(played, np.where(indoor, 0.0, wind), np.nan),
        "cold_out": np.where(played, np.where(indoor, 0.0, np.clip(50.0 - temp, 0.0, None)), np.nan),
    }


def _side_values(ev: pd.DataFrame, table: pd.DataFrame, cols: List[str]) -> Dict[str, np.ndarray]:
    """Per event: each column of a (event, team) table for the home (``_a``) and away (``_b``) side."""
    tb = table[["event_id", "team_id", *cols]].drop_duplicates(["event_id", "team_id"], keep="last")
    out: Dict[str, np.ndarray] = {}
    for side, col in (("a", "home_id"), ("b", "away_id")):
        m = ev[["event_id", col]].merge(tb, left_on=["event_id", col], right_on=["event_id", "team_id"], how="left")
        for c in cols:
            out[f"{c}_{side}"] = m[c].to_numpy(dtype=float)
    return out


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    out[~np.isfinite(out)] = np.nan
    return out


def _qb_features(ev: pd.DataFrame, extra: pd.DataFrame, qb_stats: pd.DataFrame, batch: np.ndarray,
                 played: np.ndarray, lr: float, cfg: FootballConfig) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Quarterback quality before each game.

    A quarterback's rating is his EPA per play (dropbacks and his own carries)
    over his earlier games, with old games fading and strong shrinkage towards
    what inexperienced quarterbacks have produced so far.  Per side:

    * ``qb_inc`` - rating of the incumbent, i.e. the starter of the team's
      previous game (knowable pregame);
    * ``qb_base`` - running average rating of the quarterbacks who started the
      team's recent games, i.e. the quarterback play already inside the team
      ratings;
    * ``qb_act`` / ``qb_change`` - the same for the quarterback who actually
      started this game (retrospective).
    """
    n = len(ev)
    out = {f"{k}_{s}": np.full(n, np.nan) for k in ("qb_inc", "qb_base", "qb_act", "qb_change") for s in "ab"}
    rows_of = qb_stats.groupby("event_id", sort=False).indices
    qb_id = qb_stats["player_id"].to_numpy()
    qb_plays = qb_stats["plays"].to_numpy(dtype=float)
    qb_epa = qb_stats["epa"].to_numpy(dtype=float)
    names = dict(zip(qb_stats["player_id"], qb_stats["player_name"]))
    eids, homes, aways = ev["event_id"].to_numpy(), ev["home_id"].to_numpy(), ev["away_id"].to_numpy()
    starters = {"a": extra["home_qb_id"].to_numpy(dtype=object), "b": extra["away_qb_id"].to_numpy(dtype=object)}
    for side, col in (("a", "home_qb_name"), ("b", "away_qb_name")):
        names.update({q: nm for q, nm in zip(starters[side], extra[col]) if isinstance(q, str)})
    epa_sum: Dict[str, float] = {}
    play_sum: Dict[str, float] = {}
    career: Dict[str, float] = {}
    incumbent: Dict[str, str] = {}
    base: Dict[str, float] = {}
    prior = 0.0                              # EPA per play averages zero by construction
    novice_epa = novice_plays = 0.0

    def rating(q: str) -> float:
        return (epa_sum.get(q, 0.0) + cfg.qb_prior_plays * prior) / (play_sum.get(q, 0.0) + cfg.qb_prior_plays)

    pending: List[int] = []

    def flush() -> None:
        """Apply a finished slate: statistics first, then who started."""
        nonlocal prior, novice_epa, novice_plays
        for i in pending:
            for r in rows_of.get(eids[i], ()):
                q = qb_id[r]
                if career.get(q, 0.0) < cfg.qb_prior_plays:
                    novice_epa += qb_epa[r]
                    novice_plays += qb_plays[r]
                epa_sum[q] = cfg.qb_decay * epa_sum.get(q, 0.0) + qb_epa[r]
                play_sum[q] = cfg.qb_decay * play_sum.get(q, 0.0) + qb_plays[r]
                career[q] = career.get(q, 0.0) + qb_plays[r]
        if novice_plays >= cfg.qb_prior_min_plays:
            prior = novice_epa / novice_plays
        for i in pending:
            for side, team in (("a", homes[i]), ("b", aways[i])):
                q = starters[side][i]
                if isinstance(q, str):
                    incumbent[team] = q
                    base[team] = rating(q) if team not in base else base[team] + lr * (rating(q) - base[team])
        pending.clear()

    for i in range(n):
        if i and batch[i] != batch[i - 1]:
            flush()
        for side, team in (("a", homes[i]), ("b", aways[i])):
            inc = incumbent.get(team)
            if inc is not None:
                out[f"qb_inc_{side}"][i] = rating(inc)
                out[f"qb_base_{side}"][i] = base[team]
            q = starters[side][i]
            if played[i] and isinstance(q, str):
                out[f"qb_act_{side}"][i] = rating(q)
                out[f"qb_change_{side}"][i] = float(inc is not None and q != inc)
        if played[i]:
            pending.append(i)
    flush()
    table = {
        "prior": float(prior),
        "qbs": {q: {"name": names.get(q, q), "rating": round(rating(q), 5), "plays": round(career[q], 1)}
                for q in career},
        "teams": {t: {"incumbent": incumbent[t], "base": round(base[t], 5)} for t in incumbent},
    }
    return out, table


def expected_qb_values(row: pd.Series, inc_a: float, inc_b: float) -> pd.Series:
    """``row`` with the expected starters' ratings replaced (used when a
    starting quarterback is announced or ruled out)."""
    new = row.copy()
    new["qb_inc_a"], new["qb_inc_b"] = inc_a, inc_b
    new["qb_inc_diff"] = inc_a - inc_b
    new["qb_inc_adj_diff"] = (inc_a - row["qb_base_a"]) - (inc_b - row["qb_base_b"])
    new["qb_inc_sum"] = inc_a + inc_b
    return new


def _tune_slow_lr(a_idx, b_idx, va, vb, season_idx, batch, home_edge, tune_mask, n_entities,
                  cfg: FootballConfig) -> float:
    """Slow update rate with the lowest one-step-ahead error on the burn-in seasons."""
    best_lr, best = cfg.slow_lr_grid[len(cfg.slow_lr_grid) // 2], np.inf
    if tune_mask.sum() < cfg.min_tune_events:
        return best_lr
    end = int(np.flatnonzero(tune_mask).max()) + 1
    sl = slice(0, end)
    for lr in cfg.slow_lr_grid:
        r = offdef_pregame(a_idx[sl], b_idx[sl], va[sl], vb[sl], season_idx[sl], batch[sl], home_edge[sl],
                           lr=lr, carry=cfg.season_carry, n_entities=n_entities)
        exp_a = r["mean"] + r["off_a"] + r["def_b"] + r["home"] * home_edge[sl] / 2
        exp_b = r["mean"] + r["off_b"] + r["def_a"] - r["home"] * home_edge[sl] / 2
        m = tune_mask[sl] & np.isfinite(va[sl])
        mse = float(np.mean((va[sl][m] - exp_a[m]) ** 2 + (vb[sl][m] - exp_b[m]) ** 2))
        if mse < best:
            best, best_lr = mse, lr
    return best_lr


def build_frame(store: Any, league: str, market: str = "winner", *, now: Any = None,
                cfg: Optional[FootballConfig] = None) -> FeatureFrame:
    """Feature frame for one market of any football league.

    Needs only events and results; the optional tables ``games`` (lines,
    quarterbacks, stadium), ``team_stats`` and ``qb_stats`` add their feature
    families when they are on file.
    """
    if market not in MARKETS:
        raise NotImplementedError(f"football has no feature builder for market {market!r}")
    cfg = cfg or FootballConfig()
    outcome, classes, target_col = MARKETS[market]
    now_ts = pd.Timestamp(utcnow() if now is None else parse_ts(now))
    ev = _load_events(store, league, now_ts)
    played = ev["played"].to_numpy()
    n = len(ev)
    extra = pd.DataFrame({"event_id": ev["event_id"]})
    if store.has_table(league, "games"):
        extra = extra.merge(store.read_table(league, "games").drop_duplicates("event_id"), on="event_id", how="left")
    has = lambda *cols: set(cols) <= set(extra.columns)  # noqa: E731

    ent_a = extra["home_entity"].fillna(ev["home_id"]) if has("home_entity") else ev["home_id"]
    ent_b = extra["away_entity"].fillna(ev["away_id"]) if has("away_entity") else ev["away_id"]
    entities = pd.Index(pd.unique(pd.concat([ent_a, ent_b])))
    a_idx, b_idx = entities.get_indexer(ent_a), entities.get_indexer(ent_b)
    season_num = pd.to_numeric(ev["season"], errors="coerce").astype(int).to_numpy()
    seasons_sorted = np.sort(np.unique(season_num))
    season_idx = np.searchsorted(seasons_sorted, season_num)
    batch = slate_batches(ev["start_time_utc"])
    home_edge = np.where(ev["neutral_site"].fillna(0).astype(int).to_numpy() == 1, 0.0, 1.0)
    hs = np.where(played, ev["home_score"].to_numpy(dtype=float), np.nan)
    aw = np.where(played, ev["away_score"].to_numpy(dtype=float), np.nan)
    margin, total = hs - aw, hs + aw
    score_a = np.where(played, np.where(margin > 0, 1.0, np.where(margin < 0, 0.0, 0.5)), np.nan)

    # ---- update rates are tuned on the burn-in seasons only (the first one is warm-up)
    burn = seasons_sorted[: cfg.burn_in_seasons]
    tune_mask = np.isin(season_num, burn[1:]) & played
    if tune_mask.sum() < cfg.min_tune_events:
        elo_params, elo_grid = EloParams(k=20.0, home_adv=50.0, carry=0.67, margin_scale=7.0), []
    else:
        elo_params, elo_grid = tune_elo(a_idx, b_idx, score_a, season_idx, batch, home_edge, tune_mask=tune_mask,
                                        margin=margin, margin_grid=cfg.elo_margin_grid)
    elo_a, elo_b, _ = elo_pregame(a_idx, b_idx, score_a, season_idx, batch, home_edge, elo_params,
                                  margin=margin, n_entities=len(entities))
    slow_lr = _tune_slow_lr(a_idx, b_idx, hs, aw, season_idx, batch, home_edge, tune_mask, len(entities), cfg)

    def offdef(va: np.ndarray, vb: np.ndarray, lr: float) -> Dict[str, np.ndarray]:
        return offdef_pregame(a_idx, b_idx, va, vb, season_idx, batch, home_edge, lr=lr,
                              carry=cfg.season_carry, n_entities=len(entities))

    def edge(r: Dict[str, np.ndarray]) -> np.ndarray:      # A's offence v B's defence, minus the reverse
        return (r["off_a"] + r["def_b"]) - (r["off_b"] + r["def_a"])

    def both(r: Dict[str, np.ndarray]) -> np.ndarray:      # what the two offences are expected to add up to
        return 2.0 * r["mean"] + r["off_a"] + r["def_b"] + r["off_b"] + r["def_a"]

    pts_slow, pts_fast = offdef(hs, aw, slow_lr), offdef(hs, aw, cfg.fast_lr)
    day = _local_days(ev, extra)
    rest = _rest_days(ev, day, cfg)
    starts = ev["start_time_utc"]
    last_of_batch = starts.groupby(batch).transform("max")
    prev_batch_last = pd.Series(pd.unique(last_of_batch)).shift().to_numpy()

    df = pd.DataFrame({
        "event_id": ev["event_id"],
        "start_time_utc": starts,
        "block": season_num,
        "season": ev["season"],
        "status": ev["status"],
        "stage": ev["stage"],
        "side_a_id": ev["home_id"],
        "side_b_id": ev["away_id"],
        # Latest information any feature can contain: the last kickoff of earlier slates.
        "info_cutoff_utc": pd.to_datetime(pd.Series(prev_batch_last[batch], index=ev.index), utc=True),
        "margin": margin,
        "total_points": total,
        "home_edge": home_edge,
        "elo_a": elo_a, "elo_b": elo_b,
        "elo_diff": elo_a - elo_b,
        "abs_elo_diff": np.abs(elo_a - elo_b + elo_params.home_adv * home_edge),
        "pts_slow_diff": edge(pts_slow), "pts_fast_diff": edge(pts_fast),
        "league_pts": 2.0 * pts_slow["mean"],
        "pts_total_exp": both(pts_slow), "pts_total_fast": both(pts_fast),
        "rest_days_a": rest["rest_a"], "rest_days_b": rest["rest_b"],
        "rest_diff": rest["rest_a"] - rest["rest_b"],
        "bye_a": (rest["rest_a"] >= cfg.bye_rest_days).astype(float),
        "bye_b": (rest["rest_b"] >= cfg.bye_rest_days).astype(float),
        "short_a": (rest["rest_a"] <= cfg.short_rest_days).astype(float),
        "short_b": (rest["rest_b"] <= cfg.short_rest_days).astype(float),
        "season_games_a": rest["season_games_a"], "season_games_b": rest["season_games_b"],
        "is_playoff": ev["stage"].isin(["playoffs", "postseason"]).astype(float),
        "n_pending_prior": _pending_prior(ev, now_ts, cfg),
    })
    families = {"core"}

    if store.has_table(league, "team_stats"):
        families.add("stats")
        v = _side_values(ev, store.read_table(league, "team_stats"),
                         ["plays", "epa", "dropbacks", "pass_epa", "carries", "rush_epa", "turnovers"])
        v = {k: np.where(played, x, np.nan) for k, x in v.items()}
        epa_a, epa_b = _ratio(v["epa_a"], v["plays_a"]), _ratio(v["epa_b"], v["plays_b"])
        epa_slow, epa_fast = offdef(epa_a, epa_b, slow_lr), offdef(epa_a, epa_b, cfg.fast_lr)
        df["epa_slow_diff"], df["epa_fast_diff"], df["epa_slow_sum"] = edge(epa_slow), edge(epa_fast), both(epa_slow)
        df["pass_edge"] = edge(offdef(_ratio(v["pass_epa_a"], v["dropbacks_a"]),
                                      _ratio(v["pass_epa_b"], v["dropbacks_b"]), slow_lr))
        df["rush_edge"] = edge(offdef(_ratio(v["rush_epa_a"], v["carries_a"]),
                                      _ratio(v["rush_epa_b"], v["carries_b"]), slow_lr))
        df["to_edge"] = -edge(offdef(v["turnovers_a"], v["turnovers_b"], slow_lr))
        df["pace_sum"] = both(offdef(v["plays_a"], v["plays_b"], slow_lr))
        no_stats = (epa_slow["n_a"] == 0) | (epa_slow["n_b"] == 0)
        for col in ("epa_slow_diff", "epa_fast_diff", "epa_slow_sum", "pass_edge", "rush_edge", "to_edge", "pace_sum"):
            df[col] = np.where(no_stats, np.nan, df[col])
        df["dq_no_team_stats"] = no_stats.astype(float)

    qb_table: Dict[str, Any] = {}
    if has("home_qb_id", "away_qb_id") and store.has_table(league, "qb_stats"):
        families.add("qb")
        qb, qb_table = _qb_features(ev, extra, store.read_table(league, "qb_stats"), batch, played, slow_lr, cfg)
        for key, val in qb.items():
            df[key] = val
        df["qb_inc_diff"] = df["qb_inc_a"] - df["qb_inc_b"]
        df["qb_inc_adj_diff"] = (df["qb_inc_a"] - df["qb_base_a"]) - (df["qb_inc_b"] - df["qb_base_b"])
        df["qb_inc_sum"] = df["qb_inc_a"] + df["qb_inc_b"]
        df["qb_act_diff"] = df["qb_act_a"] - df["qb_act_b"]
        df["qb_act_adj_diff"] = (df["qb_act_a"] - df["qb_base_a"]) - (df["qb_act_b"] - df["qb_base_b"])
        df["qb_act_sum"] = df["qb_act_a"] + df["qb_act_b"]
        df["dq_no_qb_history"] = df["qb_inc_diff"].isna().astype(float)
    if has("home_abbr", "away_abbr", "stadium_id"):
        families.add("site")
        for key, val in _site_features(ev, extra).items():
            df[key] = val
    if has("div_game"):
        families.add("context")
        df["div_game"] = pd.to_numeric(extra["div_game"], errors="coerce").to_numpy(dtype=float)
    if has("roof", "surface", "temp", "wind"):
        families.add("roof")
        for key, val in _roof_features(extra, played).items():
            df[key] = val

    # ---- the bet: target, line, and which rows belong in this market's frame
    target = (extra["target"].fillna(1).astype(int).to_numpy() == 1) if has("target") else np.ones(n, dtype=bool)
    line = np.full(n, np.nan)
    if market == "winner":
        signed = margin
    else:
        source = {"spread": "spread_line", "total": "total_line"}[market]
        if has(source):
            raw_line = pd.to_numeric(extra[source], errors="coerce").to_numpy(dtype=float)
            # Spread: the stored number is how many points the home side is
            # favoured by; the bet's line is the home handicap, its negative.
            line = -raw_line if market == "spread" else raw_line
        signed = margin + line if market == "spread" else total - line
    df["line"] = line
    df["y"] = np.where(played & ~np.isnan(signed) & (signed != 0), (signed > 0).astype(float), np.nan)
    level = played & (signed == 0)                   # tie (winner) or push (spread, total): no binary outcome
    no_line = np.isnan(line) & (market != "winner")
    keep = target & ~level & ~no_line & (played | ev["upcoming"].to_numpy())
    push_rows = df[level & target].reset_index(drop=True)
    counts = {"n_level": int((level & target).sum()), "n_played": int((played & target).sum()),
              "n_played_without_line": int((played & target & no_line).sum()),
              "n_upcoming_without_line": int((ev["upcoming"].to_numpy() & target & no_line).sum())}
    df = df[keep].reset_index(drop=True)

    feature_cols = [c for fam, cols in FEATURES[market].items() if fam in families for c in cols]
    retro_cols = [c for fam, cols in RETROSPECTIVE[market].items() if fam in families for c in cols]
    level_word = "tied games" if market == "winner" else "pushes"
    limitations = [
        f"{counts['n_level']} {level_word} out of {counts['n_played']} played games have no binary outcome and "
        "are excluded from fitting and scoring (they still feed the ratings).",
        "No pregame injury reports, announced starters or depth charts exist in the historical data, so the "
        "backtest cannot include late availability news. The selectable models assume each team starts the "
        "quarterback who started its previous game.",
    ]
    if market != "winner":
        limitations.append(
            "The line of each historical bet is the single undated number in the schedule file; the source "
            "does not say which bookmaker it is from or whether it is an opening or a closing number.")
        if counts["n_upcoming_without_line"]:
            limitations.append(f"{counts['n_upcoming_without_line']} upcoming games have no line on file yet "
                               "and are not in this market's frame.")
    if "stats" in families:
        limitations.append("EPA comes from nflfastR's expected-points model, a fixed play-value scale that was "
                           "itself estimated on seasons inside the backtest period.")
    return FeatureFrame(
        league=league, market=market, outcome=outcome, classes=classes, df=df,
        feature_cols=feature_cols, retrospective_cols=retro_cols, feature_notes=dict(FEATURE_LABELS),
        limitations=limitations, point_in_time=True,
        extra={
            "elo_params": elo_params.__dict__, "elo_tuning_grid_size": len(elo_grid), "slow_lr": slow_lr,
            "fast_lr": cfg.fast_lr, "burn_in_seasons": [int(s) for s in burn], "families": sorted(families),
            "tuning_note": "Elo and rating update rates were tuned on the burn-in seasons only.",
            "target_col": target_col, **counts,
            # Not written to validation reports: the lookup tables used at prediction time.
            "rosters": {"qb": qb_table, "level_rows": push_rows},
        },
    )
