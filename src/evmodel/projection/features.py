"""Walk-forward projection features.

For every row of a game-log table this module computes what could have been
known *before* that game: the player's expected playing time and expected
per-minute (or per-game) production, using only that player's earlier games.
Because each row only looks backwards, the same code serves back-tests (every
historical row is an honest out-of-sample forecast) and live use (append one
empty "next game" row per player and read off its features).

The estimator, for one stat ``x`` with opportunity ``m`` (minutes; or 1 per
game for sports without a minutes notion)::

    rate   = (sum_j w_j x_j + k * prior_rate) / (sum_j w_j m_j + k)
    opp    = (sum_j v_j m_j + k_m * prior_opp) / (sum_j v_j + k_m)
    mean   = rate * opp

``w_j`` and ``v_j`` halve every ``rate_halflife`` / ``opp_halflife`` games
back in time and are further multiplied by ``*_carry`` for each season
boundary crossed, so last season counts, but less.  ``k`` and ``k_m`` shrink
small samples towards the prior: the average rate of the player's position
group in the *previous* season (so priors never see the future either).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = ["FeatureParams", "position_group", "compute_features", "append_next_game_rows"]


@dataclass(frozen=True)
class FeatureParams:
    """Hyper-parameters of the walk-forward estimator for one sport."""

    #: base stat columns to project
    stats: Tuple[str, ...]
    #: opportunity column (``"minutes"``), or ``None`` for per-game rates
    opportunity: Optional[str] = "minutes"
    #: half-life (in games) of the weights on past production
    rate_halflife: float = 20.0
    #: weight multiplier applied to games from earlier seasons, per boundary
    rate_carry: float = 0.5
    #: prior strength in opportunity units (minutes, or games if per-game)
    rate_prior_strength: float = 100.0
    #: half-life (in games) for projecting opportunity (minutes)
    opp_halflife: float = 6.0
    opp_carry: float = 0.3
    #: prior strength for opportunity, in games
    opp_prior_strength: float = 1.0
    #: a layoff longer than this many days counts like a season boundary
    layoff_days: Optional[float] = 21.0
    #: positions kept (None = all); rows of other positions are dropped
    positions: Optional[Tuple[str, ...]] = None
    #: opportunity assumed for a player with no history at all
    default_opportunity: float = 18.0


_NBA_GROUPS = {
    "PG": "G", "SG": "G", "G": "G",
    "SF": "F", "PF": "F", "F": "F", "GF": "F",
    "C": "C", "FC": "C",
}
_NHL_GROUPS = {"C": "F", "L": "F", "R": "F", "LW": "F", "RW": "F", "F": "F", "D": "D", "G": "G"}


def position_group(sport: str, position: pd.Series) -> pd.Series:
    """Coarse position group used for priors (unknown positions -> ``"UNK"``)."""
    pos = position.fillna("").astype(str).str.upper().str.strip()
    if sport == "nba":
        return pos.map(_NBA_GROUPS).fillna("UNK")
    if sport == "nhl":
        return pos.map(_NHL_GROUPS).fillna("UNK")
    return pos.where(pos != "", "UNK")


def _recursive_prior_sums(
    values: np.ndarray,
    halflife: float,
    carry: float,
    boundary: np.ndarray,
    player: np.ndarray,
    rank: np.ndarray,
    n_players: int,
) -> np.ndarray:
    """Decay-weighted sums over each player's *earlier* games.

    For row ``i`` the result is ``sum_j w_ij * values_j`` over earlier rows
    ``j`` of the same player, where the most recent game has weight 1, each
    game further back multiplies the weight by ``0.5 ** (1 / halflife)``, and
    every season boundary crossed multiplies it by ``carry``.

    Implemented as the recursion ``S_i = carry**b_i * (lam * S_{i-1} +
    values_{i-1})``, advanced one game at a time for all players at once
    (rows are grouped by their game number ``rank`` within the player).
    """
    n, k = values.shape
    lam = 0.5 ** (1.0 / halflife)
    out = np.zeros((n, k))
    state = np.zeros((n_players, k))
    order = np.argsort(rank, kind="stable")
    sorted_rank = rank[order]
    max_rank = int(sorted_rank[-1]) if n else -1
    starts = np.searchsorted(sorted_rank, np.arange(max_rank + 2))
    discount = np.where(boundary > 0, carry, 1.0)
    for r in range(max_rank + 1):
        rows = order[starts[r]:starts[r + 1]]
        if rows.size == 0:
            continue
        players = player[rows]
        pre = state[players] * discount[rows, None]
        out[rows] = pre
        state[players] = pre * lam + values[rows]
    return out


def _league_priors(
    df: pd.DataFrame, stats: Sequence[str], opp: np.ndarray, group_col: str
) -> Tuple[Dict[Tuple[int, str], np.ndarray], Dict[Tuple[int, str], float]]:
    """Per (season, group): average rate per opportunity unit and average opportunity."""
    work = df[["season", group_col]].copy()
    work["_opp"] = opp
    for s in stats:
        work[s] = df[s].to_numpy(float)
    grouped = work.groupby(["season", group_col], observed=True)
    totals = grouped[list(stats) + ["_opp"]].sum()
    counts = grouped.size()
    rates: Dict[Tuple[int, str], np.ndarray] = {}
    opps: Dict[Tuple[int, str], float] = {}
    for key, row in totals.iterrows():
        denom = row["_opp"]
        rates[key] = (row[list(stats)].to_numpy(float) / denom) if denom > 0 else np.zeros(len(stats))
        opps[key] = float(denom / counts[key]) if counts[key] > 0 else 0.0
    return rates, opps


def compute_features(logs: pd.DataFrame, params: FeatureParams, sport: Optional[str] = None) -> pd.DataFrame:
    """Pre-game projections for every row of a game-log table.

    Rows flagged ``_placeholder`` (see :func:`append_next_game_rows`) receive
    features but contribute nothing to any estimate.

    Returns a frame indexed like the (position-filtered, sorted) logs with the
    identity columns plus:

    ``proj_opp``       projected opportunity (minutes) for the game
    ``n_prior``        number of earlier games by the player
    ``eff_games``      effective (decay-weighted) number of earlier games
    ``days_rest``      days since the player's previous game (NaN if none)
    ``rate_<stat>``    projected production per opportunity unit
    ``mu_<stat>``      projected mean of the stat
    ``prior_in_sample`` True when the prior came from the row's own season
                       (only possible in the first season of the table)
    """
    if logs.empty:
        raise ValueError("no game logs to compute features from")
    sport = sport or str(logs["sport"].iloc[0])
    df = logs.copy()
    if "_placeholder" not in df.columns:
        df["_placeholder"] = False
    df["_placeholder"] = df["_placeholder"].fillna(False).astype(bool)
    if "position" not in df.columns:
        df["position"] = ""
    if params.positions is not None:
        df = df[df["position"].astype(str).str.upper().isin(params.positions)]
    missing = [s for s in params.stats if s not in df.columns]
    if missing:
        raise KeyError(f"game logs lack stat columns {missing}")

    # Identify players by provider id when available (names can collide).
    pid = df["player_id"].astype(str)
    df["_pid"] = pid.where(pid.str.len() > 0, df["player_key"])
    df = df.sort_values(["_pid", "game_date", "game_id"], kind="stable").reset_index(drop=True)
    df["_group"] = position_group(sport, df["position"])
    n = len(df)

    placeholder = df["_placeholder"].to_numpy()
    live = ~placeholder
    stats = list(params.stats)
    x = df[stats].to_numpy(float)
    x = np.where(np.isnan(x), 0.0, x)
    if params.opportunity is not None:
        opp = df[params.opportunity].to_numpy(float)
        opp = np.where(np.isnan(opp), 0.0, opp)
    else:
        opp = np.ones(n)
    x[placeholder] = 0.0
    opp_obs = np.where(live, opp, 0.0)
    played = live.astype(float)

    player_codes, uniques = pd.factorize(df["_pid"], sort=False)
    n_players = len(uniques)
    new_player = np.ones(n, dtype=bool)
    new_player[1:] = player_codes[1:] != player_codes[:-1]
    rank = df.groupby("_pid", sort=False).cumcount().to_numpy()
    season = df["season"].to_numpy(float)
    dates = df["game_date"].to_numpy("datetime64[D]")
    gap = np.full(n, np.nan)
    gap[1:] = (dates[1:] - dates[:-1]).astype(float)
    gap[new_player] = np.nan
    boundary = np.zeros(n)
    boundary[1:] = season[1:] != season[:-1]
    if params.layoff_days is not None:
        with np.errstate(invalid="ignore"):
            boundary = np.maximum(boundary, (gap > params.layoff_days).astype(float))
    boundary[new_player] = 0.0

    rate_channels = np.column_stack([x, opp_obs, played])
    rate_sums = _recursive_prior_sums(
        rate_channels, params.rate_halflife, params.rate_carry, boundary, player_codes, rank, n_players
    )
    s_x, s_m, s_g = rate_sums[:, : len(stats)], rate_sums[:, len(stats)], rate_sums[:, len(stats) + 1]
    opp_sums = _recursive_prior_sums(
        np.column_stack([opp_obs, played]), params.opp_halflife, params.opp_carry, boundary,
        player_codes, rank, n_players,
    )
    o_num, o_den = opp_sums[:, 0], opp_sums[:, 1]
    n_prior = _recursive_prior_sums(
        played[:, None], 1e12, 1.0, np.zeros(n), player_codes, rank, n_players
    )[:, 0]

    # Priors: previous season's averages for the player's position group.
    rates, opps = _league_priors(df[live], stats, opp_obs[live], "_group")
    seasons_available = sorted({int(k[0]) for k in rates})
    first_season = seasons_available[0] if seasons_available else None
    groups = df["_group"].to_numpy()
    prior_rate = np.zeros((n, len(stats)))
    prior_opp = np.full(n, params.default_opportunity if params.opportunity is not None else 1.0)
    in_sample = np.zeros(n, dtype=bool)
    combos = pd.MultiIndex.from_arrays([df["season"].astype(int), df["_group"]]).unique()
    for season_value, group in combos:
        source = int(season_value) - 1
        used_own = False
        while (source, group) not in rates and first_season is not None and source > first_season:
            source -= 1
        if (source, group) not in rates:
            if (int(season_value), group) in rates:
                source, used_own = int(season_value), True
            else:
                continue
        mask = (season == season_value) & (groups == group)
        prior_rate[mask] = rates[(source, group)]
        if params.opportunity is not None and opps.get((source, group), 0.0) > 0.0:
            prior_opp[mask] = opps[(source, group)]
        in_sample[mask] = used_own

    k = params.rate_prior_strength
    denom = (s_m + k)[:, None]
    # With no history and no prior strength the estimate falls back to the prior.
    rate = np.where(denom > 0, (s_x + k * prior_rate) / np.where(denom > 0, denom, 1.0), prior_rate)
    if params.opportunity is not None:
        k_m = params.opp_prior_strength
        opp_denom = o_den + k_m
        proj_opp = np.where(
            opp_denom > 0, (o_num + k_m * prior_opp) / np.where(opp_denom > 0, opp_denom, 1.0), prior_opp
        )
    else:
        proj_opp = np.ones(n)

    out = df[[c for c in ("sport", "season", "game_id", "game_date", "player_id", "player", "player_key",
                           "team", "opponent", "is_home", "minutes", "position", "season_type")
              if c in df.columns]].copy()
    out["_placeholder"] = placeholder
    out["proj_opp"] = proj_opp
    out["n_prior"] = np.rint(n_prior).astype(int)
    out["eff_games"] = s_g
    out["days_rest"] = gap
    out["prior_in_sample"] = in_sample
    for i, s in enumerate(stats):
        out[f"rate_{s}"] = rate[:, i]
        out[f"mu_{s}"] = rate[:, i] * proj_opp
    for s in stats:
        out[s] = df[s].to_numpy(float)
    return out


def append_next_game_rows(
    logs: pd.DataFrame,
    as_of: Optional[pd.Timestamp] = None,
    active_within_days: float = 21.0,
    season: Optional[int] = None,
) -> pd.DataFrame:
    """Add one empty "next game" row per recently active player.

    The rows are flagged ``_placeholder``; running :func:`compute_features`
    on the result yields each player's projection for a game on ``as_of``.
    Players whose last game is more than ``active_within_days`` before
    ``as_of`` are skipped (injured, traded out of the league, retired).
    """
    if logs.empty:
        raise ValueError("no game logs")
    as_of = pd.Timestamp(as_of) if as_of is not None else logs["game_date"].max() + pd.Timedelta(days=1)
    if as_of.tzinfo is not None:
        as_of = as_of.tz_convert(None)
    as_of = as_of.normalize()
    history = logs[logs["game_date"] < as_of]
    if history.empty:
        raise ValueError(f"no game logs before {as_of.date()}")
    pid = history["player_id"].astype(str)
    key = pid.where(pid.str.len() > 0, history["player_key"])
    last = history.loc[history.assign(_k=key).groupby("_k")["game_date"].idxmax()].copy()
    last = last[(as_of - last["game_date"]).dt.days <= active_within_days]
    nxt = last.copy()
    stat_like = [c for c in nxt.columns if pd.api.types.is_float_dtype(nxt[c])]
    nxt[stat_like] = np.nan
    nxt["game_date"] = as_of
    nxt["game_id"] = "next"
    nxt["season"] = int(season) if season is not None else int(history["season"].max())
    nxt["_placeholder"] = True
    base = history.copy()
    base["_placeholder"] = False
    return pd.concat([base, nxt], ignore_index=True)
