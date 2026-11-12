"""The projection model: game logs in, predictive distributions out.

``project_players`` is the entry point for live use: give it recent game logs
and it returns, for every active player and every market of the sport, a
:class:`~evmodel.projection.base.Projection` - a full distribution from
which the probability of any over/under line can be read.

The mean comes from the walk-forward estimator in
:mod:`evmodel.projection.features` (recent per-minute production x expected
minutes, shrunk towards positional norms); the shape around the mean comes
from the market's fitted :class:`~evmodel.distributions.FamilySpec`.

What this model does **not** know: injuries and lineup changes announced
since a player's last game, matchups, blowout risk, weather.  Sportsbooks
price those in.  Treat the model as a second opinion to the market consensus,
not a replacement for it.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..markets import MarketDef, markets_for_sport
from .base import Projection, ProjectionKey
from .features import FeatureParams, append_next_game_rows, compute_features

__all__ = [
    "SPORT_PARAMS",
    "feature_params",
    "walk_forward",
    "market_mean",
    "market_actual",
    "eligible_rows",
    "project_players",
    "projections_frame",
    "align_to_quotes",
]

#: Estimator settings per sport, tuned on held-out seasons of real game logs
#: (see ``evmodel calibrate --help`` and the README for how).
SPORT_PARAMS: Dict[str, FeatureParams] = {
    "nba": FeatureParams(
        stats=("points", "rebounds", "assists", "threes", "steals", "blocks", "turnovers"),
        opportunity="minutes",
        rate_halflife=25.0,
        rate_carry=0.6,
        rate_prior_strength=50.0,
        opp_halflife=6.0,
        opp_carry=0.3,
        opp_prior_strength=0.5,
        layoff_days=45.0,
        default_opportunity=18.0,
    ),
    "nhl": FeatureParams(
        stats=("goals", "assists", "points", "shots_on_goal", "blocked_shots", "saves"),
        opportunity="minutes",
        rate_halflife=140.0,
        rate_carry=0.4,
        rate_prior_strength=240.0,
        opp_halflife=16.0,
        opp_carry=0.8,
        opp_prior_strength=1.0,
        layoff_days=45.0,
        default_opportunity=14.0,
    ),
    "nfl": FeatureParams(
        stats=(
            "passing_yards", "passing_tds", "pass_attempts", "completions", "interceptions",
            "rushing_yards", "carries", "rushing_tds",
            "receptions", "targets", "receiving_yards", "receiving_tds",
        ),
        opportunity=None,
        rate_halflife=14.0,
        rate_carry=0.4,
        rate_prior_strength=5.0,
        layoff_days=None,
        positions=("QB", "RB", "WR", "TE", "FB"),
    ),
}


def feature_params(sport: str) -> FeatureParams:
    """Estimator settings for a sport (raises ``KeyError`` if unsupported)."""
    sport = sport.lower()
    if sport not in SPORT_PARAMS:
        raise KeyError(f"no projection model for sport {sport!r}; supported: {sorted(SPORT_PARAMS)}")
    return SPORT_PARAMS[sport]


def walk_forward(logs: pd.DataFrame, sport: Optional[str] = None, params: Optional[FeatureParams] = None) -> pd.DataFrame:
    """Pre-game projections for every historical row (for back-tests)."""
    sport = (sport or str(logs["sport"].iloc[0])).lower()
    return compute_features(logs, params or feature_params(sport), sport)


def market_mean(features: pd.DataFrame, market: MarketDef) -> pd.Series:
    """Projected mean of a market: the sum of its component stats' means,
    times the market's fitted correction factor."""
    return features[[f"mu_{s}" for s in market.stats]].sum(axis=1) * market.mean_scale


def market_actual(features: pd.DataFrame, market: MarketDef) -> pd.Series:
    """Realised value of a market for each row."""
    return features[list(market.stats)].sum(axis=1, skipna=False)


def eligible_rows(
    features: pd.DataFrame, market: MarketDef, min_games: int = 8, mean_floor: float = 1.0
) -> pd.Series:
    """Rows where a book would plausibly post this market.

    Requires some history, a projection of at least ``mean_floor`` times the
    market's ``min_mean`` and, where the market is position specific, a
    matching position.
    """
    floor = mean_floor * market.min_mean
    mask = (features["n_prior"] >= min_games) & (market_mean(features, market) >= floor)
    if market.positions and "position" in features.columns:
        mask &= features["position"].astype(str).str.upper().isin(market.positions)
    return mask


def project_players(
    logs: pd.DataFrame,
    sport: Optional[str] = None,
    as_of: Optional[pd.Timestamp] = None,
    markets: Optional[Iterable[str]] = None,
    min_games: int = 5,
    active_within_days: float = 21.0,
    params: Optional[FeatureParams] = None,
    mean_adjustments: Optional[Mapping[ProjectionKey, float]] = None,
) -> Dict[ProjectionKey, Projection]:
    """Project every recently active player's next game.

    Parameters
    ----------
    logs:
        Normalised game logs (ideally the current and previous season).
    as_of:
        Date of the games being projected; only earlier games are used.
        Defaults to the day after the latest game in ``logs``.
    markets:
        Market keys to project (default: all registered for the sport).
    min_games:
        Players with fewer earlier games are skipped.
    mean_adjustments:
        Optional multipliers ``{(player_key, market): factor}`` applied to the
        projected mean - the hook for information the model lacks (a starter
        ruled out, a minutes restriction, a pace-up matchup).

    Returns
    -------
    dict
        ``{(player_key, market_key): Projection}``.  If two active players
        share a name, the one projected for the larger role is kept.
    """
    sport = (sport or str(logs["sport"].iloc[0])).lower()
    params = params or feature_params(sport)
    wanted = set(markets) if markets is not None else None
    definitions = [m for m in markets_for_sport(sport) if wanted is None or m.key in wanted]
    definitions = [m for m in definitions if all(s in params.stats for s in m.stats)]
    if not definitions:
        raise ValueError(f"no projectable markets for sport {sport!r}")

    features = compute_features(
        append_next_game_rows(logs, as_of=as_of, active_within_days=active_within_days), params, sport
    )
    nxt = features[features["_placeholder"] & (features["n_prior"] >= min_games)]
    adjustments = mean_adjustments or {}

    out: Dict[ProjectionKey, Projection] = {}
    for market in definitions:
        means = market_mean(nxt, market).to_numpy(float)
        positions = nxt["position"].astype(str).str.upper().to_numpy() if "position" in nxt.columns else None
        for i, row in enumerate(nxt.itertuples(index=False)):
            mean = float(means[i])
            if not np.isfinite(mean) or mean <= 0.0:
                continue
            if market.positions and positions is not None and positions[i] not in market.positions:
                continue
            key = (row.player_key, market.key)
            mean *= float(adjustments.get(key, 1.0))
            existing = out.get(key)
            if existing is not None and existing.mean >= mean:
                continue
            out[key] = Projection(
                sport=sport,
                player_key=row.player_key,
                player=row.player,
                market=market.key,
                mean=mean,
                spec=market.spec,
                n_games=int(row.n_prior),
                team=str(row.team),
                details={
                    "proj_opportunity": float(row.proj_opp),
                    "eff_games": float(row.eff_games),
                    "days_rest": float(row.days_rest) if np.isfinite(row.days_rest) else float("nan"),
                },
            )
    return out


def projections_frame(projections: Mapping[ProjectionKey, Projection]) -> pd.DataFrame:
    """Projections as a table (one row per player and market)."""
    if not projections:
        return pd.DataFrame(columns=["sport", "player", "player_key", "team", "market", "mean", "median", "p10", "p90", "sd", "n_games"])
    rows = [p.to_dict() for p in projections.values()]
    return pd.DataFrame(rows).sort_values(["player", "market"]).reset_index(drop=True)


def align_to_quotes(
    projections: Mapping[ProjectionKey, Projection], quotes: pd.DataFrame
) -> Dict[ProjectionKey, Projection]:
    """Re-key projections to the player names used by an odds feed.

    Stats feeds and odds feeds spell players differently (``"B. Tkachuk"`` vs
    ``"Brady Tkachuk"``, nicknames, missing suffixes).  Names that already
    agree are passed through; the rest are resolved with
    :class:`evmodel.names.NameMatcher`, which refuses ambiguous matches.
    Returns only projections for players that appear in ``quotes``.
    """
    from ..names import NameMatcher

    if not projections or quotes.empty:
        return {}
    by_player: Dict[str, List[Projection]] = {}
    for projection in projections.values():
        by_player.setdefault(projection.player_key, []).append(projection)
    names = {key: items[0].player for key, items in by_player.items()}
    matcher = NameMatcher(list(names.values()))
    out: Dict[ProjectionKey, Projection] = {}
    quoted = quotes.drop_duplicates("player_key")[["player_key", "player"]]
    for quote_key, quote_name in zip(quoted["player_key"], quoted["player"]):
        stats_key = quote_key if quote_key in by_player else matcher.match(quote_name)
        if stats_key is None or stats_key not in by_player:
            continue
        for projection in by_player[stats_key]:
            out[(quote_key, projection.market)] = (
                projection if stats_key == quote_key else replace(projection, player_key=quote_key)
            )
    return out
