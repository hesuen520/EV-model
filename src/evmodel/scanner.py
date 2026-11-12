"""The scanner: turn a snapshot of quotes into a ranked list of +EV bets.

For every price a book is offering, the scanner asks two independent sources
what the bet is really worth:

* the **market**: the de-vigged consensus of the reference books
  (:class:`evmodel.market.MarketEngine`), leaving out the book being evaluated;
* the **model**: the projection engine's predictive distribution, when
  projections are supplied.

The two are blended in log-odds space (the market gets most of the weight by
default - it knows about injuries and news the model does not), expected
value and a fractional-Kelly stake are computed, and the bets that clear the
configured thresholds are returned best first.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from .config import Config
from .ev import edge as _edge
from .ev import expected_value, fractional_kelly, kelly_fraction
from .market import Consensus, MarketEngine, expit, logit
from .markets import canonical_market_key, find_market
from .odds import decimal_to_american, prob_to_american
from .projection.base import Projection, ProjectionKey
from .schema import OPPORTUNITY_COLUMNS

__all__ = ["evaluate", "scan", "apply_staking", "LOW_CONFIDENCE_FLAGS"]

#: Flags that mark an estimate as less trustworthy (stricter EV threshold,
#: smaller stake).
LOW_CONFIDENCE_FLAGS = frozenset(
    {"line_translated", "market_avg", "model_only", "uncalibrated_translation"}
)


def _american(decimal: float) -> float:
    return float(round(decimal_to_american(decimal)))


def _fair_american(prob: float) -> float:
    if not 0.0 < prob < 1.0:
        return float("nan")
    return float(round(prob_to_american(prob)))


def _blend(
    consensus: Optional[Consensus],
    projection: Optional[Projection],
    line: float,
    model_weight: float,
    full_weight_games: int,
):
    """Combine market and model.

    Returns ``(P(over|no push), P(push), source, market_c, model_c, low, high)``
    where ``low``/``high`` bound ``P(over|no push)`` across de-vig methods.
    """
    market_c = consensus.cond_over if consensus is not None else float("nan")
    model_c = float("nan")
    model_push = 0.0
    if projection is not None:
        over, model_push, under = projection.probs(line)
        if over + under > 0.0:
            model_c = over / (over + under)

    have_market = consensus is not None
    have_model = not math.isnan(model_c)
    if have_market and have_model:
        scale = min(1.0, projection.n_games / max(full_weight_games, 1))
        w = min(max(model_weight * scale, 0.0), 1.0)
        cond = expit((1.0 - w) * logit(market_c) + w * logit(model_c))
        low = expit((1.0 - w) * logit(consensus.cond_over_lo) + w * logit(model_c))
        high = expit((1.0 - w) * logit(consensus.cond_over_hi) + w * logit(model_c))
        push = (1.0 - w) * consensus.push + w * model_push
        source = "blend" if w > 0.0 else "market"
    elif have_market:
        cond, push, source = market_c, consensus.push, "market"
        low, high = consensus.cond_over_lo, consensus.cond_over_hi
    elif have_model:
        cond, push, source = model_c, model_push, "model"
        low = high = model_c
    else:
        return None
    return cond, push, source, market_c, model_c, low, high


def evaluate(
    quotes: pd.DataFrame,
    config: Optional[Config] = None,
    projections: Optional[Mapping[ProjectionKey, Projection]] = None,
    engine: Optional[MarketEngine] = None,
) -> pd.DataFrame:
    """Value every offered side in a quotes snapshot (no filtering).

    Returns one row per (book, prop, line, side) that could be valued, with
    the columns of :data:`evmodel.schema.OPPORTUNITY_COLUMNS`.  ``stake_units``
    is left at zero; :func:`scan` fills it in.
    """
    config = config or Config()
    scan_cfg = config.scan
    engine = engine or MarketEngine(quotes, config.consensus, dfs_books=scan_cfg.dfs_books)
    q = engine.quotes
    dfs = {k.lower(): float(v) for k, v in scan_cfg.dfs_books.items()}
    targets = {b.lower() for b in scan_cfg.target_books} if scan_cfg.target_books else None
    projections = projections or {}

    records: List[Dict[str, object]] = []
    for row in q.itertuples(index=False):
        if targets is not None and row.book not in targets:
            continue
        if row.is_alternate and not scan_cfg.include_alternates:
            continue
        base, _ = canonical_market_key(row.market)
        consensus = engine.consensus(
            row.sport, row.event_id, row.player_key, base, row.line, exclude_book=row.book
        )
        projection = projections.get((row.player_key, base))
        blended = _blend(
            consensus, projection, row.line, scan_cfg.model_weight, scan_cfg.model_full_weight_games
        )
        if blended is None:
            continue
        cond_over, push, source, market_c, model_c, cond_low, cond_high = blended

        flags: List[str] = []
        if source == "model":
            flags.append("model_only")
        if consensus is not None:
            if consensus.kind == "market_avg":
                flags.append("market_avg")
            if consensus.line_gap > 0.0:
                flags.append("line_translated")
                definition = find_market(row.sport, base)
                if definition is None or not definition.calibrated:
                    flags.append("uncalibrated_translation")
            if consensus.n_refs == 1:
                flags.append("single_ref")
            if consensus.disagreement > scan_cfg.disagreement_flag:
                flags.append("refs_disagree")
        if row.is_alternate:
            flags.append("alt_line")
        is_dfs = row.book in dfs
        if is_dfs:
            flags.append("dfs")

        event = f"{row.away_team} @ {row.home_team}" if row.home_team or row.away_team else ""
        for side, offered in (("over", row.over_price), ("under", row.under_price)):
            if offered is None or (isinstance(offered, float) and math.isnan(offered)):
                continue
            price = 1.0 / dfs[row.book] if is_dfs else float(offered)
            cond = cond_over if side == "over" else 1.0 - cond_over
            worst = cond_low if side == "over" else 1.0 - cond_high
            p_win = cond * (1.0 - push)
            side_flags = list(flags)
            if not math.isnan(market_c) and not math.isnan(model_c):
                m_side = market_c if side == "over" else 1.0 - market_c
                x_side = model_c if side == "over" else 1.0 - model_c
                breakeven = 1.0 / price
                if (m_side - breakeven) * (x_side - breakeven) < 0.0:
                    side_flags.append("model_disagrees")
            records.append(
                {
                    "sport": row.sport,
                    "event_id": row.event_id,
                    "commence_time": row.commence_time,
                    "event": event,
                    "player": row.player,
                    "player_key": row.player_key,
                    "market": base,
                    "line": float(row.line),
                    "side": side,
                    "book": row.book,
                    "price": price,
                    "price_american": _american(price),
                    "fair_prob": p_win,
                    "push_prob": push,
                    "fair_price_american": _fair_american(cond),
                    "ev": expected_value(p_win, price, push),
                    "ev_worst": expected_value(worst * (1.0 - push), price, push),
                    "edge": _edge(p_win, price, push),
                    "kelly": kelly_fraction(p_win, price, push),
                    "stake_units": 0.0,
                    "market_prob": (market_c if side == "over" else 1.0 - market_c),
                    "model_prob": (model_c if side == "over" else 1.0 - model_c),
                    "source": source,
                    "n_refs": consensus.n_refs if consensus is not None else 0,
                    "ref_books": consensus.describe_refs() if consensus is not None else "",
                    "ref_weight": consensus.weight if consensus is not None else 0.0,
                    "ref_hold": consensus.hold if consensus is not None else float("nan"),
                    "ref_disagreement": consensus.disagreement if consensus is not None else float("nan"),
                    "line_gap": consensus.line_gap if consensus is not None else float("nan"),
                    "projection": projection.mean if projection is not None else float("nan"),
                    "flags": ",".join(side_flags),
                    "last_update": row.last_update,
                }
            )
    out = pd.DataFrame.from_records(records, columns=OPPORTUNITY_COLUMNS)
    if out.empty:
        out["commence_time"] = pd.to_datetime(out["commence_time"], utc=True)
        out["last_update"] = pd.to_datetime(out["last_update"], utc=True)
    return out


def _is_low_confidence(flags: str) -> bool:
    return bool(LOW_CONFIDENCE_FLAGS.intersection(f for f in str(flags).split(",") if f))


def apply_staking(opportunities: pd.DataFrame, config: Optional[Config] = None) -> pd.DataFrame:
    """Size each bet (fractional Kelly, capped) and enforce exposure limits.

    Bets are processed best-EV first; each is cut back so that the totals per
    player, per game and per scan stay within the configured limits.  Props
    on the same player or game are strongly correlated, so independent Kelly
    stakes would over-bet them.  Bets whose stake falls below ``min_units``
    are dropped.
    """
    config = config or Config()
    st = config.staking
    if opportunities.empty:
        return opportunities.copy()
    df = opportunities.sort_values("ev", ascending=False, kind="stable").copy()

    per_player: Dict[tuple, float] = {}
    per_event: Dict[str, float] = {}
    total = 0.0
    stakes: List[float] = []
    for row in df.itertuples(index=False):
        fraction = fractional_kelly(row.fair_prob, row.price, row.push_prob, fraction=st.kelly_fraction)
        if _is_low_confidence(row.flags):
            fraction *= st.low_confidence_multiplier
        units = min(fraction / st.unit_size, st.max_units)
        player = (row.event_id, row.player_key)
        room = [
            st.max_units_per_player - per_player.get(player, 0.0),
            st.max_units_per_event - per_event.get(row.event_id, 0.0),
        ]
        if st.max_units_total is not None:
            room.append(st.max_units_total - total)
        units = max(min(units, *room), 0.0)
        if st.round_to > 0:
            units = math.floor(units / st.round_to + 1e-9) * st.round_to
        if units < st.min_units:
            units = 0.0
        per_player[player] = per_player.get(player, 0.0) + units
        per_event[row.event_id] = per_event.get(row.event_id, 0.0) + units
        total += units
        stakes.append(round(units, 6))
    df["stake_units"] = stakes
    return df[df["stake_units"] > 0.0].reset_index(drop=True)


def scan(
    quotes: pd.DataFrame,
    config: Optional[Config] = None,
    projections: Optional[Mapping[ProjectionKey, Projection]] = None,
    now: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Find +EV bets in a quotes snapshot.

    Parameters
    ----------
    quotes:
        Quotes table (see :mod:`evmodel.schema`).
    config:
        Thresholds, book weights and staking rules.
    projections:
        Optional ``{(player_key, market): Projection}`` from the projection
        engine, blended with the market consensus.
    now:
        Current time (UTC) for the ``min_minutes_to_start`` filter; defaults
        to the wall clock.

    Returns
    -------
    DataFrame
        One row per recommended bet, best EV first, with stake sizes.
    """
    config = config or Config()
    cfg = config.scan
    valued = evaluate(quotes, config, projections)
    if valued.empty:
        return valued

    american = valued["price_american"]
    keep = (american >= cfg.min_price_american) & (american <= cfg.max_price_american)

    low = valued["flags"].map(_is_low_confidence)
    model_only = valued["source"] == "model"
    threshold = np.where(low, cfg.min_ev_low_confidence, cfg.min_ev)
    threshold = np.where(model_only, np.maximum(threshold, cfg.min_ev_model_only), threshold)
    keep &= valued["ev"] >= threshold
    if cfg.min_ev_worst_case is not None:
        keep &= valued["ev_worst"] >= cfg.min_ev_worst_case
    if not cfg.allow_model_only:
        keep &= ~model_only

    if cfg.min_minutes_to_start is not None:
        now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
        if now.tzinfo is None:
            now = now.tz_localize("UTC")
        starts = valued["commence_time"]
        cutoff = now + pd.Timedelta(minutes=float(cfg.min_minutes_to_start))
        keep &= starts.isna() | (starts >= cutoff)

    picks = valued[keep].sort_values("ev", ascending=False, kind="stable")
    if cfg.best_price_only and not picks.empty:
        picks = picks.drop_duplicates(["event_id", "player_key", "market", "side"], keep="first")
        # Never recommend both sides of the same prop.
        picks = picks.drop_duplicates(["event_id", "player_key", "market"], keep="first")
    return apply_staking(picks.reset_index(drop=True), config)
