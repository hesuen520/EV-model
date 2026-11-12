"""The market engine: fair probabilities from other books' prices.

The idea behind "+EV betting" against a sportsbook is that some books are much
better at pricing than others.  If a sharp book makes a player's points
over 24.5 a 54% proposition once its margin is removed, and another book still
pays +105 on that over, the second book's price is a positive-expectation bet
*if the sharp book is right*.

For every prop this module

1. takes each reference book's two-way price and removes the vig
   (:mod:`evmodel.devig`), giving that book's fair ``P(over)`` at its own line;
2. when a reference book's line differs from the line being evaluated, moves
   the probability across lines through the market's distribution family
   (:meth:`evmodel.distributions.FamilySpec.solve_mean`);
3. combines the reference books in log-odds space with their weights.

A price is only ever called +EV on the strength of *other* books: the book
being evaluated cannot create its own edge.  Its own quote can, however,
shrink an edge - see :meth:`MarketEngine.consensus`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ConsensusConfig
from .devig import devig, hold as _hold
from .distributions import FamilySpec, is_integer_line
from .markets import DEFAULT_SPEC, canonical_market_key, find_market
from .schema import normalize_quotes, prop_id

__all__ = ["Reference", "Consensus", "MarketEngine", "logit", "expit", "pool_logit"]

_P_CLIP = 1e-6
_LINE_TOL = 1e-9


def logit(p: float) -> float:
    p = min(max(p, _P_CLIP), 1.0 - _P_CLIP)
    return math.log(p / (1.0 - p))


def expit(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def pool_logit(probs: List[float], weights: List[float]) -> float:
    """Weighted average of probabilities in log-odds space."""
    total = sum(weights)
    if not total > 0.0:
        raise ValueError("weights must sum to a positive number")
    return expit(sum(w * logit(p) for p, w in zip(probs, weights)) / total)


@dataclass
class Reference:
    """One book's de-vigged view of a prop, at that book's own line."""

    book: str
    line: float
    fair_over: float  # P(over | no push) under the configured de-vig method
    fair_over_lo: float  # lowest P(over | no push) any de-vig method gives
    fair_over_hi: float  # highest
    hold: float
    weight: float  # configured sharpness weight (0 for non-reference books)
    over_price: float
    under_price: float
    last_update: Optional[pd.Timestamp] = None
    _spec: FamilySpec = field(default=DEFAULT_SPEC, repr=False)
    _mean: Optional[float] = field(default=None, repr=False)

    @property
    def mean(self) -> float:
        """Mean of the stat implied by this quote under the market's family.

        Raises ``ValueError`` if the family cannot reproduce the quote.
        """
        if self._mean is None:
            self._mean = self._spec.solve_mean(self.line, self.fair_over)
        return self._mean


@dataclass(frozen=True)
class Consensus:
    """Fair probabilities for one line of one prop."""

    line: float
    over: float
    push: float
    under: float
    #: bounds on P(over | no push) across de-vig methods: the choice of
    #: method is a modelling assumption, and these show how much it matters
    cond_over_lo: float
    cond_over_hi: float
    kind: str  # "sharp" (weighted reference books) or "market_avg" (fallback)
    n_refs: int
    weight: float
    hold: float
    disagreement: float  # weighted sd of the references' P(over | no push)
    line_gap: float  # largest |reference line - line| used
    refs: Tuple[Tuple[str, float, float, float], ...]  # (book, line, P(over|no push) at target, weight)

    @property
    def cond_over(self) -> float:
        """P(over | no push)."""
        denom = self.over + self.under
        return self.over / denom if denom > 0 else 0.5

    def prob(self, side: str) -> float:
        """P(win) of a side, unconditional (pushes are neither win nor loss)."""
        if side == "over":
            return self.over
        if side == "under":
            return self.under
        raise ValueError(f"side must be 'over' or 'under', got {side!r}")

    def cond_prob(self, side: str) -> float:
        """P(win | no push) of a side."""
        c = self.cond_over
        if side == "over":
            return c
        if side == "under":
            return 1.0 - c
        raise ValueError(f"side must be 'over' or 'under', got {side!r}")

    def worst_cond_prob(self, side: str) -> float:
        """P(win | no push) of a side under the de-vig method least kind to it."""
        if side == "over":
            return self.cond_over_lo
        if side == "under":
            return 1.0 - self.cond_over_hi
        raise ValueError(f"side must be 'over' or 'under', got {side!r}")

    def describe_refs(self) -> str:
        return ",".join(f"{book}:{line:g}" for book, line, _p, _w in self.refs)


class MarketEngine:
    """Builds reference quotes from a snapshot and answers consensus queries."""

    def __init__(
        self,
        quotes: pd.DataFrame,
        config: Optional[ConsensusConfig] = None,
        dfs_books: Optional[Dict[str, float]] = None,
    ) -> None:
        self.config = config or ConsensusConfig()
        self.quotes = normalize_quotes(quotes)
        excluded = {b.lower() for b in self.config.exclude_reference_books}
        excluded |= {b.lower() for b in (dfs_books or {})}
        self._excluded = excluded
        self._weights = {k.lower(): float(v) for k, v in self.config.book_weights.items()}
        self._refs: Dict[str, List[Reference]] = {}
        self._specs: Dict[str, FamilySpec] = {}
        self._build()

    # ------------------------------------------------------------------ build
    def spec_for(self, sport: str, market: str) -> FamilySpec:
        """Distribution family used for a market (generic default if unknown)."""
        definition = find_market(sport, market)
        return definition.spec if definition is not None else DEFAULT_SPEC

    def _build(self) -> None:
        q = self.quotes
        if q.empty:
            return
        cfg = self.config
        two_sided = q[q["over_price"].notna() & q["under_price"].notna()].copy()
        two_sided = two_sided[~two_sided["book"].isin(self._excluded)]
        if two_sided.empty:
            return
        two_sided["market_base"] = two_sided["market"].map(lambda m: canonical_market_key(m)[0])

        if cfg.max_quote_age_minutes is not None and two_sided["last_update"].notna().any():
            newest = q["last_update"].max()
            cutoff = newest - pd.Timedelta(minutes=float(cfg.max_quote_age_minutes))
            two_sided = two_sided[two_sided["last_update"].isna() | (two_sided["last_update"] >= cutoff)]

        q_over = 1.0 / two_sided["over_price"].to_numpy(float)
        q_under = 1.0 / two_sided["under_price"].to_numpy(float)
        total = q_over + q_under
        two_sided["_q_over"] = q_over
        two_sided["_q_under"] = q_under
        two_sided["_hold"] = 1.0 - 1.0 / total
        ok = (total >= cfg.min_ref_total) & (two_sided["_hold"].to_numpy() <= cfg.max_ref_hold)
        two_sided = two_sided[ok]
        if two_sided.empty:
            return

        group_cols = ["sport", "event_id", "player_key", "market_base", "book"]
        for (sport, event_id, pkey, market, book), rows in two_sided.groupby(group_cols, sort=False):
            # One reference per book and prop: its main line (non-alternate if
            # available), the one priced closest to even.
            main = rows[~rows["is_alternate"]]
            candidates = main if not main.empty else rows
            balance = (candidates["_q_over"] - candidates["_q_under"]).abs()
            row = candidates.loc[balance.idxmin()]
            implied = [row["_q_over"], row["_q_under"]]
            fair_over = devig(implied, cfg.devig_method)[0]
            # For two outcomes the multiplicative and power methods bracket
            # every other method, so they give the range of fair values.
            extremes = [fair_over, devig(implied, "multiplicative")[0], devig(implied, "power")[0]]
            pid = prop_id(event_id, pkey, market)
            spec = self._specs.setdefault(pid, self.spec_for(sport, market))
            last_update = row["last_update"] if pd.notna(row["last_update"]) else None
            self._refs.setdefault(pid, []).append(
                Reference(
                    book=book,
                    line=float(row["line"]),
                    fair_over=float(fair_over),
                    fair_over_lo=float(min(extremes)),
                    fair_over_hi=float(max(extremes)),
                    hold=float(row["_hold"]),
                    weight=self._weights.get(book, 0.0),
                    over_price=float(row["over_price"]),
                    under_price=float(row["under_price"]),
                    last_update=last_update,
                    _spec=spec,
                )
            )

    # ---------------------------------------------------------------- queries
    def references(self, event_id: str, player_key: str, market: str) -> List[Reference]:
        """All reference quotes for a prop (one per book)."""
        base, _ = canonical_market_key(market)
        return list(self._refs.get(prop_id(event_id, player_key, base), []))

    def consensus(
        self,
        sport: str,
        event_id: str,
        player_key: str,
        market: str,
        line: float,
        exclude_book: Optional[str] = None,
    ) -> Optional[Consensus]:
        """Fair probabilities at ``line``, or ``None`` if there is no basis.

        ``exclude_book`` is the book whose price is being evaluated.  A
        consensus must be supported by *other* books (the minimum-reference
        rules ignore the target), and those books must outweigh the target
        itself by ``min_ref_weight_ratio``.  When ``include_target_book`` is set, the
        target's own de-vigged quote is then pooled in at its own weight:
        a sharp book's price is evidence too, so apparent edges against a
        sharp book shrink, while a zero-weight soft book never influences
        its own fair value.
        """
        cfg = self.config
        base, _ = canonical_market_key(market)
        everyone = list(self._refs.get(prop_id(event_id, player_key, base), []))
        others = [r for r in everyone if r.book != exclude_book]
        if not others:
            return None
        spec = self.spec_for(sport, base)
        own = [r for r in everyone if r.book == exclude_book] if cfg.include_target_book else []
        target_weight = self._weights.get(str(exclude_book).lower(), 0.0) if exclude_book else 0.0
        needed_weight = max(cfg.min_ref_weight, cfg.min_ref_weight_ratio * target_weight)

        def pool(kind: str, translate: bool) -> Optional[Consensus]:
            """Pool the other books; if they qualify, let the target's own view count too."""
            if kind == "sharp":
                weighted = [(r, r.weight) for r in others if r.weight > 0.0]
                own_weighted = [(r, r.weight) for r in own if r.weight > 0.0]
            else:
                weighted = [(r, 1.0) for r in others]
                own_weighted = [(r, 1.0) for r in own]
            basis = self._pool(weighted, spec, line, kind, translate)
            if basis is None:
                return None
            if kind == "sharp":
                if basis.n_refs < cfg.min_ref_books or basis.weight < needed_weight - 1e-12:
                    return None
            elif basis.n_refs < cfg.fallback_min_books:
                return None
            if not own_weighted:
                return basis
            # The count, weight and reference list keep describing the *other*
            # books; the target's own quote only tempers the probabilities.
            full = self._pool(weighted + own_weighted, spec, line, kind, translate)
            if full is None:
                return basis
            return replace(
                full,
                n_refs=basis.n_refs,
                weight=basis.weight,
                hold=basis.hold,
                disagreement=basis.disagreement,
                line_gap=basis.line_gap,
                refs=basis.refs,
            )

        # Tiers, most trustworthy first.  References quoting exactly this line
        # need no distributional assumption, so translated references are
        # only brought in when the same-line ones are not enough.
        for kind in ("sharp", "market_avg"):
            # The unweighted market average is only a fallback for judging
            # books that carry no reference weight themselves; it never
            # overrules a weighted (sharper) book.
            if kind == "market_avg" and (not cfg.fallback_market_average or target_weight > 0.0):
                break
            for translate in (False, True):
                result = pool(kind, translate)
                if result is not None:
                    return result
        return None

    def _pool(
        self,
        weighted: List[Tuple[Reference, float]],
        spec: FamilySpec,
        line: float,
        kind: str,
        allow_translation: bool = True,
    ) -> Optional[Consensus]:
        cfg = self.config
        integer_line = is_integer_line(line)
        probs: List[float] = []
        lows: List[float] = []
        highs: List[float] = []
        pushes: List[float] = []
        weights: List[float] = []
        base_weights: List[float] = []
        holds: List[float] = []
        used: List[Tuple[str, float, float, float]] = []
        max_gap = 0.0
        for ref, weight in weighted:
            gap = abs(ref.line - line)
            needs_dist = gap >= _LINE_TOL or integer_line
            if gap >= _LINE_TOL and not allow_translation:
                continue
            dist = None
            if needs_dist:
                try:
                    dist = spec.dist(ref.mean)
                except ValueError:
                    continue  # the family cannot reproduce this quote: skip it
            if gap < _LINE_TOL:
                cond = ref.fair_over
                push = dist.prob_push(line) if dist is not None else 0.0
                w = weight
            else:
                sd = dist.sd
                gap_sd = gap / sd if sd > 0 else math.inf
                if gap_sd > cfg.max_line_gap_sd:
                    continue
                over, push, under = dist.probs(line)
                if over + under <= 0.0:
                    continue
                cond = over / (over + under)
                w = weight * math.exp(-gap_sd / cfg.line_gap_scale)
                max_gap = max(max_gap, gap)
            if w <= 0.0:
                continue
            # Carry the reference's de-vig range to this line as a log-odds shift.
            base = logit(ref.fair_over)
            lows.append(expit(logit(cond) + logit(ref.fair_over_lo) - base))
            highs.append(expit(logit(cond) + logit(ref.fair_over_hi) - base))
            probs.append(cond)
            pushes.append(push)
            weights.append(w)
            base_weights.append(weight)
            holds.append(ref.hold)
            used.append((ref.book, ref.line, cond, w))
        if not probs:
            return None
        total = sum(weights)
        cond_over = pool_logit(probs, weights)
        push = sum(w * p for w, p in zip(weights, pushes)) / total
        mean_p = sum(w * p for w, p in zip(weights, probs)) / total
        disagreement = math.sqrt(sum(w * (p - mean_p) ** 2 for w, p in zip(weights, probs)) / total)
        return Consensus(
            line=float(line),
            over=cond_over * (1.0 - push),
            push=push,
            under=(1.0 - cond_over) * (1.0 - push),
            cond_over_lo=min(pool_logit(lows, weights), cond_over),
            cond_over_hi=max(pool_logit(highs, weights), cond_over),
            kind=kind,
            n_refs=len(probs),
            weight=sum(base_weights),
            hold=sum(w * h for w, h in zip(weights, holds)) / total,
            disagreement=disagreement,
            line_gap=max_gap,
            refs=tuple(used),
        )

    def implied_mean(
        self, sport: str, event_id: str, player_key: str, market: str, exclude_book: Optional[str] = None
    ) -> Optional[float]:
        """Weighted market-implied mean of the stat (the market's "projection")."""
        refs = [r for r in self.references(event_id, player_key, market) if r.book != exclude_book]
        means = []
        for r in refs:
            try:
                means.append((r.mean, r.weight))
            except ValueError:
                continue
        sharp = [(m, w) for m, w in means if w > 0.0]
        pool = sharp or [(m, 1.0) for m, _w in means]
        if not pool:
            return None
        total = sum(w for _m, w in pool)
        return sum(m * w for m, w in pool) / total
