"""Price comparison: implied probability, no-vig, fair odds, edge, expected
value - and paper tracking.  Nothing here places a bet.

The module takes a sports-based probability that already exists in the ledger
and compares it with a timestamped, executable price.  It never feeds
anything back into the prediction engine, which cannot import it.

Vocabulary used throughout:

* **raw implied probability** - the price's own break-even: ``1 / decimal
  odds`` for a sportsbook, the ask plus fee for a contract.  Includes the
  venue's margin.
* **no-vig probability** - the margin removed, which needs every outcome of
  the same market from the same venue at the same quote time.  When a side is
  missing the no-vig figure is unavailable (never guessed).
* **fair odds** - ``1 / model probability``.
* **edge** - model probability minus the break-even probability.
* **expected value** - expected profit per unit staked, from the actual
  payout: decimal odds with a refunded push for sportsbooks; ask plus fee for
  a contract paying 1.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from evmodel.devig import devig
from evmodel.ev import expected_value
from evmodel.odds import parse_odds

from ..config import EvConfig, Settings
from ..core.ids import norm_text, stable_id
from ..core.timeutil import age_hours, hst_date, hst_day_bounds_utc, iso_z, parse_ts, utcnow
from .rules import FEES, RULES, comparable, fee_for, settle_outcome

__all__ = ["PriceImportError", "import_prices", "raw_implied", "no_vig", "evaluate", "evaluate_one",
           "open_paper_positions", "settle_paper_positions", "paper_report", "PRICE_TEMPLATE"]

PRICE_TEMPLATE = (
    "# One row per quoted outcome. quoted_at needs a zone. Sportsbook rows use 'price' (American or decimal);\n"
    "# prediction-market rows use 'ask' (and optionally 'bid'), in dollars between 0 and 1.\n"
    "# selection: a participant name, 'draw', 'over' or 'under'. rules_key and fee_key: see `sportspred ev rules`.\n"
    "league,event_id,home,away,date,market,selection,line,venue_kind,venue,price,ask,bid,quoted_at,rules_key,"
    "fee_key,is_executable\n"
    "nba,,Washington Wizards,Detroit Pistons,2026-11-12,winner,Detroit Pistons,,sportsbook,examplebook,-320,,,"
    "2026-11-12T22:00:00Z,moneyline_incl_ot,sportsbook,1\n"
    "nba,,Washington Wizards,Detroit Pistons,2026-11-12,winner,Washington Wizards,,sportsbook,examplebook,+260,,,"
    "2026-11-12T22:00:00Z,moneyline_incl_ot,sportsbook,1\n"
    "nba,,Washington Wizards,Detroit Pistons,2026-11-12,winner,Detroit Pistons,,prediction_market,kalshi,,0.78,0.76,"
    "2026-11-12T22:00:00Z,prediction_market_winner_incl_ot,kalshi_taker,1\n"
)


class PriceImportError(ValueError):
    def __init__(self, path: Any, errors: Sequence[str]) -> None:
        self.errors = list(errors)
        super().__init__(f"{path}: {len(self.errors)} problem(s); nothing was imported\n  "
                         + "\n  ".join(self.errors[:25]))


# ---------------------------------------------------------------- import
def import_prices(store: Any, path: Any, *, source_provider: Optional[str] = None) -> Dict[str, int]:
    """Import timestamped quotes from CSV (template: ``sportspred ev template``).

    The event is identified by ``event_id`` or by ``league`` + ``home`` +
    ``away`` + ``date`` (the Pacific/Honolulu or UTC calendar date of the
    start).  A quote that cannot be tied to exactly one event is an error.
    """
    if isinstance(path, pd.DataFrame):
        df, path = path.astype(str).replace({"None": "", "nan": ""}), "<frame>"
    else:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, comment="#")
    df.columns = [c.strip().lower() for c in df.columns]
    need = ["league", "market", "selection", "venue_kind", "venue", "quoted_at"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise PriceImportError(path, [f"missing required column(s): {', '.join(missing)}"])
    errors: List[str] = []
    rows: List[Dict[str, Any]] = []
    names = store.participant_names()
    cache: Dict[str, pd.DataFrame] = {}
    collected = iso_z(utcnow())
    for i, row in df.iterrows():
        line_no = i + 2
        get = lambda k: (str(row.get(k, "")).strip() or None)   # noqa: E731
        league = get("league")
        if league not in cache:
            cache[league] = store.events(league, with_results=False, include_synthetic=True)
        ev = cache[league]
        match = ev[(ev["event_id"] == get("event_id")) | (ev["provider_event_id"] == get("event_id"))] \
            if get("event_id") else ev[
                ev["home_id"].map(lambda x: norm_text(names.get(x, ""))).eq(norm_text(get("home")))
                & ev["away_id"].map(lambda x: norm_text(names.get(x, ""))).eq(norm_text(get("away")))
                & (ev["start_time_utc"].dt.strftime("%Y-%m-%d").eq(get("date"))
                   | ev["start_time_utc"].dt.tz_convert("Pacific/Honolulu").dt.strftime("%Y-%m-%d").eq(get("date")))]
        if len(match) != 1:
            errors.append(f"line {line_no}: {len(match)} events match (need exactly one)")
            continue
        event = match.iloc[0]
        try:
            quoted = iso_z(parse_ts(get("quoted_at")))
        except ValueError as exc:
            errors.append(f"line {line_no}: quoted_at: {exc}")
            continue
        kind = get("venue_kind")
        if kind not in ("sportsbook", "prediction_market", "exchange"):
            errors.append(f"line {line_no}: venue_kind must be sportsbook, prediction_market or exchange")
            continue
        selection = get("selection") or ""
        sel_id = selection.lower() if selection.lower() in ("draw", "over", "under") else next(
            (pid for pid in (event["home_id"], event["away_id"]) if norm_text(names.get(pid, "")) == norm_text(selection)),
            None)
        if sel_id is None:
            errors.append(f"line {line_no}: selection {selection!r} is not a side of this event, draw, over or under")
            continue
        rules_key, fee_key = get("rules_key"), get("fee_key")
        if rules_key not in RULES:
            errors.append(f"line {line_no}: rules_key {rules_key!r} is unknown (see `sportspred ev rules`)")
        if fee_key and fee_key not in FEES:
            errors.append(f"line {line_no}: fee_key {fee_key!r} is unknown (see `sportspred ev fees`)")
        decimal = contract = bid = None
        try:
            if kind == "prediction_market":
                contract = float(get("ask"))
                bid = None if get("bid") is None else float(get("bid"))
                if not 0.0 < contract < 1.0 or (bid is not None and not 0.0 <= bid <= contract):
                    raise ValueError("ask must be in (0, 1) and bid must not exceed it")
            else:
                decimal = parse_odds(get("price"))
        except (TypeError, ValueError) as exc:
            errors.append(f"line {line_no}: price: {exc}")
            continue
        line = get("line")
        rows.append({
            "quoted_at_utc": quoted, "collected_at_utc": max(collected, quoted), "event_id": event["event_id"],
            "league": league, "market": get("market"), "line": None if line is None else float(line),
            "selection": sel_id, "venue_kind": kind, "venue": get("venue"), "price_decimal": decimal,
            "contract_price": contract, "is_executable": 0 if get("is_executable") == "0" else 1,
            "rules_key": rules_key, "detail_json": json.dumps({"fee_key": fee_key or ("sportsbook" if decimal else None),
                                                                "bid": bid}),
            "is_synthetic": int(event["is_synthetic"]),
        })
    if errors:
        raise PriceImportError(path, errors)
    import hashlib
    from pathlib import Path
    if path == "<frame>":
        source = store.add_source(source_provider or "price_feed", None, sha256=hashlib.sha256(
            df.to_csv(index=False).encode()).hexdigest(), license="see provider terms")
    else:
        data = Path(path).read_bytes()
        source = store.add_source("csv_prices", str(Path(path).resolve()), sha256=hashlib.sha256(data).hexdigest(),
                                  n_bytes=len(data), license="user supplied")
    n = 0
    with store.conn() as con:
        for r in rows:
            pid = stable_id("prc", r["event_id"], r["market"], r["selection"], r["line"] if r["line"] is not None else "",
                            r["venue"], r["quoted_at_utc"])
            cur = con.execute(
                """INSERT OR IGNORE INTO prices (price_id, quoted_at_utc, collected_at_utc, event_id, league, market,
                       line, selection, venue_kind, venue, price_decimal, contract_price, is_executable, rules_key,
                       detail_json, source_id, is_synthetic)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (pid, r["quoted_at_utc"], r["collected_at_utc"], r["event_id"], r["league"], r["market"], r["line"],
                 r["selection"], r["venue_kind"], r["venue"], r["price_decimal"], r["contract_price"],
                 r["is_executable"], r["rules_key"], r["detail_json"], source, r["is_synthetic"]))
            n += cur.rowcount
    return {"quotes_new": n, "quotes_already_on_file": len(rows) - n}


# ------------------------------------------------------------------ maths
def raw_implied(quote: Mapping[str, Any]) -> float:
    """Break-even probability of the quote itself, margin and fee included."""
    if quote["venue_kind"] == "prediction_market":
        fee = fee_for(json.loads(quote["detail_json"] or "{}").get("fee_key")).per_contract(float(quote["contract_price"]))
        return float(quote["contract_price"]) + fee
    return 1.0 / float(quote["price_decimal"])


def no_vig(quotes: pd.DataFrame, quote: Mapping[str, Any], method: str) -> Optional[float]:
    """Margin-free probability of ``quote``'s outcome, or ``None`` when the
    other outcomes of the same market, venue and quote time are not on file."""
    same = quotes[(quotes["event_id"] == quote["event_id"]) & (quotes["market"] == quote["market"])
                  & (quotes["venue"] == quote["venue"]) & (quotes["quoted_at_utc"] == quote["quoted_at_utc"])
                  & (quotes["line"].fillna(-1e9) == (quote["line"] if quote["line"] == quote["line"] and quote["line"] is not None else -1e9))]
    need = 3 if str(quote["market"]).startswith("winner_3way") else 2
    if same["selection"].nunique() != need or len(same) != need:
        return None
    if quote["venue_kind"] == "prediction_market":
        implied = same["contract_price"].astype(float).tolist()
    else:
        implied = (1.0 / same["price_decimal"].astype(float)).tolist()
    if sum(implied) <= 1.0:          # no margin to remove (or an arbitrage): leave as quoted
        fair = [x / sum(implied) for x in implied]
    else:
        fair = devig(implied, method)
    return float(fair[same["selection"].tolist().index(quote["selection"])])


def event_state_reasons(pred: Mapping[str, Any], event: Optional[Mapping[str, Any]], now_iso: str) -> List[str]:
    """Why a recorded prediction can no longer be acted on, judged against its
    event as it stands now (the prediction itself is a snapshot)."""
    if event is None:
        return ["the event is no longer on file"]
    start = iso_z(event["start_time_utc"])
    reasons: List[str] = []
    if event["status"] != "scheduled":
        reasons.append(f"event is now {event['status']}")
    if start != pred["event_start_utc"]:
        reasons.append("the event's start has changed since this prediction; re-run predict")
    if min(start, pred["event_start_utc"]) <= now_iso:
        reasons.append("event has started")
    return reasons


def evaluate_one(pred: Mapping[str, Any], quote: Mapping[str, Any], quotes: pd.DataFrame, cfg: EvConfig,
                 now_iso: str, event_reasons: Sequence[str] = ()) -> Dict[str, Any]:
    """Compare one ledger prediction with one quote of the same selection."""
    p = float(pred["prob"])
    probs = json.loads(pred["probs_json"])
    p_push = float(probs.get("push", 0.0))
    rules = RULES[quote["rules_key"]]
    out: Dict[str, Any] = {
        "prediction_id": pred["prediction_id"], "price_id": quote["price_id"], "league": pred["league"],
        "event_id": pred["event_id"], "market": pred["market"], "line": pred["line"],
        "selection_label": pred["selection_label"], "venue": quote["venue"], "venue_kind": quote["venue_kind"],
        "quoted_at_utc": quote["quoted_at_utc"], "rules_key": quote["rules_key"], "model_prob": p,
        "fair_odds_decimal": 1.0 / p, "rec_status": pred["rec_status"],
    }
    reasons: List[str] = list(event_reasons)
    ok, why = comparable(pred["market"], rules, pred.get("sport"))
    if not ok:
        return {**out, "status": "not comparable", "eligible": False, "reasons": [why]}
    implied = raw_implied(quote)
    fair_mkt = no_vig(quotes, quote, cfg.devig_method)
    if quote["venue_kind"] == "prediction_market":
        cost = implied                                            # ask + fee, per contract paying 1
        ev = (p + 0.5 * p_push if rules.draw == "half" else p) - cost
        ev_per_unit, breakeven = ev / cost, cost
        price_text = f"ask {float(quote['contract_price']):.3f} + fee {cost - float(quote['contract_price']):.4f}"
    else:
        d = float(quote["price_decimal"])
        ev_per_unit = expected_value(p, d, p_push)
        fee = fee_for(json.loads(quote["detail_json"] or "{}").get("fee_key"))
        if fee.kind == "commission":
            ev_per_unit -= fee.rate * p * (d - 1.0)               # commission on net winnings
        breakeven = (1.0 - p_push) / d
        price_text = f"decimal {d:.3f}"
    out.update({
        "price": price_text, "raw_implied_prob": implied, "no_vig_prob": fair_mkt,
        "breakeven_prob": breakeven, "edge": p - breakeven, "ev_per_unit": ev_per_unit,
    })
    if not int(quote["is_executable"]):
        reasons.append("quote is marked not executable")
    age_min = age_hours(quote["quoted_at_utc"], now_iso) * 60.0
    if age_min > cfg.max_quote_age_minutes:
        reasons.append(f"quote is {age_min:.0f} minutes old (maximum {cfg.max_quote_age_minutes:g})")
    if quote["quoted_at_utc"] >= pred["event_start_utc"]:
        reasons.append("quote was taken at or after the event start")
    if pred["rec_status"] != "FINAL":
        reasons.append(f"prediction status is {pred['rec_status']}, not FINAL")
    for preset in cfg.presets:
        if preset.enabled and pred["league"] in preset.leagues and pred["market"] in preset.markets:
            if not preset.raw_implied_min <= implied <= preset.raw_implied_max:
                reasons.append(
                    f"raw implied probability {implied:.3f} is outside the {preset.name} band "
                    f"{preset.raw_implied_min:.2f}-{preset.raw_implied_max:.2f}")
    if out["edge"] < cfg.min_edge:
        reasons.append(f"edge {out['edge']:+.3f} is below the minimum {cfg.min_edge:.3f}")
    out.update({"status": "evaluated", "eligible": not reasons, "reasons": reasons})
    return out


def _quotes(store: Any, event_ids: Sequence[str]) -> pd.DataFrame:
    if not len(event_ids):
        return store.query("SELECT * FROM prices WHERE 0")
    marks = ",".join("?" * len(event_ids))
    return store.query(f"SELECT * FROM prices WHERE event_id IN ({marks}) ORDER BY quoted_at_utc", list(event_ids))


def evaluate(store: Any, *, settings: Optional[Settings] = None, day: Any = None, now: Any = None,
             include_synthetic: bool = False) -> pd.DataFrame:
    """Evaluate the latest prediction of every event on an HST slate against
    the most recent quote of its selection at each venue (quoted at or before
    ``now``).  A prediction with no quote gets ``status = "EV unavailable"``
    - a price is never assumed."""
    settings = settings or Settings()
    now_dt = utcnow() if now is None else parse_ts(now)
    now_iso = iso_z(now_dt)
    preds = store.predictions(slate=day or hst_date(now_dt), latest_only=True, include_synthetic=include_synthetic)
    preds = preds[preds["rec_status"] != "PASS"]
    quotes = _quotes(store, preds["event_id"].unique().tolist())
    quotes = quotes[quotes["quoted_at_utc"] <= now_iso]
    current = store.events(event_ids=preds["event_id"].unique().tolist(), with_results=False,
                           include_synthetic=True).set_index("event_id")
    rows: List[Dict[str, Any]] = []
    for _, pred in preds.iterrows():
        event = current.loc[pred["event_id"]] if pred["event_id"] in current.index else None
        stale = event_state_reasons(pred, event, now_iso)
        mine = quotes[(quotes["event_id"] == pred["event_id"]) & (quotes["market"] == pred["market"])
                      & (quotes["selection"] == pred["selection"])]
        if pred["line"] == pred["line"] and pred["line"] is not None:
            mine = mine[mine["line"] == pred["line"]]              # the exact line, never a nearby one
        if mine.empty:
            rows.append({"prediction_id": pred["prediction_id"], "league": pred["league"],
                         "event_id": pred["event_id"], "market": pred["market"], "line": pred["line"],
                         "selection_label": pred["selection_label"], "model_prob": pred["prob"],
                         "fair_odds_decimal": 1.0 / float(pred["prob"]), "rec_status": pred["rec_status"],
                         "status": "EV unavailable", "eligible": False,
                         "reasons": ["no price on file for this selection at this exact line"]})
            continue
        for _, quote in mine.sort_values("quoted_at_utc").groupby("venue").tail(1).iterrows():
            rows.append(evaluate_one(pred, quote, quotes, settings.ev, now_iso, stale))
    return pd.DataFrame(rows)


# ------------------------------------------------------------ paper tracking
def open_paper_positions(store: Any, evaluations: pd.DataFrame, *, settings: Optional[Settings] = None,
                         now: Any = None) -> int:
    """Record eligible evaluations as PAPER positions.  No order is sent
    anywhere; this only writes rows to the local store."""
    settings = settings or Settings()
    if evaluations.empty:
        return 0
    n = 0
    with store.conn() as con:
        for _, e in evaluations[evaluations["eligible"]].iterrows():
            pid = stable_id("pos", e["prediction_id"], e["price_id"])
            detail = {k: (None if isinstance(e[k], float) and e[k] != e[k] else e[k]) for k in
                      ("model_prob", "raw_implied_prob", "no_vig_prob", "breakeven_prob", "edge", "ev_per_unit",
                       "fair_odds_decimal", "price", "venue", "venue_kind", "rules_key")}
            cur = con.execute(
                "INSERT OR IGNORE INTO paper_positions (position_id, created_at_utc, prediction_id, price_id, stake, "
                "detail_json, mode) VALUES (?,?,?,?,?,?, 'paper')",
                (pid, iso_z(now or utcnow()), e["prediction_id"], e["price_id"], float(settings.ev.paper_stake),
                 json.dumps(detail, default=str)))
            n += cur.rowcount
    return n


def settle_paper_positions(store: Any, *, now: Any = None) -> pd.DataFrame:
    """Settle open paper positions from verified results under each quote's
    rules.  A changed outcome (result correction) appends a new settlement
    linked to the one it replaces."""
    pos = store.query(
        "SELECT p.*, q.rules_key, q.venue_kind, q.price_decimal, q.contract_price, q.detail_json AS quote_detail, "
        "q.selection, q.market, q.line, q.event_id FROM paper_positions p JOIN prices q ON q.price_id = p.price_id")
    if pos.empty:
        return pd.DataFrame()
    events = store.events(event_ids=pos["event_id"].unique().tolist(), with_results=True,
                          include_synthetic=True).set_index("event_id")
    prior = store.query("SELECT * FROM settlements ORDER BY settlement_id")
    last = {r["position_id"]: r for _, r in prior.iterrows()}
    out = []
    for _, p in pos.iterrows():
        ev = {**events.loc[p["event_id"]].to_dict(), "event_id": p["event_id"]}
        outcome, note = settle_outcome(RULES[p["rules_key"]], p["market"], p["selection"],
                                       None if p["line"] != p["line"] else p["line"], ev)
        if outcome == "unsettled":
            continue
        stake = float(p["stake"])
        if p["venue_kind"] == "prediction_market":
            cost = float(p["contract_price"]) + fee_for(json.loads(p["quote_detail"] or "{}").get("fee_key")) \
                .per_contract(float(p["contract_price"]))
            contracts = stake / cost
            payout = {"win": contracts, "loss": 0.0, "half": 0.5 * contracts}.get(outcome, stake)
        else:
            d = float(p["price_decimal"])
            fee = fee_for(json.loads(p["quote_detail"] or "{}").get("fee_key"))
            win = stake * d - (fee.rate * stake * (d - 1.0) if fee.kind == "commission" else 0.0)
            payout = {"win": win, "loss": 0.0, "half": 0.5 * (win + stake)}.get(outcome, stake)
        prev = last.get(p["position_id"])
        if prev is not None and prev["outcome"] == outcome and abs(float(prev["payout"]) - payout) < 1e-9:
            continue
        with store.conn() as con:
            con.execute(
                "INSERT INTO settlements (position_id, settled_at_utc, outcome, payout, pnl, rules_key, rule_notes, "
                "result_source_id, supersedes_settlement_id) VALUES (?,?,?,?,?,?,?,?,?)",
                (p["position_id"], iso_z(now or utcnow()), outcome, payout, payout - stake, p["rules_key"], note,
                 ev.get("result_source_id"), None if prev is None else int(prev["settlement_id"])))
        out.append({"position_id": p["position_id"], "outcome": outcome, "stake": stake, "payout": payout,
                    "pnl": payout - stake, "resettled": prev is not None, "note": note})
    return pd.DataFrame(out)


def paper_report(store: Any) -> Dict[str, Any]:
    """Price-dependent paper results.  Kept apart from prediction accuracy
    and calibration, which `sportspred review` reports without any price."""
    pos = store.query("SELECT * FROM paper_positions")
    st = store.query("SELECT * FROM settlements ORDER BY settlement_id")
    latest = st.loc[~st["position_id"].duplicated(keep="last")] if len(st) else st
    out: Dict[str, Any] = {"mode": "paper", "positions": int(len(pos)), "settled": int(len(latest)),
                           "open": int(len(pos) - len(latest))}
    if len(latest):
        staked = float(pos.set_index("position_id").loc[latest["position_id"], "stake"].sum())
        out.update({"outcomes": latest["outcome"].value_counts().to_dict(), "staked": staked,
                    "pnl": float(latest["pnl"].sum()), "roi": float(latest["pnl"].sum() / staked) if staked else None,
                    "note": "Small samples say little: at typical prices hundreds of positions are needed before "
                            "a return is distinguishable from luck."})
        ev = [json.loads(x).get("ev_per_unit") for x in pos.set_index("position_id").loc[latest["position_id"], "detail_json"]]
        out["expected_roi_at_entry"] = float(np.mean([x for x in ev if x is not None])) if ev else None
    return out
