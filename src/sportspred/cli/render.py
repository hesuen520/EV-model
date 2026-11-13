"""Plain-text rendering of predictions for the terminal."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping

import pandas as pd

from ..core.timeutil import fmt_hst

__all__ = ["prediction_from_row", "render_prediction", "render_slate_table"]

_JSON_FIELDS = ("probs", "reasons", "checks", "confidence", "factors", "features", "data_quality", "sources",
                "display")


def prediction_from_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """A ledger row (JSON text columns) as a prediction record (objects)."""
    rec = dict(row)
    for key in _JSON_FIELDS:
        if f"{key}_json" in rec:
            rec[key] = json.loads(rec.pop(f"{key}_json"))
    return rec


def _pct(x: Any) -> str:
    return "n/a" if x is None or x != x else f"{100 * float(x):.1f}%"


def _line(value: Any) -> str:
    """' +3.5' for a market with a line; nothing when there is none (stored as NULL, read back as NaN)."""
    return "" if value is None or value != value else f" {value:+g}"


def render_slate_table(preds: Iterable[Mapping[str, Any]]) -> str:
    rows = []
    for p in preds:
        d = p["display"]
        rows.append({
            "start (HST)": d["start_hst"], "league": d["league"],
            "event": f"{d['side_b']} at {d['side_a']}" if d.get("side_a_role") == "home"
            else f"{d['side_a']} v {d['side_b']}",
            "market": p["market"] + _line(p.get("line")),
            "selection": p["selection_label"], "prob": _pct(p.get("prob")),
            "model range": f"{_pct(p.get('prob_low'))}-{_pct(p.get('prob_high'))}",
            "status": p["rec_status"], "confidence": p["confidence"].get("label", ""),
            "validation": p["validation_status"],
        })
    if not rows:
        return "No predictions."
    return pd.DataFrame(rows).to_string(index=False)


def _bullets(title: str, items: List[str]) -> List[str]:
    if not items:
        return []
    return [f"  {title}:"] + [f"    - {x}" for x in items]


def render_prediction(p: Mapping[str, Any]) -> str:
    d, c = p["display"], p["confidence"]
    head = f"{d['side_b']} at {d['side_a']}" if d.get("side_a_role") == "home" else f"{d['side_a']} v {d['side_b']}"
    venue = ", ".join(x for x in (d.get("venue"), d.get("venue_city")) if x) or "venue not published"
    site = "" if d.get("neutral_site") is None else (" (neutral site)" if d["neutral_site"] else "")
    line = _line(p.get("line"))
    lines = [
        f"{d['competition']} | {head}",
        f"  Start: {d['start_hst']} | Venue: {venue}{site}",
        f"  Market: {p['market']}{line} ({d.get('market_note') or ''})",
        f"  Status: {p['rec_status']} - {d.get('status_definition', '')}",
        f"  Selection: {p['selection_label']}  model probability {_pct(p.get('prob'))}"
        f"  (validation status: {p['validation_status']}; model {p['model_version']})",
    ]
    ev_unc = c.get("event_uncertainty", {})
    mod_unc = c.get("model_uncertainty", {})
    lines += [
        "  Uncertainty:",
        f"    - event: the selection loses {_pct(ev_unc.get('prob_selection_loses'))} of the time even if the "
        "probability is exactly right",
        f"    - model: {_pct(p.get('prob_low'))} to {_pct(p.get('prob_high'))} "
        f"({mod_unc.get('method', 'n/a')})",
        f"    - data quality: {p['data_quality'].get('grade', 'n/a')}",
        f"  Confidence: {c.get('label')} - {c.get('basis', '')}",
    ]
    lines += _bullets("Why this status", p["reasons"])
    f = p["factors"]
    lines += _bullets("Supporting factors", f.get("supporting", []))
    lines += _bullets("Opponent strength", f.get("opponent_strength", []))
    lines += _bullets("Upset risks", f.get("upset_risks", []))
    lines += _bullets("Availability assumptions", f.get("availability_assumptions", []))
    lines += _bullets("Missing data", f.get("missing_data", []) + p["data_quality"].get("missing_features", []))
    lines += _bullets("Checks", [f"{k['label']} ({k['subject']}): {k['state']} - {k['detail']}" for k in p["checks"]])
    lines += _bullets("Sources", [
        f"{s['provider']}: {s['url']} (collected {fmt_hst(s['collected_at_utc'])}"
        + (f", published {fmt_hst(s['published_at_utc'])}" if s.get("published_at_utc") else "") + ")"
        for s in p["sources"]])
    lines.append(f"  Predicted: {d.get('predicted_at_hst') or d.get('last_check_hst', 'n/a')} | newest source check: "
                 f"{d.get('last_source_check_hst') or 'no source read'} | prediction id "
                 f"{p.get('prediction_id', 'n/a')}")
    return "\n".join(lines)
