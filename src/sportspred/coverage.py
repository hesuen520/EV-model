"""The coverage table: what is actually operational, computed from the store.

A (sport, league, market) status is never typed in by hand.  It follows from
what the plugin declares and from what this installation has ingested and
validated, so the table cannot claim more than has been demonstrated.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pandas as pd

from .core.enums import COVERAGE_DEFINITIONS, CoverageStatus
from .sports import all_leagues, get_league
from .storage.store import is_demo_league

__all__ = ["coverage_status", "coverage_table", "coverage_markdown"]


def _latest_runs(store: Any) -> Dict[tuple, Dict[str, Any]]:
    runs = store.validation_runs()
    out: Dict[tuple, Dict[str, Any]] = {}
    for _, row in runs.iterrows():   # ordered by time; later rows overwrite earlier ones
        out[(row["league"], row["market"])] = row.to_dict()
    return out


def _status(league: str, implemented: bool, run: Optional[Dict[str, Any]], n_real_events: int) -> tuple:
    if not implemented:
        return CoverageStatus.UNAVAILABLE, "no model implemented for this market"
    if is_demo_league(league):
        return CoverageStatus.DEMO_ONLY, "synthetic demo league"
    if run is None:
        if n_real_events:
            return CoverageStatus.ADAPTER_ONLY, f"{n_real_events} real events ingested; not validated yet"
        return CoverageStatus.ADAPTER_ONLY, "no real data ingested in this installation"
    if int(run["is_synthetic"]):
        return CoverageStatus.DEMO_ONLY, "only validated on synthetic data"
    if int(run["gates_passed"]):
        return CoverageStatus.VALIDATED, "all validation gates passed"
    report = json.loads(run["report_json"])
    failed = [g["gate"] for g in report.get("gates", []) if not g["passed"]]
    return CoverageStatus.EXPERIMENTAL, "gates not met: " + ", ".join(failed)


def coverage_status(store: Any, league: str, market: str) -> CoverageStatus:
    plugin, spec = get_league(league)
    decl = spec.market(market)
    run = _latest_runs(store).get((league, market))
    n = int(store.query("SELECT COUNT(*) AS n FROM events WHERE league=? AND is_synthetic=0", [league])["n"][0])
    return _status(league, decl.implemented, run, n)[0]


def coverage_table(store: Any) -> pd.DataFrame:
    """One row per declared (sport, league, market)."""
    runs = _latest_runs(store)
    counts = store.query(
        "SELECT league, COUNT(*) AS n, MAX(start_time_utc) AS last FROM events "
        "WHERE is_synthetic=0 AND status='final' GROUP BY league")
    n_by = dict(zip(counts["league"], counts["n"]))
    last_by = dict(zip(counts["league"], counts["last"]))
    rows: List[Dict[str, Any]] = []
    for plugin, spec in all_leagues():
        for decl in spec.markets:
            run = runs.get((spec.key, decl.key))
            status, basis = _status(spec.key, decl.implemented, run, int(n_by.get(spec.key, 0)))
            row: Dict[str, Any] = {
                "sport": spec.sport, "league": spec.key, "competition": spec.name, "gender": spec.gender,
                "level": spec.level, "discipline": spec.discipline or "", "market": decl.key,
                "status": status.value, "basis": basis,
                "real_events_final": int(n_by.get(spec.key, 0)),
                "results_through": (last_by.get(spec.key) or "")[:10],
                "data_source": spec.real_data_source or "CSV import only",
                "model": "", "final_test_n": None, "final_test_accuracy": None,
                "final_test_log_loss": None, "validated_at": "",
            }
            if run is not None:
                report = json.loads(run["report_json"])
                sel = report.get("final_test", {}).get("selected", {})
                # A yes-or-no question about every starter of a race is mostly "no": its accuracy
                # says nothing, so it is left out here (the report gives per-race figures).
                per_entrant = decl.key.startswith("finish:")
                row.update({
                    "model": report.get("selected_model") or "",
                    "final_test_n": sel.get("n"),
                    "final_test_accuracy": None if per_entrant else sel.get("accuracy"),
                    "final_test_log_loss": sel.get("log_loss"), "validated_at": run["created_at_utc"][:10],
                })
            rows.append(row)
    return pd.DataFrame(rows)


def coverage_markdown(store: Any) -> str:
    table = coverage_table(store)
    lines = [
        "# Coverage",
        "",
        "Generated from the registry and this installation's store. A status is earned by "
        "ingested data and recorded validation runs; it is never set by hand.",
        "",
        "| Status | Meaning |",
        "|---|---|",
    ]
    lines += [f"| {s.value} | {text} |" for s, text in COVERAGE_DEFINITIONS.items()]
    lines += ["", "| Sport | League | Market | Status | Basis | Final-test n | Accuracy | Log loss | Results through | Data source |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        acc = "" if pd.isna(r["final_test_accuracy"]) else f"{100 * r['final_test_accuracy']:.1f}%"
        ll = "" if pd.isna(r["final_test_log_loss"]) else f"{r['final_test_log_loss']:.4f}"
        n = "" if pd.isna(r["final_test_n"]) else f"{int(r['final_test_n'])}"
        lines.append(
            f"| {r['sport']} | {r['competition']} (`{r['league']}`) | {r['market']} | **{r['status']}** | "
            f"{r['basis']} | {n} | {acc} | {ll} | {r['results_through']} | {r['data_source']} |")
    return "\n".join(lines) + "\n"
