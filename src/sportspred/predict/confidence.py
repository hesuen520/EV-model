"""Confidence labels tied to observed out-of-sample results.

A label is looked up from the validation run's confidence table: the band the
selection's probability falls in, how often selections in that band actually
won out of sample, and how many events that rests on.  Without enough
validated events in the band the label is "Unrated" - never a guess.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from ..config import ConfidenceConfig

__all__ = ["confidence_for"]


def confidence_for(prob: float, report: Optional[Mapping[str, Any]], cfg: ConfidenceConfig) -> Dict[str, Any]:
    if report is None or not report.get("confidence_table"):
        return {"label": "Unrated", "basis": "no validation run on file for this league and market"}
    band = None
    for row in report["confidence_table"]:
        if row["prob_low"] - 1e-9 <= prob < row["prob_high"] - 1e-9 or (row["prob_high"] >= 1.0 and prob >= row["prob_low"]):
            band = row
    if band is None:
        return {"label": "Unrated", "basis": f"no validated selections near probability {prob:.2f}"}
    info = {
        "band": [band["prob_low"], band["prob_high"]], "n": band["n"], "observed": band["observed"],
        "observed_low": band["observed_low"], "observed_high": band["observed_high"],
        "mean_predicted": band["mean_predicted"],
    }
    if band["n"] < cfg.min_band_events:
        label = "Unrated"
        basis = (f"only {band['n']} out-of-sample selections in the "
                 f"{band['prob_low']:.2f}-{band['prob_high']:.2f} band (minimum {cfg.min_band_events})")
    else:
        if band["observed_low"] >= cfg.high_observed_low:
            label = "High"
        elif band["observed_low"] >= cfg.medium_observed_low:
            label = "Medium"
        else:
            label = "Low"
        basis = (
            f"out of sample, selections in the {band['prob_low']:.2f}-{band['prob_high']:.2f} band won "
            f"{100 * band['observed']:.1f}% (95% interval {100 * band['observed_low']:.1f}-"
            f"{100 * band['observed_high']:.1f}%, n={band['n']})"
        )
    return {"label": label, "basis": basis, **info}
