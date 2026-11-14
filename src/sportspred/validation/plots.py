"""Validation charts as dependency-free SVG.

Two charts, used by the dashboard and by the generated validation documents:

* a **reliability (calibration) chart** - predicted probability of the
  selected outcome against how often it happened out of sample, one dot per
  probability band, sized by the number of events, with a 95% Wilson interval
  and the perfect-calibration diagonal as a thin reference;
* an **improvement-by-season chart** - log-loss improvement of the selected
  model over a baseline in every season, so a result that only holds on
  average is visible as such.

Colours are CSS custom properties with light and dark values, so the same SVG
is readable in both themes.  Every mark carries a native ``<title>`` tooltip
and ``data-tip`` text for the dashboard's hover layer; the numbers are also
given as a table by the caller.
"""

from __future__ import annotations

import html
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

__all__ = ["CHART_CSS", "reliability_svg", "improvement_svg", "standalone"]

#: Roles used by the charts (reference palette: blue slot 1, neutral chrome).
_LIGHT = "--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--series:#2a78d6"
_DARK = "--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--series:#3987e5"
_RULES = """
.viz svg{display:block;width:100%;height:auto}
.viz text{font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
.viz .bg{fill:var(--surface)}.viz .grid{stroke:var(--grid);stroke-width:1}.viz .axis{stroke:var(--axis);stroke-width:1}
.viz .ref{stroke:var(--muted);stroke-width:1}.viz .tick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.viz .lab{fill:var(--ink2);font-size:12px}.viz .title{fill:var(--ink);font-size:13px;font-weight:600}
.viz .whisker{stroke:var(--series);stroke-width:2;stroke-linecap:round;opacity:.45}
.viz .dot{fill:var(--series);stroke:var(--surface);stroke-width:2}.viz .bar{fill:var(--series)}
.viz .hit{fill:transparent;cursor:default}.viz .hit:focus{outline:none}
.viz .hit:hover + .mark,.viz .hit:focus + .mark{opacity:.75}
"""
CHART_CSS = (
    ".viz{" + _LIGHT + "}\n"
    "@media (prefers-color-scheme: dark){:root:where(:not([data-theme=\"light\"])) .viz{" + _DARK + "}}\n"
    ":root[data-theme=\"dark\"] .viz{" + _DARK + "}" + _RULES
)
_STANDALONE_CSS = (".viz{" + _LIGHT + "}@media (prefers-color-scheme: dark){.viz{" + _DARK + "}}"
                   + _RULES.replace(".viz svg{display:block;width:100%;height:auto}", ""))


def _esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def reliability_svg(rows: Sequence[Mapping[str, Any]], *, title: str = "Calibration of selections",
                    width: int = 460, height: int = 340) -> str:
    """``rows``: reliability table rows (``mean_predicted, observed,
    observed_low, observed_high, n``)."""
    rows = [r for r in rows if r.get("n")]
    if not rows:
        return '<div class="viz"><p class="lab">No out-of-sample selections to plot.</p></div>'
    left, right, top, bottom = 48, 16, 30, 44
    lo = math.floor(min(min(r["mean_predicted"], r["observed_low"]) for r in rows) * 10) / 10
    lo = max(0.0, min(lo, 0.9))
    w, h = width - left - right, height - top - bottom
    sx = lambda v: left + (v - lo) / (1 - lo) * w            # noqa: E731
    sy = lambda v: top + (1 - (v - lo) / (1 - lo)) * h       # noqa: E731
    n_max = max(r["n"] for r in rows)
    ticks = [round(lo + i * (1 - lo) / 5, 2) for i in range(6)]
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_esc(title)}">',
           f'<rect class="bg" width="{width}" height="{height}"/>',
           f'<text class="title" x="{left}" y="18">{_esc(title)}</text>']
    for t in ticks:
        out.append(f'<line class="grid" x1="{left}" x2="{left + w}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>')
        out.append(f'<text class="tick" x="{left - 6}" y="{sy(t) + 4:.1f}" text-anchor="end">{100 * t:.0f}%</text>')
        out.append(f'<text class="tick" x="{sx(t):.1f}" y="{top + h + 16}" text-anchor="middle">{100 * t:.0f}%</text>')
    out.append(f'<line class="axis" x1="{left}" x2="{left + w}" y1="{top + h}" y2="{top + h}"/>')
    out.append(f'<line class="ref" x1="{sx(lo):.1f}" y1="{sy(lo):.1f}" x2="{sx(1):.1f}" y2="{sy(1):.1f}"/>')
    out.append(f'<text class="tick" x="{sx(1) - 4:.1f}" y="{sy(1) + 14:.1f}" text-anchor="end">perfect calibration</text>')
    out.append(f'<text class="lab" x="{left + w / 2:.1f}" y="{height - 6}" text-anchor="middle">'
               'Predicted probability of the selection</text>')
    out.append(f'<text class="lab" transform="translate(13 {top + h / 2:.1f}) rotate(-90)" text-anchor="middle">'
               'How often it won</text>')
    for r in rows:
        x, y = sx(r["mean_predicted"]), sy(r["observed"])
        radius = 4 + 5 * math.sqrt(r["n"] / n_max)
        tip = (f"Predicted {_pct(r['mean_predicted'])}, won {_pct(r['observed'])} "
               f"(95% interval {_pct(r['observed_low'])} to {_pct(r['observed_high'])}), {r['n']:,} selections")
        out.append(f'<line class="whisker" x1="{x:.1f}" x2="{x:.1f}" y1="{sy(r["observed_low"]):.1f}" '
                   f'y2="{sy(r["observed_high"]):.1f}"/>')
        out.append(f'<circle class="hit" tabindex="0" cx="{x:.1f}" cy="{y:.1f}" r="{radius + 8:.1f}" '
                   f'data-tip="{_esc(tip)}"><title>{_esc(tip)}</title></circle>')
        out.append(f'<circle class="dot mark" cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" pointer-events="none"/>')
    out.append("</svg>")
    return '<div class="viz">' + "".join(out) + "</div>"


def improvement_svg(rows: Sequence[Mapping[str, Any]], *, reference: str, title: Optional[str] = None,
                    width: int = 460, height: int = 300) -> str:
    """``rows``: ``block, n, improvement`` (log loss of the reference minus
    log loss of the selected model; above zero means the model was better)."""
    rows = list(rows)
    if not rows:
        return '<div class="viz"><p class="lab">No seasons to plot.</p></div>'
    title = title or f"Log-loss improvement over {reference}, by season"
    left, right, top, bottom = 52, 12, 30, 40
    w, h = width - left - right, height - top - bottom
    vmax = max(max(abs(r["improvement"]) for r in rows), 1e-6)
    step = 10 ** math.floor(math.log10(vmax))
    step = step * (5 if vmax / step >= 5 else 2 if vmax / step >= 2 else 1) / 2
    hi = math.ceil(max(r["improvement"] for r in rows) / step) * step
    lo = math.floor(min(min(r["improvement"] for r in rows), 0.0) / step) * step
    hi = max(hi, step)
    sy = lambda v: top + (hi - v) / (hi - lo) * h            # noqa: E731
    band = w / len(rows)
    bar = min(24.0, band * 0.62)
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_esc(title)}">',
           f'<rect class="bg" width="{width}" height="{height}"/>',
           f'<text class="title" x="{left}" y="18">{_esc(title)}</text>']
    v = lo
    while v <= hi + 1e-12:
        out.append(f'<line class="grid" x1="{left}" x2="{left + w}" y1="{sy(v):.1f}" y2="{sy(v):.1f}"/>')
        out.append(f'<text class="tick" x="{left - 6}" y="{sy(v) + 4:.1f}" text-anchor="end">{v:+.3f}</text>')
        v += step
    out.append(f'<line class="axis" x1="{left}" x2="{left + w}" y1="{sy(0):.1f}" y2="{sy(0):.1f}"/>')
    every = max(1, math.ceil(len(rows) / 9))
    for i, r in enumerate(rows):
        cx = left + band * (i + 0.5)
        y0, y1 = sy(0), sy(r["improvement"])
        up = r["improvement"] >= 0
        y, hh = (y1, y0 - y1) if up else (y0, y1 - y0)
        hh = max(hh, 1.0)
        rad = min(4.0, hh, bar / 2)
        x = cx - bar / 2
        # Rounded at the data end, square at the baseline.
        if up:
            path = (f"M{x:.1f},{y + hh:.1f} V{y + rad:.1f} Q{x:.1f},{y:.1f} {x + rad:.1f},{y:.1f} H{x + bar - rad:.1f} "
                    f"Q{x + bar:.1f},{y:.1f} {x + bar:.1f},{y + rad:.1f} V{y + hh:.1f} Z")
        else:
            path = (f"M{x:.1f},{y:.1f} V{y + hh - rad:.1f} Q{x:.1f},{y + hh:.1f} {x + rad:.1f},{y + hh:.1f} "
                    f"H{x + bar - rad:.1f} Q{x + bar:.1f},{y + hh:.1f} {x + bar:.1f},{y + hh - rad:.1f} V{y:.1f} Z")
        tip = (f"{r['block']}: {r['improvement']:+.4f} log loss versus {reference} "
               f"({'better' if up else 'worse'}), {r['n']:,} events")
        out.append(f'<rect class="hit" tabindex="0" x="{cx - band / 2:.1f}" y="{top}" width="{band:.1f}" height="{h}" '
                   f'data-tip="{_esc(tip)}"><title>{_esc(tip)}</title></rect>')
        out.append(f'<path class="bar mark" d="{path}" pointer-events="none"/>')
        if i % every == 0:
            out.append(f'<text class="tick" x="{cx:.1f}" y="{top + h + 16}" text-anchor="middle">{_esc(r["block"])}</text>')
    out.append(f'<text class="lab" x="{left + w / 2:.1f}" y="{height - 6}" text-anchor="middle">'
               'Season (above zero: the model beat the reference)</text>')
    out.append("</svg>")
    return '<div class="viz">' + "".join(out) + "</div>"


def standalone(viz_html: str) -> str:
    """A chart as a self-contained ``.svg`` document (styles inlined, light
    and dark aware)."""
    start = viz_html.index("<svg")
    end = viz_html.rindex("</svg>") + len("</svg>")
    svg = viz_html[start:end]
    head, rest = svg.split(">", 1)
    return f'{head} xmlns="http://www.w3.org/2000/svg" class="viz"><style>{_STANDALONE_CSS}</style>{rest}'
