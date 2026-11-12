"""A self-contained HTML board of the scanner's picks.

``write_report(picks, "board.html")`` produces one file with no external
dependencies beyond optional web fonts: open it in any browser or send it to
your phone.  Each pick is drawn with its *edge bar*: where the book's price
says the bet breaks even, where the fair probability sits, and the gap
between the two - which is the entire reason for the bet.
"""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from .markets import find_market
from .odds import format_american

__all__ = ["render_report", "write_report", "REPORT_CSS", "pick_rows_html"]

_FLAG_TEXT: Dict[str, str] = {
    "line_translated": "fair price translated from a different line",
    "uncalibrated_translation": "translation uses an unfitted distribution",
    "market_avg": "no sharp book: average of other books",
    "single_ref": "only one reference book",
    "refs_disagree": "reference books disagree",
    "model_only": "projection model only",
    "model_disagrees": "model and market disagree on the side",
    "alt_line": "alternate line",
    "dfs": "pick'em leg: valued against entry break-even",
    "books_split": "both sides look +EV at different books: check for news or a stale price",
}
_FLAG_SHORT: Dict[str, str] = {
    "line_translated": "line moved",
    "uncalibrated_translation": "unfitted shape",
    "market_avg": "market avg",
    "single_ref": "1 ref",
    "refs_disagree": "refs split",
    "model_only": "model only",
    "model_disagrees": "model split",
    "alt_line": "alt line",
    "dfs": "pick'em",
    "books_split": "books split",
}

#: Stylesheet shared by the standalone report and anything that embeds its rows.
REPORT_CSS = """
/* Layout: one column of pick "tickets"; each ticket = who / edge bar / numbers. */
:root {
  --bg: #f3f6f3; --surface: #ffffff; --line: #dbe3dd; --ink: #13201a; --muted: #5a6a61;
  --edge: #0c7a52; --edge-soft: #d5efe2; --warn: #9a5b00; --warn-soft: #fbecd0; --track: #e7ece8;
  --display: "Barlow Condensed", "Arial Narrow", "Helvetica Neue", Arial, sans-serif;
  --text: "Barlow", "Helvetica Neue", Helvetica, Arial, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0e1412; --surface: #151d1a; --line: #27332e; --ink: #e7efea; --muted: #93a39a;
    --edge: #45d39b; --edge-soft: #173a2c; --warn: #e3ad45; --warn-soft: #3a2d12; --track: #212b27;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #0e1412; --surface: #151d1a; --line: #27332e; --ink: #e7efea; --muted: #93a39a;
  --edge: #45d39b; --edge-soft: #173a2c; --warn: #e3ad45; --warn-soft: #3a2d12; --track: #212b27;
  color-scheme: dark;
}
body { background: var(--bg); color: var(--ink); font-family: var(--text); font-size: 16px; line-height: 1.45; margin: 0; }
.wrap { max-width: 980px; margin: 0 auto; padding-block: 28px 56px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
h1, h2 { font-family: var(--display); font-weight: 700; letter-spacing: 0.01em; text-wrap: balance; margin: 0; }
h1 { font-size: clamp(2.1rem, 6vw, 3.1rem); line-height: 1; text-transform: uppercase; }
h2 { font-size: 1.5rem; text-transform: uppercase; margin-top: 40px; }
.sub { color: var(--muted); margin: 8px 0 0; max-width: 62ch; }
.banner { background: var(--warn-soft); color: var(--warn); border: 1px solid var(--warn); border-radius: 6px;
  padding: 10px 14px; margin-top: 18px; font-weight: 600; }
.stats { display: flex; flex-wrap: wrap; gap: 8px 32px; margin: 22px 0 6px; padding: 14px 0;
  border-top: 2px solid var(--ink); border-bottom: 1px solid var(--line); }
.stat b { display: block; font-family: var(--display); font-size: 1.9rem; line-height: 1.05; font-variant-numeric: tabular-nums; }
.stat span { color: var(--muted); font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.07em; }
.picks { display: flex; flex-direction: column; gap: 10px; margin-top: 18px; }
.pick { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px;
  display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(0, 1.35fr) auto; gap: 8px 22px; align-items: center; }
.who { min-width: 0; }
.who .player { font-family: var(--display); font-size: 1.45rem; font-weight: 600; line-height: 1.1; overflow-wrap: anywhere; }
.who .bet { font-weight: 600; margin-top: 2px; }
.who .bet em { font-style: normal; color: var(--edge); }
.who .game { color: var(--muted); font-size: 0.85rem; overflow-wrap: anywhere; }
.bar { min-width: 0; }
.track { position: relative; height: 10px; background: var(--track); border-radius: 5px; margin: 18px 0 6px; }
.gap { position: absolute; top: 0; bottom: 0; background: var(--edge); border-radius: 5px; opacity: 0.85; }
.tick { position: absolute; top: -5px; bottom: -5px; width: 2px; background: var(--ink); }
.tick.fair { background: var(--edge); width: 3px; }
.ticklabels { display: flex; justify-content: space-between; gap: 10px; color: var(--muted); font-size: 0.82rem;
  font-variant-numeric: tabular-nums; }
.ticklabels b { color: var(--ink); font-weight: 600; }
.ticklabels .f b { color: var(--edge); }
.nums { display: grid; grid-template-columns: repeat(3, auto); gap: 2px 18px; text-align: right;
  font-variant-numeric: tabular-nums; }
.nums div span { display: block; color: var(--muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.07em; }
.nums div b { font-family: var(--display); font-size: 1.5rem; font-weight: 600; line-height: 1.1; }
.nums .ev b { color: var(--edge); }
.meta { grid-column: 1 / -1; display: flex; flex-wrap: wrap; gap: 6px 8px; align-items: center;
  border-top: 1px dashed var(--line); padding-top: 9px; color: var(--muted); font-size: 0.82rem; }
.chip { border: 1px solid var(--line); border-radius: 999px; padding: 1px 9px; white-space: nowrap; }
.chip.flag { color: var(--warn); border-color: var(--warn); background: var(--warn-soft); }
.chip.book { color: var(--ink); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
.empty { background: var(--surface); border: 1px dashed var(--line); border-radius: 8px; padding: 26px; color: var(--muted); }
.notes { color: var(--muted); max-width: 68ch; }
.notes p { margin: 10px 0; }
.notes b { color: var(--ink); }
table.legend { border-collapse: collapse; margin-top: 10px; font-size: 0.9rem; }
table.legend td { padding: 4px 14px 4px 0; vertical-align: top; color: var(--muted); }
table.legend td:first-child { white-space: nowrap; }
@media (max-width: 720px) {
  .pick { grid-template-columns: minmax(0, 1fr); }
  .nums { text-align: left; grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .track { margin-top: 10px; }
}
"""

_FONTS = (
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    'family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@600;700&display=swap">'
)


def _e(value: object) -> str:
    return html.escape("" if value is None else str(value))


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _market_label(sport: str, market: str) -> str:
    definition = find_market(sport, market)
    return definition.label if definition is not None else market.replace("player_", "").replace("_", " ").title()


def _bar(breakeven: float, fair: float) -> str:
    """The edge bar: break-even tick, fair tick and the gap between them."""
    lo = max(0.0, min(breakeven, fair) - 0.12)
    hi = min(1.0, max(breakeven, fair) + 0.12)
    span = max(hi - lo, 1e-9)

    def pos(p: float) -> float:
        return 100.0 * (min(max(p, lo), hi) - lo) / span

    left, right = sorted((pos(breakeven), pos(fair)))
    return (
        '<div class="track" role="img" '
        f'aria-label="price breaks even at {breakeven:.1%}, fair probability {fair:.1%}">'
        f'<div class="gap" style="left:{left:.2f}%;width:{max(right - left, 0.6):.2f}%"></div>'
        f'<div class="tick" style="left:{pos(breakeven):.2f}%"></div>'
        f'<div class="tick fair" style="left:calc({pos(fair):.2f}% - 1px)"></div></div>'
        '<div class="ticklabels">'
        f'<span>price needs <b>{breakeven:.1%}</b></span>'
        f'<span class="f">fair <b>{fair:.1%}</b></span></div>'
    )


def pick_rows_html(picks: pd.DataFrame, limit: Optional[int] = None) -> str:
    """HTML for the list of picks (the ``.picks`` block), without page chrome."""
    if picks.empty:
        return '<div class="empty">No bets clear the thresholds in this snapshot. That is the usual state of a market: most prices are fair or worse.</div>'
    rows: List[str] = []
    for row in (picks.head(limit) if limit else picks).itertuples(index=False):
        push = float(row.push_prob) if _finite(row.push_prob) else 0.0
        breakeven = (1.0 - push) / float(row.price)
        fair = float(row.fair_prob)
        flags = [f for f in str(row.flags).split(",") if f and f != "nan"]
        chips = "".join(
            f'<span class="chip flag" title="{_e(_FLAG_TEXT.get(f, f))}">{_e(_FLAG_SHORT.get(f, f))}</span>' for f in flags
        )
        refs = str(row.ref_books) if isinstance(row.ref_books, str) and row.ref_books else ""
        ref_text = f"vs {_e(refs.replace(',', ', '))}" if refs else "model projection"
        if _finite(row.projection):
            ref_text += f" &middot; model mean {float(row.projection):.1f}"
        start = ""
        if isinstance(row.commence_time, pd.Timestamp) and not pd.isna(row.commence_time):
            start = row.commence_time.strftime(" &middot; %b %-d %H:%M UTC")
        fair_price = format_american(row.fair_price_american) if _finite(row.fair_price_american) else "n/a"
        rows.append(
            '<article class="pick">'
            '<div class="who">'
            f'<div class="player">{_e(row.player)}</div>'
            f'<div class="bet"><em>{_e(str(row.side).title())} {float(row.line):g}</em> {_e(_market_label(row.sport, row.market))}</div>'
            f'<div class="game">{_e(row.event)}{start}</div>'
            "</div>"
            f'<div class="bar">{_bar(breakeven, fair)}</div>'
            '<div class="nums">'
            f'<div><span>Price</span><b>{_e(format_american(row.price_american))}</b></div>'
            f'<div class="ev"><span>EV</span><b>{100.0 * float(row.ev):+.1f}%</b></div>'
            f'<div><span>Units</span><b>{float(row.stake_units):.2f}</b></div>'
            "</div>"
            '<div class="meta">'
            f'<span class="chip book">{_e(row.book)}</span>'
            f"<span>fair price {_e(fair_price)}</span><span>{ref_text}</span>{chips}"
            "</div></article>"
        )
    return '<div class="picks">' + "".join(rows) + "</div>"


def render_report(
    picks: pd.DataFrame,
    quotes: Optional[pd.DataFrame] = None,
    title: str = "Prop Board",
    simulated: bool = False,
    fragment: bool = False,
) -> str:
    """Render picks as an HTML page.

    ``fragment=True`` returns the page without ``<html>``/``<head>``/``<body>``
    wrappers (for embedding or publishing through a host that adds its own).
    """
    n = len(picks)
    stats = [
        (f"{n}", "bets"),
        (f"{100.0 * picks['ev'].mean():+.1f}%" if n else "n/a", "average EV"),
        (f"{picks['stake_units'].sum():.1f}" if n else "0", "units at risk"),
    ]
    if quotes is not None and not quotes.empty:
        props = quotes.groupby(["event_id", "player_key", "market"]).ngroups
        stats += [(f"{props:,}", "props scanned"), (f"{quotes['book'].nunique()}", "books compared")]
    stat_html = "".join(f'<div class="stat"><b>{_e(v)}</b><span>{_e(k)}</span></div>' for v, k in stats)
    stamp = pd.Timestamp.now(tz="UTC").strftime("%b %-d, %Y %H:%M UTC")
    banner = (
        '<div class="banner">Simulated slate. The players, games and prices below are fictional and exist '
        "only to demonstrate the scanner. Nothing here is a real bet.</div>"
        if simulated
        else ""
    )
    used_flags = sorted({f for value in picks.get("flags", []) for f in str(value).split(",") if f and f != "nan"})
    legend = "".join(
        f'<tr><td><span class="chip flag">{_e(_FLAG_SHORT.get(f, f))}</span></td><td>{_e(_FLAG_TEXT.get(f, f))}</td></tr>'
        for f in used_flags
    )
    legend_html = f'<table class="legend">{legend}</table>' if legend else ""
    body = f"""
<div class="wrap">
  <header>
    <h1>{_e(title)}</h1>
    <p class="sub">Prices that pay more than the sharp-book consensus says they should. Generated {stamp}.</p>
    {banner}
  </header>
  <div class="stats">{stat_html}</div>
  {pick_rows_html(picks)}
  <h2>Reading a ticket</h2>
  <div class="notes">
    <p><b>The bar</b> is the whole argument. The dark tick is the win probability the book's price needs to break
    even. The green tick is the fair probability: the vig-free consensus of sharper books, blended with the
    projection model when one is available. The filled gap between them is your edge.</p>
    <p><b>EV</b> is expected profit per unit staked if the fair probability is right. <b>Units</b> is a
    fractional-Kelly stake (one unit is 1% of bankroll by default), capped and cut back for bets on the same
    player or game.</p>
    <p>Prices move. Check that the line and price are still available before betting, and skip anything that
    moved against you. A snapshot cannot see injury news that broke after it was taken.</p>
    {legend_html}
    <p>Expected value is an estimate, not a promise. Even a real 3% edge loses over hundreds of bets a
    meaningful share of the time. Bet only what you can afford to lose.</p>
  </div>
</div>"""
    head = f"<title>{_e(title)}</title>\n{_FONTS}\n<style>{REPORT_CSS}</style>"
    if fragment:
        return f'{head}\n<meta name="claude:viewport-fit-sides" content="cover">\n{body}'
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"{head}\n</head>\n<body>{body}\n</body>\n</html>\n"
    )


def write_report(
    picks: pd.DataFrame,
    path: Union[str, Path],
    quotes: Optional[pd.DataFrame] = None,
    title: str = "Prop Board",
    simulated: bool = False,
) -> Path:
    """Write the HTML board to ``path`` and return it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(picks, quotes, title, simulated), encoding="utf-8")
    return path
