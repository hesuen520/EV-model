"""Static, self-contained dashboard: one HTML file, no external assets.

``build_dashboard(store, settings)`` reads everything from the store and
returns the page.  Sections: slate (every field a prediction carries), combos,
coverage, validation (with calibration and season-stability charts), ledger
and reviews, EV (paper), data health.  Times are Pacific/Honolulu.  Synthetic
content is only included on request and is then banner-labelled.
"""

from __future__ import annotations

import html
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from ..config import Settings
from ..core.enums import COVERAGE_DEFINITIONS, REC_STATUS_DEFINITIONS
from ..core.timeutil import fmt_hst, hst_date, iso_z, parse_ts, utcnow
from ..coverage import coverage_table
from ..validation.plots import CHART_CSS, improvement_svg, reliability_svg

__all__ = ["build_dashboard", "validation_views", "slate_views"]

NOTICE = ("Probabilities are estimates from historical evidence. A well calibrated 70% selection still loses "
          "three times in ten. Nothing here guarantees a winning pick or a profit.")

_JSON = ("probs", "reasons", "checks", "confidence", "factors", "features", "data_quality", "sources", "display")


def _e(x: Any) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def _pct(x: Any, digits: int = 1) -> str:
    return "n/a" if x is None or x != x else f"{100 * float(x):.{digits}f}%"


def _num(x: Any, digits: int = 4) -> str:
    return "n/a" if x is None or x != x else f"{float(x):.{digits}f}"


def _chip(text: str, kind: str) -> str:
    return f'<span class="chip {kind}"><i></i>{_e(text)}</span>'


_STATUS_KIND = {"FINAL": "good", "PROVISIONAL": "warn", "CONDITIONAL": "serious", "PASS": "muted",
                "validated": "good", "experimental": "warn", "demo-only": "muted", "adapter-only": "muted",
                "unavailable": "muted"}


def _table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str], labels: Optional[Mapping[str, str]] = None,
           raw: Sequence[str] = ()) -> str:
    if not rows:
        return '<p class="muted">Nothing to show.</p>'
    labels = labels or {}
    head = "".join(f"<th>{_e(labels.get(c, c.replace('_', ' ')))}</th>" for c in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{r.get(c, '') if c in raw else _e(r.get(c, ''))}</td>" for c in columns) + "</tr>"
        for r in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _bullets(title: str, items: Sequence[str]) -> str:
    if not items:
        return ""
    return f"<h4>{_e(title)}</h4><ul>" + "".join(f"<li>{_e(i)}</li>" for i in items) + "</ul>"


# ------------------------------------------------------------------- slate
def slate_views(store: Any, *, include_synthetic: bool = False) -> Dict[str, List[Dict[str, Any]]]:
    """Latest prediction per event and market, grouped by HST slate date."""
    df = store.predictions(latest_only=True, include_synthetic=include_synthetic)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for rec in df.to_dict("records"):
        for key in _JSON:
            rec[key] = json.loads(rec.pop(f"{key}_json"))
        out.setdefault(hst_date(rec["event_start_utc"]).isoformat(), []).append(rec)
    for day in out:
        out[day].sort(key=lambda p: (p["event_start_utc"], p["league"], p["market"]))
    return dict(sorted(out.items()))


def _prediction_card(p: Mapping[str, Any], now_iso: Optional[str] = None) -> str:
    d, c, f, q = p["display"], p["confidence"], p["factors"], p["data_quality"]
    started = ('<p class="started">This event has started. What follows is the pregame record, not a live '
               'selection.</p>' if now_iso and str(p["event_start_utc"]) <= now_iso else "")
    title = (f"{d.get('side_b')} at {d.get('side_a')}" if d.get("side_a_role") == "home"
             else f"{d.get('side_a')} v {d.get('side_b')}")
    venue = ", ".join(x for x in (d.get("venue"), d.get("venue_city")) if x) or "venue not published"
    if d.get("neutral_site"):
        venue += " (neutral site)"
    line = "" if p.get("line") is None or p["line"] != p["line"] else f" {p['line']:+g}"
    unc = c.get("event_uncertainty", {})
    mod = c.get("model_uncertainty", {})
    sources = "".join(
        f'<li><a href="{_e(s["url"])}" rel="noopener">{_e(s["provider"])}</a> - collected {_e(fmt_hst(s["collected_at_utc"]))}'
        + (f', published {_e(fmt_hst(s["published_at_utc"]))}' if s.get("published_at_utc") else "") + "</li>"
        for s in p["sources"] if s.get("url") and str(s["url"]).startswith("http"))
    checks = _table([{"check": f"{k['label']} ({k['subject']})", "state": k["state"], "detail": k["detail"]}
                     for k in p["checks"]], ["check", "state", "detail"])
    synthetic = '<p class="synthetic">SYNTHETIC DEMO - not a real event</p>' if p.get("is_synthetic") else ""
    return f"""
<article class="card" data-league="{_e(p['league'])}" data-status="{_e(p['rec_status'])}">
  {synthetic}{started}
  <header><div><h3>{_e(title)}</h3>
    <p class="muted">{_e(d.get('competition'))} · {_e(d.get('start_hst'))} · {_e(venue)}</p></div>
    {_chip(p['rec_status'], _STATUS_KIND.get(p['rec_status'], 'muted'))}</header>
  <div class="pick"><div><span class="k">Market</span><b>{_e(p['market'])}{_e(line)}</b><small>{_e(d.get('market_note') or '')}</small></div>
    <div><span class="k">Selection</span><b>{_e(p['selection_label'])}</b></div>
    <div><span class="k">Model probability</span><b class="big">{_pct(p.get('prob'))}</b>
      <small>model range {_pct(p.get('prob_low'))} - {_pct(p.get('prob_high'))}</small></div>
    <div><span class="k">Confidence</span><b>{_e(c.get('label'))}</b>
      <small>{_chip(p['validation_status'], _STATUS_KIND.get(p['validation_status'], 'muted'))}</small></div></div>
  <p class="status-def">{_e(REC_STATUS_DEFINITIONS.get(p['rec_status'], '') if isinstance(p['rec_status'], str) else '')}</p>
  {_bullets('Why this status', p['reasons'])}
  <details><summary>Evidence, uncertainty, checks and sources</summary>
    <h4>Uncertainty</h4><ul>
      <li>Event: the selection loses {_pct(unc.get('prob_selection_loses'))} of the time even if the probability is exactly right.</li>
      <li>Model: {_pct(p.get('prob_low'))} to {_pct(p.get('prob_high'))} ({_e(mod.get('method', 'n/a'))}).</li>
      <li>Data quality: {_e(q.get('grade', 'n/a'))}. Confidence basis: {_e(c.get('basis', 'n/a'))}.</li></ul>
    {_bullets('Supporting factors', f.get('supporting', []))}
    {_bullets('Opponent strength', f.get('opponent_strength', []))}
    {_bullets('Upset risks', f.get('upset_risks', []))}
    {_bullets('Availability assumptions', f.get('availability_assumptions', []))}
    {_bullets('Missing data', list(f.get('missing_data', [])) + list(q.get('missing_features', [])))}
    <h4>Pregame checks</h4>{checks}
    <h4>Sources</h4><ul>{sources or '<li>none recorded</li>'}</ul>
    <p class="muted">Last check {_e(d.get('last_check_hst'))} · model {_e(p['model_version'])} · prediction {_e(p['prediction_id'])}</p>
  </details>
</article>"""


def _slate_section(slates: Mapping[str, List[Dict[str, Any]]], combos: pd.DataFrame, labels: Mapping[str, str],
                   today: str, now_iso: Optional[str] = None) -> str:
    if not slates:
        return '<p class="muted">No predictions recorded yet. Run <code>sportspred predict --league nba</code>.</p>'
    days = list(slates)
    # Open on the first slate that still has an event to come.
    pending = [d for d in days if now_iso and any(str(x["event_start_utc"]) > now_iso for x in slates[d])]
    current = pending[0] if pending else (today if today in slates else next((d for d in days if d >= today), days[-1]))
    opts = "".join(f'<option value="{d}"{" selected" if d == current else ""}>{d} ({len(slates[d])})</option>' for d in days)
    leagues = sorted({p["league"] for ps in slates.values() for p in ps})
    lg_opts = '<option value="">All leagues</option>' + "".join(f'<option value="{_e(l)}">{_e(l)}</option>' for l in leagues)
    st_opts = '<option value="">All statuses</option>' + "".join(
        f'<option value="{s}">{s}</option>' for s in ("FINAL", "CONDITIONAL", "PROVISIONAL", "PASS"))
    blocks = []
    for day, preds in slates.items():
        day_combos = combos[combos["slate_date_hst"] == day] if len(combos) else combos
        latest = day_combos.sort_values("seq").drop_duplicates("role", keep="last") if len(day_combos) else day_combos
        combo_html = ""
        for _, cb in latest.iterrows():
            legs = [labels.get(x, x) for x in json.loads(cb["leg_ids_json"])]
            prob = "" if pd.isna(cb["joint_prob"]) else f" - all-win probability {_pct(cb['joint_prob'])} (conservative {_pct(cb['joint_prob_low'])})"
            combo_html += (f"<li>{_chip(cb['rec_status'], _STATUS_KIND.get(cb['rec_status'], 'muted'))} "
                           f"<b>{_e(cb['role'].replace('_', ' '))}</b>{_e(prob)}"
                           + (f"<br><span class='muted'>{_e(cb['message'])}</span>" if cb["message"] else "")
                           + ("<br>" + " + ".join(_e(x) for x in legs) if legs else "") + "</li>")
        combos_block = (f"<section class='combos'><h3>Combos</h3><ul>{combo_html}</ul>"
                        "<p class='muted'>Adding a leg can never raise the probability that every selection wins.</p></section>"
                        if combo_html else "<section class='combos'><h3>Combos</h3><p class='muted'>None recorded for this "
                        "date. Run <code>sportspred combos</code>.</p></section>")
        blocks.append(f'<div class="slate" data-day="{day}"{"" if day == current else " hidden"}>{combos_block}'
                      + "".join(_prediction_card(p, now_iso) for p in preds) + "</div>")
    return (f'<div class="filters"><label>Date (HST) <select id="f-day">{opts}</select></label>'
            f'<label>League <select id="f-league">{lg_opts}</select></label>'
            f'<label>Status <select id="f-status">{st_opts}</select></label></div>' + "".join(blocks))


# --------------------------------------------------------------- validation
def _history_note(runs: Sequence[Mapping[str, Any]]) -> str:
    if len(runs) < 2:
        return ""
    earlier = ", ".join(f"{_e(str(x['created_at_utc'])[:16].replace('T', ' '))}Z {_e(x['selected_model'])} "
                        f"log loss {_num(x['log_loss'])}" for x in runs[:-1])
    return (f" This league and market has been validated {len(runs)} times (re-runs followed data or defect fixes), "
            f"so its final test period has been read {len(runs)} times in total. Earlier runs: {earlier}.")


def validation_views(store: Any, *, include_synthetic: bool = False) -> List[Dict[str, Any]]:
    runs = store.validation_runs(include_synthetic=include_synthetic)
    latest = runs.drop_duplicates(["league", "market"], keep="last") if len(runs) else runs
    return [json.loads(r) for r in latest["report_json"]]


def validation_history(store: Any, *, include_synthetic: bool = False) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """Every recorded run per (league, market), oldest first.

    Each run reads its final test period once, but a league that was run
    again (for example after a defect was fixed) has had that period read more
    than once.  Showing the history keeps that visible."""
    runs = store.validation_runs(include_synthetic=include_synthetic)
    out: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for _, run in runs.iterrows():
        rep = json.loads(run["report_json"])
        sel = rep.get("final_test", {}).get("selected", {})
        out.setdefault((run["league"], run["market"]), []).append({
            "created_at_utc": run["created_at_utc"], "run_id": rep.get("run_id"),
            "selected_model": rep.get("selected_model"), "n": sel.get("n"), "accuracy": sel.get("accuracy"),
            "log_loss": sel.get("log_loss"), "gates_passed": bool(rep.get("gates_passed")),
        })
    return out


def _validation_section(reports: Sequence[Mapping[str, Any]],
                        history: Optional[Mapping[Tuple[str, str], Sequence[Mapping[str, Any]]]] = None) -> str:
    if not reports:
        return '<p class="muted">No validation runs on file. Run <code>sportspred validate --league nba</code>.</p>'
    out = []
    for r in sorted(reports, key=lambda x: (x["sport"], x["league"], x["market"])):
        f, sel = r.get("final_test", {}), r.get("final_test", {}).get("selected", {})
        status = "demo-only" if r.get("is_synthetic") else ("validated" if r.get("gates_passed") else "experimental")
        cands = [{"model": c["name"] + (" (selected)" if c["name"] == r.get("selected_model") else "")
                  + (" [retrospective]" if c.get("uses_retrospective") else ""),
                  "n": c["dev"]["n"], "accuracy": _pct(c["dev"]["accuracy"]), "log_loss": _num(c["dev"]["log_loss"]),
                  "brier": _num(c["dev"]["brier"]), "calibration_error": _num(c["dev"]["ece"])}
                 for c in r.get("candidates", [])]
        finals = [{"model": f"{r.get('selected_model')} (selected)", "n": sel.get("n"), "accuracy":
                   f"{_pct(sel.get('accuracy'))} [{_pct(sel['accuracy_ci'][0])}, {_pct(sel['accuracy_ci'][1])}]"
                   if sel.get("accuracy_ci") else _pct(sel.get("accuracy")),
                   "log_loss": f"{_num(sel.get('log_loss'))} [{_num(sel['log_loss_ci'][0])}, {_num(sel['log_loss_ci'][1])}]"
                   if sel.get("log_loss_ci") else _num(sel.get("log_loss")), "brier": _num(sel.get("brier"))}]
        finals += [{"model": f"baseline: {n}", "n": m.get("n"), "accuracy": _pct(m.get("accuracy")),
                    "log_loss": _num(m.get("log_loss")), "brier": _num(m.get("brier"))}
                   for n, m in (f.get("baselines") or {}).items()]
        finals += [{"model": f"retrospective upper bound: {n}", "n": m.get("n"), "accuracy": _pct(m.get("accuracy")),
                    "log_loss": _num(m.get("log_loss")), "brier": _num(m.get("brier"))}
                   for n, m in (f.get("retrospective_experiments") or {}).items()]
        bench = r.get("market_benchmark")
        if bench:
            finals.append({"model": f"market benchmark: {bench['label']} ({bench['timing']})", "n": bench["n"],
                           "accuracy": _pct(bench["market"].get("accuracy")),
                           "log_loss": _num(bench["market"].get("log_loss")), "brier": _num(bench["market"].get("brier"))})
        passes = [{"min_probability": f"{p['min_prob']:.2f}", "selections": p["n_selected"], "coverage": _pct(p.get("coverage")),
                   "accuracy": _pct(p.get("accuracy")), "interval": f"{_pct(p.get('accuracy_low'))} - {_pct(p.get('accuracy_high'))}"
                   if p.get("n_selected") else "", "mean_predicted": _pct(p.get("mean_predicted"))}
                  for p in f.get("pass_rule", [])]
        rel = f.get("reliability_selected_outcome", [])
        stab = (r.get("dev", {}).get("stability_vs_baselines") or {})
        ref = next((k for k in stab if k != "base_rate"), next(iter(stab), None))
        charts = f'<figure>{reliability_svg(rel, title="Calibration of selections (final test)")}' \
                 + "<details><summary>Table</summary>" + _table(
                     [{"predicted": _pct(x["mean_predicted"]), "won": _pct(x["observed"]),
                       "interval": f"{_pct(x['observed_low'])} - {_pct(x['observed_high'])}", "n": x["n"]} for x in rel],
                     ["predicted", "won", "interval", "n"]) + "</details></figure>"
        if ref:
            rows = stab[ref]["by_block"]
            charts += f"<figure>{improvement_svg(rows, reference=ref)}<details><summary>Table</summary>" + _table(
                [{"season": x["block"], "n": x["n"], "improvement": f"{x['improvement']:+.4f}"} for x in rows],
                ["season", "n", "improvement"]) + "</details></figure>"
        gates = "".join(f"<li>{'✓' if g['passed'] else '✗'} <b>{_e(g['gate'].replace('_', ' '))}</b>: {_e(g['detail'])}</li>"
                        for g in r.get("gates", []))
        out.append(f"""
<details class="val" {'open' if len(reports) == 1 else ''}><summary><b>{_e(r['league'])}</b> · {_e(r['market'])}
  {_chip(status, _STATUS_KIND[status])} <span class="muted">final test n={_e(sel.get('n'))}, accuracy {_pct(sel.get('accuracy'))},
  log loss {_num(sel.get('log_loss'))}</span></summary>
  <p class="muted">{_e(r['data']['n_completed'])} completed events, {_e(r['data']['first_event'][:10])} to
  {_e(r['data']['last_event'][:10])}. {len(r['protocol']['dev_folds'])} walk-forward folds; final test blocks
  {_e(', '.join(f.get('blocks', [])))} were read once in this run, after the model was selected on development folds.
  {_e(r.get('selection_note', ''))}.{_history_note((history or {}).get((r['league'], r['market']), []))}</p>
  <div class="charts">{charts}</div>
  <h4>Final test (not used for selection)</h4>{_table(finals, ['model', 'n', 'accuracy', 'log_loss', 'brier'])}
  <h4>Development folds (pooled out-of-sample)</h4>{_table(cands, ['model', 'n', 'accuracy', 'log_loss', 'brier', 'calibration_error'])}
  <h4>Performance and coverage after the PASS rule (final test)</h4>{_table(passes, ['min_probability', 'selections', 'coverage', 'accuracy', 'interval', 'mean_predicted'])}
  <h4>Gates</h4><ul class="gates">{gates}</ul>
  {_bullets('Limitations', r.get('limitations', []))}
</details>""")
    return "".join(out)


# ----------------------------------------------------------------- the page
_CSS = """
:root{--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;--accent:#2a78d6;
--good:#0ca30c;--warn:#fab219;--serious:#ec835a;--critical:#d03b3b}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--line:#2c2c2a;--accent:#3987e5}}
:root[data-theme="dark"]{--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--line:#2c2c2a;--accent:#3987e5}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}h1{font-size:22px;margin:8px 0 2px}h2{font-size:18px;margin:24px 0 8px}
h3{font-size:16px;margin:0}h4{font-size:13px;margin:14px 0 4px;color:var(--ink2);text-transform:uppercase;letter-spacing:.04em}
p{margin:6px 0}.muted,small{color:var(--ink2);font-size:13px}code{background:var(--line);padding:1px 5px;border-radius:4px;font-size:13px}
a{color:var(--accent)}nav{display:flex;gap:4px;flex-wrap:wrap;border-bottom:1px solid var(--line);margin:14px 0}
nav button{background:none;border:0;border-bottom:2px solid transparent;color:var(--ink2);padding:8px 12px;font:inherit;cursor:pointer}
nav button[aria-selected="true"]{color:var(--ink);border-bottom-color:var(--accent);font-weight:600}
.notice{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:10px 12px;color:var(--ink2);font-size:13px}
.synthetic{background:var(--critical);color:#fff;font-weight:700;padding:6px 10px;border-radius:6px;margin:0 0 8px}
.started{color:var(--muted);border-left:3px solid var(--line);padding:2px 10px;margin:0 0 8px;font-size:13px}
.card,.val,.combos{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px;margin:12px 0}
.card header{display:flex;justify-content:space-between;gap:12px;align-items:flex-start}
.pick{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:12px 0}
.pick div{display:flex;flex-direction:column}.k{font-size:12px;color:var(--ink2);text-transform:uppercase;letter-spacing:.04em}
.big{font-size:26px;line-height:1.2}.status-def{color:var(--ink2);font-size:13px;border-left:3px solid var(--line);padding-left:8px}
.chip{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--line);border-radius:999px;padding:2px 9px;font-size:12px;font-weight:600;white-space:nowrap;color:var(--ink)}
.chip i{width:8px;height:8px;border-radius:50%;background:var(--muted)}.chip.good i{background:var(--good)}
.chip.warn i{background:var(--warn)}.chip.serious i{background:var(--serious)}.chip.muted{color:var(--ink2)}
ul{margin:4px 0;padding-left:20px}li{margin:2px 0}.gates{list-style:none;padding-left:0}
.scroll{overflow-x:auto;max-width:100%}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--ink2);font-weight:600;white-space:nowrap}
details>summary{cursor:pointer;color:var(--ink2);margin-top:8px}.val>summary{color:var(--ink);font-size:15px}
.filters{display:flex;gap:12px;flex-wrap:wrap;margin:8px 0}.filters label{font-size:13px;color:var(--ink2);display:flex;flex-direction:column;gap:2px}
select{font:inherit;padding:5px 8px;border:1px solid var(--line);border-radius:6px;background:var(--surface);color:var(--ink)}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}figure{margin:0}
#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--page);padding:6px 9px;border-radius:6px;font-size:12px;max-width:260px;z-index:9;display:none}
[hidden]{display:none!important}
"""

_JS = """
const tabs=[...document.querySelectorAll('nav button')];
function show(id){tabs.forEach(b=>b.setAttribute('aria-selected',b.dataset.tab===id));
document.querySelectorAll('main>section').forEach(s=>s.hidden=s.id!==id);try{localStorage.setItem('sp-tab',id)}catch(e){}}
tabs.forEach(b=>b.addEventListener('click',()=>show(b.dataset.tab)));
let start='slate';try{start=localStorage.getItem('sp-tab')||'slate'}catch(e){}
show(document.getElementById(start)?start:'slate');
const day=document.getElementById('f-day'),lg=document.getElementById('f-league'),st=document.getElementById('f-status');
function filter(){if(!day)return;document.querySelectorAll('.slate').forEach(s=>s.hidden=s.dataset.day!==day.value);
document.querySelectorAll('.slate .card').forEach(c=>c.hidden=(lg.value&&c.dataset.league!==lg.value)||(st.value&&c.dataset.status!==st.value))}
[day,lg,st].forEach(el=>el&&el.addEventListener('change',filter));filter();
const tip=document.getElementById('tip');
function place(e,t){tip.textContent=t;tip.style.display='block';const r=tip.getBoundingClientRect();
let x=(e.clientX||0)+12,y=(e.clientY||0)+12;if(x+r.width>innerWidth-8)x=innerWidth-r.width-8;if(y+r.height>innerHeight-8)y=(e.clientY||0)-r.height-12;
tip.style.left=Math.max(8,x)+'px';tip.style.top=Math.max(8,y)+'px'}
document.querySelectorAll('[data-tip]').forEach(el=>{el.addEventListener('pointermove',e=>place(e,el.dataset.tip));
el.addEventListener('pointerleave',()=>tip.style.display='none');
el.addEventListener('focus',()=>{const b=el.getBoundingClientRect();place({clientX:b.left+b.width/2,clientY:b.top},el.dataset.tip)});
el.addEventListener('blur',()=>tip.style.display='none')});
"""


def build_dashboard(store: Any, settings: Optional[Settings] = None, *, include_synthetic: bool = False,
                    now: Any = None) -> str:
    settings = settings or Settings()
    now = utcnow() if now is None else parse_ts(now)
    slates = slate_views(store, include_synthetic=include_synthetic)
    combos = store.combos(include_synthetic=include_synthetic)
    labels = {p["prediction_id"]: f"{p['selection_label']} ({p['league']} {p['market']}, {_pct(p.get('prob'))})"
              for ps in slates.values() for p in ps}
    all_preds = store.predictions(include_synthetic=include_synthetic)
    labels.update({r["prediction_id"]: f"{r['selection_label']} ({r['league']} {r['market']})"
                   for _, r in all_preds.iterrows() if r["prediction_id"] not in labels})
    cov = coverage_table(store)
    cov_rows = [{"sport": r["sport"], "league": f"{r['competition']} ({r['league']})", "market": r["market"],
                 "status": _chip(r["status"], _STATUS_KIND[r["status"]]), "basis": _e(r["basis"]),
                 "final_test_n": "" if pd.isna(r["final_test_n"]) else int(r["final_test_n"]),
                 "accuracy": "" if pd.isna(r["final_test_accuracy"]) else _pct(r["final_test_accuracy"]),
                 "log_loss": "" if pd.isna(r["final_test_log_loss"]) else _num(r["final_test_log_loss"]),
                 "results_through": r["results_through"], "data_source": r["data_source"]}
                for _, r in cov[cov["status"] != "unavailable"].iterrows()]
    unavailable = cov[cov["status"] == "unavailable"]
    defs = "".join(f"<li>{_chip(s.value, _STATUS_KIND[s.value])} {_e(t)}</li>" for s, t in COVERAGE_DEFINITIONS.items())
    rec_defs = "".join(f"<li>{_chip(s.value, _STATUS_KIND[s.value])} {_e(t)}</li>" for s, t in REC_STATUS_DEFINITIONS.items())

    from ..ledger import lessons, review_summary
    summary = review_summary(store, include_synthetic=include_synthetic)
    reviews = store.reviews().tail(40)
    review_rows = [{"prediction": labels.get(r["prediction_id"], r["prediction_id"]), "category": r["category"],
                    "note": r["notes"]} for _, r in reviews.iterrows()]
    lesson_rows = [{"status": r["status"], "title": r["title"], "proposed_change": r["proposed_change"],
                    "test_plan": r["test_plan"]} for _, r in lessons(store).iterrows()]
    ledger_ok, problems = store.verify_ledger()

    n_prices = int(store.query("SELECT COUNT(*) AS n FROM prices")["n"][0])
    if n_prices:
        from ..ev import paper_report
        ev_html = (f"<p>{n_prices} quotes on file. Paper tracking only - nothing is ever placed.</p><pre>"
                   + _e(json.dumps(paper_report(store), indent=1, default=str)) + "</pre>"
                   "<p class='muted'>Run <code>sportspred ev evaluate</code> for per-selection figures.</p>")
    else:
        ev_html = ("<p><b>EV unavailable - no prices on file.</b> The EV module is optional and separate: import "
                   "timestamped quotes with <code>sportspred ev import-prices</code>. A price is never assumed.</p>")

    issues = store.issues()
    issue_rows = [{"league": k[0], "kind": k[1], "count": int(v)} for k, v in
                  issues.groupby(["league", "kind"]).size().items()] if len(issues) else []
    src = store.sources()
    src = src[src["is_synthetic"] == 0] if not include_synthetic else src
    src_rows = [{"provider": p, "files": int(len(g)), "last_collected_hst": fmt_hst(g["collected_at_utc"].max()),
                 "last_published_hst": fmt_hst(g["published_at_utc"].dropna().max()) if g["published_at_utc"].notna().any() else "not stated"}
                for p, g in src.groupby("provider")]
    banner = '<p class="synthetic">This page includes SYNTHETIC DEMO DATA - not real events, results or probabilities.</p>' \
        if include_synthetic else ""
    n_final = sum(1 for ps in slates.values() for p in ps if p["rec_status"] == "FINAL")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>sportspred dashboard</title>
<style>{_CSS}{CHART_CSS}</style></head><body><main>
<h1>sportspred</h1><p class="muted">Generated {_e(fmt_hst(now))} · all times Pacific/Honolulu (HST, UTC-10) ·
ledger {'intact' if ledger_ok else 'PROBLEM: ' + _e('; '.join(problems))} · {len(all_preds)} predictions recorded, {n_final} currently FINAL</p>
{banner}<p class="notice">{_e(NOTICE)}</p>
<nav role="tablist">{''.join(f'<button role="tab" data-tab="{i}">{t}</button>' for i, t in (('slate', 'Slate & combos'), ('coverage', 'Coverage'), ('validation', 'Validation'), ('ledger', 'Ledger & review'), ('ev', 'EV (paper)'), ('health', 'Data health')))}</nav>
<section id="slate"><h2>Slate</h2><ul class="gates">{rec_defs}</ul>{_slate_section(slates, combos, labels, hst_date(now).isoformat(), iso_z(now))}</section>
<section id="coverage" hidden><h2>Coverage</h2><p class="muted">Every status is derived from ingested data and recorded
validation runs; none is set by hand.</p><ul class="gates">{defs}</ul>
{_table(cov_rows, ['sport', 'league', 'market', 'status', 'basis', 'final_test_n', 'accuracy', 'log_loss', 'results_through', 'data_source'], raw=['status', 'basis'])}
<h4>Unavailable (declared, no model)</h4><p class="muted">{_e(', '.join(sorted(set(unavailable['league'] + ' ' + unavailable['market']))) or 'none')}</p></section>
<section id="validation" hidden><h2>Validation</h2><p class="muted">Chronological walk-forward validation with a reserved
final test period. Accuracy, calibration and price-dependent value are separate questions; this section is about the first two.</p>
{_validation_section(validation_views(store, include_synthetic=include_synthetic),
                     validation_history(store, include_synthetic=include_synthetic))}</section>
<section id="ledger" hidden><h2>Ledger &amp; review</h2><p class="muted">Predictions are append-only and hash-chained; updates
are new rows linked to the ones they supersede. A loss alone does not show the analysis was wrong.</p>
<pre>{_e(json.dumps(summary, indent=1, default=str))}</pre><h4>Recent reviews</h4>{_table(review_rows, ['prediction', 'category', 'note'])}
<h4>Lessons (testable proposals)</h4>{_table(lesson_rows, ['status', 'title', 'proposed_change', 'test_plan'])}</section>
<section id="ev" hidden><h2>EV (paper)</h2>{ev_html}</section>
<section id="health" hidden><h2>Data health</h2><h4>Sources</h4>{_table(src_rows, ['provider', 'files', 'last_collected_hst', 'last_published_hst'])}
<h4>Logged data issues</h4>{_table(issue_rows, ['league', 'kind', 'count'])}</section>
</main><div id="tip" role="tooltip"></div><script>{_JS}</script></body></html>"""
