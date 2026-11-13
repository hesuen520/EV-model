"""Core commands: coverage, ingest, validate, train, predict, slate, ledger."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Optional

import click
import pandas as pd

from ..core.enums import REC_STATUS_DEFINITIONS
from ..core.timeutil import fmt_hst, hst_date, hst_day_bounds_utc, parse_ts, utcnow
from . import Context, pass_ctx
from .render import prediction_from_row, render_prediction, render_slate_table


def _seasons(text: Optional[str]):
    if not text:
        return None
    out = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def _fmt(x, digits=4):
    return "n/a" if x is None or x != x else f"{x:.{digits}f}"


def print_validation(report: dict) -> None:
    f, d = report["final_test"], report["dev"]
    click.echo(f"\n{report['league']} {report['market']} - validation run {report.get('run_id', '')}")
    click.echo(f"  events: {report['data']['n_completed']} completed, "
               f"{report['data']['first_event'][:10]} .. {report['data']['last_event'][:10]}")
    click.echo(f"  protocol: {len(report['protocol']['dev_folds'])} walk-forward folds; final test blocks "
               f"{', '.join(f['blocks'])} (never used for selection)")
    click.echo("\n  Development folds (pooled out-of-sample):")
    rows = [{"model": c["name"] + (" *" if c["name"] == report["selected_model"] else "")
             + (" [retrospective]" if c["uses_retrospective"] else ""),
             "n": c["dev"]["n"], "accuracy": _fmt(c["dev"]["accuracy"]), "log_loss": _fmt(c["dev"]["log_loss"], 5),
             "brier": _fmt(c["dev"]["brier"], 5), "ece": _fmt(c["dev"]["ece"])} for c in report["candidates"]]
    click.echo("  " + pd.DataFrame(rows).to_string(index=False).replace("\n", "\n  "))
    click.echo(f"  selected: {report['selected_model']} ({report['selection_note']})")
    for name, st in (d.get("stability_vs_baselines") or {}).items():
        click.echo(f"  season-by-season: beats {name} in {st['blocks_improved']} of {st['blocks']} development seasons")
    s = f["selected"]
    click.echo(f"\n  Final test ({s['first_event'][:10]} .. {s['last_event'][:10]}, n={s['n']}):")
    click.echo(f"    accuracy {_fmt(s['accuracy'])} [{_fmt(s['accuracy_ci'][0])}, {_fmt(s['accuracy_ci'][1])}]"
               f"  log loss {_fmt(s['log_loss'], 5)} [{_fmt(s['log_loss_ci'][0], 5)}, {_fmt(s['log_loss_ci'][1], 5)}]"
               f"  Brier {_fmt(s['brier'], 5)}  ECE {_fmt(s['ece'])}")
    for name, m in f["baselines"].items():
        click.echo(f"    baseline {name}: accuracy {_fmt(m['accuracy'])}  log loss {_fmt(m['log_loss'], 5)}"
                   f"  Brier {_fmt(m['brier'], 5)}")
    for name, m in f["retrospective_experiments"].items():
        click.echo(f"    retrospective {name}: accuracy {_fmt(m['accuracy'])}  log loss {_fmt(m['log_loss'], 5)}"
                   "  (upper bound; not a realistic backtest)")
    if report.get("market_benchmark"):
        b = report["market_benchmark"]
        click.echo(f"    market benchmark ({b['label']}, {b['timing']}, n={b['n']}): log loss "
                   f"{_fmt(b['market']['log_loss'], 5)} vs model {_fmt(b['model_same_events']['log_loss'], 5)}")
    click.echo("\n  After PASS rule (final test):")
    pr = pd.DataFrame(f["pass_rule"])
    cols = [c for c in ("min_prob", "n_selected", "coverage", "accuracy", "accuracy_low", "accuracy_high",
                        "mean_predicted") if c in pr]
    click.echo("  " + pr[cols].round(4).to_string(index=False).replace("\n", "\n  "))
    click.echo("\n  Gates:")
    for g in report["gates"]:
        click.echo(f"    [{'x' if g['passed'] else ' '}] {g['gate']}: {g['detail']}")
    if report.get("is_synthetic"):
        click.echo("  => DEMO-ONLY (synthetic data can never be validated)")
    else:
        click.echo(f"  => {'VALIDATED' if report['gates_passed'] else 'EXPERIMENTAL'}")
    for lim in report.get("limitations", []):
        click.echo(f"  limitation: {lim}")


def register(main: click.Group) -> None:
    @main.command()
    @click.option("--md", "md_path", type=click.Path(dir_okay=False), default=None,
                  help="Also write the table as Markdown to this file.")
    @click.option("--all", "show_all", is_flag=True, help="Include unavailable markets.")
    @pass_ctx
    def coverage(ctx: Context, md_path: Optional[str], show_all: bool) -> None:
        """What is validated, experimental, demo-only, adapter-only or unavailable."""
        from ..coverage import coverage_markdown, coverage_table

        table = coverage_table(ctx.store)
        view = table if show_all else table[table["status"] != "unavailable"]
        cols = ["sport", "league", "market", "status", "basis", "final_test_n", "final_test_accuracy",
                "final_test_log_loss", "results_through"]
        click.echo(view[cols].to_string(index=False))
        if md_path:
            Path(md_path).parent.mkdir(parents=True, exist_ok=True)
            Path(md_path).write_text(coverage_markdown(ctx.store))
            click.echo(f"\nwrote {md_path}")

    @main.command()
    @click.option("--league", required=True)
    @click.option("--seasons", default=None, help='e.g. "2015-2027" or "2025,2026".')
    @click.option("--refresh/--no-refresh", default=None, help="Force or forbid re-downloading.")
    @pass_ctx
    def ingest(ctx: Context, league: str, seasons: Optional[str], refresh: Optional[bool]) -> None:
        """Fetch real data for a league from its public source."""
        from ..pipeline import ingest_league
        from ..sports import DataUnavailable

        try:
            report = ingest_league(ctx.store, league, settings=ctx.settings, seasons=_seasons(seasons),
                                   refresh=refresh)
        except DataUnavailable as exc:
            raise click.ClickException(
                f"{exc}\nNo data was substituted. Use `sportspred import events --league {league} --csv ...` "
                "to load results yourself.") from exc
        click.echo(str(report))

    @main.command("check-leakage")
    @click.option("--league", required=True)
    @click.option("--market", default="winner", show_default=True)
    @click.option("--cutoffs", default=3, show_default=True, help="How many past instants to rebuild features at.")
    @pass_ctx
    def check_leakage(ctx: Context, league: str, market: str, cutoffs: int) -> None:
        """Point-in-time test: rebuild features as of past instants and compare.

        Features of the first slate after each instant are built knowing only
        what was on file before it, and must equal the values obtained from the
        full history.  Any difference means a feature used information from
        its own slate or later.  Read-only.
        """
        from ..sports import get_league
        from ..validation.leakage import asof_invariance

        plugin, _ = get_league(league)
        result = asof_invariance(plugin, ctx.store, league, market, n_cutoffs=cutoffs)
        if result.empty:
            raise click.ClickException("no comparable events: ingest results for this league first")
        for cutoff, part in result.groupby("cutoff"):
            click.echo(f"as of {cutoff:%Y-%m-%d %H:%M:%S}Z: {int(part['n_events'].max())} events, {len(part)} features, "
                       f"largest difference {part['max_abs_diff'].max():.3g}")
        bad = result[~result["ok"]]
        if len(bad):
            click.echo(bad.sort_values("max_abs_diff", ascending=False).head(10).to_string(index=False))
            raise click.ClickException(f"{league} {market}: LEAKAGE - {bad['feature'].nunique()} feature(s) differ")
        click.echo(f"{league} {market}: no leakage found at {result['cutoff'].nunique()} cut-offs")

    @main.command()
    @click.option("--league", required=True)
    @click.option("--market", default="winner", show_default=True)
    @click.option("--json", "json_path", type=click.Path(dir_okay=False), default=None)
    @pass_ctx
    def validate(ctx: Context, league: str, market: str, json_path: Optional[str]) -> None:
        """Chronological walk-forward validation with a reserved final test."""
        from ..pipeline import validate_league

        report, _ = validate_league(ctx.store, league, market, settings=ctx.settings)
        print_validation(report)
        if json_path:
            Path(json_path).parent.mkdir(parents=True, exist_ok=True)
            Path(json_path).write_text(json.dumps(report, indent=1, default=str))
            click.echo(f"\nwrote {json_path}")

    @main.command()
    @click.option("--league", required=True)
    @click.option("--market", default="winner", show_default=True)
    @pass_ctx
    def train(ctx: Context, league: str, market: str) -> None:
        """Fit the production model selected by the latest validation run."""
        from ..pipeline import train_league

        model = train_league(ctx.store, league, market, settings=ctx.settings)
        click.echo(f"trained {model.model_version}: {model.n_train} training events, "
                   f"{model.n_calib} calibration events, {len(model.replicas)} bootstrap replicas")

    @main.command()
    @click.option("--league", required=True, multiple=True)
    @click.option("--market", default="winner", show_default=True)
    @click.option("--horizon-hours", type=float, default=None)
    @click.option("--now", default=None, hidden=True, help="Override the clock (ISO UTC); for tests.")
    @click.option("--detail", is_flag=True, help="Print the full explanation of each prediction.")
    @click.option("--dry-run", is_flag=True, help="Do not write to the ledger.")
    @pass_ctx
    def predict(ctx: Context, league, market: str, horizon_hours: Optional[float], now: Optional[str],
                detail: bool, dry_run: bool) -> None:
        """Generate pregame predictions for upcoming events and record them."""
        from ..predict import PredictionError, predict_league

        preds = []
        for lg in league:
            try:
                preds += predict_league(ctx.store, lg, market, settings=ctx.settings, now=now,
                                        horizon_hours=horizon_hours, write=not dry_run)
            except PredictionError as exc:
                raise click.ClickException(str(exc)) from exc
        click.echo(render_slate_table(preds))
        new = sum(1 for p in preds if not p.get("unchanged") and "prediction_id" in p)
        click.echo(f"\n{len(preds)} predictions ({new} written to the ledger, "
                   f"{sum(1 for p in preds if p.get('unchanged'))} unchanged).")
        if detail:
            for p in preds:
                click.echo("\n" + render_prediction(p))

    @main.command()
    @click.option("--date", "day", default=None, help="Pacific/Honolulu date YYYY-MM-DD (default: today HST).")
    @click.option("--league", multiple=True)
    @click.option("--detail", is_flag=True)
    @click.option("--history", is_flag=True, help="Show superseded predictions too.")
    @pass_ctx
    def slate(ctx: Context, day: Optional[str], league, detail: bool, history: bool) -> None:
        """Show recorded predictions for one Pacific/Honolulu day."""
        d = date.fromisoformat(day) if day else hst_date(utcnow())
        lo, hi = hst_day_bounds_utc(d)
        df = ctx.store.predictions(league=list(league) or None, start_from=lo, start_before=hi,
                                   latest_only=not history)
        preds = [prediction_from_row(r) for r in df.to_dict("records")]
        preds.sort(key=lambda p: (p["event_start_utc"], p["league"]))
        click.echo(f"Slate for {d.isoformat()} (Pacific/Honolulu)\n")
        click.echo(render_slate_table(preds))
        if detail:
            for p in preds:
                click.echo("\n" + render_prediction(p))
        click.echo("\nStatus definitions:")
        for status, text in REC_STATUS_DEFINITIONS.items():
            click.echo(f"  {status.value}: {text}")

    @main.group()
    def ledger() -> None:
        """Inspect the immutable prediction ledger."""

    @ledger.command("verify")
    @pass_ctx
    def ledger_verify(ctx: Context) -> None:
        """Check the hash chain and immutability triggers."""
        ok, problems = ctx.store.verify_ledger()
        n = len(ctx.store.predictions(include_synthetic=True))
        if ok:
            click.echo(f"ledger intact: {n} predictions, hash chain verified")
        else:
            for p in problems:
                click.echo(f"PROBLEM: {p}")
            raise click.ClickException("ledger verification failed")

    @ledger.command("export")
    @click.argument("path", type=click.Path(dir_okay=False))
    @click.option("--format", "fmt", type=click.Choice(["csv", "json"]), default="csv")
    @pass_ctx
    def ledger_export(ctx: Context, path: str, fmt: str) -> None:
        """Export every prediction (including superseded ones)."""
        df = ctx.store.predictions(include_synthetic=False)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if fmt == "csv":
            df.to_csv(path, index=False)
        else:
            Path(path).write_text(json.dumps([prediction_from_row(r) for r in df.to_dict("records")],
                                             indent=1, default=str))
        click.echo(f"wrote {len(df)} predictions to {path}")
