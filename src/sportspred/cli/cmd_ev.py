"""EV commands.  Optional, separate, paper tracking only."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Optional

import click
import pandas as pd

from . import Context, pass_ctx

_NOTICE = ("EV is an estimate that is only as good as the probability and the price behind it. "
           "Nothing here guarantees a profit; nothing here places a bet.")


def register(main: click.Group) -> None:
    @main.group()
    def ev() -> None:
        """Compare recorded probabilities with timestamped prices (paper only)."""

    @ev.command("template")
    @click.argument("path", type=click.Path(dir_okay=False))
    def ev_template(path: str) -> None:
        """Write an example prices CSV."""
        from ..ev import PRICE_TEMPLATE

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(PRICE_TEMPLATE)
        click.echo(f"wrote {path}")

    @ev.command("import-prices")
    @click.option("--csv", "path", type=click.Path(exists=True, dir_okay=False), default=None)
    @click.option("--odds-api", "league", default=None,
                  help="Fetch one league from The Odds API instead (needs ODDS_API_KEY; spends credits).")
    @click.option("--markets", default="h2h", show_default=True)
    @pass_ctx
    def ev_import(ctx: Context, path: Optional[str], league: Optional[str], markets: str) -> None:
        """Store timestamped quotes from a CSV file or The Odds API."""
        from ..ev import PriceImportError, import_prices
        from ..sports import DataUnavailable

        try:
            if league:
                from ..ev.odds_api import fetch_odds_api

                out = import_prices(ctx.store, fetch_odds_api(league, markets=markets), source_provider="the_odds_api")
            elif path:
                out = import_prices(ctx.store, path)
            else:
                raise click.ClickException("give --csv FILE or --odds-api LEAGUE")
        except (PriceImportError, DataUnavailable) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in out.items()))

    @ev.command("evaluate")
    @click.option("--date", "day", default=None, help="Pacific/Honolulu date YYYY-MM-DD (default: today HST).")
    @click.option("--now", default=None, hidden=True)
    @click.option("--open-paper", is_flag=True, help="Record eligible evaluations as paper positions.")
    @pass_ctx
    def ev_evaluate(ctx: Context, day: Optional[str], now: Optional[str], open_paper: bool) -> None:
        """Implied probability, no-vig, fair odds, edge and EV for a slate."""
        from ..ev import evaluate, open_paper_positions

        df = evaluate(ctx.store, settings=ctx.settings, day=date.fromisoformat(day) if day else None, now=now)
        if df.empty:
            click.echo("No selections on this slate.")
            return
        for _, r in df.iterrows():
            head = f"{r['league']} {r['market']}{'' if pd.isna(r.get('line')) else f' {r.line:+g}'}: {r['selection_label']}"
            if r["status"] != "evaluated":
                click.echo(f"{head} | model {r['model_prob']:.3f} (fair odds {r['fair_odds_decimal']:.2f}) | "
                           f"{r['status']}: {'; '.join(r['reasons'])}")
                continue
            nv = "unavailable (other sides not quoted)" if pd.isna(r["no_vig_prob"]) else f"{r['no_vig_prob']:.3f}"
            click.echo(
                f"{head} @ {r['venue']} ({r['price']}, quoted {r['quoted_at_utc']})\n"
                f"    model {r['model_prob']:.3f} | fair odds {r['fair_odds_decimal']:.2f} | raw implied "
                f"{r['raw_implied_prob']:.3f} | no-vig {nv} | edge {r['edge']:+.3f} | EV per unit {r['ev_per_unit']:+.3f}\n"
                f"    {'ELIGIBLE' if r['eligible'] else 'not eligible: ' + '; '.join(r['reasons'])}")
        if open_paper:
            n = open_paper_positions(ctx.store, df[df["status"] == "evaluated"], settings=ctx.settings, now=now)
            click.echo(f"\n{n} paper position(s) opened. No real order exists or can be placed from here.")
        click.echo("\n" + _NOTICE)

    @ev.group("paper")
    def paper() -> None:
        """Paper positions: settle and report (kept apart from prediction accuracy)."""

    @paper.command("settle")
    @pass_ctx
    def paper_settle(ctx: Context) -> None:
        from ..ev import settle_paper_positions

        df = settle_paper_positions(ctx.store)
        click.echo("Nothing to settle." if df.empty else df.to_string(index=False))

    @paper.command("report")
    @pass_ctx
    def paper_report_cmd(ctx: Context) -> None:
        from ..ev import paper_report

        click.echo(json.dumps(paper_report(ctx.store), indent=1, default=str))
        click.echo("\nPrediction accuracy and calibration are reported separately by `sportspred review`.")

    @ev.command("rules")
    def ev_rules() -> None:
        """Settlement rule sets a quote can be governed by."""
        from ..ev import RULES

        for r in RULES.values():
            click.echo(f"{r.key}: {r.description}" + (f" [{r.note}]" if r.note else ""))

    @ev.command("fees")
    def ev_fees() -> None:
        """Fee schedules, with source and the date they were read."""
        from ..ev import FEES

        for f in FEES.values():
            src = f" (source {f.source}, read {f.verified})" if f.source else ""
            click.echo(f"{f.key}: {f.note}{src}")
