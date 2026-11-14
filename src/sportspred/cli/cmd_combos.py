"""Combo commands: build a slate's combos, validate the selection process."""

from __future__ import annotations

from datetime import date
from typing import Optional

import click
import pandas as pd

from ..core.timeutil import fmt_hst
from . import Context, pass_ctx


def _show(cand, indent: str = "  ") -> None:
    click.echo(f"{indent}all-win probability {cand.prob:.3f} (conservative {cand.joint.prob_low:.3f}); "
               f"{cand.joint.method}")
    for leg in cand.legs:
        line = "" if leg.line is None else f" {leg.line:+g}"
        click.echo(f"{indent}  - {leg.label} [{leg.market}{line}] {leg.prob:.3f}  ({leg.event_label}, "
                   f"{fmt_hst(leg.start_utc)}, {leg.league}, {leg.rec_status})")
    for note in sorted({p.note for p in cand.joint.pairs}):
        click.echo(f"{indent}  dependency: {note}")


def _print(result, title: str) -> None:
    click.echo(f"\n{title}: {result.status.value}")
    click.echo(f"  {result.message}")
    if result.three_leg:
        click.echo("\n  Three-leg combo (primary):")
        _show(result.three_leg[0], "    ")
        for alt in result.three_leg[1:]:
            click.echo("    alternative:")
            _show(alt, "      ")
    if result.two_leg:
        click.echo("\n  Two-leg alternative:")
        _show(result.two_leg[0], "    ")
    if result.singles:
        click.echo("\n  Strong singles:")
        for leg in result.eligible[:8]:
            click.echo(f"    - {leg.label} {leg.prob:.3f}  ({leg.event_label}, {fmt_hst(leg.start_utc)}, "
                       f"{leg.league} {leg.market}, {leg.rec_status})")
    for k, cands in sorted(result.larger.items()):
        click.echo(f"\n  {k}-leg (only while supported):")
        _show(cands[0], "    ")
    for note in result.notes:
        click.echo(f"\n  note: {note}")
    for ex in result.rejected_examples[:3]:
        click.echo(f"  rejected combination {ex['legs']}: {'; '.join(ex['reasons'])}")


def register(main: click.Group) -> None:
    @main.group(invoke_without_command=True)
    @click.option("--date", "day", default=None, help="Pacific/Honolulu date YYYY-MM-DD (default: today HST).")
    @click.option("--league", multiple=True)
    @click.option("--dry-run", is_flag=True, help="Do not record the combos.")
    @click.option("--now", default=None, hidden=True)
    @click.pass_context
    def combos(ctx: click.Context, day: Optional[str], league, dry_run: bool, now: Optional[str]) -> None:
        """Strongest supported three-leg combo for a slate, with alternatives."""
        if ctx.invoked_subcommand is not None:
            return
        from ..combos import build_slate_combos

        app: Context = ctx.obj
        result = build_slate_combos(app.store, date.fromisoformat(day) if day else None, settings=app.settings,
                                    leagues=list(league) or None, write=not dry_run, now=now)
        click.echo(f"Combos for {result.slate_date_hst} (Pacific/Honolulu)")
        click.echo(f"Rules: legs must be {' or '.join(result.rules['leg_statuses'])}, coverage "
                   f"{' or '.join(result.rules['allowed_coverage'])}, probability >= {result.rules['min_leg_prob']:.2f}; "
                   "legs with unmodelled dependencies are never combined.")
        _print(result, "Recommendation")
        if result.excluded:
            click.echo("\n  Not eligible:")
            for ex in result.excluded:
                click.echo(f"    - {ex['leg']}: {ex['reason']}")
        if result.preview is not None:
            _print(result.preview, "PROVISIONAL PREVIEW (not a recommendation)")

    @combos.command("validate")
    @click.option("--league", required=True)
    @click.option("--market", default="winner", show_default=True)
    @pass_ctx
    def combos_validate(ctx: Context, league: str, market: str) -> None:
        """Back-test the combo selection rule over historical slates."""
        from ..combos import combo_backtest, tune_min_leg_prob

        try:
            oos = ctx.store.read_table(league, f"oos_{market}")
        except FileNotFoundError as exc:
            raise click.ClickException(f"{exc}\nRun `sportspred validate --league {league} --market {market}` first.")
        cfg = ctx.settings.combos
        result = combo_backtest(oos, min_leg_prob=cfg.min_leg_prob)
        cols = ["legs", "n_combos", "coverage", "mean_predicted_all_win", "observed_all_win", "observed_low",
                "observed_high", "observed_minus_predicted"]
        for period, rows in result.items():
            df = pd.DataFrame(rows)
            click.echo(f"\n{league} {market} - {period.replace('_', ' ')} slates, leg minimum {cfg.min_leg_prob:.2f} "
                       "(top-k legs by probability per slate):")
            click.echo(df[[c for c in cols if c in df]].round(4).to_string(index=False))
        click.echo("\nLeg-minimum grid for the three-leg rule (development slates only; "
                   "the final test is never used to choose a rule):")
        grid = tune_min_leg_prob(oos, size=cfg.primary_size)
        click.echo(grid[[c for c in ["min_leg_prob", "n_combos", "coverage", "mean_predicted_all_win",
                                     "observed_all_win", "observed_minus_predicted"] if c in grid]]
                   .round(4).to_string(index=False))
        click.echo("\nNote: historical pregame check states do not exist, so this covers the probability and "
                   "selection logic, not the FINAL-status filter applied live.")
        ctx.store.write_table(league, f"combo_backtest_{market}",
                              pd.concat([pd.DataFrame(v).assign(period=k) for k, v in result.items()]),
                              mode="replace")
