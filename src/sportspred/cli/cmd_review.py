"""Result grading, reviews, lessons and monitoring commands."""

from __future__ import annotations

import json
from typing import Optional

import click
import pandas as pd

from . import Context, pass_ctx


def register(main: click.Group) -> None:
    @main.command()
    @pass_ctx
    def grade(ctx: Context) -> None:
        """Grade recorded predictions against verified results (no prices involved)."""
        from ..ledger import grade_predictions

        written = grade_predictions(ctx.store)
        if written.empty:
            click.echo("Nothing new to grade. (Run `sportspred ingest` first to pick up final scores.)")
            return
        click.echo(written[["league", "market", "selection_label", "prob", "outcome", "regrade"]]
                   .to_string(index=False))
        click.echo(f"\n{len(written)} grades written.")

    @main.command()
    @click.option("--propose-lessons", is_flag=True, help="Draft testable lessons from review patterns.")
    @pass_ctx
    def review(ctx: Context, propose_lessons: bool) -> None:
        """Review graded predictions: variance, model error, data issue or availability change."""
        from ..ledger import propose_lessons as propose
        from ..ledger import review_predictions, review_summary

        new = review_predictions(ctx.store)
        if not new.empty:
            for _, r in new.iterrows():
                click.echo(f"[{r['category']}] {r['league']} {r['selection_label']} "
                           f"(p={r['prob']:.3f}, {r['outcome']}): {r['note']}")
        summary = review_summary(ctx.store)
        click.echo("\nSummary of graded predictions:")
        click.echo(json.dumps(summary, indent=1, default=str))
        click.echo("\nA loss alone does not show the analysis was wrong; 'model_error' requires a statistically "
                   "significant shortfall across many predictions.")
        if propose_lessons:
            ids = propose(ctx.store)
            click.echo(f"\n{len(ids)} lesson(s) proposed. See `sportspred lessons list`.")

    @main.group()
    def lessons() -> None:
        """Lessons: testable change proposals, validated before promotion."""

    @lessons.command("list")
    @pass_ctx
    def lessons_list(ctx: Context) -> None:
        from ..ledger import lessons as all_lessons

        df = all_lessons(ctx.store)
        if df.empty:
            click.echo("No lessons recorded.")
            return
        for _, r in df.iterrows():
            click.echo(f"{r['lesson_id']} [{r['status']}] {r['title']}\n  hypothesis: {r['hypothesis']}\n"
                       f"  proposed change: {r['proposed_change']}\n  test plan: {r['test_plan']}\n"
                       f"  validation run: {r['validation_run_id'] or 'none'}")

    @lessons.command("add")
    @click.option("--title", required=True)
    @click.option("--hypothesis", required=True)
    @click.option("--change", "proposed_change", required=True)
    @click.option("--test-plan", required=True)
    @click.option("--league", default=None)
    @click.option("--market", default=None)
    @pass_ctx
    def lessons_add(ctx: Context, title: str, hypothesis: str, proposed_change: str, test_plan: str,
                    league: Optional[str], market: Optional[str]) -> None:
        from ..ledger import add_lesson

        click.echo(add_lesson(ctx.store, title=title, hypothesis=hypothesis, proposed_change=proposed_change,
                              test_plan=test_plan, league=league, market=market))

    @lessons.command("status")
    @click.argument("lesson_id")
    @click.argument("status", type=click.Choice(["testing", "supported", "rejected", "promoted"]))
    @click.option("--validation-run", default=None, help="Run that tested the change (required to support/promote).")
    @pass_ctx
    def lessons_status(ctx: Context, lesson_id: str, status: str, validation_run: Optional[str]) -> None:
        from ..ledger import set_lesson_status

        try:
            set_lesson_status(ctx.store, lesson_id, status, validation_run_id=validation_run)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"{lesson_id} -> {status}")

    @main.command()
    @click.option("--league", required=True)
    @click.option("--market", default="winner", show_default=True)
    @click.option("--window", default=200, show_default=True)
    @pass_ctx
    def monitor(ctx: Context, league: str, market: str, window: int) -> None:
        """Calibration drift of live selections and drift of model inputs."""
        from ..ledger import calibration_monitor, feature_drift
        from ..sports import get_league

        click.echo(json.dumps(calibration_monitor(ctx.store, league, market, window=window), indent=1, default=str))
        preds = ctx.store.predictions(league=league, market=market, latest_only=True)
        if preds.empty:
            click.echo("\nNo recorded predictions: input drift cannot be assessed yet.")
            return
        recent = pd.DataFrame([json.loads(x) for x in preds["features_json"]])
        plugin, _ = get_league(league)
        frame = plugin.feature_frame(ctx.store, league, market)
        drift = feature_drift(frame.completed, recent, [c for c in frame.feature_cols if c in recent])
        click.echo(f"\nInput drift: {len(recent)} recorded predictions versus {len(frame.completed)} historical events")
        click.echo("No drift statistics (too few predictions)." if drift.empty else drift.round(4).to_string(index=False))
