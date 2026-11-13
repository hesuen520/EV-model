"""CSV import, templates and the synthetic demo."""

from __future__ import annotations

from typing import Optional

import click

from . import Context, pass_ctx


def register(main: click.Group) -> None:
    @main.command()
    @click.argument("directory", type=click.Path(file_okay=False))
    def templates(directory: str) -> None:
        """Write example CSV files for every import (see docs/CSV_IMPORT.md)."""
        from ..adapters.csv_import import write_templates

        for path in write_templates(directory):
            click.echo(f"wrote {path}")

    @main.group("import")
    def import_group() -> None:
        """Import events, results, availability or check records from CSV."""

    def _run(fn, *args, **kwargs) -> None:
        from ..adapters.csv_import import CsvImportError

        try:
            click.echo(", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in fn(*args, **kwargs).items()))
        except (CsvImportError, KeyError) as exc:
            raise click.ClickException(str(exc)) from exc

    @import_group.command("events")
    @click.option("--league", required=True)
    @click.option("--csv", "path", required=True, type=click.Path(exists=True, dir_okay=False))
    @click.option("--tz", default=None, help="Zone of timestamps that carry none, e.g. America/New_York.")
    @click.option("--create-participants", is_flag=True, help="Allow names not seen before (first import).")
    @pass_ctx
    def import_events(ctx: Context, league: str, path: str, tz: Optional[str], create_participants: bool) -> None:
        """Schedule rows, with results where the row has them."""
        from ..adapters import csv_import

        _run(csv_import.import_events, ctx.store, league, path, tz=tz, create_participants=create_participants)

    @import_group.command("results")
    @click.option("--league", required=True)
    @click.option("--csv", "path", required=True, type=click.Path(exists=True, dir_okay=False))
    @pass_ctx
    def import_results(ctx: Context, league: str, path: str) -> None:
        """Results for events already on file."""
        from ..adapters import csv_import

        _run(csv_import.import_results, ctx.store, league, path)

    @import_group.command("availability")
    @click.option("--league", required=True)
    @click.option("--csv", "path", required=True, type=click.Path(exists=True, dir_okay=False))
    @click.option("--tz", default=None)
    @click.option("--league-wide", is_flag=True,
                  help="The file is a complete league report: unlisted teams with an upcoming event are "
                       "recorded as 'checked, nothing listed'.")
    @click.option("--check-key", default="injury_report", show_default=True,
                  help="Which pregame check this file satisfies, e.g. starting_goalie or probable_pitcher.")
    @click.option("--published-at", default=None,
                  help="When the report this file transcribes was published. The check is aged from this time; "
                       "without it, from the collection time.")
    @click.option("--collected-at", default=None,
                  help="When the source was read (default: now). State it when importing a file saved earlier.")
    @pass_ctx
    def import_availability(ctx: Context, league: str, path: str, tz: Optional[str], league_wide: bool,
                            check_key: str, published_at: Optional[str], collected_at: Optional[str]) -> None:
        """Injury / line-up statements as published; also logs the pregame check."""
        from ..adapters import csv_import
        from ..core.timeutil import parse_ts

        when = None
        if collected_at:
            try:
                when = parse_ts(collected_at, assume_tz=tz)
            except Exception as exc:  # noqa: BLE001 - reported to the user as a usage error
                raise click.ClickException(f"--collected-at: {exc}") from exc
        _run(csv_import.import_availability, ctx.store, league, path, tz=tz, league_wide=league_wide,
             check_key=check_key, collected_at=when, report_published_at=published_at)
        click.echo("Re-run `sportspred predict` to update statuses.")

    @import_group.command("checks")
    @click.option("--league", required=True)
    @click.option("--csv", "path", required=True, type=click.Path(exists=True, dir_okay=False))
    @click.option("--tz", default=None)
    @pass_ctx
    def import_checks(ctx: Context, league: str, path: str, tz: Optional[str]) -> None:
        """Records that a pregame check was done (and what it found)."""
        from ..adapters import csv_import

        _run(csv_import.import_checks, ctx.store, league, path, tz=tz)

    @main.command()
    @click.option("--dir", "directory", default="data/demo", show_default=True,
                  help="Separate data directory for the demo; real data is never touched.")
    @pass_ctx
    def demo(ctx: Context, directory: str) -> None:
        """Offline end-to-end run on SYNTHETIC data (clearly labelled)."""
        import copy

        from ..adapters.synthetic import SYNTHETIC_BANNER
        from ..combos import build_slate_combos
        from ..pipeline import open_store, train_league, validate_league
        from ..predict import predict_league
        from ..sports import get_league
        from .cmd_core import print_validation
        from .render import render_slate_table

        settings = copy.deepcopy(ctx.settings)
        settings.paths.data_dir = directory
        settings.selection.allowed_coverage = ["validated", "experimental", "demo-only"]
        settings.combos.allowed_coverage = ["validated", "demo-only"]
        settings.selection.horizon_hours = 72.0
        store = open_store(settings)
        bar = "=" * len(SYNTHETIC_BANNER)
        click.echo(f"{bar}\n{SYNTHETIC_BANNER}\n{bar}")
        for league in ("demo_basketball", "demo_duel"):
            plugin, _ = get_league(league)
            click.echo(str(plugin.ingest(store, league)))
            report, _ = validate_league(store, league, "winner", settings=settings)
            print_validation(report)
            train_league(store, league, "winner", settings=settings)
            preds = predict_league(store, league, "winner", settings=settings)
            click.echo("\n" + render_slate_table(preds))
        result = build_slate_combos(store, settings=settings, include_synthetic=True, write=True)
        click.echo(f"\nCombos: {result.status.value} - {result.message}")
        ok, _ = store.verify_ledger()
        click.echo(f"Ledger intact: {ok}. Store: {directory}/store")
        click.echo(f"\n{bar}\n{SYNTHETIC_BANNER}\nDemo predictions are PROVISIONAL at best: no availability checks exist "
                   f"for invented teams.\n{bar}")
