"""Live feed commands (official/unofficial APIs).  Adapter-only where the
host cannot be reached: the command says so and points to the CSV import."""

from __future__ import annotations

from typing import Optional

import click

from . import Context, pass_ctx


def register(main: click.Group) -> None:
    @main.group()
    def live() -> None:
        """Fetch near-real-time availability and scores (ESPN site API)."""

    def _fetch(ctx: Context, url: str, relpath: str):
        from ..adapters.espn import fetch_json
        from ..adapters.http import Fetcher
        from ..sports import DataUnavailable

        try:
            return fetch_json(url, Fetcher(ctx.settings.paths.raw, retries=1), relpath=relpath)
        except DataUnavailable as exc:
            raise click.ClickException(
                f"{exc}\nNothing was recorded. Import the report yourself instead:\n"
                "  sportspred templates <dir>   then   sportspred import availability --league <key> --csv <file>"
            ) from exc

    @live.command("injuries")
    @click.option("--league", required=True)
    @click.option("--horizon-hours", default=48.0, show_default=True,
                  help="Log the injury-report check for teams with an event starting within this window.")
    @pass_ctx
    def live_injuries(ctx: Context, league: str, horizon_hours: float) -> None:
        """League-wide injury list -> availability rows and injury_report checks."""
        from ..adapters.espn import import_injuries, injuries_url

        payload, got = _fetch(ctx, injuries_url(league), f"{league}/injuries.json")
        sid = got.register(ctx.store, notes=f"{league} injuries")
        out = import_injuries(ctx.store, league, payload, collected_at=got.collected_at_utc, source_id=sid,
                              horizon_hours=horizon_hours)
        click.echo(f"{league}: {out['reports']} availability statements stored; injury-report check logged for "
                   f"{out['teams_reported']} listed and {out['teams_nothing_listed']} unlisted teams with upcoming "
                   f"events; {out['unknown_teams']} rows for teams not in the store were skipped.")
        click.echo("Re-run `sportspred predict` to update statuses.")

    @live.command("nba-injury-report")
    @pass_ctx
    def live_nba_report(ctx: Context) -> None:
        """Official NBA injury report (public archive of every published version)."""
        from ..adapters.http import Fetcher
        from ..adapters.nba_injury_report import fetch_and_import
        from ..core.timeutil import fmt_hst
        from ..sports import DataUnavailable
        from ..sports.basketball.adapter import SDV_LEAGUES, current_season_label

        try:
            out = fetch_and_import(ctx.store, Fetcher(ctx.settings.paths.raw, retries=2),
                                   current_season_label(SDV_LEAGUES["nba"]))
        except DataUnavailable as exc:
            raise click.ClickException(f"{exc}\nNothing was recorded.") from exc
        if out["version_utc"] is None:
            click.echo("The archive holds no report version yet.")
            return
        click.echo(f"Newest report version in the archive: published {fmt_hst(out['version_utc'])}.")
        click.echo(f"{out['statements']} availability statements stored; injury-report check logged as reported for "
                   f"{out['teams_reported']} teams, nothing listed for {out['teams_nothing_listed']}, not yet "
                   f"submitted for {out['teams_not_submitted']}"
                   + (f" ({', '.join(out['not_submitted'])})" if out.get("not_submitted") else "") + ".")
        click.echo("Freshness is judged from the publication time above. The archive lags the league's live report, "
                   "so a version this old may not satisfy the re-check window. Re-run `sportspred predict`.")

    @live.command("scoreboard")
    @click.option("--league", required=True)
    @click.option("--date", "day", default=None, help="US schedule date YYYYMMDD (default: ESPN's current day).")
    @pass_ctx
    def live_scoreboard(ctx: Context, league: str, day: Optional[str]) -> None:
        """Daily scoreboard -> event status, start-time changes and final scores."""
        from ..adapters.espn import import_scoreboard, scoreboard_url

        payload, got = _fetch(ctx, scoreboard_url(league, day), f"{league}/scoreboard_{day or 'today'}.json")
        sid = got.register(ctx.store, notes=f"{league} scoreboard {day or 'today'}")
        out = import_scoreboard(ctx.store, league, payload, source_id=sid)
        click.echo(f"{league}: {out['events']} events refreshed, {out['results']} results recorded, "
                   f"{out['skipped']} games skipped (teams not in the store).")
