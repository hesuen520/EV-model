"""``sportspred props``: projections behind the player-statistic markets."""

from __future__ import annotations

from typing import Optional

import click
import pandas as pd

from . import Context, pass_ctx


def projection_table(store, league: str, stat: str, *, settings, now: Optional[str] = None,
                     horizon_hours: Optional[float] = None, team: Optional[str] = None) -> pd.DataFrame:
    """One row per modelled player with a game inside the horizon: expected
    minutes, the projected statistic and the middle 80% of outcomes.  Nothing
    is recorded; a selection needs an imported threshold and goes through
    ``sportspred predict``."""
    from ..core.timeutil import fmt_hst, parse_ts, utcnow
    from ..predict.engine import PredictionError, load_promoted
    from ..sports import get_league

    market = f"player_prop:{stat}"
    plugin, spec = get_league(league)
    spec.market(market)
    now_dt = utcnow() if now is None else parse_ts(now)
    now_ts = pd.Timestamp(now_dt)
    horizon = settings.selection.horizon_hours if horizon_hours is None else horizon_hours
    model = load_promoted(store, league, market)
    frame = plugin.feature_frame(store, league, market, now=now_dt)
    rows = frame.upcoming
    rows = rows[(rows["start_time_utc"] > now_ts) & (rows["start_time_utc"] <= now_ts + pd.Timedelta(hours=horizon))]
    names = store.participant_names()
    if team:
        wanted = team.casefold()
        rows = rows[[wanted in str(names.get(t, t)).casefold() for t in rows["side_a_id"]]]
    if rows.empty:
        return pd.DataFrame()
    project = getattr(model.candidate, "projection", None)
    if not callable(project):
        raise PredictionError(f"the model in use for {league} {market} ({model.model_name}) is a baseline with no "
                              "projection to show")
    proj = project(rows)
    lines = rows["line"].to_numpy(dtype=float)
    behind = []
    for missing, pending in zip(rows.get("n_box_missing", pd.Series(0, index=rows.index)),
                                rows.get("n_pending_prior", pd.Series(0, index=rows.index))):
        notes = (["box score missing"] if missing else []) + (["earlier game not final"] if pending else [])
        behind.append("; ".join(notes))
    out = pd.DataFrame({
        "start (HST)": [fmt_hst(t) for t in rows["start_time_utc"]],
        "team": [names.get(t, t) for t in rows["side_a_id"]],
        "opponent": [names.get(t, t) for t in rows["side_b_id"]],
        "player": rows["player_name"].to_numpy(),
        "expected minutes": proj["exposure"].round(1).to_numpy() if "exposure" in proj else float("nan"),
        f"projected {stat}": proj["mean"].round(1).to_numpy(),
        "10th pct": proj["q10"].astype(int).to_numpy(), "median": proj["q50"].astype(int).to_numpy(),
        "90th pct": proj["q90"].astype(int).to_numpy(),
        "threshold on file": ["" if v != v else f"{v:g}" for v in lines],
        "inputs behind": behind,
    })
    return out.sort_values(["start (HST)", "team", f"projected {stat}"], ascending=[True, True, False]).reset_index(drop=True)


def register(main: click.Group) -> None:
    @main.command()
    @click.option("--league", required=True)
    @click.option("--stat", default="points", show_default=True,
                  help="points | rebounds | assists | threes | pra")
    @click.option("--team", default=None, help="Only teams whose name contains this text.")
    @click.option("--horizon-hours", type=float, default=None)
    @click.option("--now", default=None, hidden=True, help="Override the clock (ISO UTC); for tests.")
    @pass_ctx
    def props(ctx: Context, league: str, stat: str, team: Optional[str], horizon_hours: Optional[float],
              now: Optional[str]) -> None:
        """Projected player statistics for upcoming games (not selections).

        Shows what the model expects from each modelled player: minutes, the
        projected statistic and the range of likely outcomes.  Nothing is
        recorded.  A selection needs a threshold: import one with
        `sportspred import prop-lines`, then run
        `sportspred predict --league LEAGUE --market player_prop:STAT`.
        """
        from ..predict import PredictionError

        try:
            table = projection_table(ctx.store, league, stat, settings=ctx.settings, now=now,
                                     horizon_hours=horizon_hours, team=team)
        except (PredictionError, KeyError) as exc:
            raise click.ClickException(str(exc)) from exc
        if table.empty:
            click.echo("No modelled players with a game inside the horizon.")
            return
        click.echo(table.to_string(index=False))
        click.echo("\nProjections only: conditional on the player appearing, before any injury report, and not a "
                   "selection. These markets are experimental - they have not been scored against market thresholds.")
        if table["inputs behind"].ne("").any():
            click.echo("Rows marked in 'inputs behind' rest on inputs that are a game old: ingest the missing box "
                       "score, or wait for the earlier game to finish.")
