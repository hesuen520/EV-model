"""Command-line interface: ``evmodel --help``."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import click
import numpy as np
import pandas as pd

from . import __version__
from .config import Config, load_config
from .devig import METHODS, devig, hold, overround
from .ev import edge as edge_fn
from .ev import expected_value, fractional_kelly, kelly_fraction
from .markets import find_market, markets_for_sport
from .odds import format_american, parse_odds, prob_to_american

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 500)


def _pct(value: float, digits: int = 1) -> str:
    return "" if value is None or not np.isfinite(value) else f"{100.0 * value:+.{digits}f}%"


def _prob(value: float) -> str:
    return "" if value is None or not np.isfinite(value) else f"{100.0 * value:.1f}%"


def _american(decimal: float) -> str:
    from .odds import decimal_to_american

    return format_american(decimal_to_american(decimal))


def _parse_seasons(text: Optional[str]) -> Optional[List[int]]:
    """``"2024,2025"`` or ``"2019-2023"`` -> list of ints."""
    if not text:
        return None
    seasons: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            seasons.extend(range(int(start), int(end) + 1))
        elif part:
            seasons.append(int(part))
    return seasons


def _picks_table(picks: pd.DataFrame, top: Optional[int] = None) -> str:
    if picks.empty:
        return "No bets clear the thresholds."
    rows = picks.head(top) if top else picks
    out = pd.DataFrame(
        {
            "player": rows["player"],
            "market": [
                (find_market(s, m).label if find_market(s, m) else m) for s, m in zip(rows["sport"], rows["market"])
            ],
            "bet": [f"{side} {line:g}" for side, line in zip(rows["side"], rows["line"])],
            "book": rows["book"],
            "price": rows["price_american"].map(format_american),
            "fair": rows["fair_price_american"].map(lambda v: format_american(v) if np.isfinite(v) else ""),
            "win%": rows["fair_prob"].map(_prob),
            "EV": rows["ev"].map(_pct),
            "units": rows["stake_units"].map(lambda v: f"{v:.2f}"),
            "refs": rows["n_refs"],
            "src": rows["source"],
            "flags": rows["flags"],
        }
    )
    return out.to_string(index=False)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="evmodel")
def main() -> None:
    """Find +EV player-prop bets.

    \b
    Quick start
      evmodel demo                         end-to-end run on a simulated slate
      evmodel devig -- -125 +105           remove the vig from a two-way price
      evmodel project --sport nba          model projections from real game logs
      evmodel scan --odds-csv odds.csv     find +EV bets in prices you typed in
      evmodel fetch-odds --sport nba       pull live props (needs ODDS_API_KEY)
    """


# ----------------------------------------------------------------- calculators
@main.command("devig")
@click.argument("prices", nargs=-1, required=True)
@click.option("--method", type=click.Choice(METHODS + ("all",)), default="all", show_default=True)
def devig_cmd(prices: Tuple[str, ...], method: str) -> None:
    """Remove the vig from a market.  PRICES are every outcome's odds.

    American (-125 +105), decimal (1.80 2.05) or fractional (4/5) prices are
    accepted.  Put ``--`` before negative prices: ``evmodel devig -- -125 +105``.
    """
    if len(prices) < 2:
        raise click.UsageError("give the price of every outcome (at least two)")
    decimals = [parse_odds(p) for p in prices]
    implied = [1.0 / d for d in decimals]
    click.echo(f"implied: {'  '.join(_prob(q) for q in implied)}   overround {overround(implied):+.2%}   hold {hold(implied):.2%}")
    methods = METHODS if method == "all" else (method,)
    for name in methods:
        fair = devig(implied, name)
        text = "  ".join(f"{_prob(p)} ({format_american(prob_to_american(p))})" for p in fair)
        click.echo(f"{name:>15}: {text}")


@main.command("ev")
@click.option("--prob", type=float, required=True, help="Your win probability, e.g. 0.55.")
@click.option("--price", required=True, help="The price offered, e.g. -110 or 1.91.")
@click.option("--push", type=float, default=0.0, show_default=True, help="Probability of a push.")
@click.option("--kelly", "kelly_mult", type=float, default=0.25, show_default=True, help="Kelly fraction.")
@click.option("--bankroll", type=float, default=None, help="Bankroll, to show the stake in money.")
def ev_cmd(prob: float, price: str, push: float, kelly_mult: float, bankroll: Optional[float]) -> None:
    """Expected value and Kelly stake of a single bet."""
    decimal = parse_odds(price)
    ev = expected_value(prob, decimal, push)
    stake = fractional_kelly(prob, decimal, push, fraction=kelly_mult)
    click.echo(f"price {_american(decimal)} (decimal {decimal:.3f}), break-even {_prob((1 - push) / decimal)}")
    click.echo(f"EV per unit staked: {_pct(ev, 2)}   edge: {_pct(edge_fn(prob, decimal, push), 2)}")
    click.echo(f"full Kelly: {kelly_fraction(prob, decimal, push):.2%} of bankroll   {kelly_mult:g}x Kelly: {stake:.2%}")
    if bankroll:
        click.echo(f"stake on a {bankroll:,.0f} bankroll: {stake * bankroll:,.2f}")


@main.command("markets")
@click.option("--sport", default=None, help="Limit to one sport.")
def markets_cmd(sport: Optional[str]) -> None:
    """List supported prop markets and their fitted distributions."""
    from .markets import MARKETS

    rows = []
    for (s, key), m in MARKETS.items():
        if sport and s != sport.lower():
            continue
        spec = m.spec
        rows.append(
            {
                "sport": s, "market": key, "label": m.label, "stats": "+".join(m.stats),
                "family": spec.family + (f" x{spec.scale:g}" if spec.scale != 1 else ""),
                "c": spec.c, "phi": spec.phi, "alpha": spec.alpha,
                "mean_scale": m.mean_scale, "fitted": "yes" if m.calibrated else "no",
            }
        )
    click.echo(pd.DataFrame(rows).to_string(index=False))


@main.command("dfs")
@click.option("--operator", default=None, help="prizepicks or underdog (default: both).")
def dfs_cmd(operator: Optional[str]) -> None:
    """Break-even leg probabilities for pick'em entries."""
    from .dfs import PAYOUTS, breakeven_table

    book = {operator.lower(): PAYOUTS[operator.lower()]} if operator else PAYOUTS
    table = breakeven_table(book)
    table["breakeven"] = table["breakeven"].map(_prob)
    table["equivalent_price"] = table.pop("equivalent_decimal").map(_american)
    click.echo(table.to_string(index=False))
    click.echo("\nPayout tables change and vary by state - check the multipliers in your app.")


# ------------------------------------------------------------------------ data
@main.command("fetch-stats")
@click.option("--sport", required=True, type=click.Choice(["nba", "nhl", "nfl"]))
@click.option("--seasons", default=None, help="e.g. 2025,2026 or 2019-2026 (default: last two).")
@click.option("--cache-dir", default="data/raw", show_default=True)
@click.option("--refresh", is_flag=True, help="Re-download even if cached.")
def fetch_stats_cmd(sport: str, seasons: Optional[str], cache_dir: str, refresh: bool) -> None:
    """Download player game logs (free public data, no key needed)."""
    from .providers.stats import load_gamelogs

    logs = load_gamelogs(sport, _parse_seasons(seasons), cache_dir, refresh=True if refresh else None)
    click.echo(
        f"{len(logs):,} player-games, {logs['player_key'].nunique():,} players, "
        f"{logs['game_date'].min().date()} to {logs['game_date'].max().date()} (cached under {cache_dir})"
    )
    if logs.attrs.get("missing_seasons"):
        click.echo(f"not available: {logs.attrs['missing_seasons']}")


@main.command("fetch-odds")
@click.option("--sport", required=True, help="nba, nfl, nhl, mlb, wnba, ncaab, ncaaf")
@click.option("--markets", default=None, help="Comma-separated market keys (default: all registered).")
@click.option("--regions", default="us,us2,eu", show_default=True, help="eu adds Pinnacle; us_dfs adds pick'em apps.")
@click.option("--bookmakers", default=None, help="Comma-separated bookmaker keys (overrides regions).")
@click.option("--max-events", type=int, default=None, help="Only the next N games (saves credits).")
@click.option("--out", "out_path", default="data/odds/latest.json", show_default=True)
@click.option("--yes", is_flag=True, help="Skip the credit-cost confirmation.")
def fetch_odds_cmd(sport: str, markets: Optional[str], regions: str, bookmakers: Optional[str],
                   max_events: Optional[int], out_path: str, yes: bool) -> None:
    """Pull live player-prop odds from The Odds API (needs ODDS_API_KEY)."""
    from .providers.odds_api import OddsApiClient, OddsApiError, default_markets, estimate_cost

    try:
        client = OddsApiClient()
    except OddsApiError as exc:
        raise click.ClickException(
            f"{exc.message}. Get a key at https://the-odds-api.com and run:  export ODDS_API_KEY=your_key"
        )
    market_list = [m.strip() for m in markets.split(",")] if markets else default_markets(sport)
    region_list = [r.strip() for r in regions.split(",") if r.strip()]
    book_list = [b.strip() for b in bookmakers.split(",")] if bookmakers else None
    try:
        events = client.events(sport)
    except OddsApiError as exc:
        raise click.ClickException(str(exc))
    if max_events:
        events = events[:max_events]
    n_regions = max(1, -(-len(book_list) // 10)) if book_list else len(region_list)
    cost = estimate_cost(len(events), len(market_list), n_regions)
    click.echo(f"{len(events)} events x {len(market_list)} markets x {n_regions} region(s): up to {cost} credits")
    if not events:
        return
    if not yes and not click.confirm("Fetch?", default=True):
        return
    try:
        quotes = client.fetch_props(sport, market_list, region_list, book_list, max_events, snapshot_path=out_path)
    except OddsApiError as exc:
        raise click.ClickException(str(exc))
    click.echo(
        f"{len(quotes):,} quotes from {quotes['book'].nunique()} books saved to {out_path}; "
        f"credits remaining: {client.usage.remaining}"
    )


@main.command("odds-template")
@click.argument("path", default="odds.csv")
def odds_template_cmd(path: str) -> None:
    """Write an example odds CSV you can fill in by hand."""
    from .providers.csv_odds import write_template

    click.echo(f"wrote {write_template(path)}")


# ------------------------------------------------------------------ projections
def _load_logs(sport: str, cache_dir: str, seasons: Optional[Sequence[int]] = None, offline: bool = False) -> pd.DataFrame:
    from .providers.stats import load_gamelogs

    return load_gamelogs(sport, seasons, cache_dir, refresh=False if offline else None)


@main.command("project")
@click.option("--sport", required=True, type=click.Choice(["nba", "nhl", "nfl"]))
@click.option("--player", default=None, help="Only players whose name contains this text.")
@click.option("--market", default=None, help="Only this market key, e.g. player_points.")
@click.option("--line", type=float, default=None, help="Show over/under probabilities at this line.")
@click.option("--as-of", default=None, help="Project games on this date (YYYY-MM-DD); default: next day.")
@click.option("--top", type=int, default=30, show_default=True)
@click.option("--cache-dir", default="data/raw", show_default=True)
@click.option("--offline", is_flag=True, help="Use cached data only.")
@click.option("--out", "out_path", default=None, help="Write all projections to this CSV.")
def project_cmd(sport: str, player: Optional[str], market: Optional[str], line: Optional[float],
                as_of: Optional[str], top: int, cache_dir: str, offline: bool, out_path: Optional[str]) -> None:
    """Model projections from real game logs."""
    from .projection.model import project_players, projections_frame

    logs = _load_logs(sport, cache_dir, offline=offline)
    projections = project_players(logs, sport, as_of=as_of, markets=[market] if market else None)
    table = projections_frame(projections)
    click.echo(f"game logs through {logs['game_date'].max().date()}; {table['player_key'].nunique()} players projected")
    if player:
        table = table[table["player"].str.contains(player, case=False, regex=False)]
    if table.empty:
        click.echo("no matching projections")
        return
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_path, index=False)
        click.echo(f"wrote {len(table):,} projections to {out_path}")
    show = table.sort_values("mean", ascending=False) if market else table
    cols = ["player", "team", "market", "mean", "median", "p10", "p90", "n_games"]
    view = show[cols].head(top).copy()
    if line is not None:
        probs = [projections[(k, m)].probs(line) for k, m in zip(show["player_key"].head(top), show["market"].head(top))]
        view[f"P(over {line:g})"] = [_prob(p[0]) for p in probs]
        view[f"P(under {line:g})"] = [_prob(p[2]) for p in probs]
        view["fair over"] = [
            format_american(prob_to_american(p[0] / (p[0] + p[2]))) if 0 < p[0] / (p[0] + p[2]) < 1 else "" for p in probs
        ]
    view["mean"] = view["mean"].map(lambda v: f"{v:.2f}")
    click.echo(view.to_string(index=False))


def _projections_for(quotes: pd.DataFrame, cache_dir: str, offline: bool) -> Dict:
    """Projections for every sport in a quotes table that has a model."""
    from .projection.model import SPORT_PARAMS, align_to_quotes, project_players

    projections: Dict = {}
    for sport in sorted(set(quotes["sport"]) & set(SPORT_PARAMS)):
        try:
            logs = _load_logs(sport, cache_dir, offline=offline)
        except Exception as exc:  # network or cache problems must not stop a scan
            click.echo(f"[{sport}] no game logs ({exc}); scanning without the model", err=True)
            continue
        sub = quotes[quotes["sport"] == sport]
        starts = sub["commence_time"].dropna()
        as_of = starts.min().tz_convert("America/New_York").tz_localize(None).normalize() if len(starts) else None
        if as_of is not None and as_of <= logs["game_date"].min():
            as_of = None
        projected = project_players(logs, sport, as_of=as_of)
        aligned = align_to_quotes(projected, sub)
        click.echo(f"[{sport}] model projections for {len({k[0] for k in aligned})} quoted players "
                   f"(logs through {logs['game_date'].max().date()})", err=True)
        projections.update(aligned)
    return projections


def _load_quotes(odds: Optional[str], odds_csv: Optional[str], sample: bool, sport: Optional[str], seed: int):
    from .providers.csv_odds import read_odds_csv
    from .providers.odds_api import load_snapshot
    from .providers.sample import make_slate

    chosen = [bool(odds), bool(odds_csv), sample]
    if sum(chosen) != 1:
        raise click.UsageError("choose exactly one of --odds, --odds-csv or --sample")
    if odds:
        return load_snapshot(odds), None
    if odds_csv:
        return read_odds_csv(odds_csv, sport=sport), None
    world = make_slate(sport or "nba", seed=seed)
    return world.quotes, world


@main.command("scan")
@click.option("--odds", default=None, help="Snapshot JSON written by fetch-odds.")
@click.option("--odds-csv", default=None, help="CSV of prices (see odds-template).")
@click.option("--sample", is_flag=True, help="Use a simulated slate (not real odds).")
@click.option("--sport", default=None, help="Sport for --odds-csv / --sample.")
@click.option("--config", "config_path", default=None, help="TOML config file.")
@click.option("--min-ev", type=float, default=None, help="Minimum EV, e.g. 0.03.")
@click.option("--books", default=None, help="Only look for bets at these books (comma separated).")
@click.option("--no-model", is_flag=True, help="Market consensus only; skip projections.")
@click.option("--model-only", is_flag=True, help="Also report bets supported only by the model.")
@click.option("--top", type=int, default=40, show_default=True)
@click.option("--out", "out_path", default=None, help="Write the picks to this CSV.")
@click.option("--html", "html_path", default=None, help="Write an HTML report.")
@click.option("--cache-dir", default="data/raw", show_default=True)
@click.option("--offline", is_flag=True, help="Use cached game logs only.")
@click.option("--seed", type=int, default=0, help="Seed for --sample.")
def scan_cmd(odds: Optional[str], odds_csv: Optional[str], sample: bool, sport: Optional[str],
             config_path: Optional[str], min_ev: Optional[float], books: Optional[str], no_model: bool,
             model_only: bool, top: int, out_path: Optional[str], html_path: Optional[str],
             cache_dir: str, offline: bool, seed: int) -> None:
    """Find +EV bets in a set of prices."""
    from .scanner import evaluate, scan

    config = load_config(config_path)
    if min_ev is not None:
        config.scan.min_ev = min_ev
    if books:
        config.scan.target_books = [b.strip().lower() for b in books.split(",") if b.strip()]
    if model_only:
        config.scan.allow_model_only = True
    quotes, world = _load_quotes(odds, odds_csv, sample, sport, seed)
    if quotes.empty:
        raise click.ClickException("no usable quotes found")
    if sample:
        click.echo("SIMULATED slate: fictional players and prices, for demonstration only.\n", err=True)
    projections = {} if (no_model or sample) else _projections_for(quotes, cache_dir, offline)
    picks = scan(quotes, config, projections)
    props = quotes.groupby(["event_id", "player_key", "market"]).ngroups
    click.echo(f"{len(quotes):,} quotes, {props:,} props, {quotes['book'].nunique()} books -> {len(picks)} bets")
    click.echo(_picks_table(picks, top))
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        picks.to_csv(out_path, index=False)
        click.echo(f"\nwrote {out_path}")
    if html_path:
        from .report import write_report

        write_report(picks, html_path, quotes=quotes, simulated=sample)
        click.echo(f"wrote {html_path}")


@main.command("demo")
@click.option("--sport", default="nba", type=click.Choice(["nba", "nfl"]), show_default=True)
@click.option("--seed", type=int, default=7, show_default=True)
@click.option("--html", "html_path", default=None, help="Also write an HTML report.")
def demo_cmd(sport: str, seed: int, html_path: Optional[str]) -> None:
    """End-to-end run on a simulated slate (no key or network needed)."""
    from .providers.sample import make_slate
    from .scanner import scan

    world = make_slate(sport, seed=seed)
    picks = scan(world.quotes)
    click.echo("SIMULATED slate: fictional players and prices. Nothing here is a real bet.\n")
    click.echo(f"{len(world.quotes):,} quotes from {world.quotes['book'].nunique()} simulated books -> {len(picks)} bets\n")
    click.echo(_picks_table(picks, 25))
    if not picks.empty:
        true_ev = world.true_ev(picks)
        click.echo(
            f"\nBecause the slate is simulated the truth is known: predicted EV {_pct(picks['ev'].mean(), 2)}, "
            f"true EV {_pct(true_ev.mean(), 2)}, {int((true_ev > 0).sum())}/{len(picks)} bets truly +EV."
        )
    if html_path:
        from .report import write_report

        write_report(picks, html_path, quotes=world.quotes, simulated=True)
        click.echo(f"wrote {html_path}")


# ------------------------------------------------------------------ validation
@main.command("calibrate")
@click.option("--sport", required=True, type=click.Choice(["nba", "nhl", "nfl"]))
@click.option("--fit", "fit_seasons", default=None, help="Seasons to fit on, e.g. 2017-2021.")
@click.option("--val", "val_seasons", default=None, help="Seasons to select the family on.")
@click.option("--test", "test_seasons", default=None, help="Held-out seasons for the report.")
@click.option("--markets", default=None, help="Comma-separated market keys.")
@click.option("--write", is_flag=True, help="Save the fitted specs into the package.")
@click.option("--out", "out_path", default=None, help="Save the fitted specs to this JSON instead.")
@click.option("--cache-dir", default="data/raw", show_default=True)
@click.option("--offline", is_flag=True)
def calibrate_cmd(sport: str, fit_seasons: Optional[str], val_seasons: Optional[str], test_seasons: Optional[str],
                  markets: Optional[str], write: bool, out_path: Optional[str], cache_dir: str, offline: bool) -> None:
    """Fit each market's distribution on real game logs and test it out of sample."""
    from .projection.fit import DEFAULT_SPLITS, fit_markets, save_specs, specs_payload, summary_frame

    splits = DEFAULT_SPLITS[sport]
    fit_s = _parse_seasons(fit_seasons) or list(splits["fit"])
    val_s = _parse_seasons(val_seasons) or list(splits["val"])
    test_s = _parse_seasons(test_seasons) if test_seasons is not None else list(splits["test"])
    seasons = range(min(fit_s) - 2, max(fit_s + val_s + (test_s or [])) + 1)
    logs = _load_logs(sport, cache_dir, seasons, offline)
    click.echo(f"{len(logs):,} player-games; fit {fit_s[0]}-{fit_s[-1]}, select {val_s[0]}-{val_s[-1]}, "
               f"test {test_s[0]}-{test_s[-1]}" if test_s else "no test seasons")
    fits = fit_markets(sport, logs, fit_s, val_s, test_s,
                       markets=[m.strip() for m in markets.split(",")] if markets else None,
                       progress=lambda message: click.echo("  " + message, err=True))
    click.echo(summary_frame(fits).round(4).to_string(index=False))
    if write or out_path:
        payload = specs_payload(fits, sport, {"fit_seasons": fit_s, "val_seasons": val_s, "test_seasons": test_s})
        click.echo(f"saved to {save_specs(payload, out_path)}")


@main.command("backtest")
@click.option("--sport", required=True, type=click.Choice(["nba", "nhl", "nfl"]))
@click.option("--seasons", default=None, help="Seasons to score (default: the held-out test seasons).")
@click.option("--markets", default=None, help="Comma-separated market keys.")
@click.option("--detail", is_flag=True, help="Print each market's reliability table.")
@click.option("--cache-dir", default="data/raw", show_default=True)
@click.option("--offline", is_flag=True)
def backtest_cmd(sport: str, seasons: Optional[str], markets: Optional[str], detail: bool,
                 cache_dir: str, offline: bool) -> None:
    """Walk-forward calibration of the projection model on real results.

    Every game is projected from earlier games only, then the model's
    over/under probabilities are compared with what happened.
    """
    from .projection.fit import DEFAULT_SPLITS, evaluate_markets, summary_frame

    score = _parse_seasons(seasons) or list(DEFAULT_SPLITS[sport]["test"])
    logs = _load_logs(sport, cache_dir, range(min(score) - 3, max(score) + 1), offline)
    results = evaluate_markets(sport, logs, score, markets=[m.strip() for m in markets.split(",")] if markets else None)
    if not results:
        raise click.ClickException("not enough data to evaluate")
    table = summary_frame(results)
    click.echo(f"seasons {score[0]}-{score[-1]}: probabilities at lines set around the model's median")
    click.echo(table.round(4).to_string(index=False))
    click.echo(
        "\nece = average gap between predicted and observed over-frequency (0.01 = one percentage point);\n"
        "brier_baseline = score of always predicting the base rate (lower brier is better)."
    )
    if detail:
        for key, result in results.items():
            click.echo(f"\n{key}\n{result.reliability.round(4).to_string(index=False)}")


@main.command("simulate")
@click.option("--slates", type=int, default=40, show_default=True)
@click.option("--sport", default="nba", type=click.Choice(["nba", "nfl"]), show_default=True)
@click.option("--market-noise", type=float, default=0.015, show_default=True,
              help="Error shared by every book (information nobody has priced).")
@click.option("--config", "config_path", default=None)
@click.option("--seed", type=int, default=0, show_default=True)
def simulate_cmd(slates: int, sport: str, market_noise: float, config_path: Optional[str], seed: int) -> None:
    """Check the scanner's EV claims in a simulated market where truth is known."""
    from .simulation import simulate_market

    result = simulate_market(slates, sport, load_config(config_path), market_noise=market_noise, seed=seed)
    summary = result.summary()
    if not summary.get("n_bets"):
        click.echo("no bets were recommended")
        return
    click.echo("SIMULATION - synthetic books and outcomes, not real betting results.\n")
    click.echo(f"{summary['n_bets']} bets over {slates} slates ({summary['bets_per_slate']:.1f} per slate), "
               f"{summary['units_staked']:.1f} units staked")
    click.echo(f"predicted EV {_pct(summary['predicted_ev'], 2)}   true EV {_pct(summary['true_ev'], 2)}   "
               f"realized ROI {_pct(summary['realized_roi'], 2)}   truly +EV {summary['share_truly_positive']:.0%}")
    click.echo("\nby predicted edge:\n" + result.by_ev_bucket().round(4).to_string())
    click.echo("\nby book:\n" + result.by("book").round(4).to_string())


# ------------------------------------------------------------------------ bets
@main.group("bets")
def bets_group() -> None:
    """Track placed bets, closing-line value and results."""


_LEDGER = click.option("--ledger", default="data/bets.csv", show_default=True, help="Ledger CSV file.")


@bets_group.command("add")
@_LEDGER
@click.option("--player", required=True)
@click.option("--market", required=True)
@click.option("--line", type=float, required=True)
@click.option("--side", type=click.Choice(["over", "under"]), required=True)
@click.option("--price", required=True, help="Price obtained, e.g. -110 or 1.91.")
@click.option("--stake", type=float, required=True)
@click.option("--book", default="")
@click.option("--sport", default="")
@click.option("--fair-prob", type=float, default=None, help="Your win probability at bet time.")
@click.option("--notes", default="")
def bets_add(ledger: str, player: str, market: str, line: float, side: str, price: str, stake: float,
             book: str, sport: str, fair_prob: Optional[float], notes: str) -> None:
    """Record a bet you placed."""
    from .tracking import BetLedger

    bet_id = BetLedger(ledger).add(
        player=player, market=market, line=line, side=side, price=parse_odds(price), stake=stake,
        book=book, sport=sport, fair_prob=fair_prob, notes=notes,
    )
    click.echo(f"recorded {bet_id}")


@bets_group.command("add-picks")
@_LEDGER
@click.argument("picks_csv")
@click.option("--unit", type=float, required=True, help="Money value of one unit.")
@click.option("--top", type=int, default=None, help="Only the first N picks.")
def bets_add_picks(ledger: str, picks_csv: str, unit: float, top: Optional[int]) -> None:
    """Record every pick from a scan output CSV as placed (stake = units x unit)."""
    from .tracking import BetLedger, DuplicateBetError

    picks = pd.read_csv(picks_csv)
    book = BetLedger(ledger)
    added = skipped = 0
    for _, row in (picks.head(top) if top else picks).iterrows():
        try:
            book.add(row, stake=float(row["stake_units"]) * unit)
            added += 1
        except DuplicateBetError:
            skipped += 1
    click.echo(f"recorded {added} bets" + (f", skipped {skipped} already in the ledger" if skipped else ""))


@bets_group.command("close")
@_LEDGER
@click.argument("bet_id")
@click.option("--over", "over_price", default=None, help="Closing over price.")
@click.option("--under", "under_price", default=None, help="Closing under price.")
@click.option("--line", type=float, default=None, help="Closing line if it moved.")
def bets_close(ledger: str, bet_id: str, over_price: Optional[str], under_price: Optional[str], line: Optional[float]) -> None:
    """Record closing prices to measure closing-line value."""
    from .tracking import BetLedger

    book = BetLedger(ledger)
    bet = book.to_frame().set_index("bet_id").loc[bet_id]
    market = find_market(str(bet["sport"]), str(bet["market"]))
    result = book.record_close(
        bet_id,
        closing_over_price=parse_odds(over_price) if over_price else None,
        closing_under_price=parse_odds(under_price) if under_price else None,
        closing_line=line,
        spec=market.spec if market else None,
    )
    click.echo(f"closing fair prob {_prob(result['closing_fair_prob'])}   CLV (price) {_pct(result['clv_price'], 2)}   "
               f"CLV (EV at close) {_pct(result['clv_ev'], 2)}")


@bets_group.command("settle")
@_LEDGER
@click.argument("bet_id")
@click.option("--actual", type=float, default=None, help="The stat's final value.")
@click.option("--result", type=click.Choice(["win", "loss", "push", "void"]), default=None)
def bets_settle(ledger: str, bet_id: str, actual: Optional[float], result: Optional[str]) -> None:
    """Settle a bet from the final stat value (or an explicit result)."""
    from .tracking import BetLedger

    if actual is None and result is None:
        raise click.UsageError("give --actual or --result")
    pnl = BetLedger(ledger).settle(bet_id, result=result, actual=actual)
    click.echo(f"settled: pnl {pnl:+.2f}")


@bets_group.command("list")
@_LEDGER
@click.option("--open", "only_open", is_flag=True, help="Only unsettled bets.")
def bets_list(ledger: str, only_open: bool) -> None:
    """Show recorded bets."""
    from .tracking import BetLedger

    book = BetLedger(ledger)
    frame = book.open_bets() if only_open else book.to_frame()
    if frame.empty:
        click.echo("no bets")
        return
    cols = ["bet_id", "player", "market", "line", "side", "book", "price", "stake", "ev", "clv_ev", "result", "pnl"]
    click.echo(frame[cols].to_string(index=False))


@bets_group.command("summary")
@_LEDGER
@click.option("--by", default=None, help="Break results down by a column (market, book, sport, side).")
def bets_summary(ledger: str, by: Optional[str]) -> None:
    """Results next to what the model expected, with the sampling error."""
    from .tracking import BetLedger, breakdown, summarize

    frame = BetLedger(ledger).to_frame()
    stats = summarize(frame)
    if not stats.get("n_bets"):
        click.echo("no bets")
        return

    def num(key: str, fmt: str = "{:.2f}") -> str:
        value = stats.get(key)
        return "n/a" if value is None or (isinstance(value, float) and not np.isfinite(value)) else fmt.format(value)

    click.echo(f"bets {stats['n_bets']} (settled {stats['n_settled']}, open {stats['n_open']})   "
               f"record {stats['wins']}-{stats['losses']}-{stats['pushes']}")
    ci_low, ci_high = stats.get("roi_ci", (float("nan"), float("nan")))
    ci = f"{ci_low:+.2%} to {ci_high:+.2%}" if np.isfinite(ci_low) and np.isfinite(ci_high) else "n/a"
    click.echo(f"staked {num('total_staked')}   pnl {num('pnl', '{:+.2f}')}   ROI {num('roi', '{:+.2%}')}   95% CI {ci}")
    click.echo(f"expected ROI {num('expected_roi', '{:+.2%}')}   luck {num('luck', '{:+.2f}')}   "
               f"z vs expectation {num('z_score')}   P(no edge) {num('p_value_profit', '{:.3f}')}")
    click.echo(f"CLV: avg EV at close {num('avg_clv_ev', '{:+.2%}')}   beat the close {num('clv_beat_rate', '{:.0%}')} "
               f"of {stats.get('clv_n', 0)} tracked")
    click.echo(f"bets needed to confirm this edge at 2 sigma: {num('bets_needed', '{:,.0f}')}")
    if by:
        click.echo("\n" + breakdown(frame, by=by).round(4).to_string())


if __name__ == "__main__":  # pragma: no cover
    main()
