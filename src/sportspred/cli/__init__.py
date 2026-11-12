"""Command-line interface: ``sportspred --help``.

Command modules named ``cmd_*`` in this package are discovered automatically;
each exposes ``register(group)`` and adds its own commands.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Optional

import click
import pandas as pd

from .. import __version__
from ..config import Settings, load_settings

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_colwidth", 80)


class Context:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._store = None

    @property
    def store(self):
        if self._store is None:
            from ..pipeline import open_store

            self._store = open_store(self.settings)
        return self._store


pass_ctx = click.make_pass_decorator(Context)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="sportspred")
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False), default=None,
              help="TOML settings file (default: $SPORTSPRED_CONFIG, else built-in defaults).")
@click.option("--data-dir", default=None, help="Data directory (default: $SPORTSPRED_DATA_DIR or ./data).")
@click.option("-v", "--verbose", is_flag=True, help="Log progress to stderr.")
@click.pass_context
def main(ctx: click.Context, config_path: Optional[str], data_dir: Optional[str], verbose: bool) -> None:
    """Multi-sport pregame probabilities from verified sports evidence.

    Probabilities are estimates. Nothing here guarantees a winning pick or a profit.
    """
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings(config_path)
    if data_dir:
        settings.paths.data_dir = data_dir
    ctx.obj = Context(settings)


def _discover() -> None:
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("cmd_"):
            module = importlib.import_module(f"{__name__}.{info.name}")
            module.register(main)


_discover()

__all__ = ["main", "Context", "pass_ctx"]
