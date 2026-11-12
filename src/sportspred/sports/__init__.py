"""Sport plugins and the registry that discovers them.

Each sub-package of :mod:`sportspred.sports` that defines a module-level
``PLUGIN`` (a :class:`~sportspred.sports.base.SportPlugin` instance) is picked
up automatically, so adding a sport means adding a package - no shared file
has to be edited.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from functools import lru_cache
from typing import Dict, List, Tuple

from .base import (
    AvailabilityEffect,
    Candidate,
    CheckSpec,
    Explanation,
    FeatureFrame,
    IngestReport,
    LeagueSpec,
    MarketDecl,
    SplitPlan,
    SportPlugin,
)

__all__ = [
    "AvailabilityEffect",
    "Candidate",
    "CheckSpec",
    "Explanation",
    "FeatureFrame",
    "IngestReport",
    "LeagueSpec",
    "MarketDecl",
    "SplitPlan",
    "SportPlugin",
    "all_plugins",
    "all_leagues",
    "get_league",
    "DataUnavailable",
]

log = logging.getLogger(__name__)


class DataUnavailable(RuntimeError):
    """A real data source could not be reached or is not on file.

    Raised instead of falling back to invented numbers.  The message should
    tell the user which CSV import fills the gap.
    """


@lru_cache(maxsize=1)
def all_plugins() -> Tuple[SportPlugin, ...]:
    plugins: List[SportPlugin] = []
    for info in pkgutil.iter_modules(__path__):
        if not info.ispkg:
            continue
        try:
            module = importlib.import_module(f"{__name__}.{info.name}")
        except Exception:  # a broken plugin must not take the platform down
            log.exception("sport plugin %s failed to import", info.name)
            continue
        plugin = getattr(module, "PLUGIN", None)
        if isinstance(plugin, SportPlugin):
            plugins.append(plugin)
    plugins.sort(key=lambda p: p.sport)
    seen: Dict[str, str] = {}
    for plugin in plugins:
        for spec in plugin.leagues:
            if spec.key in seen:
                raise RuntimeError(
                    f"league key {spec.key!r} is declared by both {seen[spec.key]} and {plugin.sport}"
                )
            seen[spec.key] = plugin.sport
    return tuple(plugins)


def all_leagues() -> List[Tuple[SportPlugin, LeagueSpec]]:
    return [(plugin, spec) for plugin in all_plugins() for spec in plugin.leagues]


def get_league(key: str) -> Tuple[SportPlugin, LeagueSpec]:
    for plugin, spec in all_leagues():
        if spec.key == key:
            return plugin, spec
    known = ", ".join(sorted(spec.key for _, spec in all_leagues()))
    raise KeyError(f"unknown league {key!r}; known leagues: {known}")
