"""Settings: every tunable rule in one place, overridable from a TOML file.

``Settings()`` gives the defaults.  ``load_settings(path)`` reads a TOML file
(see ``examples/sportspred.toml``); the path may also come from the
``SPORTSPRED_CONFIG`` environment variable and the data directory from
``SPORTSPRED_DATA_DIR``.  Credentials are **only** ever read from environment
variables (see ``.env.example``) and never stored in a config file.

The numbers here are *selection and freshness rules*, not model weights.
Model weights are learned in validation; these rules are explicit, editable,
and their effect on historical performance is reported by ``sportspred
validate`` (see the PASS-rule table).
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

__all__ = [
    "PathsConfig",
    "SelectionConfig",
    "ConfidenceConfig",
    "ComboConfig",
    "EvPreset",
    "EvConfig",
    "ValidationSettings",
    "Settings",
    "load_settings",
    "credential",
]


@dataclass
class PathsConfig:
    data_dir: str = "data"

    @property
    def store(self) -> Path:
        return Path(self.data_dir) / "store"

    @property
    def raw(self) -> Path:
        return Path(self.data_dir) / "raw"

    @property
    def reports(self) -> Path:
        return Path(self.data_dir) / "reports"


@dataclass
class SelectionConfig:
    #: A selection needs at least this probability; below it the status is PASS.
    #: Keyed by outcome type.  The validated hit-rate at these thresholds is in
    #: every validation report.
    min_prob_binary: float = 0.60
    min_prob_three_way: float = 0.45
    min_prob_multi: float = 0.20
    min_prob_over_under: float = 0.55
    #: Per (league, market) overrides, e.g. {"nba:winner": 0.62}.
    min_prob_overrides: Dict[str, float] = field(default_factory=dict)
    #: Coverage statuses allowed to produce selections (anything else is PASS).
    allowed_coverage: List[str] = field(default_factory=lambda: ["validated", "experimental"])
    #: PASS when the lower end of the model-uncertainty interval of a binary
    #: selection is not above a coin flip.
    require_interval_above_even: bool = True
    #: Only events starting within this many hours get predictions.
    horizon_hours: float = 36.0
    #: Results on file count as out of date when the newest one is further
    #: before the event than this multiple of the league's typical longest
    #: yearly break (the median, over the years on file, of each year's longest
    #: gap between result days).  Ordinary off-seasons pass; a source that has
    #: simply stopped does not.
    results_break_margin: float = 1.5
    #: Freshness overrides per check key (hours), e.g. {"injury_report": 12}.
    max_age_hours: Dict[str, float] = field(default_factory=dict)
    recheck_within_hours: Dict[str, float] = field(default_factory=dict)


@dataclass
class ConfidenceConfig:
    """Labels are assigned from *observed* out-of-sample hit rates of the
    probability band a selection falls in, never from the raw probability."""

    min_band_events: int = 150
    high_observed_low: float = 0.70     # lower 95% bound of the band's observed hit rate
    medium_observed_low: float = 0.60


@dataclass
class ComboConfig:
    primary_size: int = 3
    max_size: int = 6
    #: Statuses a leg must have to be part of a recommended combo.
    leg_statuses: List[str] = field(default_factory=lambda: ["FINAL"])
    #: Show clearly labelled PROVISIONAL previews built from PROVISIONAL legs.
    show_provisional_preview: bool = True
    min_leg_prob: float = 0.65
    #: Larger combos are only shown while the joint probability stays above this.
    min_joint_prob_larger: float = 0.25
    alternatives_per_size: int = 3
    #: Legs from different events may be treated as independent only when the
    #: validated same-slate residual correlation is within this bound.
    max_abs_residual_correlation: float = 0.03
    allowed_coverage: List[str] = field(default_factory=lambda: ["validated"])


@dataclass
class EvPreset:
    """Eligibility band on the *raw* market-implied probability of a selection."""

    name: str = "us_major_team_win"
    leagues: List[str] = field(default_factory=lambda: ["nba", "nhl", "nfl"])
    markets: List[str] = field(default_factory=lambda: ["winner"])
    raw_implied_min: float = 0.40
    raw_implied_max: float = 0.70
    enabled: bool = True


@dataclass
class EvConfig:
    devig_method: str = "multiplicative"
    max_quote_age_minutes: float = 30.0
    min_edge: float = 0.02
    presets: List[EvPreset] = field(default_factory=lambda: [EvPreset()])
    paper_stake: float = 1.0
    mode: str = "paper"                 # the only supported mode


@dataclass
class ValidationSettings:
    n_boot: int = 500
    seed: int = 20261112
    min_final_events: int = 300
    min_dev_folds: int = 3
    max_ece: float = 0.03
    n_replicas: int = 30


@dataclass
class Settings:
    paths: PathsConfig = field(default_factory=PathsConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    combos: ComboConfig = field(default_factory=ComboConfig)
    ev: EvConfig = field(default_factory=EvConfig)
    validation: ValidationSettings = field(default_factory=ValidationSettings)

    def min_prob(self, league: str, market: str, outcome: str) -> float:
        key = f"{league}:{market}"
        if key in self.selection.min_prob_overrides:
            return float(self.selection.min_prob_overrides[key])
        return {
            "binary": self.selection.min_prob_binary,
            "three_way": self.selection.min_prob_three_way,
            "multi": self.selection.min_prob_multi,
            "over_under": self.selection.min_prob_over_under,
        }.get(str(outcome), self.selection.min_prob_binary)


def _apply(obj: Any, data: Mapping[str, Any], path: str = "") -> None:
    known = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"unknown setting {path}{key!r}; known: {sorted(known)}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, Mapping):
            _apply(current, value, f"{path}{key}.")
        elif key == "presets" and isinstance(value, list):
            presets = []
            for item in value:
                preset = EvPreset()
                _apply(preset, item, f"{path}presets.")
                presets.append(preset)
            setattr(obj, key, presets)
        else:
            if isinstance(current, float) and isinstance(value, int) and not isinstance(value, bool):
                value = float(value)
            if current is not None and not isinstance(current, (dict, list)) and not isinstance(value, type(current)):
                raise ValueError(
                    f"setting {path}{key} expects {type(current).__name__}, got {type(value).__name__}")
            setattr(obj, key, value)


def load_settings(path: Optional[Union[str, Path]] = None) -> Settings:
    """Defaults, overridden by a TOML file and then by environment variables."""
    settings = Settings()
    path = path or os.environ.get("SPORTSPRED_CONFIG")
    if path:
        with open(path, "rb") as fh:
            _apply(settings, tomllib.load(fh))
    if os.environ.get("SPORTSPRED_DATA_DIR"):
        settings.paths.data_dir = os.environ["SPORTSPRED_DATA_DIR"]
    for preset in settings.ev.presets:
        if not (0.0 <= preset.raw_implied_min < preset.raw_implied_max <= 1.0):
            raise ValueError(f"EV preset {preset.name!r} has an invalid implied-probability band")
    if settings.ev.mode != "paper":
        raise ValueError("only paper-tracking mode is supported; this platform never places bets")
    return settings


def credential(name: str) -> Optional[str]:
    """A secret from the environment (``None`` when unset).  Never logged."""
    value = os.environ.get(name)
    return value if value else None
