"""The contract every sport plugin implements.

A plugin owns everything sport-specific: where its data comes from, how
point-in-time features are built, which outcome structure a market has, which
candidate models are worth comparing and which pregame checks must be complete
before a selection may be called FINAL.  Everything else (validation, the
prediction engine, combos, the ledger, the dashboard) is shared and only talks
to plugins through the types in this module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..core.enums import OutcomeType

__all__ = [
    "MarketDecl",
    "LeagueSpec",
    "CheckSpec",
    "FeatureFrame",
    "Candidate",
    "SplitPlan",
    "Explanation",
    "AvailabilityEffect",
    "IngestReport",
    "SportPlugin",
    "FRAME_REQUIRED",
]

#: Columns every feature frame must carry.
FRAME_REQUIRED = ("event_id", "start_time_utc", "block", "y", "side_a_id", "side_b_id")


@dataclass(frozen=True)
class MarketDecl:
    """A market a league declares, and whether a model exists for it."""

    key: str                       # winner | winner_3way | spread | total | player_prop | finish
    outcome: OutcomeType
    implemented: bool = True       # features + candidate models exist
    includes_overtime: Optional[bool] = None
    note: str = ""


@dataclass(frozen=True)
class LeagueSpec:
    key: str                       # nba, wnba, ncaam, atp, atp_challenger, ...
    sport: str
    name: str
    gender: str = "men"            # men | women | mixed | open
    level: str = "pro"             # competition level, never mixed inside one league key
    discipline: Optional[str] = None   # singles | doubles | ...
    markets: Tuple[MarketDecl, ...] = ()
    real_data_source: str = ""     # reachable real source used by the adapter; "" = CSV import only
    live_feeds: str = ""           # official feeds an adapter exists for (may be unreachable here)
    notes: str = ""

    def market(self, key: str) -> MarketDecl:
        for m in self.markets:
            if m.key == key:
                return m
        raise KeyError(f"league {self.key!r} declares no market {key!r}")


@dataclass(frozen=True)
class CheckSpec:
    """A pregame check that must be complete before a selection is FINAL."""

    key: str                       # injury_report | lineup | starting_goalie | probable_pitcher | weather | ...
    label: str
    scope: str = "team"            # "team": needed for each side; "event": once per event
    max_age_hours: float = 24.0    # information older than this is stale
    recheck_within_hours: Optional[float] = None   # must have been checked this close to the start
    required: bool = True
    why: str = ""


@dataclass
class FeatureFrame:
    """Point-in-time model inputs: one row per event, completed and upcoming.

    ``df`` columns (see :data:`FRAME_REQUIRED`):

    * ``event_id``; ``start_time_utc`` (tz-aware); ``block`` - the chronological
      block used for fold boundaries (usually the season), sortable;
    * ``y`` - integer index into ``classes`` (NaN for events without a result);
    * ``side_a_id`` / ``side_b_id`` - for two-sided events.  ``classes`` for a
      binary market is ``("b", "a")`` so ``P[:, 1]`` is side A's probability.
      Side A is the home side where one exists, otherwise assigned by a rule
      that cannot depend on the result;
    * ``info_cutoff_utc`` (optional) - latest instant of any information used
      by the row's features; validation asserts it precedes the start.

    ``feature_cols`` are safe for honest backtests.  ``retrospective_cols``
    use information that was not knowable pregame (for example who actually
    played) and may only feed candidates flagged ``uses_retrospective``.
    """

    league: str
    market: str
    outcome: OutcomeType
    classes: Tuple[str, ...]
    df: pd.DataFrame
    feature_cols: List[str]
    retrospective_cols: List[str] = field(default_factory=list)
    feature_notes: Dict[str, str] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)
    point_in_time: bool = True     # False when even the standard features are a retrospective reconstruction
    extra: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        missing = [c for c in FRAME_REQUIRED if c not in self.df.columns]
        if missing:
            raise ValueError(f"feature frame lacks required columns {missing}")
        if self.df["event_id"].duplicated().any():
            raise ValueError("feature frame has duplicate event_id rows")
        if getattr(self.df["start_time_utc"].dt, "tz", None) is None:
            raise ValueError("feature frame start_time_utc must be timezone-aware")
        unknown = [c for c in self.feature_cols + self.retrospective_cols if c not in self.df.columns]
        if unknown:
            raise ValueError(f"feature columns missing from frame: {unknown}")
        overlap = set(self.feature_cols) & set(self.retrospective_cols)
        if overlap:
            raise ValueError(f"columns flagged both point-in-time and retrospective: {sorted(overlap)}")
        y = self.df["y"].dropna()
        if len(y) and (y.min() < 0 or y.max() >= len(self.classes) or (y != y.round()).any()):
            raise ValueError("target y must be an integer index into classes")
        if "info_cutoff_utc" in self.df.columns:
            late = self.df["info_cutoff_utc"].notna() & (self.df["info_cutoff_utc"] >= self.df["start_time_utc"])
            if late.any():
                bad = self.df.loc[late, "event_id"].head(3).tolist()
                raise ValueError(
                    f"{int(late.sum())} rows use information from at/after their own start (e.g. {bad})"
                )

    @property
    def completed(self) -> pd.DataFrame:
        return self.df[self.df["y"].notna()]

    @property
    def upcoming(self) -> pd.DataFrame:
        return self.df[self.df["y"].isna()]


class Candidate(ABC):
    """A model (or baseline) that can be fitted on past rows and asked for
    outcome probabilities on later rows.  Implementations must hold no state
    between :meth:`fit` calls other than what ``fit`` itself learns."""

    name: str = "candidate"
    description: str = ""
    is_baseline: bool = False
    complexity: int = 1            # lower = simpler; used by the one-standard-error rule
    uses_retrospective: bool = False
    calibration: str = "platt"     # platt | isotonic | none

    @abstractmethod
    def fit(self, train: pd.DataFrame, classes: Sequence[str]) -> "Candidate":
        """Learn from completed rows only.  Returns ``self``."""

    @abstractmethod
    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        """``(n_rows, n_classes)`` probabilities, each row summing to 1."""

    @abstractmethod
    def clone(self) -> "Candidate":
        """An unfitted copy with the same configuration."""

    def feature_names(self) -> List[str]:
        return []

    def contributions(self, rows: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Per-row additive contributions on the log-odds scale towards side A
        (binary markets), one column per feature, if the model has them."""
        return None


@dataclass
class SplitPlan:
    """How a league's history is cut into chronological blocks."""

    block_col: str = "block"
    n_final_blocks: int = 1        # most recent full blocks reserved as the untouched final test
    min_train_blocks: int = 3      # blocks of training data before the first dev fold
    n_calib_blocks: int = 1        # blocks between train and test used only for calibration/tuning
    burn_in_blocks: int = 0        # earliest blocks dropped from model fitting (rating warm-up)
    min_block_events: int = 50     # a trailing block smaller than this joins the previous one
    max_dev_folds: Optional[int] = None


@dataclass
class Explanation:
    supporting: List[str] = field(default_factory=list)
    upset_risks: List[str] = field(default_factory=list)
    opponent_strength: List[str] = field(default_factory=list)
    availability_assumptions: List[str] = field(default_factory=list)
    missing_data: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, List[str]]:
        return {
            "supporting": self.supporting,
            "upset_risks": self.upset_risks,
            "opponent_strength": self.opponent_strength,
            "availability_assumptions": self.availability_assumptions,
            "missing_data": self.missing_data,
        }


@dataclass
class AvailabilityEffect:
    """What published availability does to one event's model inputs.

    ``row`` has confirmed absences/returns applied.  ``worst_case_row`` assumes
    every *uncertain* player on the selected side sits out (``None`` when the
    plugin cannot quantify that); it is used to decide whether a selection is
    CONDITIONAL on those players.
    """

    row: pd.Series
    applied: List[str] = field(default_factory=list)
    uncertain_a: List[str] = field(default_factory=list)   # uncertain players that matter, side A
    uncertain_b: List[str] = field(default_factory=list)
    worst_case_rows: Dict[str, pd.Series] = field(default_factory=dict)   # {"a": row if A's doubts sit, "b": ...}
    unmatched: List[str] = field(default_factory=list)
    quantified: bool = False


@dataclass
class IngestReport:
    league: str
    n_events: int = 0
    n_results: int = 0
    n_upcoming: int = 0
    first_event: Optional[str] = None
    last_event: Optional[str] = None
    sources: List[str] = field(default_factory=list)
    tables: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"{self.league}: {self.n_events} events ({self.n_results} with results, "
            f"{self.n_upcoming} upcoming) {self.first_event or '?'} .. {self.last_event or '?'}"
        ]
        lines += [f"  table {k}: {v} rows" for k, v in self.tables.items()]
        lines += [f"  warning: {w}" for w in self.warnings]
        return "\n".join(lines)


class SportPlugin(ABC):
    """Base class of a sport plugin.  Subclasses set ``sport`` and ``leagues``."""

    sport: str = ""
    leagues: Tuple[LeagueSpec, ...] = ()

    def league(self, key: str) -> LeagueSpec:
        for spec in self.leagues:
            if spec.key == key:
                return spec
        raise KeyError(f"{self.sport} plugin has no league {key!r}")

    # -- data -----------------------------------------------------------------
    @abstractmethod
    def ingest(self, store: Any, league: str, *, seasons: Optional[Sequence[int]] = None,
               refresh: Optional[bool] = None, fetcher: Any = None) -> IngestReport:
        """Fetch real data for ``league`` into the store.  Must raise
        ``DataUnavailable`` (never substitute invented numbers) when the source
        cannot be reached."""

    # -- modelling ------------------------------------------------------------
    @abstractmethod
    def feature_frame(self, store: Any, league: str, market: str, *, now: Any = None) -> FeatureFrame:
        """Point-in-time features for every completed event and for events that
        are still upcoming as of ``now``."""

    @abstractmethod
    def candidates(self, league: str, market: str, frame: FeatureFrame) -> List[Candidate]:
        """Baselines first, then models in increasing complexity."""

    def split_plan(self, league: str, market: str, frame: FeatureFrame) -> SplitPlan:
        return SplitPlan()

    def required_checks(self, league: str, market: str) -> List[CheckSpec]:
        return []

    def market_benchmark(self, store: Any, league: str, market: str) -> Optional[pd.DataFrame]:
        """Optional market-implied probabilities for completed events, used
        only as a comparison in validation reports.  Columns: ``event_id``,
        one probability column per class, ``benchmark_label``,
        ``benchmark_timing`` (e.g. "closing")."""
        return None

    def explain(self, league: str, market: str, row: pd.Series, frame: FeatureFrame,
                contributions: Optional[Mapping[str, float]] = None) -> Explanation:
        return Explanation()

    def apply_availability(self, league: str, market: str, row: pd.Series, frame: FeatureFrame,
                           availability: pd.DataFrame) -> AvailabilityEffect:
        """Fold availability statements (as known at prediction time) into the
        row.  The default cannot quantify anything: it only lists uncertain
        players so the engine can make the selection CONDITIONAL on them."""
        effect = AvailabilityEffect(row=row)
        if availability is None or availability.empty:
            return effect
        for _, rep in availability.iterrows():
            text = f"{rep['player_name']} ({rep['status']})"
            if rep["status"] in ("doubtful", "questionable", "probable", "uncertain"):
                if rep.get("team_id") == row["side_a_id"]:
                    effect.uncertain_a.append(text)
                elif rep.get("team_id") == row["side_b_id"]:
                    effect.uncertain_b.append(text)
            else:
                effect.unmatched.append(text + " - reported but not used by this model")
        return effect

    def display_names(self, store: Any, league: str) -> Dict[str, str]:
        return store.participant_names()
