"""sportspred - a multi-sport pregame probability platform.

The pieces, in the order an event flows through them:

* :mod:`sportspred.adapters` - ingestion from public datasets, official feeds
  (where reachable) and documented CSV imports, each with provenance;
* :mod:`sportspred.storage` - durable SQLite + parquet store with stable IDs,
  entity matching, de-duplication and an append-only prediction ledger;
* :mod:`sportspred.sports` - one plugin per sport: point-in-time features and
  candidate models suited to that sport's outcome structure;
* :mod:`sportspred.validation` - chronological walk-forward validation with a
  reserved final test period, baselines, calibration and uncertainty;
* :mod:`sportspred.predict` - pregame predictions with FINAL / PROVISIONAL /
  CONDITIONAL / PASS statuses;
* :mod:`sportspred.props`, :mod:`sportspred.combos` - props and multi-leg
  combinations with explicit dependency handling;
* :mod:`sportspred.ev` - the optional, separate price comparison module;
* :mod:`sportspred.ledger` - grading, settlement, reviews, lessons, drift.

The prediction engine never reads betting odds.  Prices only enter through
:mod:`sportspred.ev`, after the sports-based probability exists.

Nothing here guarantees a winning pick or a profit.  Probabilities are
estimates; a well calibrated 70% selection still loses three times in ten.
"""

__version__ = "0.2.0"

__all__ = ["__version__"]
