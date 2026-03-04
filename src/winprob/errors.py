"""Centralized error taxonomy per AGENTS.md section 9.

All domain errors inherit from ``WinProbError`` so callers can catch the
entire family with a single ``except WinProbError`` clause.  Each subclass
maps directly to an entry in the AGENTS.md error taxonomy:

- IngestionError — raw data download or parsing failures
- APIError — MLB Stats API communication failures (alias for MLBAPIError)
- CoverageError — crosswalk coverage below required threshold
- SchemaError — unexpected column sets, types, or missing mandatory fields
- DriftComputationError — snapshot diff or metrics computation failures
- SnapshotIntegrityError — immutable snapshot corruption or schema violations
"""

from __future__ import annotations


class WinProbError(Exception):
    """Base class for all MLB win-probability domain errors."""


class IngestionError(WinProbError):
    """Failure during data ingestion (download, parse, or persist)."""


class APIError(WinProbError):
    """MLB Stats API communication failure (canonical name per AGENTS.md §9)."""


class CoverageError(WinProbError):
    """Crosswalk coverage fell below the required threshold."""


class SchemaError(WinProbError):
    """Data does not conform to expected schema (missing columns, wrong types)."""


class DriftComputationError(WinProbError):
    """Prediction drift computation or metrics aggregation failed."""


class SnapshotIntegrityError(WinProbError):
    """Immutable prediction snapshot is corrupt or violates its schema contract."""
