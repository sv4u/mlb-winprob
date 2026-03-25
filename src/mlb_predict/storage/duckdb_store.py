"""Hybrid DuckDB storage layer — single-file analytical database backed by Parquet exports.

DuckDB provides ~10-50x faster reads for analytical queries (feature loading,
training data concatenation, season filtering) compared to scanning individual
Parquet files with pandas.  Parquet files remain the canonical export format for
interoperability, snapshots, and checksum verification.

Usage
-----
    store = DuckDBStore()               # uses default path data/processed/mlb_predict.duckdb
    store.ingest_parquet("features", "data/processed/features/features_2025.parquet", season=2025)
    df = store.query_features(seasons=[2020, 2021, 2022])
    store.close()
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_DB_PATH = _REPO_ROOT / "data" / "processed" / "mlb_predict.duckdb"

_lock = threading.Lock()
_instance: DuckDBStore | None = None


def get_store(db_path: Path | None = None) -> "DuckDBStore":
    """Return the singleton DuckDBStore, creating it on first call."""
    global _instance
    if _instance is None or _instance._closed:
        with _lock:
            if _instance is None or _instance._closed:
                _instance = DuckDBStore(db_path or _DEFAULT_DB_PATH)
    return _instance


class DuckDBStore:
    """Hybrid DuckDB + Parquet storage for MLB prediction data.

    Ingests Parquet files into DuckDB tables for fast analytical queries.
    Exports back to Parquet for snapshot immutability and interoperability.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self._db_path))
        self._closed = False
        self._ensure_schema()
        logger.info("DuckDB store opened: %s", self._db_path)

    def _scalar(self, query: str, params: list[Any] | None = None) -> Any:
        """Execute a query and return the first column of the first row."""
        row = self._conn.execute(query, params or []).fetchone()
        if row is None:
            raise RuntimeError(f"Expected a result row from: {query}")
        return row[0]

    def _ensure_schema(self) -> None:
        """Create metadata tables.  The features table schema is derived from
        the first ingested Parquet file so it matches all ~143 columns."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS ingest_log (
                table_name VARCHAR,
                source_file VARCHAR,
                season INTEGER,
                row_count INTEGER,
                ingested_at TIMESTAMP DEFAULT current_timestamp
            )
        """)

    def _features_table_exists(self) -> bool:
        return (
            self._scalar(
                "SELECT count(*) FROM information_schema.tables "
                "WHERE table_catalog = current_database() "
                "AND table_schema = 'main' AND table_name = 'features'"
            )
            > 0
        )

    def feature_count(self) -> int:
        """Return the number of rows in the features table, or 0 if it doesn't exist."""
        if not self._features_table_exists():
            return 0
        return self._scalar("SELECT count(*) FROM features")

    def close(self) -> None:
        """Close the DuckDB connection."""
        if not self._closed:
            self._conn.close()
            self._closed = True
            logger.info("DuckDB store closed")

    def ingest_parquet(
        self,
        table: str,
        parquet_path: str | Path,
        *,
        season: int | None = None,
        replace_season: bool = True,
    ) -> int:
        """Load a Parquet file into a DuckDB table.

        If replace_season is True and season is provided, existing rows for
        that season are deleted before insertion (upsert semantics).
        Returns the number of rows inserted.
        """
        parquet_path = Path(parquet_path)
        if not parquet_path.exists():
            logger.warning("Parquet file not found: %s", parquet_path)
            return 0

        if table == "features":
            return self._ingest_features(parquet_path, season=season, replace_season=replace_season)

        return self._ingest_generic(
            table, parquet_path, season=season, replace_season=replace_season
        )

    def _ingest_features(
        self,
        parquet_path: Path,
        *,
        season: int | None = None,
        replace_season: bool = True,
    ) -> int:
        """Ingest a feature Parquet file.

        On first call the features table is created from the Parquet schema so
        that all ~143 columns are preserved.  Subsequent calls INSERT into the
        existing table.
        """
        path_str = str(parquet_path.resolve())
        source_name = parquet_path.name
        if self._features_table_exists():
            if replace_season and season is not None:
                self._conn.execute("DELETE FROM features WHERE season = ?", [season])
            self._conn.execute(
                """
                INSERT INTO features
                SELECT *, ? AS _source_file
                FROM read_parquet(?)
                """,
                [source_name, path_str],
            )
        else:
            self._conn.execute(
                """
                CREATE TABLE features AS
                SELECT *, ? AS _source_file
                FROM read_parquet(?)
                """,
                [source_name, path_str],
            )

        count = self._scalar(
            "SELECT count(*) FROM features WHERE _source_file = ?",
            [parquet_path.name],
        )

        self._conn.execute(
            "INSERT INTO ingest_log VALUES (?, ?, ?, ?, current_timestamp)",
            ["features", str(parquet_path), season, count],
        )
        logger.info("Ingested %d rows from %s into features", count, parquet_path.name)
        return count

    def _ingest_generic(
        self,
        table: str,
        parquet_path: Path,
        *,
        season: int | None = None,
        replace_season: bool = True,
    ) -> int:
        """Ingest any Parquet file into a named table (auto-schema from Parquet).

        Returns the number of rows actually inserted (not the total table size).
        """
        path_str = str(parquet_path.resolve())
        table_exists = (
            self._scalar(
                "SELECT count(*) FROM information_schema.tables "
                "WHERE table_catalog = current_database() "
                "AND table_schema = 'main' AND table_name = ?",
                [table],
            )
            > 0
        )

        if not table_exists:
            self._conn.execute(
                f"CREATE TABLE {table} AS SELECT * FROM read_parquet(?)",
                [path_str],
            )
            inserted = self._scalar(f"SELECT count(*) FROM {table}")
        else:
            if replace_season and season is not None:
                try:
                    self._conn.execute(f"DELETE FROM {table} WHERE season = ?", [season])
                except duckdb.BinderException:
                    pass
            before = self._scalar(f"SELECT count(*) FROM {table}")
            self._conn.execute(
                f"INSERT INTO {table} SELECT * FROM read_parquet(?)",
                [path_str],
            )
            after = self._scalar(f"SELECT count(*) FROM {table}")
            inserted = after - before

        self._conn.execute(
            "INSERT INTO ingest_log VALUES (?, ?, ?, ?, current_timestamp)",
            [table, str(parquet_path), season, inserted],
        )
        return inserted

    def ingest_all_features(self, features_dir: Path | None = None) -> int:
        """Bulk-ingest all feature Parquet files from the features directory.

        Drops and recreates the features table from the first file so the schema
        always matches the Parquet layout (all ~143 columns).
        """
        features_dir = features_dir or (_REPO_ROOT / "data" / "processed" / "features")
        if not features_dir.exists():
            logger.warning("Features directory not found: %s", features_dir)
            return 0

        parquet_files = sorted(features_dir.glob("features_*.parquet"))
        if not parquet_files:
            logger.warning("No feature Parquet files found in %s", features_dir)
            return 0

        if self._features_table_exists():
            self._conn.execute("DROP TABLE features")

        total = 0
        for idx, f in enumerate(parquet_files):
            path_str = str(f.resolve())
            if idx == 0:
                self._conn.execute(
                    """
                    CREATE TABLE features AS
                    SELECT *, ? AS _source_file
                    FROM read_parquet(?)
                    """,
                    [f.name, path_str],
                )
            else:
                self._conn.execute(
                    """
                    INSERT INTO features
                    SELECT *, ? AS _source_file
                    FROM read_parquet(?)
                    """,
                    [f.name, path_str],
                )

            count_result = self._conn.execute(
                "SELECT count(*) FROM features WHERE _source_file = ?", [f.name]
            ).fetchone()
            count = count_result[0] if count_result else 0
            total += count
            logger.debug("  %s: %d rows", f.name, count)

        logger.info("Bulk-ingested %d total rows from %d files", total, len(parquet_files))
        return total

    def query_features(
        self,
        *,
        seasons: list[int] | None = None,
        game_type: str | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Query feature data as a pandas DataFrame.

        This is the primary read path — replaces scanning Parquet files with a
        DuckDB query that is 10-50x faster for multi-season loads.
        """
        if not self._features_table_exists():
            logger.warning("Features table does not exist yet — returning empty DataFrame")
            return pd.DataFrame()

        col_clause = ", ".join(columns) if columns else "*"
        conditions: list[str] = []
        params: list[Any] = []

        if seasons:
            placeholders = ", ".join("?" for _ in seasons)
            conditions.append(f"season IN ({placeholders})")
            params.extend(seasons)
        if game_type:
            conditions.append("game_type = ?")
            params.append(game_type)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT {col_clause} FROM features {where} ORDER BY season, date"

        return self._conn.execute(query, params).fetchdf()

    def query_training_data(
        self,
        *,
        seasons: list[int] | None = None,
        include_spring: bool = True,
    ) -> dict[int, pd.DataFrame]:
        """Load training data grouped by season — optimized for expanding-window CV.

        Returns a dict mapping season -> DataFrame, matching the interface
        expected by train.py._load_all_feature_files().
        """
        if not self._features_table_exists():
            logger.warning("Features table does not exist yet — returning empty dict")
            return {}

        conditions = ["home_win IS NOT NULL"]
        params: list[Any] = []

        if seasons:
            placeholders = ", ".join("?" for _ in seasons)
            conditions.append(f"season IN ({placeholders})")
            params.extend(seasons)
        if not include_spring:
            conditions.append("(is_spring IS NULL OR is_spring = 0.0)")

        where = f"WHERE {' AND '.join(conditions)}"
        df = self._conn.execute(
            f"SELECT * FROM features {where} ORDER BY season, date", params
        ).fetchdf()

        if "is_spring" not in df.columns:
            df["is_spring"] = 0.0
        else:
            df["is_spring"] = df["is_spring"].fillna(0.0)

        result: dict[int, pd.DataFrame] = {}
        for season, group in df.groupby("season"):
            result[int(season)] = group.reset_index(drop=True)
        return result

    def table_stats(self) -> dict[str, Any]:
        """Return row counts and metadata for dashboard display."""
        tables = self._conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_catalog = current_database() AND table_schema = 'main'"
        ).fetchall()
        stats: dict[str, Any] = {"db_path": str(self._db_path)}
        for (tbl,) in tables:
            if tbl == "ingest_log":
                continue
            count = self._scalar(f"SELECT count(*) FROM {tbl}")
            stats[tbl] = {"rows": count}
            if tbl == "features":
                seasons = self._conn.execute(
                    "SELECT DISTINCT season FROM features ORDER BY season"
                ).fetchall()
                stats[tbl]["seasons"] = [s[0] for s in seasons]
        return stats

    def export_parquet(self, table: str, output_path: Path) -> None:
        """Export a DuckDB table back to Parquet for interoperability."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        has_source_file = (
            self._scalar(
                "SELECT count(*) FROM information_schema.columns "
                "WHERE table_catalog = current_database() "
                "AND table_schema = 'main' AND table_name = ? AND column_name = '_source_file'",
                [table],
            )
            > 0
        )
        exclude = " EXCLUDE (_source_file)" if has_source_file else ""
        out_str = str(output_path.resolve())
        self._conn.execute(
            f"COPY (SELECT *{exclude} FROM {table}) TO ? (FORMAT PARQUET)",
            [out_str],
        )
        logger.info("Exported %s to %s", table, output_path)

    def execute(self, query: str, params: list[Any] | None = None) -> Any:
        """Execute a raw DuckDB SQL query (for advanced use)."""
        if params:
            return self._conn.execute(query, params)
        return self._conn.execute(query)
