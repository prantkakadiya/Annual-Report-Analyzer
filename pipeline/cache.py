"""Cache module for SQLite-based caching and FAISS index/chunk persistence."""
import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any

import faiss

from pipeline.schema import AnalysisResult, DocumentChunk, ReportMetadata

logger = logging.getLogger(__name__)


class ReportCache:
    """Handles caching of reports, metadata, document chunks, and FAISS indices.

    Error-handling contract:
    - Write operations (save_*) log and silently swallow errors so the
      pipeline can continue even if caching fails.
    - Read operations (get_*, load_*, list_*) log and return ``None`` / empty
      list on failure so callers never crash on a cache miss.
    - Trivial helpers (make_company_id, get_company_data_dir) propagate
      exceptions — callers are expected to handle them.
    """

    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = Path(data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "cache.db"

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reports (
                    company_id   TEXT PRIMARY KEY,
                    company_name TEXT,
                    fiscal_year  TEXT,
                    analysis_data TEXT
                )
                """
            )
            conn.commit()
        logger.info("Initialized ReportCache at %s", self.data_dir)

    # ------------------------------------------------------------------
    # Helpers (trivial — exceptions propagate)
    # ------------------------------------------------------------------

    def make_company_id(self, company_name: str, fiscal_year: str) -> str:
        """Creates a slugified company ID."""
        base = f"{company_name}_{fiscal_year}"
        return re.sub(r"[^a-z0-9]+", "_", base.lower()).strip("_")

    def get_company_data_dir(self, company_id: str) -> Path:
        """Returns the data directory path for a given company, creating it if needed."""
        path = self.data_dir / company_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    # Report read / write (errors swallowed on write, None on read)
    # ------------------------------------------------------------------

    def get_report(self, company_name: str, fiscal_year: str) -> AnalysisResult | None:
        """Returns cached report result or ``None`` if not found or corrupted."""
        company_id = self.make_company_id(company_name, fiscal_year)
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT analysis_data FROM reports WHERE company_id = ?",
                    (company_id,),
                ).fetchone()

            if row is None:
                logger.info("Cache miss for report %s", company_id)
                return None

            result = AnalysisResult.model_validate_json(row[0])
            logger.info("Cache hit for report %s", company_id)
            return result
        except Exception:
            logger.exception("Error reading cached report %s", company_id)
            return None

    def save_report(self, result: AnalysisResult) -> None:
        """Upserts a report into the cache. Errors are logged but not raised."""
        company_name = result.metadata.company_name
        fiscal_year = result.metadata.fiscal_year
        company_id = self.make_company_id(company_name, fiscal_year)

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO reports (company_id, company_name, fiscal_year, analysis_data)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(company_id) DO UPDATE SET
                        company_name  = excluded.company_name,
                        fiscal_year   = excluded.fiscal_year,
                        analysis_data = excluded.analysis_data
                    """,
                    (company_id, company_name, fiscal_year, result.model_dump_json()),
                )
                conn.commit()
            logger.info("Cached report %s", company_id)
        except Exception:
            logger.exception("Failed to cache report %s", company_id)

    # ------------------------------------------------------------------
    # Chunk persistence
    # ------------------------------------------------------------------

    def save_chunks(self, company_id: str, chunks: list[DocumentChunk]) -> None:
        """Saves document chunks as JSONL. Errors are logged but not raised."""
        try:
            chunks_path = self.get_company_data_dir(company_id) / "chunks.jsonl"
            with open(chunks_path, "w", encoding="utf-8") as f:
                for chunk in chunks:
                    f.write(chunk.model_dump_json() + "\n")
            logger.info("Saved %d chunks to %s", len(chunks), chunks_path)
        except Exception:
            logger.exception("Failed to save chunks for %s", company_id)

    def load_chunks(self, company_id: str) -> list[DocumentChunk] | None:
        """Loads document chunks from JSONL, or ``None`` on failure."""
        try:
            chunks_path = self.get_company_data_dir(company_id) / "chunks.jsonl"
            if not chunks_path.exists():
                logger.info("Chunks file not found for %s", company_id)
                return None

            chunks = []
            with open(chunks_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        chunks.append(DocumentChunk.model_validate_json(line))
            logger.info("Loaded %d chunks for %s", len(chunks), company_id)
            return chunks
        except Exception:
            logger.exception("Failed to load chunks for %s", company_id)
            return None

    # ------------------------------------------------------------------
    # FAISS index persistence
    # ------------------------------------------------------------------

    def save_faiss_index(self, company_id: str, index: Any) -> None:
        """Saves FAISS index to disk. Errors are logged but not raised."""
        try:
            index_path = self.get_company_data_dir(company_id) / "faiss.index"
            faiss.write_index(index, str(index_path))
            logger.info("Saved FAISS index to %s", index_path)
        except Exception:
            logger.exception("Failed to save FAISS index for %s", company_id)

    def load_faiss_index(self, company_id: str) -> Any | None:
        """Loads FAISS index from disk, or ``None`` on failure."""
        try:
            index_path = self.get_company_data_dir(company_id) / "faiss.index"
            if not index_path.exists():
                logger.info("FAISS index not found for %s", company_id)
                return None

            index = faiss.read_index(str(index_path))
            logger.info("Loaded FAISS index for %s", company_id)
            return index
        except Exception:
            logger.exception("Failed to load FAISS index for %s", company_id)
            return None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def has_cached_analysis(self, company_name: str, fiscal_year: str | None = None) -> bool:
        """Quick check if a report exists in cache."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if fiscal_year:
                    company_id = self.make_company_id(company_name, fiscal_year)
                    row = conn.execute(
                        "SELECT 1 FROM reports WHERE company_id = ?", (company_id,)
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT 1 FROM reports WHERE company_name = ?", (company_name,)
                    ).fetchone()
                return row is not None
        except Exception:
            logger.exception("Error checking cache for %s", company_name)
            return False

    def list_cached_reports(self) -> list[ReportMetadata]:
        """Lists metadata for all cached reports."""
        reports: list[ReportMetadata] = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                for (data,) in conn.execute("SELECT analysis_data FROM reports"):
                    try:
                        reports.append(
                            AnalysisResult.model_validate_json(data).metadata
                        )
                    except Exception:
                        logger.warning("Skipping corrupted cache entry")
            logger.info("Listed %d cached reports", len(reports))
        except Exception:
            logger.exception("Error listing cached reports")
        return reports
