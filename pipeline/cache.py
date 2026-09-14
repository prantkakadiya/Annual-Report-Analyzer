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
    """Handles caching of reports, metadata, document chunks, and FAISS indices."""
    
    def __init__(self, data_dir: str = "data") -> None:
        """Initializes SQLite cache and creates tables if they do not exist."""
        try:
            self.data_dir = Path(data_dir).resolve()
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = self.data_dir / "cache.db"
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    CREATE TABLE IF NOT EXISTS reports (
                        company_id TEXT PRIMARY KEY,
                        company_name TEXT,
                        fiscal_year TEXT,
                        analysis_data TEXT
                    )
                    '''
                )
                conn.commit()
            logger.info(f"Initialized ReportCache at {self.data_dir}")
        except Exception as e:
            logger.exception(f"Failed to initialize ReportCache: {e}")
            raise
    
    def make_company_id(self, company_name: str, fiscal_year: str) -> str:
        """Creates a slugified company ID."""
        try:
            base = f"{company_name}_{fiscal_year}"
            slug = re.sub(r'[^a-z0-9]+', '_', base.lower()).strip('_')
            return slug
        except Exception as e:
            logger.exception("Error making company_id")
            raise

    def get_company_data_dir(self, company_id: str) -> Path:
        """Returns the data directory path for a given company, creating it if needed."""
        try:
            path = self.data_dir / company_id
            path.mkdir(parents=True, exist_ok=True)
            return path
        except Exception as e:
            logger.exception(f"Error creating company data dir for {company_id}")
            raise

    def get_report(self, company_name: str, fiscal_year: str) -> AnalysisResult | None:
        """Returns cached report result or None if not found or corrupted."""
        company_id = self.make_company_id(company_name, fiscal_year)
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT analysis_data FROM reports WHERE company_id = ?", 
                    (company_id,)
                )
                row = cursor.fetchone()
            
            if row:
                try:
                    result = AnalysisResult.model_validate_json(row[0])
                    logger.info(f"Cache hit for report {company_id}")
                    return result
                except Exception as parse_e:
                    logger.exception(f"Corrupted cache data for {company_id}: {parse_e}")
                    return None
            else:
                logger.info(f"Cache miss for report {company_id}")
                return None
        except Exception as e:
            logger.exception(f"Error retrieving report {company_id} from cache: {e}")
            return None

    def save_report(self, result: AnalysisResult) -> None:
        """Upserts a report into the cache."""
        company_name = result.metadata.company_name
        fiscal_year = result.metadata.fiscal_year
        company_id = self.make_company_id(company_name, fiscal_year)
        
        try:
            analysis_data = result.model_dump_json()
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    INSERT INTO reports (company_id, company_name, fiscal_year, analysis_data)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(company_id) DO UPDATE SET
                    company_name = excluded.company_name,
                    fiscal_year = excluded.fiscal_year,
                    analysis_data = excluded.analysis_data
                    ''',
                    (company_id, company_name, fiscal_year, analysis_data)
                )
                conn.commit()
            logger.info(f"Successfully cached report {company_id}")
        except Exception as e:
            logger.exception(f"Failed to save report {company_id} to cache: {e}")

    def save_chunks(self, company_id: str, chunks: list[DocumentChunk]) -> None:
        """Saves document chunks as JSONL."""
        try:
            company_dir = self.get_company_data_dir(company_id)
            chunks_path = company_dir / "chunks.jsonl"
            
            with open(chunks_path, 'w', encoding='utf-8') as f:
                for chunk in chunks:
                    f.write(chunk.model_dump_json() + '\n')
            logger.info(f"Saved {len(chunks)} chunks to {chunks_path}")
        except Exception as e:
            logger.exception(f"Failed to save chunks for {company_id}: {e}")

    def load_chunks(self, company_id: str) -> list[DocumentChunk] | None:
        """Loads document chunks from JSONL."""
        try:
            company_dir = self.get_company_data_dir(company_id)
            chunks_path = company_dir / "chunks.jsonl"
            
            if not chunks_path.exists():
                logger.info(f"Chunks file not found for {company_id}")
                return None
                
            chunks = []
            with open(chunks_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        chunks.append(DocumentChunk.model_validate_json(line))
            
            logger.info(f"Loaded {len(chunks)} chunks for {company_id}")
            return chunks
        except Exception as e:
            logger.exception(f"Failed to load chunks for {company_id}: {e}")
            return None

    def save_faiss_index(self, company_id: str, index: Any) -> None:
        """Saves FAISS index to disk."""
        try:
            company_dir = self.get_company_data_dir(company_id)
            index_path = company_dir / "faiss.index"
            faiss.write_index(index, str(index_path))
            logger.info(f"Saved FAISS index to {index_path}")
        except Exception as e:
            logger.exception(f"Failed to save FAISS index for {company_id}: {e}")

    def load_faiss_index(self, company_id: str) -> Any | None:
        """Loads FAISS index from disk."""
        try:
            company_dir = self.get_company_data_dir(company_id)
            index_path = company_dir / "faiss.index"
            
            if not index_path.exists():
                logger.info(f"FAISS index not found for {company_id}")
                return None
                
            index = faiss.read_index(str(index_path))
            logger.info(f"Loaded FAISS index for {company_id}")
            return index
        except Exception as e:
            logger.exception(f"Failed to load FAISS index for {company_id}: {e}")
            return None

    def has_cached_analysis(self, company_name: str, fiscal_year: str | None = None) -> bool:
        """Quick check if a report exists in cache."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if fiscal_year:
                    company_id = self.make_company_id(company_name, fiscal_year)
                    cursor.execute("SELECT 1 FROM reports WHERE company_id = ?", (company_id,))
                else:
                    cursor.execute("SELECT 1 FROM reports WHERE company_name = ?", (company_name,))
                
                return cursor.fetchone() is not None
        except Exception as e:
            logger.exception(f"Error checking cached analysis for {company_name}: {e}")
            return False

    def list_cached_reports(self) -> list[ReportMetadata]:
        """Lists metadata for all cached reports."""
        reports = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT analysis_data FROM reports")
                
                for row in cursor.fetchall():
                    try:
                        result = AnalysisResult.model_validate_json(row[0])
                        reports.append(result.metadata)
                    except Exception as parse_e:
                        logger.error(f"Failed to parse report metadata: {parse_e}")
            
            logger.info(f"Listed {len(reports)} cached reports")
            return reports
        except Exception as e:
            logger.exception(f"Error listing cached reports: {e}")
            return []
