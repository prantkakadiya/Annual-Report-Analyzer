"""Pydantic models for extracted data."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Literal

from pydantic import BaseModel

logger = logging.getLogger(__name__)

class FinancialMetric(BaseModel):
    """A single financial metric with current/previous year, consolidated/standalone, and citation."""
    name: str
    current_year_consolidated: float | None = None
    current_year_standalone: float | None = None
    previous_year_consolidated: float | None = None
    previous_year_standalone: float | None = None
    unit: str = "INR Crore"
    page_number: int | None = None
    notes: str | None = None

class FinancialData(BaseModel):
    """All extracted financial metrics for a report."""
    company_name: str
    fiscal_year: str
    metrics: list[FinancialMetric]
    extraction_timestamp: datetime

class Insight(BaseModel):
    """A single business insight with citation."""
    category: str
    content: str
    page_numbers: list[int]

class InsightData(BaseModel):
    """All extracted insights for a report."""
    company_name: str
    fiscal_year: str
    insights: list[Insight]
    extraction_timestamp: datetime

class ReportMetadata(BaseModel):
    """Metadata about a discovered/uploaded report."""
    company_name: str
    fiscal_year: str
    pdf_path: str
    source_url: str | None = None
    total_pages: int | None = None
    discovered_at: datetime
    validated: bool = False

class SearchCandidate(BaseModel):
    """A candidate URL from web search."""
    url: str
    title: str
    snippet: str | None = None
    score: float = 0.0
    is_direct_pdf: bool = False
    content_type: str | None = None
    content_length: int | None = None

class DocumentChunk(BaseModel):
    """A chunk of parsed document for RAG."""
    chunk_id: str
    text: str
    page_numbers: list[int]
    section: str | None = None
    embedding: list[float] | None = None

class ChatMessage(BaseModel):
    """A message in the RAG chat."""
    role: Literal["user", "assistant"]
    content: str
    citations: list[int] | None = None
    stats: dict[str, float] | None = None
    timestamp: datetime

class AnalysisResult(BaseModel):
    """Complete analysis result for a report - the top-level container."""
    metadata: ReportMetadata
    financial_data: FinancialData | None = None
    insights: InsightData | None = None
    sections_found: dict[str, bool] = {}
    chat_history: list[ChatMessage] = []
    analysis_version: str = "1.0.0"
