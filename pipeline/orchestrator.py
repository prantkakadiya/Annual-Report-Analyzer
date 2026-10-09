"""Framework-agnostic pipeline orchestrator.

Coordinates the full analysis pipeline (discovery → download → validation →
parsing → chunking/embedding → section splitting → extraction → insights)
without any dependency on Streamlit or other UI frameworks.

Progress is reported via an optional :class:`PipelineCallbacks` dataclass,
so any frontend can plug in its own reporting widgets.
"""
from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from pipeline.cache import ReportCache
from pipeline.config import DATA_DIR, EMBEDDING_BASE_URL, LLM_BASE_URL
from pipeline.model_manager import model_manager
from pipeline.schema import AnalysisResult, ReportMetadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Callback protocol
# ---------------------------------------------------------------------------

@dataclass
class PipelineCallbacks:
    """Optional callbacks for reporting pipeline progress.

    Every callback is optional — ``None`` means "no reporting".
    """

    on_step: Callable[[int, int, str], None] | None = None
    """Called at each major step: ``(step_number, total_steps, label)``."""

    on_status: Callable[[str], None] | None = None
    """Called for informational status messages."""

    on_download_progress: Callable[[int, int], None] | None = None
    """Called during PDF download: ``(bytes_downloaded, total_bytes)``."""

    on_embedding_progress: Callable[[int, int], None] | None = None
    """Called during embedding: ``(chunks_processed, total_chunks)``."""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _run_parsing_subprocess(pdf_path: str) -> str:
    """Runs Docling parsing in an isolated process to prevent PyTorch from permanently reserving RAM."""
    from pipeline.parsing.parse import ReportParser
    parser = ReportParser()
    parsed = parser.parse(pdf_path)
    return parsed.model_dump_json()


class PipelineOrchestrator:
    """Runs the full analysis pipeline independent of any UI framework.

    Args:
        cache: Persistent report / index cache.
        data_dir: Root directory for downloads and data files.
    """

    def __init__(
        self,
        cache: ReportCache,
        data_dir: str = DATA_DIR,
    ) -> None:
        self.cache = cache
        self.data_dir = data_dir

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        company_name: str | None = None,
        pdf_path: str | None = None,
        callbacks: PipelineCallbacks | None = None,
    ) -> tuple[AnalysisResult, "RAGIndex"]:
        """Execute the full analysis pipeline.

        Args:
            company_name: Company to analyse (used for search if *pdf_path*
                is not given).
            pdf_path: Skip discovery and use this PDF directly.
            callbacks: Optional progress callbacks.

        Returns:
            A ``(result, rag_index)`` tuple so the caller can use the
            index for follow-up chat.
        """
        # Lazy imports to avoid circular deps and heavy module loads at import time
        from pipeline.discovery.browser import ReportDownloader
        from pipeline.discovery.search import ReportSearcher
        from pipeline.discovery.validate import ReportValidator
        from pipeline.extraction.extract import FinancialExtractor
        from pipeline.extraction.insights import InsightGenerator
        from pipeline.parsing.chunking import DocumentChunker
        from pipeline.parsing.parse import ReportParser
        from pipeline.parsing.sections import SectionSplitter
        from pipeline.rag.index import RAGIndex

        cb = callbacks or PipelineCallbacks()

        step = 0
        total_steps = 8

        def _advance(label: str) -> None:
            nonlocal step
            step += 1
            logger.info("Pipeline step %d/%d: %s", step, total_steps, label)
            if cb.on_step:
                cb.on_step(step, total_steps, label)

        source_url: str | None = None

        # ── Load LLM for discovery & validation ────────────────────
        if cb.on_status:
            cb.on_status("⚙️ Loading LLM into VRAM...")
        model_manager.require_llm()

        # ── Discovery & Download ───────────────────────────────────
        if pdf_path is None:
            _advance("Searching for annual report…")
            searcher = ReportSearcher(llm_base_url=LLM_BASE_URL)
            candidates = await searcher.search(company_name)
            if not candidates:
                raise RuntimeError("No search results")

            _advance("Downloading report PDF…")
            downloader = ReportDownloader(
                download_dir=f"{self.data_dir}/downloads"
            )
            downloads = await downloader.download_from_candidates(
                candidates,
                company_name,
                progress_callback=cb.on_download_progress,
            )
            if not downloads:
                raise RuntimeError("Download failed")

            candidate, pdf_path = downloads[0]
            source_url = candidate.url
        else:
            _advance("Using uploaded PDF…")
            _advance("Skipping download (manual upload)")

        # ── Validation ─────────────────────────────────────────────
        _advance("Validating PDF…")
        validator = ReportValidator(llm_base_url=LLM_BASE_URL)
        val = await validator.validate(pdf_path, company_name)
        if not val.is_valid:
            raise RuntimeError(f"Validation failed: {val.reason}")
        fiscal_year = val.fiscal_year or "Unknown"
        company_name = company_name or val.detected_company or "Unknown Company"
        if cb.on_status:
            cb.on_status(f"✅ Confirmed: {company_name} — FY {fiscal_year}")

        # ── Cache check ────────────────────────────────────────────
        cached = self.cache.get_report(company_name, fiscal_year)
        if cached is not None:
            logger.info("Loaded from cache")
            # Still need a RAGIndex for chat
            rag_index = RAGIndex(
                embedding_base_url=EMBEDDING_BASE_URL,
                cache=self.cache,
            )
            cid = self.cache.make_company_id(company_name, fiscal_year)
            rag_index.load_index(cid)
            return cached, rag_index

        # ── Parsing ────────────────────────────────────────────────
        _advance("Parsing PDF to Markdown…")
        if cb.on_status:
            cb.on_status("⚙️ Unloading LLM to free VRAM for Docling OCR...")
        model_manager.stop_all()
        # Run parsing in an isolated subprocess so PyTorch RAM is 100% destroyed on exit
        if cb.on_status:
            cb.on_status("⚙️ Booting isolated parsing process to protect RAM...")
            
        import concurrent.futures
        import multiprocessing as mp
        from pipeline.parsing.parse import ParsedReport
        
        # We must use 'spawn' to ensure no memory is inherited from parent
        ctx = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx) as executor:
            future = executor.submit(_run_parsing_subprocess, pdf_path)
            parsed_json = future.result()
            
        parsed = ParsedReport.model_validate_json(parsed_json)

        if cb.on_status:
            cb.on_status(f"📄 {parsed.total_pages} pages extracted")

        # ── Chunking & Embedding ───────────────────────────────────
        _advance("Chunking & building embeddings…")
        if cb.on_status:
            cb.on_status("⚙️ Swapping LLM for Embedding Model in VRAM...")
        model_manager.require_embedding()

        chunker = DocumentChunker()
        chunks = chunker.chunk(parsed)
        company_id = self.cache.make_company_id(company_name, fiscal_year)

        rag_index = RAGIndex(
            embedding_base_url=EMBEDDING_BASE_URL,
            cache=self.cache,
        )
        await rag_index.build_index(
            chunks, company_id, progress_callback=cb.on_embedding_progress
        )
        if cb.on_status:
            cb.on_status(f"🔍 {len(chunks)} chunks indexed")

        # ── Section splitting ──────────────────────────────────────
        _advance("Splitting into sections…")
        splitter = SectionSplitter()
        sections = await splitter.split(parsed, rag_index=rag_index)
        standard_names = [
            "MD&A",
            "Directors' Report",
            "Risk Management",
            "Outlook & Guidance",
            "Chairman/CEO Letter",
            "Balance Sheet",
            "Statement of Profit & Loss",
            "Cash Flow Statement",
            "Notes to Accounts",
            "Corporate Governance Report"
        ]
        sections_found = {name: (name in sections) for name in standard_names}
        
        if cb.on_status:
            cb.on_status(f"📑 {sum(sections_found.values())}/{len(sections_found)} standard sections found")
            cb.on_status("\n--- SECTION SUMMARY ---")
            for name, found in sections_found.items():
                if found:
                    method = sections[name].found_by if hasattr(sections[name], 'found_by') else "regex"
                    cb.on_status(f"✅ FOUND: {name} (via {method})")
                else:
                    cb.on_status(f"❌ MISSING: {name}")
            cb.on_status("-----------------------")

        # ── Financial extraction ───────────────────────────────────
        _advance("Extracting financial metrics…")
        if cb.on_status:
            cb.on_status("⚙️ Swapping Embedding Model for LLM in VRAM...")
        model_manager.require_llm()
        extractor = FinancialExtractor(llm_base_url=LLM_BASE_URL)

        # Route verbose extraction logs to the background logger, not the UI
        extraction_logger = logger.info
        financial_data = await extractor.extract(
            sections, company_name, fiscal_year, log_callback=extraction_logger
        )
        if cb.on_status:
            cb.on_status(f"💰 {len(financial_data.metrics)} metrics extracted")

        # ── Insight generation ─────────────────────────────────────
        _advance("Generating business insights…")
        insight_gen = InsightGenerator(llm_base_url=LLM_BASE_URL)
        insights = await insight_gen.generate(
            sections, company_name, fiscal_year, financial_data
        )
        if cb.on_status:
            cb.on_status(f"💡 {len(insights.insights)} insights generated")

        # ── Assemble result ────────────────────────────────────────
        result = AnalysisResult(
            metadata=ReportMetadata(
                company_name=company_name,
                fiscal_year=fiscal_year,
                pdf_path=pdf_path,
                source_url=source_url,
                total_pages=parsed.total_pages,
                discovered_at=datetime.now(),
                validated=True,
            ),
            financial_data=financial_data,
            insights=insights,
            sections_found=sections_found,
        )

        self.cache.save_report(result)
        logger.info("Pipeline complete for %s FY %s", company_name, fiscal_year)
        return result, rag_index
