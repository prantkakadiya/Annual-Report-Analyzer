"""Annual Report Analyzer — Streamlit application.

Orchestrates the full pipeline: discovery → download → validation →
parsing → chunking/embedding → section splitting → extraction →
insight generation, with a RAG chat panel and export options.
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Logging — configure once at module load
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline imports
# ---------------------------------------------------------------------------
from pipeline.cache import ReportCache  # noqa: E402
from pipeline.discovery.browser import ReportDownloader  # noqa: E402
from pipeline.discovery.search import ReportSearcher  # noqa: E402
from pipeline.discovery.validate import ReportValidator, ValidationResult  # noqa: E402
from pipeline.extraction.extract import FinancialExtractor  # noqa: E402
from pipeline.extraction.insights import InsightGenerator  # noqa: E402
from pipeline.parsing.chunking import DocumentChunker  # noqa: E402
from pipeline.parsing.parse import ReportParser  # noqa: E402
from pipeline.parsing.sections import SectionSplitter  # noqa: E402
from pipeline.rag.rag import RAGEngine  # noqa: E402
from pipeline.schema import (  # noqa: E402
    AnalysisResult,
    ChatMessage,
    ReportMetadata,
)
from export.to_markdown import export_markdown  # noqa: E402
from export.to_pdf import export_pdf  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8080")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:8081")
DATA_DIR = "data"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Annual Report Analyzer",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_cache() -> ReportCache:
    """Return a singleton cache instance via session state."""
    if "cache" not in st.session_state:
        st.session_state.cache = ReportCache(data_dir=DATA_DIR)
    return st.session_state.cache


def _init_state() -> None:
    """Initialise session-state keys on first run."""
    defaults: dict = {
        "result": None,           # AnalysisResult | None
        "rag_engine": None,       # RAGEngine | None
        "chat_history": [],       # list[ChatMessage]
        "is_running": False,
        "candidates": None,       # list[SearchCandidate] | None
        "selected_candidate": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------
async def _run_pipeline(
    company_name: str | None = None,
    pdf_path: str | None = None,
) -> AnalysisResult:
    """Execute the full analysis pipeline with live progress updates.

    Args:
        company_name: Company to analyse.
        pdf_path: If provided, skip discovery and use this PDF directly.

    Returns:
        The completed ``AnalysisResult``.
    """
    cache = _get_cache()
    progress = st.progress(0, text="Initialising…")
    status_area = st.status("🔄 Running analysis pipeline…", expanded=True)

    step = 0
    total_steps = 8

    def _advance(label: str) -> None:
        nonlocal step
        step += 1
        pct = int(step / total_steps * 100)
        progress.progress(pct, text=label)
        status_area.write(f"**Step {step}/{total_steps}:** {label}")
        logger.info("Pipeline step %d/%d: %s", step, total_steps, label)

    fiscal_year: str = ""
    source_url: str | None = None

    # ── Step 1: Discovery / Upload ──────────────────────────────────
    if pdf_path is None:
        _advance("Searching for annual report…")
        try:
            searcher = ReportSearcher()
            candidates = await searcher.search(company_name)
            if not candidates:
                st.warning("No candidates found. Please upload a PDF manually.")
                raise RuntimeError("No search results")
        except Exception:
            logger.exception("Search failed")
            raise

        # ── Step 2: Download ────────────────────────────────────────
        _advance("Downloading report PDF…")
        downloader = ReportDownloader(download_dir=f"{DATA_DIR}/downloads")
        
        # Setup a progress bar for the download
        dl_progress = st.progress(0, text="Downloading PDF...")
        def update_dl_progress(downloaded: int, total_size: int):
            if total_size > 0:
                pct = min(downloaded / total_size, 1.0)
                dl_progress.progress(pct, text=f"Downloading PDF... {downloaded/1024/1024:.1f}MB / {total_size/1024/1024:.1f}MB")
            else:
                dl_progress.progress(0.5, text=f"Downloading PDF... {downloaded/1024/1024:.1f}MB")
                
        downloads = await downloader.download_from_candidates(
            candidates, company_name, progress_callback=update_dl_progress
        )
        dl_progress.empty()
        if not downloads:
            st.warning("Could not download any PDF. Please upload manually.")
            raise RuntimeError("Download failed")

        candidate, pdf_path = downloads[0]
        source_url = candidate.url
    else:
        _advance("Using uploaded PDF…")
        _advance("Skipping download (manual upload)")

    # ── Step 3: Validation ──────────────────────────────────────────
    _advance("Validating PDF…")
    validator = ReportValidator(llm_base_url=LLM_BASE_URL)
    val: ValidationResult = await validator.validate(pdf_path, company_name)
    if not val.is_valid:
        st.error(f"Validation failed: {val.reason}")
        logger.warning("Validation failed: %s", val.reason)
        raise RuntimeError(f"Validation failed: {val.reason}")
    fiscal_year = val.fiscal_year or "Unknown"
    company_name = company_name or val.detected_company or "Unknown Company"
    
    status_area.write(
        f"  ✅ Confirmed: **{company_name}** — FY {fiscal_year}"
    )

    # Check cache
    cached = cache.get_report(company_name, fiscal_year)
    if cached is not None:
        status_area.update(label="✅ Loaded from cache", state="complete")
        progress.progress(100, text="Done (cached)")
        st.toast("Loaded from cache!", icon="⚡")
        return cached

    # ── Step 4: Parsing ─────────────────────────────────────────────
    _advance("Parsing PDF to Markdown…")
    parser = ReportParser()
    parsed = parser.parse(pdf_path)
    status_area.write(f"  📄 {parsed.total_pages} pages extracted")

    # ── Step 5: Chunking & Embedding ────────────────────────────────
    _advance("Chunking & building embeddings…")
    chunker = DocumentChunker()
    chunks = chunker.chunk(parsed)

    company_id = cache.make_company_id(company_name, fiscal_year)

    rag = RAGEngine(
        embedding_base_url=EMBEDDING_BASE_URL,
        llm_base_url=LLM_BASE_URL,
        cache=cache,
    )
    await rag.build_index(chunks, company_id)
    st.session_state.rag_engine = rag
    status_area.write(f"  🔍 {len(chunks)} chunks indexed")

    # ── Step 6: Section splitting ───────────────────────────────────
    _advance("Splitting into sections…")
    splitter = SectionSplitter()
    sections = await splitter.split(parsed, rag_index=rag)
    sections_found = {name: (name in sections) for name in [
        "MD&A", "Directors' Report", "Balance Sheet",
        "Statement of Profit & Loss", "Cash Flow Statement",
        "Notes to Accounts", "Corporate Governance Report",
        "Auditor's Report",
    ]}
    found_count = sum(sections_found.values())
    status_area.write(
        f"  📑 {found_count}/{len(sections_found)} standard sections found"
    )

    # ── Step 7: Financial extraction ────────────────────────────────
    _advance("Extracting financial metrics…")
    extractor = FinancialExtractor(llm_base_url=LLM_BASE_URL)
    financial_data = await extractor.extract(sections, company_name, fiscal_year)
    status_area.write(
        f"  💰 {len(financial_data.metrics)} metrics extracted"
    )

    # ── Step 8: Insight generation ──────────────────────────────────
    _advance("Generating business insights…")
    insight_gen = InsightGenerator(llm_base_url=LLM_BASE_URL)
    insights = await insight_gen.generate(sections, company_name, fiscal_year)
    status_area.write(
        f"  💡 {len(insights.insights)} insights generated"
    )

    # ── Assemble result ─────────────────────────────────────────────
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

    # Persist
    cache.save_report(result)

    status_area.update(label="✅ Analysis complete!", state="complete")
    progress.progress(100, text="Done!")
    return result


# ---------------------------------------------------------------------------
# UI — Sidebar
# ---------------------------------------------------------------------------
def _render_sidebar() -> tuple[str, bool, bytes | None]:
    """Render sidebar controls; return (company_name, run_clicked, upload_bytes)."""
    with st.sidebar:
        st.header("📊 Annual Report Analyzer")
        st.caption("v1.0.0 · Powered by local LLMs")
        st.divider()

        company = st.text_input(
            "Company name",
            placeholder="e.g. Infosys, TCS, Reliance…",
        )

        col1, col2 = st.columns(2)
        with col1:
            run = st.button(
                "🔍 Analyze",
                disabled=st.session_state.is_running,
                use_container_width=True,
            )
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                for k in ["result", "rag_engine", "chat_history", "candidates"]:
                    st.session_state[k] = None if k != "chat_history" else []
                st.rerun()

        st.divider()
        st.subheader("📎 Manual Upload")
        uploaded = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            help="Upload an annual report PDF directly.",
        )

        # Cached reports
        st.divider()
        st.subheader("📁 Previous Reports")
        cache = _get_cache()
        try:
            cached_reports = cache.list_cached_reports()
        except Exception:
            cached_reports = []

        if cached_reports:
            for meta in cached_reports:
                label = f"{meta.company_name} ({meta.fiscal_year})"
                if st.button(label, key=f"cache_{meta.company_name}_{meta.fiscal_year}"):
                    result = cache.get_report(meta.company_name, meta.fiscal_year)
                    if result:
                        st.session_state.result = result
                        # Reload RAG index
                        rag = RAGEngine(
                            embedding_base_url=EMBEDDING_BASE_URL,
                            llm_base_url=LLM_BASE_URL,
                            cache=cache,
                        )
                        cid = cache.make_company_id(meta.company_name, meta.fiscal_year)
                        if rag.load_index(cid):
                            st.session_state.rag_engine = rag
                        st.rerun()
        else:
            st.caption("No cached reports yet.")

        # Export
        if st.session_state.result is not None:
            st.divider()
            st.subheader("📤 Export")
            res: AnalysisResult = st.session_state.result
            md_text = export_markdown(res)
            st.download_button(
                "📄 Download Markdown",
                data=md_text,
                file_name=f"{res.metadata.company_name}_{res.metadata.fiscal_year}.md",
                mime="text/markdown",
                use_container_width=True,
            )
            if st.button("📑 Generate PDF", use_container_width=True):
                try:
                    out_dir = Path(DATA_DIR) / "exports"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    pdf_file = str(
                        out_dir / f"{res.metadata.company_name}_{res.metadata.fiscal_year}.pdf"
                    )
                    export_pdf(res, pdf_file)
                    with open(pdf_file, "rb") as f:
                        st.download_button(
                            "⬇️ Download PDF",
                            data=f,
                            file_name=Path(pdf_file).name,
                            mime="application/pdf",
                            use_container_width=True,
                        )
                except Exception as exc:
                    logger.exception("PDF export failed")
                    st.error(f"PDF export failed: {exc}")

        upload_bytes = uploaded.read() if uploaded else None
        return company, run, upload_bytes


# ---------------------------------------------------------------------------
# UI — Results tabs
# ---------------------------------------------------------------------------
def _render_results(result: AnalysisResult) -> None:
    """Render the four result tabs."""
    meta = result.metadata
    st.header(f"{meta.company_name} — {meta.fiscal_year}")

    tab_fin, tab_ins, tab_chat, tab_info = st.tabs(
        ["💰 Financial Metrics", "💡 Insights", "💬 Report Chat", "ℹ️ Report Info"]
    )

    # ── Tab 1: Financials ───────────────────────────────────────────
    with tab_fin:
        if result.financial_data and result.financial_data.metrics:
            import pandas as pd

            rows = []
            for m in result.financial_data.metrics:
                rows.append({
                    "Metric": m.name,
                    "CY Consolidated": m.current_year_consolidated,
                    "CY Standalone": m.current_year_standalone,
                    "PY Consolidated": m.previous_year_consolidated,
                    "PY Standalone": m.previous_year_standalone,
                    "Unit": m.unit,
                    "Page": f"[p.{m.page_number}]" if m.page_number else "—",
                    "Notes": m.notes or "",
                })
            df = pd.DataFrame(rows)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "CY Consolidated": st.column_config.NumberColumn(format=",.2f"),
                    "CY Standalone": st.column_config.NumberColumn(format=",.2f"),
                    "PY Consolidated": st.column_config.NumberColumn(format=",.2f"),
                    "PY Standalone": st.column_config.NumberColumn(format=",.2f"),
                },
            )
        else:
            st.info("No financial metrics were extracted.")

    # ── Tab 2: Insights ─────────────────────────────────────────────
    with tab_ins:
        if result.insights and result.insights.insights:
            by_cat: dict[str, list[str]] = {}
            for ins in result.insights.insights:
                by_cat.setdefault(ins.category, []).append(ins.content)

            for cat, items in by_cat.items():
                with st.expander(f"**{cat}**", expanded=True):
                    for item in items:
                        st.markdown(f"- {item}")
        else:
            st.info("No insights were generated.")

    # ── Tab 3: RAG Chat ─────────────────────────────────────────────
    with tab_chat:
        _render_chat()

    # ── Tab 4: Report Info ──────────────────────────────────────────
    with tab_info:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Metadata")
            st.markdown(f"- **Company:** {meta.company_name}")
            st.markdown(f"- **Fiscal Year:** {meta.fiscal_year}")
            st.markdown(f"- **Source:** {meta.source_url or 'Manual upload'}")
            st.markdown(f"- **Pages:** {meta.total_pages or 'N/A'}")
            st.markdown(f"- **Analysed:** {meta.discovered_at.strftime('%Y-%m-%d %H:%M')}")
            st.markdown(f"- **Version:** {result.analysis_version}")

        with col_b:
            st.subheader("Sections Detected")
            for sec, found in result.sections_found.items():
                icon = "✅" if found else "❌"
                st.markdown(f"- {icon} {sec}")


def _render_chat() -> None:
    """RAG chat panel."""
    rag: RAGEngine | None = st.session_state.rag_engine

    if rag is None or rag.index is None:
        st.warning(
            "RAG index not available. Run an analysis first or load a cached report."
        )
        return

    # Display history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg.role):
            st.markdown(msg.content)

    # Input
    if question := st.chat_input("Ask a question about the report…"):
        # Show user message
        user_msg = ChatMessage(
            role="user",
            content=question,
            timestamp=datetime.now(),
        )
        st.session_state.chat_history.append(user_msg)
        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching report…"):
                try:
                    answer, cited_pages = asyncio.run(
                        rag.answer_question(
                            question,
                            chat_history=st.session_state.chat_history,
                        )
                    )
                except Exception as exc:
                    logger.exception("RAG answer failed")
                    answer = f"⚠️ Error generating answer: {exc}"
                    cited_pages = []

            st.markdown(answer)

        asst_msg = ChatMessage(
            role="assistant",
            content=answer,
            citations=cited_pages,
            timestamp=datetime.now(),
        )
        st.session_state.chat_history.append(asst_msg)

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat"):
            st.session_state.chat_history = []
            st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
from contextlib import contextmanager
import subprocess
import time

@contextmanager
def manage_embedding_server():
    """Starts the embedding server and guarantees it stops when done."""
    emb_server_process = subprocess.Popen(
        [os.path.abspath("llm_server/start_embedding_server.sh")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=os.path.abspath("llm_server")
    )
    # Wait for the model to load
    time.sleep(5)
    try:
        yield
    finally:
        emb_server_process.terminate()
        try:
            emb_server_process.wait(2)
        except subprocess.TimeoutExpired:
            emb_server_process.kill()

def main() -> None:
    _init_state()

    company_name, run_clicked, upload_bytes = _render_sidebar()

    # ── Handle manual upload ────────────────────────────────────────
    if upload_bytes and (run_clicked or st.session_state.get("_last_upload") != hash(upload_bytes)):
        st.session_state["_last_upload"] = hash(upload_bytes)
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".pdf", dir=f"{DATA_DIR}/downloads"
        )
        Path(f"{DATA_DIR}/downloads").mkdir(parents=True, exist_ok=True)
        tmp.write(upload_bytes)
        tmp.flush()
        tmp_path = tmp.name
        tmp.close()

        st.session_state.is_running = True
        try:
            with manage_embedding_server():
                result = asyncio.run(
                    _run_pipeline(company_name or None, pdf_path=tmp_path)
                )
            st.session_state.result = result
        except Exception as exc:
            logger.exception("Pipeline failed (upload)")
            st.error(f"Analysis failed: {exc}")
        finally:
            st.session_state.is_running = False

    # ── Handle search-based analysis ────────────────────────────────
    elif run_clicked and company_name:
        st.session_state.is_running = True
        try:
            with manage_embedding_server():
                result = asyncio.run(
                    _run_pipeline(company_name)
                )
            st.session_state.result = result
        except Exception as exc:
            logger.exception("Pipeline failed (search)")
            st.error(f"Analysis failed: {exc}")
        finally:
            st.session_state.is_running = False

    elif run_clicked and not company_name:
        st.warning("Please enter a company name.")

    # ── Render results ──────────────────────────────────────────────
    if st.session_state.result is not None:
        _render_results(st.session_state.result)
    else:
        # Landing page
        st.title("📊 Annual Report Analyzer")
        st.markdown(
            """
            Enter a company name in the sidebar and click **Analyze** to
            automatically find, download, and analyze the company's latest
            annual report.

            Alternatively, **upload a PDF** directly.

            All analysis runs on **local LLMs** — no data leaves your machine.
            """
        )
        st.info(
            "💡 Make sure both the LLM server (port 8080) and embedding "
            "server (port 8081) are running before starting an analysis."
        )


if __name__ == "__main__":
    main()
