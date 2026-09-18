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
from pipeline.model_manager import model_manager  # noqa: E402
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
async def _step_discovery(company_name: str) -> list:
    searcher = ReportSearcher(llm_base_url=LLM_BASE_URL)
    candidates = await searcher.search(company_name)
    if not candidates:
        st.warning("No candidates found. Please upload a PDF manually.")
        raise RuntimeError("No search results")
    return candidates

async def _step_download(candidates: list, company_name: str) -> tuple[str, str]:
    downloader = ReportDownloader(download_dir=f"{DATA_DIR}/downloads")
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
    return candidate.url, pdf_path

async def _step_validation(pdf_path: str, company_name: str | None) -> tuple[str, str]:
    validator = ReportValidator(llm_base_url=LLM_BASE_URL)
    val = await validator.validate(pdf_path, company_name)
    if not val.is_valid:
        st.error(f"Validation failed: {val.reason}")
        logger.warning("Validation failed: %s", val.reason)
        raise RuntimeError(f"Validation failed: {val.reason}")
    fiscal_year = val.fiscal_year or "Unknown"
    final_company = company_name or val.detected_company or "Unknown Company"
    return final_company, fiscal_year

async def _step_rag_indexing(parsed_report, company_name: str, fiscal_year: str, cache):
    chunker = DocumentChunker()
    chunks = chunker.chunk(parsed_report)
    company_id = cache.make_company_id(company_name, fiscal_year)
    rag = RAGEngine(
        embedding_base_url=EMBEDDING_BASE_URL,
        llm_base_url=LLM_BASE_URL,
        cache=cache,
    )
    
    emb_progress = st.progress(0, text=f"Embedding chunks... 0/{len(chunks)}")
    def update_emb_progress(processed: int, total: int):
        if total > 0:
            pct = min(processed / total, 1.0)
            emb_progress.progress(pct, text=f"Embedding chunks... {processed}/{total}")
            
    await rag.build_index(chunks, company_id, progress_callback=update_emb_progress)
    emb_progress.empty()
    return rag, chunks

async def _run_pipeline(
    company_name: str | None = None,
    pdf_path: str | None = None,
) -> AnalysisResult:
    """Execute the full analysis pipeline with live progress updates."""
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

    source_url: str | None = None

    # Load LLM for discovery & validation
    status_area.write("  ⚙️ Loading LLM into VRAM...")
    model_manager.require_llm()

    # ── Discovery & Download
    if pdf_path is None:
        _advance("Searching for annual report…")
        candidates = await _step_discovery(company_name)
        
        _advance("Downloading report PDF…")
        source_url, pdf_path = await _step_download(candidates, company_name)
    else:
        _advance("Using uploaded PDF…")
        _advance("Skipping download (manual upload)")

    # ── Validation
    _advance("Validating PDF…")
    company_name, fiscal_year = await _step_validation(pdf_path, company_name)
    status_area.write(f"  ✅ Confirmed: **{company_name}** — FY {fiscal_year}")

    # Check cache
    cached = cache.get_report(company_name, fiscal_year)
    if cached is not None:
        status_area.update(label="✅ Loaded from cache", state="complete")
        progress.progress(100, text="Done (cached)")
        st.toast("Loaded from cache!", icon="⚡")
        return cached

    # ── Parsing
    _advance("Parsing PDF to Markdown…")
    status_area.write("  ⚙️ Unloading LLM to free VRAM for Docling OCR...")
    model_manager.stop_all()
    parser = ReportParser()
    parsed = parser.parse(pdf_path)
    
    # Explicitly clear Docling/PyTorch from VRAM so the Embedding server can load
    status_area.write("  ⚙️ Clearing Docling from VRAM...")
    del parser
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
        
    status_area.write(f"  📄 {parsed.total_pages} pages extracted")

    # ── Chunking & Embedding
    _advance("Chunking & building embeddings…")
    status_area.write("  ⚙️ Swapping LLM for Embedding Model in VRAM...")
    model_manager.require_embedding()
    rag, chunks = await _step_rag_indexing(parsed, company_name, fiscal_year, cache)
    st.session_state.rag_engine = rag
    status_area.write(f"  🔍 {len(chunks)} chunks indexed")

    # ── Section splitting
    _advance("Splitting into sections…")
    splitter = SectionSplitter()
    sections = await splitter.split(parsed, rag_index=rag)
    sections_found = {name: (name in sections) for name in [
        "MD&A", "Directors' Report", "Balance Sheet",
        "Statement of Profit & Loss", "Cash Flow Statement",
        "Notes to Accounts", "Corporate Governance Report",
        "Auditor's Report",
    ]}
    status_area.write(f"  📑 {sum(sections_found.values())}/{len(sections_found)} standard sections found")

    # ── Financial extraction
    _advance("Extracting financial metrics…")
    status_area.write("  ⚙️ Swapping Embedding Model for LLM in VRAM...")
    model_manager.require_llm()
    extractor = FinancialExtractor(llm_base_url=LLM_BASE_URL)
    financial_data = await extractor.extract(sections, company_name, fiscal_year)
    status_area.write(f"  💰 {len(financial_data.metrics)} metrics extracted")

    # ── Insight generation
    _advance("Generating business insights…")
    insight_gen = InsightGenerator(llm_base_url=LLM_BASE_URL)
    insights = await insight_gen.generate(sections, company_name, fiscal_year)
    status_area.write(f"  💡 {len(insights.insights)} insights generated")

    # ── Assemble result
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
def _render_financials_tab(result: AnalysisResult) -> None:
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
        st.dataframe(
            pd.DataFrame(rows),
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

def _render_insights_tab(result: AnalysisResult) -> None:
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

def _render_info_tab(result: AnalysisResult) -> None:
    meta = result.metadata
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

def _render_results(result: AnalysisResult) -> None:
    """Render the four result tabs."""
    meta = result.metadata
    st.header(f"{meta.company_name} — {meta.fiscal_year}")

    tab_fin, tab_ins, tab_chat, tab_info = st.tabs(
        ["💰 Financial Metrics", "💡 Insights", "💬 Report Chat", "ℹ️ Report Info"]
    )

    with tab_fin:
        _render_financials_tab(result)
    with tab_ins:
        _render_insights_tab(result)
    with tab_chat:
        _render_chat()
    with tab_info:
        _render_info_tab(result)

def _render_chat() -> None:
    """RAG chat panel."""
    rag: RAGEngine | None = st.session_state.rag_engine

    if rag is None or rag.index is None:
        st.warning(
            "RAG index not available. Run an analysis first or load a cached report."
        )
        return

    # Clear chat button at the top
    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat", key="clear_chat_btn"):
            st.session_state.chat_history = []
            st.rerun()

    # Container to hold all messages above the input
    chat_container = st.container()

    # Input (always anchors to bottom naturally if below the container)
    question = st.chat_input("Ask a question about the report…")

    if question:
        user_msg = ChatMessage(
            role="user",
            content=question,
            timestamp=datetime.now(),
        )
        st.session_state.chat_history.append(user_msg)

    with chat_container:
        # 1. Render all history (which now includes the new user message if there is one)
        for msg in st.session_state.chat_history:
            with st.chat_message(msg.role):
                if getattr(msg, "tool_logs", None):
                    with st.status("🤖 Analysis Details", expanded=False):
                        for log_str in msg.tool_logs:
                            st.markdown(log_str)
                st.markdown(msg.content)
                if getattr(msg, "stats", None):
                    st.caption(
                        f"⏱️ {msg.stats['total_time']:.1f}s | "
                        f"📥 In: {msg.stats['input_tokens']} | "
                        f"📤 Out: {msg.stats['output_tokens']} | "
                        f"⚡ {msg.stats['speed']:.1f} t/s"
                    )

        # 2. If there's a new question, generate the assistant's response below the history
        if question:
            with st.chat_message("assistant"):
                
                # Replace the simple spinner with an advanced status tracker
                with st.status("🤖 Analyzing intent...", expanded=True) as status_box:
                    st_tool_placeholder = st.empty()
                    tool_text = ""
                    current_tool_logs = []
                    
                    def on_status(payload: dict):
                        nonlocal tool_text
                        if payload["type"] == "tool_name":
                            msg_str = f"🛠️ Tool Requested: {payload['content']}..."
                            status_box.update(label=msg_str)
                            current_tool_logs.append(f"**{msg_str}**")
                        elif payload["type"] == "tool_stream":
                            tool_text += payload["content"]
                            st_tool_placeholder.markdown(f"```json\n{tool_text}\n```")
                        elif payload["type"] == "info":
                            msg_str = payload["content"]
                            status_box.write(msg_str)
                            current_tool_logs.append(msg_str)
                        elif payload["type"] == "complete":
                            status_box.update(label="Analysis Complete", state="complete", expanded=False)

                message_placeholder = st.empty()
                full_response = []
                
                def on_chunk(chunk: str):
                    if not full_response:
                        message_placeholder.empty() # Clear if needed
                        status_box.update(label="Finished", state="complete", expanded=False)
                    full_response.append(chunk)
                    message_placeholder.markdown("".join(full_response) + "▌")

                stats = None
                try:
                    answer, cited_pages, stats = asyncio.run(
                        rag.answer_question(
                            question,
                            chat_history=st.session_state.chat_history,
                            stream_callback=on_chunk,
                            status_callback=on_status
                        )
                    )
                    if tool_text:
                        current_tool_logs.append(f"```json\n{tool_text}\n```")
                except Exception as exc:
                    logger.exception("RAG answer failed")
                    answer = f"⚠️ Error generating answer: {exc}"
                    cited_pages = []
                    message_placeholder.markdown(answer)

                # Finalize output without the cursor
                message_placeholder.markdown(answer)
                if stats:
                    st.caption(
                        f"⏱️ {stats['total_time']:.1f}s | "
                        f"📥 In: {stats['input_tokens']} | "
                        f"📤 Out: {stats['output_tokens']} | "
                        f"⚡ {stats['speed']:.1f} t/s"
                    )

            asst_msg = ChatMessage(
                role="assistant",
                content=answer,
                citations=cited_pages,
                stats=stats,
                tool_logs=current_tool_logs,
                timestamp=datetime.now(),
            )
            st.session_state.chat_history.append(asst_msg)
            st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
from contextlib import contextmanager
import subprocess
import time


def main() -> None:
    _init_state()

    company_name, run_clicked, upload_bytes = _render_sidebar()

    # ── Handle manual upload ────────────────────────────────────────
    if upload_bytes and (run_clicked or st.session_state.get("_last_upload") != hash(upload_bytes)):
        st.session_state["_last_upload"] = hash(upload_bytes)
        
        # Ensure the directory exists BEFORE trying to create a temp file inside it!
        Path(f"{DATA_DIR}/downloads").mkdir(parents=True, exist_ok=True)
        
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".pdf", dir=f"{DATA_DIR}/downloads"
        )
        tmp.write(upload_bytes)
        tmp.flush()
        tmp_path = tmp.name
        tmp.close()

        st.session_state.is_running = True
        try:
            if True:
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
            if True:
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
