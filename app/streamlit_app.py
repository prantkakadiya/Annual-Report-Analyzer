"""Annual Report Analyzer — Streamlit application.

Thin UI layer that delegates pipeline execution to
:class:`pipeline.orchestrator.PipelineOrchestrator` and renders results.
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
from pipeline.config import DATA_DIR, EMBEDDING_BASE_URL, LLM_BASE_URL  # noqa: E402
from pipeline.orchestrator import PipelineCallbacks, PipelineOrchestrator  # noqa: E402
from pipeline.rag.chat import RAGChatAgent  # noqa: E402
from pipeline.rag.index import RAGIndex  # noqa: E402
from pipeline.schema import AnalysisResult, ChatMessage, ReportMetadata  # noqa: E402
from export.to_markdown import export_markdown  # noqa: E402
from export.to_pdf import export_pdf  # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Annual Report Analyzer",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Session helpers
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
        "rag_index": None,        # RAGIndex | None
        "chat_history": [],       # list[ChatMessage]
        "is_running": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Pipeline execution (bridges Streamlit widgets → PipelineCallbacks)
# ---------------------------------------------------------------------------
def _run_analysis(
    company_name: str | None = None,
    pdf_path: str | None = None,
) -> None:
    """Run the full pipeline, wiring Streamlit widgets as progress callbacks."""
    cache = _get_cache()
    orchestrator = PipelineOrchestrator(cache=cache, data_dir=DATA_DIR)

    progress = st.progress(0, text="Initialising…")
    status_area = st.status("🔄 Running analysis pipeline…", expanded=True)

    def on_step(step: int, total: int, label: str) -> None:
        pct = int(step / total * 100)
        progress.progress(pct, text=label)
        status_area.write(f"**Step {step}/{total}:** {label}")

    def on_status(msg: str) -> None:
        status_area.write(f"  {msg}")

    def on_download_progress(downloaded: int, total_size: int) -> None:
        if total_size > 0:
            pct = min(downloaded / total_size, 1.0)
            _dl_bar.progress(
                pct,
                text=f"Downloading PDF... {downloaded/1024/1024:.1f}MB / "
                     f"{total_size/1024/1024:.1f}MB",
            )
        else:
            _dl_bar.progress(
                0.5,
                text=f"Downloading PDF... {downloaded/1024/1024:.1f}MB",
            )

    def on_embedding_progress(processed: int, total: int) -> None:
        if total > 0:
            pct = min(processed / total, 1.0)
            _emb_bar.progress(pct, text=f"Embedding chunks... {processed}/{total}")

    # Sub-progress bars (created lazily in the callbacks above)
    _dl_bar = st.progress(0, text="Downloading PDF...")
    _emb_bar = st.progress(0, text="Embedding chunks...")
    _dl_bar.empty()
    _emb_bar.empty()

    # Re-create so the callbacks capture fresh widgets
    _dl_bar = st.progress(0, text="Downloading PDF...")
    _emb_bar = st.progress(0, text="Embedding chunks...")

    callbacks = PipelineCallbacks(
        on_step=on_step,
        on_status=on_status,
        on_download_progress=on_download_progress,
        on_embedding_progress=on_embedding_progress,
    )

    st.session_state.is_running = True
    try:
        result, rag_index = asyncio.run(
            orchestrator.run(
                company_name=company_name,
                pdf_path=pdf_path,
                callbacks=callbacks,
            )
        )
        st.session_state.result = result
        st.session_state.rag_index = rag_index
        status_area.update(label="✅ Analysis complete!", state="complete")
        progress.progress(100, text="Done!")
    except RuntimeError as exc:
        logger.warning("Pipeline stopped: %s", exc)
        st.warning(str(exc))
    except Exception as exc:
        logger.exception("Pipeline failed")
        st.error(f"Analysis failed: {exc}")
    finally:
        st.session_state.is_running = False
        _dl_bar.empty()
        _emb_bar.empty()


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
                for k in ["result", "rag_index", "chat_history"]:
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
                        rag_index = RAGIndex(
                            embedding_base_url=EMBEDDING_BASE_URL,
                            cache=cache,
                        )
                        cid = cache.make_company_id(meta.company_name, meta.fiscal_year)
                        if rag_index.load_index(cid):
                            st.session_state.rag_index = rag_index
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
    if not result.financial_data or not result.financial_data.metrics:
        st.info("No financial metrics extracted.")
        return

    metrics = result.financial_data.metrics

    # KPI cards
    def get_kpi(name_keywords: list[str]) -> str:
        for m in metrics:
            if any(k.lower() in m.name.lower() for k in name_keywords):
                val = m.current_year_consolidated or m.current_year_standalone
                if val is not None:
                    return f"{val:,.1f} {m.unit}"
        return "—"

    rev = get_kpi(["revenue", "total income"])
    ebitda = get_kpi(["ebitda"])
    pat = get_kpi(["profit", "pat"])

    st.markdown("### 🏆 Key Highlights (Current Year)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Revenue", rev)
    c2.metric("EBITDA", ebitda)
    c3.metric("Net Profit (PAT)", pat)

    st.markdown("---")
    st.markdown("### 📊 Detailed Financial Metrics")

    def fmt(val: float | None) -> str:
        if val is None:
            return "—"
        return f"{val:,.2f}"

    rows = []
    for m in metrics:
        rows.append({
            "Metric": m.name,
            "Unit": m.unit,
            "CY Consolidated": fmt(m.current_year_consolidated),
            "CY Standalone": fmt(m.current_year_standalone),
            "PY Consolidated": fmt(m.previous_year_consolidated),
            "PY Standalone": fmt(m.previous_year_standalone),
            "Page": f"p.{m.page_number}" if m.page_number else "—",
            "Notes": m.notes or "—",
        })

    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Metric": st.column_config.TextColumn(width="medium"),
            "Notes": st.column_config.TextColumn(width="large"),
        },
    )


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


# ---------------------------------------------------------------------------
# UI — Chat
# ---------------------------------------------------------------------------
def _render_chat() -> None:
    """RAG chat panel."""
    rag_index: RAGIndex | None = st.session_state.rag_index

    if rag_index is None or rag_index.index is None:
        st.warning(
            "RAG index not available. Run an analysis first or load a cached report."
        )
        return

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat", key="clear_chat_btn"):
            st.session_state.chat_history = []
            st.rerun()

    chat_container = st.container()
    question = st.chat_input("Ask a question about the report…")

    if question:
        user_msg = ChatMessage(
            role="user",
            content=question,
            timestamp=datetime.now(),
        )
        st.session_state.chat_history.append(user_msg)

    with chat_container:
        # Render all history
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

        # Generate assistant response for new question
        if question:
            with st.chat_message("assistant"):
                with st.status("🤖 Analyzing intent...", expanded=True) as status_box:
                    st_tool_placeholder = st.empty()
                    tool_text = ""
                    current_tool_logs: list[str] = []

                    def on_status(payload: dict) -> None:
                        nonlocal tool_text
                        if payload["type"] == "tool_name":
                            msg_str = f"🛠️ Tool Requested: {payload['content']}..."
                            status_box.update(label=msg_str)
                            current_tool_logs.append(f"**{msg_str}**")
                        elif payload["type"] == "tool_stream":
                            tool_text += payload["content"]
                            st_tool_placeholder.markdown(
                                f"```json\n{tool_text}\n```"
                            )
                        elif payload["type"] == "info":
                            msg_str = payload["content"]
                            status_box.write(msg_str)
                            current_tool_logs.append(msg_str)
                        elif payload["type"] == "complete":
                            status_box.update(
                                label="Analysis Complete",
                                state="complete",
                                expanded=False,
                            )

                message_placeholder = st.empty()
                full_response: list[str] = []

                def on_chunk(chunk: str) -> None:
                    if not full_response:
                        message_placeholder.empty()
                        status_box.update(
                            label="Finished",
                            state="complete",
                            expanded=False,
                        )
                    full_response.append(chunk)
                    message_placeholder.markdown("".join(full_response) + "▌")

                stats = None
                chat_agent = RAGChatAgent(llm_base_url=LLM_BASE_URL)
                try:
                    answer, cited_pages, stats = asyncio.run(
                        chat_agent.answer_question(
                            question,
                            rag_index=rag_index,
                            chat_history=st.session_state.chat_history,
                            stream_callback=on_chunk,
                            status_callback=on_status,
                        )
                    )
                    if tool_text:
                        current_tool_logs.append(f"```json\n{tool_text}\n```")
                except Exception as exc:
                    logger.exception("RAG answer failed")
                    answer = f"⚠️ Error generating answer: {exc}"
                    cited_pages = []
                    message_placeholder.markdown(answer)

                # Finalize
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
def main() -> None:
    _init_state()

    company_name, run_clicked, upload_bytes = _render_sidebar()

    # ── Handle manual upload ────────────────────────────────────────
    if upload_bytes and (
        run_clicked
        or st.session_state.get("_last_upload") != hash(upload_bytes)
    ):
        st.session_state["_last_upload"] = hash(upload_bytes)

        Path(f"{DATA_DIR}/downloads").mkdir(parents=True, exist_ok=True)
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".pdf", dir=f"{DATA_DIR}/downloads"
        )
        tmp.write(upload_bytes)
        tmp.flush()
        tmp_path = tmp.name
        tmp.close()

        _run_analysis(company_name or None, pdf_path=tmp_path)

    # ── Handle search-based analysis ────────────────────────────────
    elif run_clicked and company_name:
        _run_analysis(company_name)

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
