# Annual Report Analyzer

> **🎓 Academic Project — Not Production-Ready**
>
> This is an academic/research project exploring local LLM-powered document analysis. It is under active development and **not suitable for production, commercial, or regulatory use**. Extracted financial data may be incomplete or incorrect. Do not rely on its output for investment or compliance decisions.

---

A local AI pipeline that takes a company name, finds its latest annual report on the web, extracts key financial data and business insights, and produces a research note with page-number citations — all running on your own machine with no cloud AI APIs.

## Who Is This For

- Students and researchers exploring **LLM-based document analysis** and **retrieval-augmented generation (RAG)**.
- Developers interested in building **local-first AI pipelines** on consumer hardware.
- Anyone curious about automated financial report processing without sending data to the cloud.

## What It Does

1. **Discovers** the company's latest annual report PDF via web search.
2. **Parses** the PDF using GPU-accelerated OCR and layout analysis.
3. **Extracts** structured financial metrics (Revenue, EBITDA, Net Profit, EPS, etc.) with page citations.
4. **Generates** narrative business insights (growth drivers, risks, revenue model).
5. **Enables Q&A** — ask free-form questions answered via RAG over the parsed document.
6. **Exports** a citation-backed research note as Markdown or PDF.

## Prerequisites

- **Python 3.11+**
- **NVIDIA GPU** with 6 GB+ VRAM (tested on RTX 3050 Laptop)
- **~16 GB system RAM**
- **llama.cpp** built with CUDA support ([build instructions](https://github.com/ggerganov/llama.cpp#build))
- **Tavily API key** (free tier: [tavily.com](https://tavily.com))
- **uv** package manager ([install](https://docs.astral.sh/uv/getting-started/installation/))

## Setup

```bash
# 1. Clone and install
git clone <repo-url> && cd annual-report-analyzer
uv sync

# 2. Install headless browser (one-time)
uv run playwright install chromium

# 3. Download model files and place them in:
#    - llm/Qwen3.5-9B-Q4_K_M.gguf
#    - Embedding model/Qwen3-Embedding-0.6B-Q8_0.gguf

# 4. Configure environment
cp .env.example .env
# Edit .env and add your TAVILY_API_KEY
```

## Usage

```bash
# Terminal 1 — Start the LLM server
./llm_server/start_server.sh

# Terminal 2 — Start the embedding server
./llm_server/start_embedding_server.sh

# Terminal 3 — Launch the app
uv run streamlit run app/streamlit_app.py
```

Then open the Streamlit UI, type a company name (e.g. "Infosys"), and click **Analyze**. You can also upload a PDF directly via the sidebar.

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Qwen3.5-9B-Instruct (Q4_K_M) via llama.cpp |
| PDF Parsing | Docling + EasyOCR (GPU-accelerated) |
| Embeddings | Qwen3-Embedding-0.6B (Q8_0) via llama.cpp |
| Vector Search | FAISS |
| Web Search | Tavily API |
| PDF Download | Playwright (headless Chromium) |
| UI | Streamlit |
| Cache | SQLite |

## Scope

- Built for standard Indian corporate annual reports (Schedule III, Division II format).
- Banks and NBFCs (Division III) are not supported.
- Processes one company's report per session.

## License

MIT
