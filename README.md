# Annual Report Analyzer

A **local, agentic pipeline** that takes a company name, finds its latest annual report on the public web, extracts key financial data and business insights, and produces a citation-backed research note — every claim traceable to a page number in the source PDF.

All analysis runs on a **locally hosted open-weight LLM** (Qwen 3.5-9B). No cloud LLM API is used at any stage of extraction or synthesis.

---

## Features

- **Automated discovery** — searches the web (Tavily API), pre-filters candidates by URL/domain heuristics, and downloads the PDF via headless browser or direct HTTP.
- **Smart validation** — confirms the PDF is the correct company's annual report by checking only the first two pages (fixed cost regardless of document length).
- **Section-aware parsing** — splits the report into standard sections (MD&A, Balance Sheet, P&L, etc.) using regex heading detection with a semantic RAG fallback.
- **Structured financial extraction** — pulls 10+ key financial metrics using JSON-schema-constrained LLM output.
- **Insight generation** — generates narrative insights (sector, revenue model, growth drivers, risks) with page citations.
- **RAG-based Q&A** — ask free-form questions answered via retrieval-augmented generation over the parsed document.
- **Citation tracking** — every number and insight carries `[p.N]` page references.
- **Local caching** — SQLite + per-report FAISS index, so re-opening a report is instant.
- **Export** — Markdown and styled PDF research-note output.

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| LLM | Qwen3.5-9B-Instruct, GGUF, Q4_K_M | Apache 2.0, strong JSON output, fits 6 GB GPU |
| LLM serving | llama.cpp / `llama-server` (GPU) | OpenAI-compatible local API, GPU/CPU layer splitting |
| Web search | Tavily API (free tier) | Simple search API, no scraping infra needed |
| Browsing / download | Playwright (headless Chromium) | Renders JS-heavy investor-relations pages |
| PDF parsing | `pymupdf4llm` | Pure Python, page-number-aware Markdown output |
| RAG — embeddings | `Qwen3-Embedding-0.6B-Q8_0.gguf` | Small, CPU-only, leaves GPU free for the main LLM |
| Embedding serving | llama.cpp / `llama-server` (CPU-only) | Separate instance on port 8081 |
| RAG — vector index | FAISS (`IndexFlatIP`) | Local, in-process similarity search |
| Cache | SQLite | Single file, no server process |
| UI | Streamlit | Fast to build, good for single-user local tools |
| Export | Markdown + PDF (`fpdf2`) | Portable text and presentable research note |
| Package management | `uv` | Fast, single tool for env + deps |

## What Gets Extracted

### Financial Metrics

Consolidated figures (primary), standalone (secondary), current and previous year:

- Revenue
- EBITDA and EBITDA margin
- Net Profit / PAT
- EPS (Basic & Diluted)
- Total Assets / Total Liabilities
- Debt/Equity ratio
- Cash Flow from Operations
- ROE / ROCE

### Business Insights

Generated from MD&A and Directors' Report, each point citing its source page:

- Sector / industry classification
- Revenue model
- Key business segments
- Growth drivers
- Key risks

## Prerequisites

- **Python 3.11+**
- **NVIDIA GPU** with 6 GB+ VRAM (tested on RTX 3050)
- **llama.cpp** built with CUDA support ([build instructions](https://github.com/ggerganov/llama.cpp#build))
- **Tavily API key** (free tier: [tavily.com](https://tavily.com))
- **uv** package manager ([install](https://docs.astral.sh/uv/getting-started/installation/))

## Installation

```bash
# 1. Clone the repository
git clone <repo-url> && cd annual-report-analyzer

# 2. Install dependencies
uv sync

# 3. Install Playwright's headless browser (one-time)
uv run playwright install chromium

# 4. Download model files (if not already present)
#    Place them in the project directory:
#    - llm/Qwen3.5-9B-Q4_K_M.gguf
#    - Embedding model/Qwen3-Embedding-0.6B-Q8_0.gguf
```

## Configuration

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```env
TAVILY_API_KEY=tvly-your-key-here

# Optional — override defaults
# LLM_BASE_URL=http://localhost:8080
# EMBEDDING_BASE_URL=http://localhost:8081
```

### Custom `llama-server` location

If `llama-server` is not on your `PATH`, set:

```bash
export LLAMA_SERVER_PATH=/path/to/llama-server
```

## Usage

### 1. Start the model servers

Open two terminals:

```bash
# Terminal 1 — Main LLM (GPU)
./llm_server/start_server.sh

# Terminal 2 — Embedding model (CPU-only)
./llm_server/start_embedding_server.sh
```

### 2. Run the app

```bash
uv run streamlit run app/streamlit_app.py
```

### 3. Analyse a company

- Type a company name (e.g. "Infosys") and click **Analyze**.
- Or upload a PDF directly via the sidebar.
- Results appear in four tabs: Financials, Insights, Chat, Report Info.
- Export to Markdown or PDF from the sidebar.

## Project Structure

```
annual-report-analyzer/
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── llm_server/
│   ├── start_server.sh              # Main LLM (GPU)
│   └── start_embedding_server.sh    # Embedding model (CPU-only)
├── pipeline/
│   ├── schema.py                    # Pydantic models for all data
│   ├── cache.py                     # SQLite + FAISS persistence
│   ├── discovery/
│   │   ├── search.py                # Tavily search + candidate scoring
│   │   ├── browser.py               # Playwright headless PDF download
│   │   └── validate.py              # PDF magic-byte + LLM validation
│   ├── parsing/
│   │   ├── parse.py                 # PDF → page-tagged Markdown
│   │   ├── sections.py              # Regex section split + semantic fallback
│   │   └── chunking.py              # Overlapping page-aware chunks
│   ├── extraction/
│   │   ├── extract.py               # JSON-schema financial extraction
│   │   └── insights.py              # Narrative insight generation
│   └── rag/
│       └── rag.py                   # Embeddings, FAISS index, RAG Q&A
├── app/
│   └── streamlit_app.py             # Main UI + pipeline orchestration
├── export/
│   ├── to_markdown.py               # Markdown research note
│   └── to_pdf.py                    # Styled PDF research note
└── data/                            # Generated at runtime (gitignored)
    ├── cache.db
    ├── downloads/
    └── <company_id>/
        ├── faiss.index
        └── chunks.jsonl
```

## Scope

Built for standard corporate annual reports following the **Schedule III, Division II** statement format (manufacturing, IT services, FMCG, etc.). Banks and NBFCs (Division III) are outside scope. Only text-based PDFs — no OCR. Each run analyses a single company's single most recent annual report.

## Troubleshooting

| Problem | Solution |
|---|---|
| **VRAM out of memory** | Lower `-ngl` in `start_server.sh` by 4–6 at a time. Remaining layers fall back to CPU automatically. |
| **Tavily timeout** | Check network. The app falls back to manual upload if search doesn't return in 8 seconds. |
| **Playwright errors** | Run `uv run playwright install chromium` again. Check that no other Chromium processes are hung. |
| **Empty financial data** | The report may use non-standard section headings. Try uploading the PDF manually and check `logs/app.log` for section-split details. |
| **Embedding server won't start** | Make sure nothing else is using port 8081. The embedding model needs ~1 GB RAM on CPU. |

## License

MIT
