# 🏗️ Architecture Guide — Annual Report Analyzer V1

**3,958 lines** · **31 `.py` files** · **4 packages** · fully local (no cloud APIs)

---

## 1. System Overview

The 30,000-foot view — what talks to what:

```mermaid
flowchart LR
    subgraph USER["👤 User"]
        Browser["Browser"]
    end

    subgraph UI["UI Layer"]
        ST["app/streamlit_app.py<br/>563L — thin UI shell"]
    end

    subgraph ORCH["Orchestration"]
        ORC["pipeline/orchestrator.py<br/>278L — pipeline runner"]
        CB["PipelineCallbacks<br/>progress reporting"]
    end

    subgraph PIPELINE["Pipeline Modules"]
        direction TB
        DISC["discovery/<br/>search · browser · validate"]
        PARSE["parsing/<br/>parse · sections · chunking"]
        EXTRACT["extraction/<br/>extract · insights"]
        RAG["rag/<br/>index · chat"]
    end

    subgraph INFRA["Infrastructure"]
        CACHE["cache.py<br/>SQLite + FAISS"]
        MM["model_manager.py<br/>VRAM lifecycle"]
        CFG["config.py<br/>centralized config"]
        SCHEMA["schema.py<br/>Pydantic models"]
    end

    subgraph SERVERS["Local Servers"]
        LLM["llama-server :8080<br/>Qwen 9B LLM"]
        EMB["llama-server :8081<br/>Qwen 0.6B Embedding"]
    end

    subgraph EXTERNAL["External"]
        TAV["Tavily API<br/>web search"]
    end

    Browser -->|HTTP| ST
    ST -->|delegates| ORC
    ORC -->|wires| CB
    ORC --> DISC
    ORC --> PARSE
    ORC --> EXTRACT
    ORC --> RAG
    ST -->|chat| RAG

    DISC -->|search| TAV
    DISC -->|LLM calls| LLM
    EXTRACT -->|LLM calls| LLM
    RAG -->|embed| EMB
    RAG -->|generate| LLM

    ORC --> CACHE
    ORC --> MM
    PIPELINE --> CFG
    PIPELINE --> SCHEMA
    MM -->|start/stop| LLM
    MM -->|start/stop| EMB
```

### Key insight
The UI layer (`streamlit_app.py`) **never touches pipeline logic directly**. It creates a [`PipelineOrchestrator`](file:///home/prant/Desktop/python/job/Annual%20Report%20Analyzer%20V1/pipeline/orchestrator.py) and wires Streamlit widgets as callbacks. This means you could replace Streamlit with a CLI, FastAPI, or anything else by just providing different callbacks.

---

## 2. Module Dependency Graph

Who imports whom — follow the arrows to understand coupling:

```mermaid
flowchart TD
    CFG["config.py"]
    SCH["schema.py"]
    MM["model_manager.py"]
    CACHE["cache.py"]

    SEARCH["discovery/search.py"]
    BROWSER["discovery/browser.py"]
    VALIDATE["discovery/validate.py"]

    PARSE["parsing/parse.py"]
    SECTIONS["parsing/sections.py"]
    CHUNK["parsing/chunking.py"]

    EXTRACT["extraction/extract.py"]
    INSIGHTS["extraction/insights.py"]

    IDX["rag/index.py"]
    CHAT["rag/chat.py"]

    ORC["orchestrator.py"]
    APP["app/streamlit_app.py"]

    EXP_MD["export/to_markdown.py"]
    EXP_PDF["export/to_pdf.py"]

    CFG --> SEARCH
    CFG --> VALIDATE
    CFG --> EXTRACT
    CFG --> INSIGHTS
    CFG --> IDX
    CFG --> CHAT
    CFG --> ORC
    CFG --> APP

    SCH --> CACHE
    SCH --> SEARCH
    SCH --> BROWSER
    SCH --> VALIDATE
    SCH --> EXTRACT
    SCH --> INSIGHTS
    SCH --> IDX
    SCH --> CHAT
    SCH --> ORC
    SCH --> APP
    SCH --> PARSE
    SCH --> SECTIONS
    SCH --> CHUNK
    SCH --> EXP_MD
    SCH --> EXP_PDF

    MM --> ORC
    MM --> CHAT
    MM --> IDX

    CACHE --> ORC
    CACHE --> APP
    CACHE --> IDX

    IDX --> CHAT
    IDX --> SECTIONS
    IDX --> ORC
    IDX --> APP

    CHAT --> APP

    ORC --> APP

    EXP_MD --> APP
    EXP_PDF --> APP
```

### Reading the graph
- **Leaves** (no outgoing arrows): `config.py`, `schema.py` — these are pure definitions, safe to edit without breaking anything downstream.
- **Root** (no incoming arrows): `streamlit_app.py` — the entry point.
- **High fan-in nodes**: `schema.py` and `config.py` are imported by almost everything. Changes here ripple everywhere.
- **Isolated clusters**: `export/` only depends on `schema.py` — completely independent from the pipeline.

---

## 3. Pipeline Sequence (the main flow)

What happens when you click "Analyze":

```mermaid
sequenceDiagram
    actor User
    participant ST as streamlit_app.py
    participant ORC as orchestrator.py
    participant MM as model_manager
    participant SEARCH as search.py
    participant DL as browser.py
    participant VAL as validate.py
    participant PARSE as parse.py
    participant CHUNK as chunking.py
    participant IDX as rag/index.py
    participant SEC as sections.py
    participant EXT as extract.py
    participant INS as insights.py
    participant CACHE as cache.py
    participant LLM as LLM :8080
    participant EMB as Embedding :8081

    User->>ST: Click "Analyze" (company name)
    ST->>ORC: run(company_name, callbacks)

    Note over ORC,MM: Phase 1: Discovery (needs LLM)
    ORC->>MM: require_llm()
    MM->>LLM: Start server
    ORC->>SEARCH: search(company_name)
    SEARCH->>LLM: Rank candidates
    SEARCH-->>ORC: candidates[]

    Note over ORC,DL: Phase 2: Download
    ORC->>DL: download_from_candidates()
    DL-->>ORC: pdf_path

    Note over ORC,VAL: Phase 3: Validation
    ORC->>VAL: validate(pdf_path)
    VAL->>LLM: Is this the right report?
    VAL-->>ORC: ValidationResult

    ORC->>CACHE: get_report() — cache check
    CACHE-->>ORC: None (miss)

    Note over ORC,MM: Phase 4: Parsing (needs VRAM for Docling)
    ORC->>MM: stop_all() — free VRAM
    ORC->>PARSE: parse(pdf_path)
    Note over PARSE: Docling OCR + layout<br/>(uses GPU via PyTorch)
    PARSE-->>ORC: ParsedReport

    Note over ORC,EMB: Phase 5: Embed + Index
    ORC->>MM: require_embedding()
    MM->>EMB: Start server
    ORC->>CHUNK: chunk(parsed)
    CHUNK-->>ORC: chunks[]
    ORC->>IDX: build_index(chunks)
    IDX->>EMB: Batch embed
    IDX-->>ORC: FAISS index ready

    Note over ORC,LLM: Phase 6: Extract + Insights
    ORC->>SEC: split(parsed, rag_index)
    SEC-->>ORC: sections{}
    ORC->>MM: require_llm()
    MM->>LLM: Start server
    ORC->>EXT: extract(sections)
    EXT->>LLM: Agentic tool-calling loop
    EXT-->>ORC: FinancialData
    ORC->>INS: generate(sections)
    INS->>LLM: Generate insights
    INS-->>ORC: InsightData

    ORC->>CACHE: save_report(result)
    ORC-->>ST: (result, rag_index)
    ST-->>User: Render tabs
```

### Debugging tip
Each phase is a **separate method call** in [`orchestrator.py`](file:///home/prant/Desktop/python/job/Annual%20Report%20Analyzer%20V1/pipeline/orchestrator.py). If the pipeline fails, check `logs/app.log` — the orchestrator logs `Pipeline step X/8: <label>` at each phase, so you can pinpoint exactly where it stopped.

---

## 4. RAG Chat Flow (the interactive part)

What happens when you ask a question in the chat tab:

```mermaid
sequenceDiagram
    actor User
    participant ST as streamlit_app.py
    participant CHAT as rag/chat.py
    participant MM as model_manager
    participant LLM as LLM :8080
    participant IDX as rag/index.py
    participant EMB as Embedding :8081

    User->>ST: "What was the revenue?"
    ST->>CHAT: answer_question(q, rag_index, history)

    CHAT->>MM: require_llm()
    CHAT->>LLM: Chat completion (with tools)

    alt LLM decides to search
        LLM-->>CHAT: tool_call: search_annual_report("revenue")
        CHAT->>IDX: search("revenue")
        IDX->>MM: require_embedding()
        IDX->>EMB: Embed query
        IDX-->>CHAT: top-k chunks with scores

        CHAT->>MM: require_llm()
        CHAT->>LLM: System + RAG context + question
        LLM-->>CHAT: Streaming answer tokens
        CHAT-->>ST: stream_callback(chunk)
    else LLM answers directly
        LLM-->>CHAT: Streaming answer tokens
        CHAT-->>ST: stream_callback(chunk)
    end

    CHAT-->>ST: (answer, cited_pages, stats)
    ST-->>User: Rendered markdown + stats
```

### Key detail
The LLM **decides** whether to use the search tool. Conversational questions ("hi", "thanks") get answered directly. Factual questions ("what was the EBITDA?") trigger the search → embed → retrieve → generate cycle.

---

## 5. VRAM Lifecycle (why `model_manager` exists)

This project runs on a **single GPU** that can't hold the LLM + Embedding model + Docling simultaneously. The `ModelManager` swaps them:

```mermaid
stateDiagram-v2
    [*] --> Idle: App starts

    Idle --> LLM_Active: require_llm()
    LLM_Active --> Idle: stop_all()
    LLM_Active --> EMB_Active: require_embedding()<br/>stops LLM first

    Idle --> EMB_Active: require_embedding()
    EMB_Active --> Idle: stop_all()
    EMB_Active --> LLM_Active: require_llm()<br/>stops EMB first

    state LLM_Active {
        [*] --> discovery
        discovery --> validation
        validation --> extraction
        extraction --> insights
    }

    state "VRAM Free<br/>(Docling OCR)" as Idle

    state EMB_Active {
        [*] --> chunking_embed
        chunking_embed --> rag_search
    }
```

### The swap sequence during pipeline:
1. **LLM loaded** → discovery, validation
2. **Stop all** → free VRAM for Docling PDF parsing (PyTorch)
3. **Embedding loaded** → chunk embedding, FAISS index
4. **LLM loaded** → extraction (agentic), insight generation
5. **During chat**: swaps between LLM ↔ Embedding per tool call

---

## 6. Data Model

How data flows through the system:

```mermaid
erDiagram
    ReportMetadata {
        string company_name
        string fiscal_year
        string pdf_path
        string source_url
        int total_pages
        datetime discovered_at
        bool validated
    }

    FinancialMetric {
        string name
        float current_year_consolidated
        float current_year_standalone
        float previous_year_consolidated
        float previous_year_standalone
        string unit
        int page_number
        string notes
    }

    FinancialData {
        string company_name
        string fiscal_year
        datetime extraction_timestamp
    }

    Insight {
        string category
        string content
        list_int page_numbers
    }

    InsightData {
        string company_name
        string fiscal_year
    }

    AnalysisResult {
        dict sections_found
        string analysis_version
    }

    DocumentChunk {
        string text
        list_int page_numbers
        string section_name
    }

    ChatMessage {
        string role
        string content
        list_int citations
        dict stats
        list_str tool_logs
        datetime timestamp
    }

    AnalysisResult ||--|| ReportMetadata : has
    AnalysisResult ||--o| FinancialData : has
    AnalysisResult ||--o| InsightData : has
    FinancialData ||--o{ FinancialMetric : contains
    InsightData ||--o{ Insight : contains
```

All models live in [`pipeline/schema.py`](file:///home/prant/Desktop/python/job/Annual%20Report%20Analyzer%20V1/pipeline/schema.py) (107 lines). `ChatMessage` lives in session state only — never persisted to cache.

---

## 7. Debugging Guide

### "Where do I look?"

| Symptom | File to check | Function / area |
|---------|--------------|-----------------|
| **Search finds wrong PDFs** | [`search.py`](file:///home/prant/Desktop/python/job/Annual%20Report%20Analyzer%20V1/pipeline/discovery/search.py) | `_rank_with_llm()` — check `_RANKING_SYSTEM_PROMPT` |
| **PDF download fails** | [`browser.py`](file:///home/prant/Desktop/python/job/Annual%20Report%20Analyzer%20V1/pipeline/discovery/browser.py) | `_try_direct_download()` then `_try_browser_download()` |
| **Validation rejects a valid PDF** | [`validate.py`](file:///home/prant/Desktop/python/job/Annual%20Report%20Analyzer%20V1/pipeline/discovery/validate.py) | `_llm_validate()` — check `_VALIDATION_SYSTEM_PROMPT` |
| **PDF parsing garbled** | [`parse.py`](file:///home/prant/Desktop/python/job/Annual%20Report%20Analyzer%20V1/pipeline/parsing/parse.py) | `parse()` — Docling converter config |
| **Sections not detected** | [`sections.py`](file:///home/prant/Desktop/python/job/Annual%20Report%20Analyzer%20V1/pipeline/parsing/sections.py) | `_regex_scan()` then `_semantic_fallback()` |
| **Wrong financial numbers** | [`extract.py`](file:///home/prant/Desktop/python/job/Annual%20Report%20Analyzer%20V1/pipeline/extraction/extract.py) | `_agentic_find_metric()` — check `_EXTRACTION_SYSTEM_PROMPT` and tool definitions |
| **Bad insights** | [`insights.py`](file:///home/prant/Desktop/python/job/Annual%20Report%20Analyzer%20V1/pipeline/extraction/insights.py) | `_generate_category()` — check `_INSIGHT_SYSTEM_PROMPT` |
| **Chat gives wrong answers** | [`chat.py`](file:///home/prant/Desktop/python/job/Annual%20Report%20Analyzer%20V1/pipeline/rag/chat.py) | `answer_question()` — check `_CHAT_SYSTEM_PROMPT` and tool schema |
| **Chat search misses info** | [`index.py`](file:///home/prant/Desktop/python/job/Annual%20Report%20Analyzer%20V1/pipeline/rag/index.py) | `search()` — check `top_k` and `threshold` params |
| **VRAM OOM / server crash** | [`model_manager.py`](file:///home/prant/Desktop/python/job/Annual%20Report%20Analyzer%20V1/pipeline/model_manager.py) | `require_llm()` / `require_embedding()` — check swap timing |
| **Pipeline hangs at a step** | [`orchestrator.py`](file:///home/prant/Desktop/python/job/Annual%20Report%20Analyzer%20V1/pipeline/orchestrator.py) | Check `logs/app.log` for last `Pipeline step X/8` |
| **Cache returns stale data** | [`cache.py`](file:///home/prant/Desktop/python/job/Annual%20Report%20Analyzer%20V1/pipeline/cache.py) | `get_report()` / `save_report()` — check `data/cache.db` |
| **UI rendering issues** | [`streamlit_app.py`](file:///home/prant/Desktop/python/job/Annual%20Report%20Analyzer%20V1/app/streamlit_app.py) | `_render_financials_tab()` / `_render_chat()` |
| **Export broken** | [`to_markdown.py`](file:///home/prant/Desktop/python/job/Annual%20Report%20Analyzer%20V1/export/to_markdown.py) / [`to_pdf.py`](file:///home/prant/Desktop/python/job/Annual%20Report%20Analyzer%20V1/export/to_pdf.py) | `export_markdown()` / `export_pdf()` |

### Quick debug workflow
```bash
# 1. Check the log for the last error
tail -50 logs/app.log

# 2. Test a single module in isolation
python -c "
from pipeline.parsing.parse import ReportParser
p = ReportParser()
r = p.parse('path/to/test.pdf')
print(f'{r.total_pages} pages')
"

# 3. Test extraction independently
python -c "
import asyncio
from pipeline.extraction.extract import FinancialExtractor
e = FinancialExtractor()
sections = {'Profit & Loss': 'your text here...'}
result = asyncio.run(e.extract(sections, 'TestCorp', '2024'))
print(result)
"

# 4. Test RAG search quality
python -c "
import asyncio
from pipeline.rag.index import RAGIndex
from pipeline.cache import ReportCache
idx = RAGIndex(cache=ReportCache())
idx.load_index('company_id')
results = asyncio.run(idx.search('revenue'))
for chunk, score in results:
    print(f'{score:.3f} | {chunk.text[:80]}...')
"
```

---

## 8. Which Diagram to Use When

| You want to… | Use diagram |
|--------------|-------------|
| Understand the big picture | **§1 System Overview** |
| Know what imports what (before editing) | **§2 Dependency Graph** |
| Debug a pipeline failure | **§3 Pipeline Sequence** |
| Debug chat quality | **§4 RAG Chat Flow** |
| Debug VRAM / OOM crashes | **§5 VRAM Lifecycle** |
| Understand the data structures | **§6 Data Model** |
| Find the right file to edit | **§7 Debugging Guide** |

---

## File Map (quick reference)

```
Annual Report Analyzer V1/
├── app/
│   └── streamlit_app.py     563L  UI layer (renders, wires callbacks)
├── pipeline/
│   ├── config.py              25L  ★ All config constants
│   ├── schema.py             107L  ★ All Pydantic data models
│   ├── orchestrator.py       278L  Pipeline runner (framework-agnostic)
│   ├── model_manager.py      132L  VRAM swap (LLM ↔ Embedding)
│   ├── cache.py              210L  SQLite + FAISS persistence
│   ├── discovery/
│   │   ├── search.py         224L  Tavily search + LLM ranking
│   │   ├── browser.py        202L  httpx + Playwright PDF download
│   │   └── validate.py       187L  LLM-based PDF validation
│   ├── parsing/
│   │   ├── parse.py          171L  Docling PDF → Markdown
│   │   ├── sections.py       245L  Regex + semantic section split
│   │   └── chunking.py       116L  Page-aware text chunking
│   ├── extraction/
│   │   ├── extract.py        370L  Agentic financial metric extraction
│   │   └── insights.py       150L  LLM insight generation
│   └── rag/
│       ├── index.py          186L  FAISS embed/index/search
│       └── chat.py           291L  Tool-calling chat agent
├── export/
│   ├── to_markdown.py        107L  Markdown report export
│   └── to_pdf.py             189L  PDF report export (fpdf2)
└── tests/                    195L  pytest suite
```
