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
            searcher = ReportSearcher(llm_base_url=LLM_BASE_URL)
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

