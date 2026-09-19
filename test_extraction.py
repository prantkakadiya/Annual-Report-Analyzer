import asyncio
import logging

# Ensure logging is printed to terminal
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

from pipeline.cache import ReportCache
from pipeline.rag.rag import RAGEngine
from pipeline.extraction.extract import FinancialExtractor

async def run_test():
    company_name = "Swiggy"
    fiscal_year = "2024-25"
    company_id = "swiggy_2024_25"

    print("Initializing Cache and RAG Engine...")
    cache = ReportCache("data")
    rag = RAGEngine(
        embedding_base_url="http://localhost:8081",
        llm_base_url="http://localhost:8080",
        cache=cache
    )
    
    print(f"Loading FAISS index for {company_id}...")
    success = rag.load_index(company_id)
    if not success:
        print("❌ Could not load FAISS index. Make sure you have analyzed this report at least once.")
        return
    print(f"✅ Loaded {len(rag.chunks)} chunks from FAISS cache.")

    print("\n🚀 Reconstructing lines from cache...")
    full_lines = []
    for chunk in rag.chunks:
        full_lines.extend(chunk.text.split('\n'))
        
    print(f"Reconstructed {len(full_lines)} lines.")
    
    print("\n🚀 Starting Financial Extraction (Agentic Search)...")
    extractor = FinancialExtractor(llm_base_url="http://localhost:8080")
    
    def console_logger(msg: str):
        # ANSI Color Codes for terminal
        if msg.startswith("🎯"):
            print(f"\n\033[1;93m{msg}\033[0m") # Bold Yellow
        elif msg.startswith("🧠"):
            print(f"\033[3;95m  {msg}\033[0m") # Italic Purple
        elif msg.startswith("🔎") or msg.startswith("📖"):
            print(f"\033[96m  {msg}\033[0m") # Cyan
        elif msg.startswith("✅"):
            print(f"\033[1;92m  {msg}\033[0m") # Bold Green
        elif msg.startswith("❌"):
            print(f"\033[1;91m  {msg}\033[0m") # Bold Red
        elif msg.startswith("📄"):
            print(f"\033[37m{msg}\033[0m") # White/Gray
        else:
            print(f"  {msg}")
            
    # Run the extractor
    financial_data = await extractor.extract(full_lines, company_name, fiscal_year, log_callback=console_logger)
    
    print(f"\n✅ Extraction Complete! Extracted {len(financial_data.metrics)} metrics:\n")
    for m in financial_data.metrics:
        print(f"--- {m.name} ---")
        print(f"  CY Consolidated: {m.current_year_consolidated}")
        print(f"  CY Standalone:   {m.current_year_standalone}")
        print(f"  PY Consolidated: {m.previous_year_consolidated}")
        print(f"  Unit:            {m.unit}")
        print(f"  Page:            {m.page_number}")
        print(f"  Notes:           {m.notes}\n")

if __name__ == "__main__":
    asyncio.run(run_test())
