import sys
import os
import asyncio
from pipeline.parsing.parse import ReportParser
from pipeline.parsing.sections import SectionSplitter
from pipeline.parsing.chunking import DocumentChunker

def run_manual_test(pdf_path: str):
    print(f"\n{'='*50}")
    print(f"🧪 MANUAL TEST: {os.path.basename(pdf_path)}")
    print(f"{'='*50}\n")
    
    # ---------------------------------------------------------
    # 1. PDF Parsing
    # ---------------------------------------------------------
    print("⏳ 1. PARSING PDF...")
    parser = ReportParser()
    parsed_report = parser.parse(pdf_path)
    
    print(f"  ✅ Successfully parsed {parsed_report.total_pages} pages!")
    if parsed_report.total_pages > 0:
        print("\n  📄 Preview of Page 1:")
        print("  " + "-"*40)
        preview_text = parsed_report.pages[0].text[:].replace('\n', '\n  ')
        print(f"  {preview_text}...\n")
    if parsed_report.total_pages > 0:
        print("\n  📄 Preview of Page 2:")
        print("  " + "-"*40)
        preview_text = parsed_report.pages[1].text[:].replace('\n', '\n  ')
        print(f"  {preview_text}...\n")
    if parsed_report.total_pages > 0:
        print("\n  📄 Preview of Page 4:")
        print("  " + "-"*40)
        preview_text = parsed_report.pages[3].text[:].replace('\n', '\n  ')
        print(f"  {preview_text}...\n")
    
    # ---------------------------------------------------------
    # 2. Section Splitting (Regex only for this test)
    # ---------------------------------------------------------
    print("⏳ 2. SPLITTING INTO SECTIONS...")
    splitter = SectionSplitter()
    # We use asyncio.run because split() is an async function (for semantic fallback)
    # We will pass rag_index=None to only test the fast regex splitting
    sections = asyncio.run(splitter.split(parsed_report, rag_index=None))
    
    print(f"  ✅ Found {len(sections)} sections using Regex:")
    for name, content in sections.items():
        print(f"     - {name} (Length: {len(content.text)} characters)")
    
    print()

    # ---------------------------------------------------------
    # 3. Document Chunking
    # ---------------------------------------------------------
    print("⏳ 3. CHUNKING DOCUMENT...")
    chunker = DocumentChunker(chunk_size=1024, overlap=200)
    chunks = chunker.chunk(parsed_report)
    
    print(f"  ✅ Sliced report into {len(chunks)} chunks!")
    if chunks:
        print("\n  🧩 Preview of Chunk #1:")
        print("  " + "-"*40)
        print(f"  Pages mapped to this chunk: {chunks[0].page_numbers}")
        chunk_preview = chunks[0].text[:300].replace('\n', '\n  ')
        print(f"  Text: {chunk_preview}...\n")

    print(f"{'='*50}")
    print("✅ MANUAL TEST COMPLETE")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python manual_test.py path/to/your/report.pdf")
        sys.exit(1)
        
    pdf_file = sys.argv[1]
    if not os.path.exists(pdf_file):
        print(f"Error: Could not find file '{pdf_file}'")
        sys.exit(1)
        
    run_manual_test(pdf_file)
