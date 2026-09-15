from pipeline.parsing.chunking import DocumentChunker
from pipeline.parsing.parse import ParsedReport

def test_chunking_engine_basic(mock_parsed_report):
    engine = DocumentChunker(chunk_size=50, overlap=10)
    chunks = engine.chunk(mock_parsed_report)
    
    assert len(chunks) > 0
    # The first chunk should have a page number mapped
    assert chunks[0].page_numbers is not None

def test_chunking_page_markers():
    engine = DocumentChunker()
    text = "<!-- PAGE 1 -->\nHello world\n<!-- PAGE 2 -->\nFoo bar"
    from pipeline.parsing.parse import ParsedReport, PageContent
    pages = [
        PageContent(page_number=1, text="Hello world"),
        PageContent(page_number=2, text="Foo bar")
    ]
    report = ParsedReport(pages=pages, full_text=text, total_pages=2)
    
    chunks = engine.chunk(report)
    
    # Check that chunks correctly captured the page numbers
    page_1_chunks = [c for c in chunks if 1 in c.page_numbers]
    page_2_chunks = [c for c in chunks if 2 in c.page_numbers]
    
    assert len(page_1_chunks) >= 1
    assert len(page_2_chunks) >= 1
    assert "Hello world" in page_1_chunks[0].text
    assert "Foo bar" in page_2_chunks[0].text
