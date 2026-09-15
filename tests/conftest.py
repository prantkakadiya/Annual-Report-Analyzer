import os
import tempfile
import pytest

@pytest.fixture
def temp_data_dir():
    """Provides a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def mock_parsed_report():
    """Returns a mock ParsedReport for parsing/chunking tests."""
    from pipeline.parsing.parse import ParsedReport, PageContent
    
    pages = [
        PageContent(page_number=1, text="Test Company Annual Report 2024."),
        PageContent(page_number=2, text="Financial Overview: Revenue was 100M.")
    ]
    
    full_text = "<!-- PAGE 1 -->\nTest Company Annual Report 2024.\n<!-- PAGE 2 -->\nFinancial Overview: Revenue was 100M."
    
    return ParsedReport(
        pages=pages,
        total_pages=2,
        full_text=full_text
    )
