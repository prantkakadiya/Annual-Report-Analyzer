import os
import tempfile
from fpdf import FPDF
from pipeline.parsing.parse import ReportParser

def create_dummy_pdf(path: str):
    """Generate a simple 2-page PDF for testing."""
    pdf = FPDF()
    
    # Page 1
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(200, 10, text="Annual Report 2024", new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.cell(200, 10, text="Welcome to the test report.", new_x="LMARGIN", new_y="NEXT")
    
    # Page 2
    pdf.add_page()
    pdf.cell(200, 10, text="Financial Overview", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(200, 10, text="Revenue: $100M", new_x="LMARGIN", new_y="NEXT")
    
    pdf.output(path)

def test_pdf_parsing():
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        # Create a real, albeit tiny, PDF
        create_dummy_pdf(tmp_path)
        
        # Parse it using the pipeline parser
        parser = ReportParser()
        parsed = parser.parse(tmp_path)
        
        # Verify structure
        assert parsed.total_pages == 2
        assert len(parsed.pages) == 2
        
        # Verify per-page extraction
        assert parsed.pages[0].page_number == 1
        assert "Annual Report 2024" in parsed.pages[0].text
        
        assert parsed.pages[1].page_number == 2
        assert "Revenue: $100M" in parsed.pages[1].text
        
        # Verify full-text compilation with page markers
        assert "<!-- PAGE 1 -->" in parsed.full_text
        assert "<!-- PAGE 2 -->" in parsed.full_text
        assert "Welcome to the test report." in parsed.full_text
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_pdf_pars():
    parser = ReportParser()
    parsed = parser.parse("/home/prant/Desktop/python/job/Annual Report Analyzer/downloads/Annual-Report-FY-2023-24-1.pdf")
    print(parsed)
    


test_pdf_pars()