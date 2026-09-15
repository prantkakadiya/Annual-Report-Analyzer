import os
from pipeline.cache import ReportCache
from pipeline.schema import AnalysisResult, ReportMetadata

def test_cache_save_and_load(temp_data_dir):
    # Initialize cache with temporary directory
    cache = ReportCache(data_dir=temp_data_dir)
    
    # Dummy data
    company = "TestCorp"
    year = "2024"
    from datetime import datetime
    result = AnalysisResult(
        metadata=ReportMetadata(
            company_name=company,
            fiscal_year=year,
            pdf_path="test.pdf",
            discovered_at=datetime.now(),
            total_pages=5
        ),
        sections_found={"Financials": True}
    )
    
    # Save to cache
    cache.save_report(result)
    
    # Load from cache
    loaded = cache.get_report(company, year)
    assert loaded is not None
    assert loaded.metadata.company_name == company
    assert loaded.metadata.fiscal_year == year
    assert loaded.sections_found == {"Financials": True}
    
    # Test miss
    miss = cache.get_report("Unknown", "2024")
    assert miss is None
