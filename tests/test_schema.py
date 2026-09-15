from pipeline.schema import FinancialMetric, Insight, AnalysisResult, ReportMetadata
from pydantic import ValidationError
import pytest

def test_financial_metric_validation():
    # Valid metric
    metric = FinancialMetric(
        name="Revenue",
        current_year_consolidated=100.5,
        unit="USD Millions",
        page_number=10
    )
    assert metric.name == "Revenue"
    assert metric.current_year_consolidated == 100.5
    
    # Missing required field
    with pytest.raises(ValidationError):
        FinancialMetric(
            current_year_consolidated=100.5,
            unit="USD Millions",
            page_number=10
        )

def test_insight_validation():
    insight = Insight(
        category="Risk",
        content="Market volatility increased.",
        page_numbers=[5, 6]
    )
    assert insight.category == "Risk"
    assert len(insight.page_numbers) == 2

def test_analysis_result_structure():
    from datetime import datetime
    meta = ReportMetadata(
        company_name="Test Co",
        fiscal_year="2024",
        pdf_path="test.pdf",
        discovered_at=datetime.now(),
        total_pages=10
    )
    result = AnalysisResult(
        metadata=meta,
        sections_found={"Financials": True}
    )
    assert result.metadata.company_name == "Test Co"
    assert result.financial_data is None
    assert result.insights is None
