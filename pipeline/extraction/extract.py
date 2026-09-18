import logging
import json
from datetime import datetime
import openai
from pydantic import ValidationError

from pipeline.schema import FinancialData, FinancialMetric
from pipeline.parsing.sections import SectionContent
from pipeline.model_manager import model_manager

logger = logging.getLogger(__name__)

__version__ = "2.0.0"

class FinancialExtractor:
    """Extracts financial metrics using Vector Search (RAG) and LLM."""
    
    def __init__(self, llm_base_url: str = "http://localhost:8080"):
        self.client = openai.AsyncOpenAI(base_url=f"{llm_base_url}/v1", api_key="not-needed")
        
    async def extract(self, rag_engine, company_name: str, fiscal_year: str) -> FinancialData:
        """Main extraction method using Vector Search (RAG)."""
        logger.info(f"Extracting financial metrics via RAG for {company_name}")
        
        queries = [
            f"What is the total revenue from operations and total income for {company_name} {fiscal_year}?",
            f"What is the EBITDA (Earnings Before Interest, Taxes, Depreciation, and Amortization) and EBITDA margin for {company_name} {fiscal_year}?",
            f"What is the Net Profit, PAT (Profit After Tax), or profit for the year attributable to owners for {company_name} {fiscal_year}?",
            f"What is the Basic and Diluted Earnings Per Share (EPS) for {company_name} {fiscal_year}?",
            f"What are the Total Assets reported in the Balance Sheet for {company_name} {fiscal_year}?",
            f"What are the Total Liabilities reported in the Balance Sheet for {company_name} {fiscal_year}?",
            f"What is the Debt to Equity ratio or debt-equity ratio for {company_name} {fiscal_year}?",
            f"What is the Net Cash generated from operating activities or Cash Flow from Operations for {company_name} {fiscal_year}?",
            f"What is the Return on Equity (ROE) or return on net worth for {company_name} {fiscal_year}?",
            f"What is the Return on Capital Employed (ROCE) for {company_name} {fiscal_year}?"
        ]

        model_manager.require_embedding()

        unique_chunks = {}
        for q in queries:
            try:
                # Limit to 3 documents per query
                results = await rag_engine.search(q, top_k=3)
                for chunk, score in results:
                    if chunk.chunk_id not in unique_chunks:
                        unique_chunks[chunk.chunk_id] = chunk
            except Exception as e:
                logger.exception(f"RAG search failed for query: {q}")

        combined_text = ""
        for i, chunk in enumerate(unique_chunks.values(), 1):
            pages = ", ".join(str(p) for p in chunk.page_numbers)
            combined_text += f"\n--- Excerpt {i} (Pages: {pages}) ---\n{chunk.text}\n"

        if not combined_text.strip():
            logger.warning("No relevant chunks found for financial metrics.")
            return FinancialData(
                company_name=company_name,
                fiscal_year=fiscal_year,
                metrics=[],
                extraction_timestamp=datetime.utcnow()
            )

        model_manager.require_llm()
        metrics = await self._extract_from_text(combined_text, company_name, fiscal_year)
        
        return FinancialData(
            company_name=company_name,
            fiscal_year=fiscal_year,
            metrics=metrics,
            extraction_timestamp=datetime.utcnow()
        )
        
    async def _extract_from_text(self, text_context: str, company_name: str, fiscal_year: str) -> list[FinancialMetric]:
        prompt = self._build_extraction_prompt(text_context, company_name, fiscal_year)
        schema = self._get_json_schema()
        
        system_prompt = """You are a meticulous financial auditor extracting quantitative data from a pool of fragmented document excerpts.
Your task is to populate a strict JSON structure.
IMPORTANT RULES:
1. ONLY use the provided excerpts. Do not hallucinate or use outside knowledge.
2. If a specific metric is completely missing from the excerpts, set its values to `null`.
3. Distinguish carefully between 'consolidated' and 'standalone' figures.
4. If a number is given in millions or billions, do not convert the raw number, just specify the unit exactly as written (e.g. 'USD Million', 'INR Crore').
5. Output ONLY valid JSON matching the schema."""

        try:
            logger.info("Attempting final LLM extraction from pooled RAG context")
            response = await self.client.chat.completions.create(
                model="qwen",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_schema", "json_schema": {"name": "financial_extraction", "schema": schema, "strict": True}},
                temperature=0.0,
                max_tokens=3000,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
            
            usage = response.usage
            if usage:
                logger.info(f"Tokens used - prompt: {usage.prompt_tokens}, completion: {usage.completion_tokens}, total: {usage.total_tokens}")
            
            content = response.choices[0].message.content
            if not content:
                logger.error("Empty response from LLM")
                return []
                
            data = json.loads(content)
            
            metrics = []
            for item in data.get("metrics", []):
                try:
                    metric = FinancialMetric(**item)
                    metrics.append(metric)
                except ValidationError as ve:
                    logger.exception("Validation error for extracted metric")
            return metrics
            
        except Exception as e:
            logger.exception("Error during LLM extraction call")
            raise e
            
    def _build_extraction_prompt(self, text_context: str, company_name: str, fiscal_year: str) -> str:
        return f"""
Please extract the following financial metrics for {company_name} ({fiscal_year}) from the provided excerpts:
- Revenue
- EBITDA and EBITDA margin
- Net Profit / PAT
- EPS (Basic and Diluted)
- Total Assets
- Total Liabilities
- Debt/Equity ratio
- Cash Flow from Operations
- ROE
- ROCE

Rules:
1. Extract EXACT numbers as reported. Do not convert units.
2. Include page numbers for EVERY value based on the page markers in the text (e.g., Pages: 45).
3. Distinguish between consolidated and standalone figures.
4. Extract both current year and previous year values.
5. If a ratio is reported directly, use that. Otherwise, note in the 'notes' field that it should be computed.

Excerpts:
{text_context}
"""

    def _get_json_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "metrics": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "current_year_consolidated": {"type": ["number", "null"]},
                            "current_year_standalone": {"type": ["number", "null"]},
                            "previous_year_consolidated": {"type": ["number", "null"]},
                            "previous_year_standalone": {"type": ["number", "null"]},
                            "unit": {"type": "string"},
                            "page_number": {"type": ["integer", "null"]},
                            "notes": {"type": ["string", "null"]}
                        },
                        "required": ["name", "unit", "current_year_consolidated", "current_year_standalone", "previous_year_consolidated", "previous_year_standalone", "page_number", "notes"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["metrics"],
            "additionalProperties": False
        }
