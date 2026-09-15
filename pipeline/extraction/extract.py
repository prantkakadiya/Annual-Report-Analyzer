import logging
import json
from datetime import datetime
import openai
from pydantic import ValidationError

from pipeline.schema import FinancialData, FinancialMetric
from pipeline.parsing.sections import SectionContent

logger = logging.getLogger(__name__)

__version__ = "1.0.0"

class FinancialExtractor:
    """Extracts financial metrics using LLM."""
    
    def __init__(self, llm_base_url: str = "http://localhost:8080"):
        self.client = openai.AsyncOpenAI(base_url=f"{llm_base_url}/v1", api_key="not-needed")
        
    async def extract(self, sections: dict[str, SectionContent], company_name: str, fiscal_year: str) -> FinancialData:
        """Main extraction method across relevant sections."""
        target_sections = ["Balance Sheet", "Statement of Profit & Loss", "Cash Flow Statement", "Financial Highlights"]
        
        all_metrics = []
        for sec_name in target_sections:
            if sec_name in sections:
                logger.info(f"Extracting from section: {sec_name}")
                try:
                    section_text = sections[sec_name].text
                    metrics = await self._extract_from_section(section_text, sec_name, company_name)
                    all_metrics.extend(metrics)
                except Exception as e:
                    logger.exception(f"Failed to extract from section {sec_name}")
        
        return FinancialData(
            company_name=company_name,
            fiscal_year=fiscal_year,
            metrics=all_metrics,
            extraction_timestamp=datetime.utcnow()
        )
        
    async def _extract_from_section(self, section_text: str, section_name: str, company_name: str) -> list[FinancialMetric]:
        prompt = self._build_extraction_prompt(section_text, section_name, company_name)
        schema = self._get_json_schema()
        
        system_prompt = """You are a financial analyst expert in extracting structured data from annual reports. Output only valid JSON.
Example output:
{
  "metrics": [
    {
      "name": "Revenue",
      "current_year_consolidated": 150000.5,
      "current_year_standalone": null,
      "previous_year_consolidated": 135000.0,
      "previous_year_standalone": null,
      "unit": "INR Crore",
      "page_number": 45,
      "notes": "Revenue from operations"
    }
  ]
}"""

        try:
            logger.info(f"Attempting extraction for {company_name} - {section_name}")
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
            
    def _build_extraction_prompt(self, section_text: str, section_name: str, company_name: str) -> str:
        return f"""
Please extract the following financial metrics for {company_name} from the provided {section_name} section of their annual report:
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
2. Include page numbers for EVERY value based on any page markers in the text.
3. Distinguish between consolidated and standalone figures. If both are available, extract both.
4. Extract both current year and previous year values.
5. If a ratio is reported directly, use that. Otherwise, note in the 'notes' field that it should be computed.
6. The unit must specify the scale and currency (e.g. INR Crore, INR Lakh, USD Million).

Text to extract from:
{section_text}
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
