import logging
import json
from datetime import datetime
import openai
from pydantic import ValidationError

from pipeline.schema import InsightData, Insight
from pipeline.parsing.sections import SectionContent
from pipeline.config import LLM_BASE_URL as DEFAULT_LLM_URL, LLM_MODEL

logger = logging.getLogger(__name__)

_INSIGHT_SYSTEM_PROMPT = """You are a financial analyst expert in extracting business insights from annual reports. Output only valid JSON.
Example output:
{
  "insights": [
    {
      "category": "Sector / industry classification",
      "content": "The company operates primarily in the IT services and consulting sector.",
      "page_numbers": [12, 14]
    }
  ]
}"""



class InsightGenerator:
    """Generates narrative insights using LLM."""
    
    def __init__(self, llm_base_url: str = DEFAULT_LLM_URL):
        self.client = openai.AsyncOpenAI(base_url=f"{llm_base_url}/v1", api_key="not-needed")
        
    async def generate(self, sections: dict[str, SectionContent], company_name: str, fiscal_year: str, financial_data: Any = None) -> InsightData:
        """Main generation method."""
        target_sections = ["MD&A", "Directors' Report", "Risk Management", "Outlook & Guidance", "Chairman/CEO Letter"]
        categories = [
            "Sector / industry classification",
            "Revenue model",
            "Segment/Business Breakdown",
            "Growth drivers",
            "Key risks",
            "CEO Sentiment & Outlook"
        ]
        
        combined_text = ""
        for sec_name in target_sections:
            if sec_name in sections:
                combined_text += f"\n\n--- {sec_name} ---\n{sections[sec_name].text}"
                
        # TRUNCATE to avoid Context Window Overflow (limit to ~20k words / ~28k tokens)
        words = combined_text.split()
        if len(words) > 20000:
            logger.warning(f"Combined sections are extremely long ({len(words)} words). Truncating to fit LLM context window.")
            combined_text = " ".join(words[:20000]) + "\n\n...[TRUNCATED FOR LENGTH]..."
                
        if not combined_text.strip():
            logger.warning("No narrative sections found (MD&A, Directors' Report, etc.) for insights generation.")
            return InsightData(
                company_name=company_name,
                fiscal_year=fiscal_year,
                insights=[],
                extraction_timestamp=datetime.utcnow()
            )
            
        financial_context = "No extracted financial data provided."
        if financial_data and hasattr(financial_data, "metrics"):
            fin_lines = []
            for m in financial_data.metrics:
                val = m.current_year_consolidated or m.current_year_standalone
                if val is not None:
                    fin_lines.append(f"- {m.name}: {val:,.2f} {m.unit}")
            if fin_lines:
                financial_context = "Extracted Financial Numbers for Context:\n" + "\n".join(fin_lines)
            
        all_insights = []
        for category in categories:
            try:
                insights = await self._generate_category(category, combined_text, company_name, financial_context)
                all_insights.extend(insights)
            except Exception as e:
                logger.exception(f"Failed to generate insights for category: {category}")
                
        return InsightData(
            company_name=company_name,
            fiscal_year=fiscal_year,
            insights=all_insights,
            extraction_timestamp=datetime.utcnow()
        )
        
    async def _generate_category(self, category: str, section_text: str, company_name: str, financial_context: str) -> list[Insight]:
        prompt = self._build_insight_prompt(category, section_text, company_name, financial_context)
        
        schema = {
            "type": "object",
            "properties": {
                "insights": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {"type": "string"},
                            "content": {"type": "string"},
                            "page_numbers": {
                                "type": "array",
                                "items": {"type": "integer"}
                            }
                        },
                        "required": ["category", "content", "page_numbers"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["insights"],
            "additionalProperties": False
        }
        
        try:
            logger.info(f"Generating insights for {company_name} - Category: {category}")
            response = await self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": _INSIGHT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_schema", "json_schema": {"name": "insight_generation", "schema": schema, "strict": True}},
                temperature=0.3,
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
            
            insights = []
            for item in data.get("insights", []):
                try:
                    insight = Insight(**item)
                    insights.append(insight)
                except ValidationError as ve:
                    logger.exception("Validation error for generated insight")
            return insights
            
        except Exception as e:
            logger.exception("Error during LLM insight generation call")
            raise e
            
    def _build_insight_prompt(self, category: str, section_text: str, company_name: str, financial_context: str) -> str:
        return f"""
Analyze the provided text from the annual report of {company_name} and generate insights for the following category: {category}.

{financial_context}

Rules:
1. Generate specific and factual insights, avoiding generic statements. Use the Extracted Financial Numbers above to back up your claims with hard data when relevant.
2. Cite page numbers for every claim in the text as [p.N], where N is the page number. Also include these in the page_numbers array.
3. If information for this category is not available in the text, say so explicitly rather than guessing, and return an empty list of insights.
4. Keep the insights clear, concise, and professional.

Text to analyze:
{section_text}
"""
