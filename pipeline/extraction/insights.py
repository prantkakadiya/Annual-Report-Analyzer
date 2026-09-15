import logging
import json
from datetime import datetime
import openai
from pydantic import ValidationError

from pipeline.schema import InsightData, Insight
from pipeline.parsing.sections import SectionContent

logger = logging.getLogger(__name__)

__version__ = "1.0.0"

class InsightGenerator:
    """Generates narrative insights using LLM."""
    
    def __init__(self, llm_base_url: str = "http://localhost:8080"):
        self.client = openai.AsyncOpenAI(base_url=f"{llm_base_url}/v1", api_key="not-needed")
        
    async def generate(self, sections: dict[str, SectionContent], company_name: str, fiscal_year: str) -> InsightData:
        """Main generation method."""
        target_sections = ["MD&A", "Directors' Report"]
        categories = [
            "Sector / industry classification",
            "Revenue model",
            "Key business segments",
            "Growth drivers",
            "Key risks"
        ]
        
        combined_text = ""
        for sec_name in target_sections:
            if sec_name in sections:
                combined_text += f"\n\n--- {sec_name} ---\n{sections[sec_name].text}"
                
        if not combined_text.strip():
            logger.warning("No MD&A or Directors' Report found for insights generation.")
            return InsightData(
                company_name=company_name,
                fiscal_year=fiscal_year,
                insights=[],
                extraction_timestamp=datetime.utcnow()
            )
            
        all_insights = []
        for category in categories:
            try:
                insights = await self._generate_category(category, combined_text, company_name)
                all_insights.extend(insights)
            except Exception as e:
                logger.exception(f"Failed to generate insights for category: {category}")
                
        return InsightData(
            company_name=company_name,
            fiscal_year=fiscal_year,
            insights=all_insights,
            extraction_timestamp=datetime.utcnow()
        )
        
    async def _generate_category(self, category: str, section_text: str, company_name: str) -> list[Insight]:
        prompt = self._build_insight_prompt(category, section_text, company_name)
        
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
        
        system_prompt = """You are a financial analyst expert in extracting business insights from annual reports. Output only valid JSON.
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

        try:
            logger.info(f"Generating insights for {company_name} - Category: {category}")
            response = await self.client.chat.completions.create(
                model="qwen",
                messages=[
                    {"role": "system", "content": system_prompt},
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
            
    def _build_insight_prompt(self, category: str, section_text: str, company_name: str) -> str:
        return f"""
Analyze the provided text from the annual report of {company_name} and generate insights for the following category: {category}.

Rules:
1. Generate specific and factual insights, avoiding generic statements.
2. Cite page numbers for every claim in the text as [p.N], where N is the page number. Also include these in the page_numbers array.
3. If information for this category is not available in the text, say so explicitly rather than guessing, and return an empty list of insights.
4. Keep the insights clear, concise, and professional.

Text to analyze:
{section_text}
"""
