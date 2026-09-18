import logging
import json
from datetime import datetime
import openai
from pydantic import ValidationError

from pipeline.schema import FinancialData, FinancialMetric
from pipeline.parsing.sections import SectionContent
from pipeline.model_manager import model_manager

logger = logging.getLogger(__name__)

__version__ = "3.0.0"

class FinancialExtractor:
    """Agentic Financial Extractor using Section Slicing and Tool Calling."""
    
    def __init__(self, llm_base_url: str = "http://localhost:8080"):
        self.client = openai.AsyncOpenAI(base_url=f"{llm_base_url}/v1", api_key="not-needed")
        
    async def extract(self, sections: dict[str, SectionContent] | list[str], company_name: str, fiscal_year: str) -> FinancialData:
        """Main extraction method using Agentic loop per metric."""
        logger.info(f"Starting Agentic Line-Search Extraction for {company_name}")
        
        full_lines = []
        if isinstance(sections, dict):
            # Only use targeted financial sections if we want strict section slicing
            target_sections = ["Balance Sheet", "Statement of Profit & Loss", "Cash Flow Statement", "Financial Highlights", "MD&A"]
            for sec_name in target_sections:
                if sec_name in sections:
                    full_lines.append(f"--- START OF SECTION: {sec_name} ---")
                    for line in sections[sec_name].text.split('\n'):
                        full_lines.append(line.strip())
                    full_lines.append(f"--- END OF SECTION: {sec_name} ---")
            
            # If no targeted sections found, fallback to ALL sections
            if not full_lines:
                for sec_name, content in sections.items():
                    for line in content.text.split('\n'):
                        full_lines.append(line.strip())
        else:
            full_lines = sections

        metrics_to_find = [
            "Revenue (or Total Income, Revenue from Operations)",
            "EBITDA (Earnings Before Interest, Taxes, Depreciation, and Amortization) and EBITDA margin",
            "Net Profit (or PAT, Profit for the year, Profit After Tax, Loss for the year)",
            "Basic and Diluted EPS (Earnings Per Share)",
            "Total Assets",
            "Total Liabilities",
            "Debt/Equity ratio",
            "Cash Flow from Operations (or Net Cash from Operating Activities)",
            "ROE (Return on Equity, return on net worth)",
            "ROCE (Return on Capital Employed)"
        ]

        all_metrics = []
        
        # Ensure LLM is running for agent loop
        model_manager.require_llm()

        for metric_desc in metrics_to_find:
            metric_result = await self._agentic_find_metric(metric_desc, full_lines, company_name, fiscal_year)
            if metric_result:
                all_metrics.append(metric_result)

        return FinancialData(
            company_name=company_name,
            fiscal_year=fiscal_year,
            metrics=all_metrics,
            extraction_timestamp=datetime.utcnow()
        )
        
    async def _agentic_find_metric(self, metric_desc: str, full_lines: list[str], company_name: str, fiscal_year: str) -> FinancialMetric | None:
        logger.info(f"Agent actively searching for metric: {metric_desc}")
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_lines",
                    "description": "Search the report sections for a keyword. Returns matching lines along with 8 lines of context above and below.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keyword": {
                                "type": "string",
                                "description": "The exact keyword or short/full phrase to search for."
                            }
                        },
                        "required": ["keyword"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "submit_metric",
                    "description": "Submit the extracted data once you have confidently found it.",
                    "parameters": self._get_json_schema()["properties"]["metrics"]["items"]
                }
            }
        ]

        messages = [
            {
                "role": "system",
                "content": (
                    f"You are an autonomous financial auditor. Your specific goal right now is to extract this exact metric for {company_name} ({fiscal_year}):\n"
                    f"**{metric_desc}**\n\n"
                    "RULES:\n"
                    "1. Use the `search_lines` tool repeatedly to explore the document. Try short forms (e.g., 'ROE', 'PAT') and full forms (e.g., 'Return on Equity').\n"
                    "2. When you find the exact numbers in the search results, use the `submit_metric` tool to submit the data and exit.\n"
                    "3. If you search 3-4 times and still cannot find the metric, use `submit_metric` with null values to give up.\n"
                    "4. Output must distinguish between consolidated and standalone if specified.\n"
                    "5. Do NOT convert units. If it says '` Million', write that exactly as the unit."
                )
            }
        ]

        extracted_metric = None
        
        # Give the agent a maximum of 5 turns to search and submit
        for turn in range(5):
            try:
                response = await self.client.chat.completions.create(
                    model="qwen",
                    messages=messages,
                    tools=tools,
                    temperature=0.0
                )
                
                msg = response.choices[0].message
                messages.append(msg)
                
                if not msg.tool_calls:
                    # If it didn't call a tool, force a break
                    break
                    
                for tc in msg.tool_calls:
                    if tc.function.name == "search_lines":
                        args = json.loads(tc.function.arguments)
                        keyword = args.get("keyword", "").strip()
                        
                        logger.info(f"Agent exploring keyword: '{keyword}'")
                        
                        results = []
                        if keyword:
                            for i, line in enumerate(full_lines):
                                if keyword.lower() in line.lower():
                                    start = max(0, i - 8)
                                    end = min(len(full_lines), i + 9)
                                    snippet = "\n".join(full_lines[start:end])
                                    
                                    # Prevent duplicate snippets
                                    if not any(snippet in r for r in results):
                                        results.append(snippet)
                        
                        if not results:
                            tool_result = f"No matches found for '{keyword}'. Try a different short form or full form."
                        else:
                            # Limit to top 3 matches to avoid blowing up context window
                            tool_result = f"Found {len(results)} matches. Showing top 3:\n\n" + "\n\n---\n\n".join(results[:3])
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": tool_result
                        })
                        
                    elif tc.function.name == "submit_metric":
                        args = json.loads(tc.function.arguments)
                        try:
                            # Clean up the name to avoid passing the giant description string
                            args["name"] = metric_desc.split("(")[0].strip()
                            extracted_metric = FinancialMetric(**args)
                        except Exception as e:
                            logger.error(f"Failed to parse submitted metric: {e}")
                        
                        return extracted_metric
            except Exception as e:
                logger.exception("Agent loop encountered an error")
                break
                
        return extracted_metric

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
