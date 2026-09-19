import logging
import json
from datetime import datetime
from typing import Callable
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
        
    async def extract(self, sections: dict[str, SectionContent] | list[str], company_name: str, fiscal_year: str, log_callback: Callable[[str], None] = None) -> FinancialData:
        """Main extraction method using Agentic loop per metric."""
        logger.info(f"Starting Agentic Line-Search Extraction for {company_name}")
        if log_callback: log_callback("Starting Agentic Extraction...")
        
        full_lines = []
        if isinstance(sections, dict):
            target_sections = ["Balance Sheet", "Statement of Profit & Loss", "Cash Flow Statement", "Financial Highlights", "MD&A"]
            for sec_name in target_sections:
                if sec_name in sections:
                    full_lines.append(f"--- START OF SECTION: {sec_name} ---")
                    for line in sections[sec_name].text.split('\n'):
                        full_lines.append(line.strip())
                    full_lines.append(f"--- END OF SECTION: {sec_name} ---")
            
            if not full_lines:
                for sec_name, content in sections.items():
                    for line in content.text.split('\n'):
                        full_lines.append(line.strip())
        else:
            full_lines = sections

        metrics_to_find = [
            "Revenue (or Total Income, Revenue from Operations)",
            "EBITDA (Earnings Before Interest, Taxes, Depreciation, and Amortization)",
            "Net Profit (or PAT, Profit for the year, Profit After Tax, Loss for the year)",
            "EPS (Basic and Diluted Earnings Per Share)",
            "Total Assets",
            "Total Liabilities",
            "Debt/Equity ratio",
            "Cash Flow from Operations (or Net Cash from Operating Activities)",
            "ROE (Return on Equity, return on net worth)",
            "ROCE (Return on Capital Employed)"
        ]

        all_metrics = {}
        
        model_manager.require_llm()

        def _is_match(sub_name: str, full_desc: str) -> bool:
            return sub_name.lower().strip() in full_desc.lower()

        while metrics_to_find:
            primary_metric = metrics_to_find[0]
            other_metrics = metrics_to_find[1:]
            
            logger.info(f"Primary Target: {primary_metric}. Other missing: {len(other_metrics)}")
            if log_callback: log_callback(f"🎯 Targeting: {primary_metric.split('(')[0].strip()}")
            
            newly_found = await self._agentic_find_metric(primary_metric, other_metrics, full_lines, company_name, fiscal_year, log_callback)
            
            for fm in newly_found:
                all_metrics[fm.name] = fm
                # Remove newly found metric from our checklist
                metrics_to_find = [m for m in metrics_to_find if not _is_match(fm.name, m)]
                
            # If the primary metric STILL wasn't found (it failed or gave up), we pop it so we don't infinite loop.
            if metrics_to_find and _is_match(primary_metric.split("(")[0].strip(), metrics_to_find[0]):
                logger.warning(f"Could not extract {primary_metric}, giving up and moving to next.")
                metrics_to_find.pop(0)

        return FinancialData(
            company_name=company_name,
            fiscal_year=fiscal_year,
            metrics=list(all_metrics.values()),
            extraction_timestamp=datetime.utcnow()
        )
        
    async def _agentic_find_metric(self, primary_metric: str, other_metrics: list[str], full_lines: list[str], company_name: str, fiscal_year: str, log_callback: Callable[[str], None] = None) -> list[FinancialMetric]:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_lines",
                    "description": "Search the report sections for a keyword. Returns matching lines along with 8 lines of context above and below, including exact line numbers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "thought": {
                                "type": "string",
                                "description": "A brief 1-sentence thought explaining why you are searching for this keyword."
                            },
                            "keyword": {
                                "type": "string",
                                "description": "The exact keyword or short/full phrase to search for."
                            }
                        },
                        "required": ["thought", "keyword"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_lines",
                    "description": "Read a specific range of lines from the document if you need to see more context beyond the 8 lines provided by search_lines.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "thought": {
                                "type": "string",
                                "description": "A brief 1-sentence thought explaining why you need to read these specific lines."
                            },
                            "start_line": {"type": "integer"},
                            "end_line": {"type": "integer"}
                        },
                        "required": ["thought", "start_line", "end_line"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "submit_metric",
                    "description": "Submit extracted data. You can call this MULTIPLE times in a row if you find multiple metrics!",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "thought": {
                                "type": "string",
                                "description": "A brief 1-sentence thought explaining your confidence in this metric submission."
                            },
                            "metric_name": {
                                "type": "string", 
                                "description": "The exact standardized name of the metric.",
                                "enum": [
                                    "Revenue", "EBITDA", "Net Profit", "EPS", 
                                    "Total Assets", "Total Liabilities", 
                                    "Debt/Equity ratio", "Cash Flow from Operations", 
                                    "ROE", "ROCE"
                                ]
                            },
                            "current_year_consolidated": {"type": "string", "description": "Value for current year (e.g., FY 2024-25). Extract exact string including commas. Leave empty if not found."},
                            "previous_year_consolidated": {"type": "string", "description": "Value for previous year (e.g., FY 2023-24). Extract exact string including commas. Leave empty if not found."},
                            "unit": {"type": "string", "description": "The unit of measurement (e.g. '` Million', '%')."},
                            "page_number": {"type": "integer", "description": "Page number where found. Leave empty if you cannot immediately see it. Do NOT waste turns searching for this."},
                            "notes": {"type": "string", "description": "Any context or notes."}
                        },
                        "required": ["thought", "metric_name"]
                    }
                }
            }
        ]

        missing_text = "\n".join([f"- {m}" for m in other_metrics])
        messages = [
            {
                "role": "system",
                "content": "You are an autonomous financial auditor."
            },
            {
                "role": "user",
                "content": (
                    f"Your PRIMARY goal right now is to extract this exact metric for {company_name} ({fiscal_year}):\n"
                    f"**{primary_metric}**\n\n"
                    f"OPPORTUNISTIC EXTRACTION:\n"
                    f"While searching, if you happen to see the data for ANY of these other missing metrics, you MUST submit them too:\n"
                    f"{missing_text}\n\n"
                    "RULES:\n"
                    "1. Use `search_lines` to explore. Try short forms (e.g., 'ROE', 'PAT') and full forms.\n"
                    "2. If a table gets cut off, use `read_lines`.\n"
                    "3. MULTIPLE METRICS: Financial tables often group Revenue, EBITDA, and Profit together. If you see multiple metrics in the same table, you MUST submit ALL of them in a single response using parallel `submit_metric` calls!\n"
                    "4. IMMEDIATELY use `submit_metric` when you find numbers. Page numbers are OPTIONAL. Leave them empty if not immediately visible. NEVER waste turns searching for page numbers!\n"
                    "5. ONLY extract the main Consolidated data. DO NOT search for standalone data.\n"
                    "6. Do NOT convert units. Write exactly as seen (e.g. '` Million').\n"
                    "7. You MUST always call a tool. Do NOT reply with conversational text outside of a tool call."
                )
            }
        ]

        extracted = []
        primary_found = False
        
        for turn in range(10):
            try:
                response = await self.client.chat.completions.create(
                    model="qwen",
                    messages=messages,
                    tools=tools,
                    temperature=0.0
                )
                
                msg = response.choices[0].message
                messages.append(msg)
                
                if msg.content:
                    if log_callback: log_callback(f"🧠 Thought: {msg.content.strip()}")
                
                if not msg.tool_calls:
                    if log_callback: log_callback("❌ Error: Agent stopped calling tools and gave up early.")
                    logger.error("The model stopped calling tools and gave up early.")
                    break
                    
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    thought = args.pop("thought", None)
                    
                    if thought and log_callback:
                        log_callback(f"🧠 Thought: {thought}")
                        
                    if tc.function.name == "search_lines":
                        keyword = args.get("keyword", "").strip()
                        if log_callback: log_callback(f"🔎 Searching: `{keyword}`")
                        logger.info(f"Agent searching keyword: '{keyword}'")
                        
                        results = []
                        if keyword:
                            for i, line in enumerate(full_lines):
                                if keyword.lower() in line.lower():
                                    start = max(0, i - 8)
                                    end = min(len(full_lines), i + 9)
                                    snippet_lines = [f"[Line {idx}] {full_lines[idx]}" for idx in range(start, end)]
                                    snippet = "\n".join(snippet_lines)
                                    if not any(snippet in r for r in results):
                                        results.append(snippet)
                        
                        if not results:
                            tool_result = f"No matches found for '{keyword}'. Try a different keyword."
                        else:
                            tool_result = f"Found {len(results)} matches. Showing top 3:\n\n" + "\n\n---\n\n".join(results[:3])
                        
                        if log_callback: log_callback(f"📄 Tool Output:\n{tool_result}\n")
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": tool_result
                        })
                        
                    elif tc.function.name == "read_lines":
                        start_line = max(0, args.get("start_line", 0))
                        end_line = min(len(full_lines), args.get("end_line", 0) + 1)
                        if log_callback: log_callback(f"📖 Reading Lines: {start_line} to {end_line}")
                        logger.info(f"Agent reading lines {start_line} to {end_line}")
                        
                        if start_line >= end_line:
                            tool_result = "Invalid line range. start_line must be less than end_line."
                        else:
                            snippet_lines = [f"[Line {idx}] {full_lines[idx]}" for idx in range(start_line, end_line)]
                            tool_result = "\n".join(snippet_lines)
                            
                        if log_callback: log_callback(f"📄 Tool Output:\n{tool_result}\n")
                            
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": tool_result
                        })
                        
                    elif tc.function.name == "submit_metric":
                        try:
                            # Use the metric_name provided by the LLM
                            name = args.pop("metric_name", primary_metric.split("(")[0].strip())
                            args["name"] = name
                            extracted.append(FinancialMetric(**args))
                            
                            if log_callback: log_callback(f"✅ Extracted: {name}")
                            logger.info(f"✅ Agent dynamically submitted metric: {name}")
                            
                            # Check if the primary target was found
                            base_primary = primary_metric.split("(")[0].strip().lower()
                            if base_primary in name.lower() or name.lower() in base_primary:
                                primary_found = True
                        except Exception as e:
                            logger.error(f"Failed to parse submitted metric: {e}")
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": "Metric submitted successfully. You can continue searching or exit if finished."
                        })
                
                # If we submitted the primary goal, we can end this session early!
                if primary_found:
                    break
                    
            except Exception as e:
                logger.exception("Agent loop encountered an error")
                if log_callback: log_callback("❌ Error: Agent loop encountered an exception.")
                break
                
        return extracted

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
