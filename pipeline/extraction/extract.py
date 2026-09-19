import logging
import json
from datetime import datetime
from typing import Callable
import openai
from pydantic import ValidationError

from pipeline.schema import FinancialData, FinancialMetric
from pipeline.parsing.sections import SectionContent
from pipeline.model_manager import model_manager
from pipeline.config import LLM_BASE_URL as DEFAULT_LLM_URL, LLM_MODEL

logger = logging.getLogger(__name__)

_EXTRACTION_SYSTEM_PROMPT = """You are a financial-document extraction agent.
You are given a text representation of a financial report. The document has numbered lines.
Your job is to extract ONE PRIMARY METRIC requested by the controller.
You have access to document-search and document-reading tools. You do not have access to the internet and must rely only on the supplied document.

EVIDENCE
Only submit a metric when the document provides sufficient evidence.
Verify, whenever possible: metric name, reporting period, values, unit, and consolidated context.
Do not guess. Do not calculate a value unless explicitly required.
Extract values exactly as shown (do not convert units).
Extract CONSOLIDATED data only unless specified otherwise.

SEARCHING & READING
- Start with likely exact terms and useful aliases (e.g., for Net Profit, search ['Net profit', 'Profit for the year', 'PAT']).
- Use the smallest number of searches necessary.
- Use `read_lines` when the search result does not contain enough context (max 250 lines).

SUBMISSION
- Use `submit_metric` to submit findings. Include exact line ranges for evidence provenance.
- Do not waste tool calls searching for page numbers.

BEHAVIOR
Be focused and economical. Do not explore the document generally.
Do not produce a final natural-language answer. Continue using tools until the primary metric is submitted or the search is exhausted.
FIND → VERIFY → SUBMIT THE PRIMARY TARGET → STOP."""

_EXTRACTION_USER_TEMPLATE = """Company: {company_name}
Fiscal year: {fiscal_year}

PRIMARY TARGET:
{primary_metric}

Your highest-priority objective is to find reliable evidence for this metric and submit it as submission_type: "primary".
The task is complete as soon as the PRIMARY TARGET has been successfully submitted.

PRIORITY:
1. Find the PRIMARY TARGET.
2. While searching for the PRIMARY TARGET, opportunistically submit other requested metrics if they appear in the same evidence you are already examining (as submission_type: "secondary").
3. Never search specifically for secondary metrics.
4. Never continue searching after the PRIMARY TARGET has been successfully submitted.

WHAT COUNTS AS AN OPPORTUNISTIC DISCOVERY:
If the PRIMARY TARGET is Revenue, and you find a table containing Revenue, EBITDA, and Profit After Tax, you may submit all three because they were found in the same evidence.
However, if Revenue is not found, you must NOT start searching separately for EBITDA or Profit After Tax.

Other metrics that may be submitted ONLY if naturally encountered while completing the primary target:
{missing_text}

Begin by searching the document for the primary target."""



PRIMARY_ALIASES = {
    "Revenue": {"revenue", "revenue from operations", "total income", "gross revenue"},
    "EBITDA": {"ebitda", "adjusted ebitda", "operating profit"},
    "Net Profit": {"net profit", "profit for the year", "pat", "profit after tax", "loss for the year", "loss after tax"},
    "EPS": {"eps", "earnings per share", "basic and diluted earnings per share"},
    "Total Assets": {"total assets", "assets"},
    "Total Liabilities": {"total liabilities", "liabilities"},
    "Debt/Equity ratio": {"debt/equity ratio", "debt to equity", "debt equity ratio"},
    "Cash Flow from Operations": {"cash flow from operations", "net cash from operating activities", "operating cash flow"},
    "ROE": {"roe", "return on equity", "return on net worth"},
    "ROCE": {"roce", "return on capital employed"}
}

def is_primary_submission(name: str, primary_metric: str) -> bool:
    primary = primary_metric.split("(")[0].strip()
    aliases = PRIMARY_ALIASES.get(primary, {primary.lower()})
    return name.lower().strip() in aliases

class FinancialExtractor:
    """Agentic Financial Extractor using Section Slicing and Tool Calling."""
    
    def __init__(self, llm_base_url: str = DEFAULT_LLM_URL):
        self.client = openai.AsyncOpenAI(base_url=f"{llm_base_url}/v1", api_key="not-needed")
        
    async def extract(self, sections: dict[str, SectionContent] | list[str], company_name: str, fiscal_year: str, log_callback: Callable[[str], None] | None = None) -> FinancialData:
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

        while metrics_to_find:
            primary_metric = metrics_to_find[0]
            other_metrics = metrics_to_find[1:]
            
            logger.info(f"Primary Target: {primary_metric}. Other missing: {len(other_metrics)}")
            if log_callback: log_callback(f"🎯 Targeting: {primary_metric.split('(')[0].strip()}")
            
            newly_found = await self._agentic_find_metric(primary_metric, other_metrics, full_lines, company_name, fiscal_year, log_callback)
            
            for fm in newly_found:
                all_metrics[fm.name] = fm
                # Remove newly found metric from our checklist
                metrics_to_find = [m for m in metrics_to_find if not is_primary_submission(fm.name, m)]
                
            # If the primary metric STILL wasn't found (it failed or gave up), we pop it so we don't infinite loop.
            if metrics_to_find and is_primary_submission(metrics_to_find[0].split("(")[0].strip(), primary_metric):
                logger.warning(f"Could not extract {primary_metric}, giving up and moving to next.")
                metrics_to_find.pop(0)

        return FinancialData(
            company_name=company_name,
            fiscal_year=fiscal_year,
            metrics=list(all_metrics.values()),
            extraction_timestamp=datetime.utcnow()
        )
        
    async def _agentic_find_metric(self, primary_metric: str, other_metrics: list[str], full_lines: list[str], company_name: str, fiscal_year: str, log_callback: Callable[[str], None] | None = None) -> list[FinancialMetric]:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_lines",
                    "description": "Search the report for multiple keywords at once. Returns matching lines along with context (1-indexed line numbers).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keywords": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of exact keywords or short/full phrases to search for (e.g. ['Revenue', 'Total income'])."
                            }
                        },
                        "required": ["keywords"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_lines",
                    "description": "Read a specific range of lines from the document (1-indexed). Maximum range: 250 lines.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "start_line": {"type": "integer", "description": "1-indexed starting line number"},
                            "end_line": {"type": "integer", "description": "1-indexed ending line number"}
                        },
                        "required": ["start_line", "end_line"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "submit_metric",
                    "description": "Submit extracted data for a metric.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "submission_type": {
                                "type": "string",
                                "enum": ["primary", "secondary"],
                                "description": "Whether you are submitting the primary target or an opportunistic secondary discovery."
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
                            "evidence_start_line": {"type": "integer", "description": "The starting line number (1-indexed) of the evidence table/text."},
                            "evidence_end_line": {"type": "integer", "description": "The ending line number (1-indexed) of the evidence table/text."},
                            "notes": {"type": "string", "description": "Any context or notes."}
                        },
                        "required": ["submission_type", "metric_name", "evidence_start_line", "evidence_end_line"]
                    }
                }
            }
        ]

        missing_text = "\n".join([f"- {m}" for m in other_metrics])
        
        user_prompt = _EXTRACTION_USER_TEMPLATE.format(
            company_name=company_name,
            fiscal_year=fiscal_year,
            primary_metric=primary_metric,
            missing_text=missing_text
        )

        messages = [
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        extracted = []
        primary_found = False
        
        for turn in range(12):
            try:
                response = await self.client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=messages,
                    tools=tools,
                    temperature=0.0
                )
                
                msg = response.choices[0].message
                messages.append(msg)
                
                if msg.content:
                    if log_callback: log_callback(f"🧠 Thought: {msg.content.strip()}")
                
                if not msg.tool_calls:
                    if log_callback: log_callback("ℹ️ Agent stopped calling tools.")
                    logger.info("The model stopped calling tools.")
                    break
                    
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                        
                    if tc.function.name == "search_lines":
                        keywords = args.get("keywords", [])
                        if not keywords:
                            keywords = [args.get("keyword", "")]
                            
                        search_display = ", ".join(f"`{k}`" for k in keywords if k)
                        if log_callback: log_callback(f"🔎 Searching: {search_display}")
                        logger.info(f"Agent searching keywords: {keywords}")
                        
                        results = []
                        for kw in keywords:
                            if not kw: continue
                            for i, line in enumerate(full_lines):
                                if kw.lower() in line.lower():
                                    start = max(0, i - 8)
                                    end = min(len(full_lines), i + 9)
                                    # 1-indexed lines for LLM
                                    snippet_lines = [f"[Line {idx + 1}] {full_lines[idx]}" for idx in range(start, end)]
                                    snippet = "\n".join(snippet_lines)
                                    if not any(snippet in r for r in results):
                                        results.append(snippet)
                        
                        if not results:
                            tool_result = f"No matches found for {keywords}. Try different keywords."
                        else:
                            total_matches = len(results)
                            display_count = min(3, total_matches)
                            tool_result = f"Found {total_matches} matches. "
                            if total_matches > display_count:
                                tool_result += f"Showing Matches 1-{display_count}. There are {total_matches - display_count} additional matches not shown.\n\n"
                            else:
                                tool_result += f"Showing all matches:\n\n"
                                
                            tool_result += "\n\n---\n\n".join(results[:display_count])
                        
                        if log_callback: log_callback(f"📄 Tool Output:\n{tool_result}\n")
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": tool_result
                        })
                        
                    elif tc.function.name == "read_lines":
                        start_line = args.get("start_line", 1)
                        end_line = args.get("end_line", 1)
                        
                        if end_line - start_line > 250:
                            tool_result = "Error: Range too large. Maximum allowed range is 250 lines. Please request a smaller range."
                        else:
                            # Convert 1-indexed LLM requests to 0-indexed internal arrays
                            start_idx = max(0, start_line - 1)
                            end_idx = min(len(full_lines), end_line)
                            
                            if log_callback: log_callback(f"📖 Reading Lines: {start_idx + 1} to {end_idx}")
                            logger.info(f"Agent reading lines {start_idx + 1} to {end_idx}")
                            
                            if start_idx >= end_idx:
                                tool_result = "Invalid line range. start_line must be less than end_line."
                            else:
                                snippet_lines = [f"[Line {idx + 1}] {full_lines[idx]}" for idx in range(start_idx, end_idx)]
                                tool_result = "\n".join(snippet_lines)
                            
                        if log_callback: log_callback(f"📄 Tool Output:\n{tool_result}\n")
                            
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": tool_result
                        })
                        
                    elif tc.function.name == "submit_metric":
                        try:
                            sub_type = args.pop("submission_type", "secondary")
                            name = args.pop("metric_name", primary_metric.split("(")[0].strip())
                            args["name"] = name
                            extracted.append(FinancialMetric(**args))
                            
                            if log_callback: log_callback(f"✅ Extracted ({sub_type}): {name}")
                            logger.info(f"✅ Agent dynamically submitted metric: {name}")
                            
                            # Only break if the primary target was submitted explicitly
                            if sub_type == "primary" and is_primary_submission(name, primary_metric):
                                primary_found = True
                        except Exception as e:
                            logger.error(f"Failed to parse submitted metric: {e}")
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": "Metric submitted successfully."
                        })
                
                # We process ALL tool calls in the turn before stopping.
                if primary_found:
                    break
                    
            except Exception as e:
                logger.exception("Agent loop encountered an error")
                if log_callback: log_callback("❌ Error: Agent loop encountered an exception.")
                break
                
        return extracted
