"""
Financial Extractor: Hybrid Architecture
Phase 1: Dynamic Table Ranking & Filtering
Phase 2: Agentic Search for missing metrics
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Callable, Any

import openai

from pipeline.model_manager import model_manager
from pipeline.schema import FinancialData, FinancialMetric
from pipeline.config import LLM_BASE_URL as DEFAULT_LLM_URL, LLM_MODEL
from pipeline.parsing.sections import SectionContent

logger = logging.getLogger(__name__)

# The global list of metrics we want to extract
TARGET_METRICS = [
    # Income Statement
    "Revenue (Total Income)",
    "Gross Profit",
    "Operating Income (EBIT)",
    "EBITDA",
    "Net Profit (PAT)",
    "Earnings Per Share (EPS)",
    # Balance Sheet
    "Total Assets",
    "Total Liabilities",
    "Total Equity",
    "Cash & Equivalents",
    "Total Debt (Borrowings)",
    # Cash Flow
    "Operating Cash Flow",
    "Investing Cash Flow",
    "Financing Cash Flow",
    "Free Cash Flow",
    # Key Ratios & Highlights
    "Gross Margin",
    "Operating Margin",
    "Net Margin",
    "Return on Equity (ROE)",
    "Return on Assets (ROA)",
    "Debt-to-Equity Ratio",
    "EPS Growth YoY"
]

METRIC_KEYWORDS = {
    # Income Statement
    "Revenue (Total Income)": ["revenue", "total income", "turnover"],
    "Gross Profit": ["gross profit"],
    "Operating Income (EBIT)": ["operating income", "operating profit", "ebit", "profit before interest"],
    "EBITDA": ["ebitda"],
    "Net Profit (PAT)": ["profit after tax", "pat", "net profit", "loss for the year", "profit for the year", "net income"],
    "Earnings Per Share (EPS)": ["earnings per share", "eps", "loss per share"],
    # Balance Sheet
    "Total Assets": ["total assets"],
    "Total Liabilities": ["total liabilities"],
    "Total Equity": ["total equity", "shareholders equity", "shareholders' equity", "net worth"],
    "Cash & Equivalents": ["cash and cash equivalents", "cash & cash equivalents", "bank balances"],
    "Total Debt (Borrowings)": ["total debt", "borrowings", "long-term debt", "short-term debt"],
    # Cash Flow
    "Operating Cash Flow": ["operating activities", "cash generated from operations"],
    "Investing Cash Flow": ["investing activities", "cash used in investing"],
    "Financing Cash Flow": ["financing activities", "cash used in financing"],
    "Free Cash Flow": ["free cash flow", "fcf"],
    # Key Ratios & Highlights
    "Gross Margin": ["gross margin", "gross profit margin"],
    "Operating Margin": ["operating margin", "ebit margin"],
    "Net Margin": ["net margin", "net profit margin"],
    "Return on Equity (ROE)": ["roe", "return on equity"],
    "Return on Assets (ROA)": ["roa", "return on assets"],
    "Debt-to-Equity Ratio": ["debt to equity", "debt-equity", "gearing"],
    "EPS Growth YoY": ["eps growth"]
}

_SYSTEM_PROMPT = """You are an expert financial data extractor.
Your task is to review a specific table from an annual report and extract ONLY the metrics requested in the "REMAINING METRICS" list.

Rules:
1. Review the Context/Heading above the table to determine if the numbers are Consolidated or Standalone, and what the units are (e.g., "in Crores", "Millions", "₹").
2. Only extract a metric if it clearly matches one of the REMAINING METRICS.
3. If a metric is NOT in the table, DO NOT make it up.
4. Call the `submit_metrics_found` tool with your findings. If the table does not contain ANY of the remaining metrics, call the tool with an empty list.
"""

_USER_PROMPT_TEMPLATE = """REMAINING METRICS TO FIND:
{remaining_metrics}

TABLE TO ANALYZE:
{table_text}
"""


class FinancialExtractor:
    """Hybrid sequential table extraction + agentic search fallback."""

    def __init__(self, llm_base_url: str = DEFAULT_LLM_URL) -> None:
        self.client = openai.AsyncOpenAI(
            base_url=f"{llm_base_url}/v1", api_key="not-needed"
        )

    def _extract_markdown_tables_with_context(
        self, text_chunks: list[str], context_lines: int = 5
    ) -> list[str]:
        tables = []
        for text in text_chunks:
            lines = text.split('\n')
            current_table = []
            start_idx = -1
            
            for i, line in enumerate(lines):
                if line.strip().startswith('|'):
                    if not current_table:
                        start_idx = i
                    current_table.append(line)
                else:
                    if len(current_table) > 2:
                        ctx_start = max(0, start_idx - context_lines)
                        context = '\n'.join(lines[ctx_start:start_idx])
                        table_str = '\n'.join(current_table)
                        tables.append(f"Context/Heading:\n{context}\n\nTable:\n{table_str}")
                    current_table = []
                    
            if len(current_table) > 2:
                ctx_start = max(0, start_idx - context_lines)
                context = '\n'.join(lines[ctx_start:start_idx])
                table_str = '\n'.join(current_table)
                tables.append(f"Context/Heading:\n{context}\n\nTable:\n{table_str}")
                
        return tables

    def _score_table(self, table_text: str, remaining_metrics: list[str]) -> tuple[int, str, int]:
        text_lower = table_text.lower()
        t1 = t2 = t3 = t4 = 0
        
        tier_1 = ["statement of profit", "balance sheet", "cash flow statement", "consolidated", "standalone", "financial results"]
        if any(kw in text_lower for kw in tier_1):
            t1 = 10
            
        for metric_name in remaining_metrics:
            keywords = METRIC_KEYWORDS.get(metric_name, [])
            for kw in keywords:
                t2 += 5 * text_lower.count(kw)
                
        tier_3 = ["crores", "millions", "lakhs", "₹", "inr", "rs."]
        for kw in tier_3:
            t3 += 2 * text_lower.count(kw)
            
        tier_4 = ["board of directors", "remuneration", "shareholding pattern", "attendance", "esop", "csr", "committee", "address"]
        for kw in tier_4:
            t4 -= 5 * text_lower.count(kw)
            
        total_score = t1 + t2 + t3 + t4
        breakdown = f"[T1: {t1} | T2: {t2} | T3: {t3} | T4: {t4}]"
        return total_score, breakdown, t2

    async def _process_table(
        self, 
        table_text: str, 
        remaining_metrics: list[str], 
        log_callback: Callable[[str], None] | None
    ) -> list[FinancialMetric]:
        tools = [{
            "type": "function",
            "function": {
                "name": "submit_metrics_found",
                "description": "Submit matching metrics found in the table. Use an empty array if none found.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "found_metrics": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "matched_target_metric": {"type": "string"},
                                    "actual_name_in_table": {"type": "string"},
                                    "current_year_consolidated": {"type": "number"},
                                    "previous_year_consolidated": {"type": "number"},
                                    "current_year_standalone": {"type": "number"},
                                    "previous_year_standalone": {"type": "number"},
                                    "unit": {"type": "string"}
                                },
                                "required": ["matched_target_metric", "actual_name_in_table", "unit"]
                            }
                        }
                    },
                    "required": ["found_metrics"]
                }
            }
        }]

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_PROMPT_TEMPLATE.format(
                remaining_metrics=json.dumps(remaining_metrics, indent=2),
                table_text=table_text
            )}
        ]

        try:
            response = await self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "submit_metrics_found"}},
                temperature=0.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            
            tool_calls = response.choices[0].message.tool_calls
            if not tool_calls:
                return []

            args_str = tool_calls[0].function.arguments
            args = json.loads(args_str)
            found = args.get("found_metrics", [])
            
            results = []
            for item in found:
                metric = FinancialMetric(
                    name=item["matched_target_metric"],
                    unit=item.get("unit", "Unknown"),
                    current_year_consolidated=item.get("current_year_consolidated"),
                    previous_year_consolidated=item.get("previous_year_consolidated"),
                    current_year_standalone=item.get("current_year_standalone"),
                    previous_year_standalone=item.get("previous_year_standalone"),
                    notes=f"Extracted as '{item.get('actual_name_in_table')}'"
                )
                results.append(metric)
                
            return results
        except Exception as e:
            if log_callback: log_callback(f"LLM Error processing table: {e}")
            return []

    async def _agentic_search_metric(self, metric_name: str, lines: list[str], log_callback: Callable | None) -> FinancialMetric | None:
        def search_document(query: str) -> str:
            query = query.lower()
            results = []
            for i, line in enumerate(lines):
                if query in line.lower():
                    start = max(0, i - 8)
                    end = min(len(lines), i + 9)
                    context = "\n".join(lines[start:end])
                    results.append(f"--- Match at line {i} ---\n{context}\n")
                    if len(results) >= 5:
                        break
            if not results:
                return f"No matches found for '{query}'."
            return "\n".join(results)
            
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_document",
                    "description": "Search for a keyword. Returns the matching line + 8 lines context.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "submit_metric",
                    "description": "Submit the extracted data if found.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "actual_name_in_document": {"type": "string"},
                            "current_year_consolidated": {"type": "number"},
                            "previous_year_consolidated": {"type": "number"},
                            "current_year_standalone": {"type": "number"},
                            "previous_year_standalone": {"type": "number"},
                            "unit": {"type": "string"}
                        },
                        "required": ["actual_name_in_document", "unit"]
                    }
                }
            }
        ]
        
        system_prompt = """You are a forensic financial extraction agent.
1. Use `search_document` to look for keywords.
2. Read the 17-line context window returned.
3. If you find the numbers, call `submit_metric`.
4. If after 3 searches you cannot find it anywhere, reply with "Not found".
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please find the financial metric: '{metric_name}'."}
        ]
        
        for iteration in range(6):
            response = await self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=tools,
                temperature=0.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
            msg = response.choices[0].message
            
            if msg.content:
                messages.append({"role": "assistant", "content": msg.content})
                if "not found" in msg.content.lower():
                    if log_callback: log_callback(f"      [Search] AI concluded metric not found.")
                    return None
                    
            if not msg.tool_calls:
                break
                
            messages.append({"role": "assistant", "tool_calls": msg.tool_calls, "content": msg.content})
            
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                if tc.function.name == "search_document":
                    q = args.get("query", "")
                    if log_callback: log_callback(f"      [Search] Query: '{q}'")
                    result_text = search_document(q)
                    
                    if log_callback: 
                        log_callback(f"      [Search Result sent to LLM] 👇\n{'-'*50}\n{result_text.strip()}\n{'-'*50}")
                        
                    messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.function.name, "content": result_text})
                elif tc.function.name == "submit_metric":
                    has_data = any([
                        args.get("current_year_consolidated") is not None,
                        args.get("previous_year_consolidated") is not None,
                        args.get("current_year_standalone") is not None,
                        args.get("previous_year_standalone") is not None
                    ])
                    if has_data:
                        return FinancialMetric(
                            name=metric_name,
                            unit=args.get("unit", "Unknown"),
                            current_year_consolidated=args.get("current_year_consolidated"),
                            previous_year_consolidated=args.get("previous_year_consolidated"),
                            current_year_standalone=args.get("current_year_standalone"),
                            previous_year_standalone=args.get("previous_year_standalone"),
                            notes=f"Found via search as '{args.get('actual_name_in_document')}'"
                        )
                    else:
                        messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.function.name, "content": "Error: Empty data. Keep searching or say 'Not found'."})
                        
        return None

    async def extract(
        self, 
        sections: dict[str, SectionContent] | list[str] | list[SectionContent], 
        company_name: str, 
        fiscal_year: str, 
        pdf_path: str | None = None,
        log_callback: Callable[[str], None] | None = None
    ) -> FinancialData:
        
        # Unpack sections to list of text strings
        text_chunks = []
        if isinstance(sections, dict):
            for v in sections.values():
                if isinstance(v, SectionContent):
                    text_chunks.append(v.text)
                else:
                    text_chunks.append(str(v))
        elif isinstance(sections, list):
            for v in sections:
                if isinstance(v, SectionContent):
                    text_chunks.append(v.text)
                else:
                    text_chunks.append(str(v))
                    
        tables = self._extract_markdown_tables_with_context(text_chunks)
        remaining_metrics = TARGET_METRICS.copy()
        extracted_metrics: list[FinancialMetric] = []
        
        # --- PHASE 1: TABLE FAST-TRACK ---
        if log_callback:
            log_callback(f"====================================")
            log_callback(f"PHASE 1: DYNAMIC TABLE SCANNING")
            log_callback(f"====================================")
            
        unvisited_tables = tables.copy()
        iteration = 0
        
        while unvisited_tables and remaining_metrics:
            iteration += 1
            
            scored_tables = []
            for t in unvisited_tables:
                total_score, breakdown, t2_score = self._score_table(t, remaining_metrics)
                if t2_score > 0:
                    scored_tables.append((total_score, breakdown, t))
            
            if not scored_tables:
                if log_callback: log_callback(f"No remaining tables contain Target Keywords. Stopping Phase 1!")
                break
                
            scored_tables.sort(key=lambda x: x[0], reverse=True) 
            best_score, breakdown, best_table = scored_tables[0]
            unvisited_tables.remove(best_table)
            
            if log_callback:
                log_callback(f"Scanning Table {iteration} (Score: {best_score}) {breakdown}")
                log_callback(f"  -> Hunting for: {', '.join(remaining_metrics)}")
            
            found_in_table = await self._process_table(best_table, remaining_metrics, log_callback)
            
            for m in found_in_table:
                has_data = any([
                    m.current_year_consolidated is not None,
                    m.previous_year_consolidated is not None,
                    m.current_year_standalone is not None,
                    m.previous_year_standalone is not None
                ])
                if has_data:
                    extracted_metrics.append(m)
                    if m.name in remaining_metrics:
                        remaining_metrics.remove(m.name)
                        if log_callback: log_callback(f"  -> Extracted: {m.name}")


        # --- PHASE 2: AGENTIC SEARCH FALLBACK ---
        if remaining_metrics:
            if log_callback:
                log_callback(f"\n====================================")
                log_callback(f"PHASE 2: TEXT SEARCH FALLBACK")
                log_callback(f"====================================")
                log_callback(f"Metrics still missing: {', '.join(remaining_metrics)}")
                
            model_manager.require_llm()
                
            full_text = "\n".join(text_chunks)
            document_lines = [line for line in full_text.split('\n') if line.strip()]
            
            for metric in list(remaining_metrics):
                if log_callback: log_callback(f"\n-> Deploying Text Search for: {metric}")
                found_metric = await self._agentic_search_metric(metric, document_lines, log_callback)
                
                if found_metric:
                    extracted_metrics.append(found_metric)
                    remaining_metrics.remove(metric)
                    if log_callback: log_callback(f"  -> SUCCESS! Found {metric} via Text Search.")
        # --- PHASE 3: CALCULATE DERIVED METRICS ---
        # If margins/ratios were not found, calculate them mathematically!
        val_map = {m.name: m for m in extracted_metrics}
        
        def calculate_and_add(target_name: str, unit: str, func) -> None:
            if target_name in remaining_metrics:
                try:
                    cy, py = func()
                    if cy is not None or py is not None:
                        new_metric = ExtractedMetric(
                            name=target_name,
                            current_year_consolidated=cy,
                            previous_year_consolidated=py,
                            unit=unit,
                            page_number=0,
                            notes="Calculated automatically from base metrics."
                        )
                        extracted_metrics.append(new_metric)
                        remaining_metrics.remove(target_name)
                        if log_callback: log_callback(f"  -> SUCCESS! Calculated {target_name} mathematically.")
                except Exception:
                    pass

        # Helper to safely get CY and PY values
        def get_vals(name):
            m = val_map.get(name)
            if not m: return None, None
            cy = m.current_year_consolidated or m.current_year_standalone
            py = m.previous_year_consolidated or m.previous_year_standalone
            return cy, py

        # 1. Net Margin = (Net Profit / Revenue) * 100
        calculate_and_add("Net Margin", "%", lambda: (
            (get_vals("Net Profit (PAT)")[0] / get_vals("Revenue (Total Income)")[0] * 100) if get_vals("Net Profit (PAT)")[0] and get_vals("Revenue (Total Income)")[0] else None,
            (get_vals("Net Profit (PAT)")[1] / get_vals("Revenue (Total Income)")[1] * 100) if get_vals("Net Profit (PAT)")[1] and get_vals("Revenue (Total Income)")[1] else None
        ))
        
        # 2. Operating Margin = (Operating Income / Revenue) * 100
        calculate_and_add("Operating Margin", "%", lambda: (
            (get_vals("Operating Income (EBIT)")[0] / get_vals("Revenue (Total Income)")[0] * 100) if get_vals("Operating Income (EBIT)")[0] and get_vals("Revenue (Total Income)")[0] else None,
            (get_vals("Operating Income (EBIT)")[1] / get_vals("Revenue (Total Income)")[1] * 100) if get_vals("Operating Income (EBIT)")[1] and get_vals("Revenue (Total Income)")[1] else None
        ))

        # 3. Return on Assets (ROA) = (Net Profit / Total Assets) * 100
        calculate_and_add("Return on Assets (ROA)", "%", lambda: (
            (get_vals("Net Profit (PAT)")[0] / get_vals("Total Assets")[0] * 100) if get_vals("Net Profit (PAT)")[0] and get_vals("Total Assets")[0] else None,
            (get_vals("Net Profit (PAT)")[1] / get_vals("Total Assets")[1] * 100) if get_vals("Net Profit (PAT)")[1] and get_vals("Total Assets")[1] else None
        ))
        
        # 4. Debt-to-Equity = Total Debt / Total Equity
        calculate_and_add("Debt-to-Equity Ratio", "x", lambda: (
            (get_vals("Total Debt (Borrowings)")[0] / get_vals("Total Equity")[0]) if get_vals("Total Debt (Borrowings)")[0] is not None and get_vals("Total Equity")[0] else None,
            (get_vals("Total Debt (Borrowings)")[1] / get_vals("Total Equity")[1]) if get_vals("Total Debt (Borrowings)")[1] is not None and get_vals("Total Equity")[1] else None
        ))

        if log_callback:
            log_callback("\n====================================")
            log_callback("📊 EXTRACTION SUMMARY")
            log_callback("====================================")
            
            found_names = [m.name for m in extracted_metrics]
            if found_names:
                log_callback(f"✅ FOUND ({len(found_names)}):")
                for name in found_names:
                    log_callback(f"  - {name}")
            else:
                log_callback("✅ FOUND (0): None")
                
            log_callback("")
            
            if remaining_metrics:
                log_callback(f"❌ MISSING ({len(remaining_metrics)}):")
                for name in remaining_metrics:
                    log_callback(f"  - {name}")
                log_callback("\n⚠️ Warning: Missing metrics were not found in tables or explicitly stated in text.")
            else:
                log_callback("🎉 All target metrics found successfully!")

        model_manager.require_llm()

        return FinancialData(
            company_name=company_name,
            fiscal_year=fiscal_year,
            metrics=extracted_metrics,
            extraction_timestamp=datetime.now()
        )
