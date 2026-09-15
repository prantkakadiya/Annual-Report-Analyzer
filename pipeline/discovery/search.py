import logging
import os
import urllib.parse
from typing import List, Tuple, Optional

import httpx
from dotenv import load_dotenv
from tavily import TavilyClient

from pipeline.schema import SearchCandidate

logger = logging.getLogger(__name__)

load_dotenv()

__version__ = "1.0.0"

class ReportSearcher:
    """Searches for annual report PDFs using Tavily."""
    
    def __init__(self, api_key: str | None = None, llm_base_url: str = "http://localhost:8080"):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            logger.error("TAVILY_API_KEY not provided or found in environment.")
            raise ValueError("Tavily API key is required.")
        self.client = TavilyClient(api_key=self.api_key)
        from openai import AsyncOpenAI
        self.llm_client = AsyncOpenAI(base_url=f"{llm_base_url}/v1", api_key="not-needed")
        
    async def search(self, company_name: str, fiscal_year: str | None = None) -> list[SearchCandidate]:
        """Main search method to find annual report candidates."""
        logger.info(f"Starting search for company: {company_name}, fiscal_year: {fiscal_year}")
        queries = self._build_queries(company_name, fiscal_year)
        
        all_candidates: dict[str, SearchCandidate] = {}
        
        for query in queries:
            try:
                logger.info(f"Executing search query: {query}")
                response = self.client.search(query, search_depth="advanced", max_results=10)
                results = response.get("results", [])
                
                for result in results:
                    url = result.get("url")
                    if not url or url in all_candidates:
                        continue
                        
                    title = result.get("title", "")
                    snippet = result.get("content", "")
                    
                    candidate = SearchCandidate(
                        url=url,
                        title=title,
                        snippet=snippet,
                        score=0.0,
                        is_direct_pdf=False
                    )
                    all_candidates[url] = candidate
                    
            except Exception as e:
                logger.exception(f"Error during search query '{query}': {e}")
                
        # Limit checks to top 15 to save time
        candidates_list = list(all_candidates.values())[:15]
        
        # Check candidates for direct PDFs
        for candidate in candidates_list:
            try:
                is_pdf, content_type, content_length = await self._check_direct_pdf(candidate.url)
                candidate.is_direct_pdf = is_pdf
                candidate.content_type = content_type
                candidate.content_length = content_length
            except Exception as e:
                logger.exception(f"Failed checking direct PDF for {candidate.url}")
                
        # Use LLM to rank the candidates
        ranked_candidates = await self._rank_with_llm(candidates_list, company_name, fiscal_year)
        return ranked_candidates
        
    def _build_queries(self, company_name: str, fiscal_year: str | None) -> list[str]:
        """Build search query variants."""
        queries = []
        if fiscal_year:
            queries.append(f'"{company_name}" annual report {fiscal_year} filetype:pdf')
            queries.append(f'"{company_name}" annual report {fiscal_year}')
            queries.append(f'"{company_name}" annual report {fiscal_year} site:bseindia.com OR site:nseindia.com')
        else:
            queries.append(f'"{company_name}" annual report latest investor relations')
            queries.append(f'"{company_name}" annual report filetype:pdf')
            queries.append(f'"{company_name}" annual report latest site:bseindia.com OR site:nseindia.com')
            
        return queries
        
    async def _rank_with_llm(self, candidates: list[SearchCandidate], company_name: str, fiscal_year: str | None) -> list[SearchCandidate]:
        import json
        if not candidates:
            return []
            
        candidates_json = []
        for i, c in enumerate(candidates):
            candidates_json.append({
                "id": i,
                "url": c.url,
                "title": c.title,
                "snippet": c.snippet,
                "is_direct_pdf": c.is_direct_pdf
            })
            
        system_prompt = "You are an expert financial researcher picking the most likely official Annual Report PDF url for a company."
        prompt = f"""
        Company: {company_name}
        Fiscal Year: {fiscal_year or 'Latest'}
        
        Search Results:
        {json.dumps(candidates_json, indent=2)}
        
        Analyze the search results. Your goal is to select the BEST URL that points to the official Annual Report published by the company.
        Prioritize results where `is_direct_pdf` is true and the domain appears official or is a recognized stock exchange (e.g., bseindia.com).
        
        Return a JSON object ranking the best candidate IDs in order of preference (best first).
        Example output:
        {{
            "ranked_ids": [3, 0, 4],
            "reason": "ID 3 is the official direct PDF for the 2024 annual report. ID 0 is a backup exchange link."
        }}
        """
        
        schema = {
            "type": "object",
            "properties": {
                "ranked_ids": {
                    "type": "array",
                    "items": {"type": "integer"}
                },
                "reason": {"type": "string"}
            },
            "required": ["ranked_ids", "reason"],
            "additionalProperties": False
        }
        
        try:
            logger.info("Asking LLM to rank search candidates...")
            response = await self.llm_client.chat.completions.create(
                model="qwen",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_schema", "json_schema": {"name": "rank_candidates", "schema": schema, "strict": True}},
                temperature=0.1,
                max_tokens=1000,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
            
            content = response.choices[0].message.content
            if content:
                data = json.loads(content)
                ranked_ids = data.get("ranked_ids", [])
                logger.info(f"LLM Ranking: {ranked_ids}. Reason: {data.get('reason')}")
                
                # Reorder candidates based on LLM ranking
                ranked_candidates = []
                used_ids = set()
                
                for idx in ranked_ids:
                    if 0 <= idx < len(candidates) and idx not in used_ids:
                        ranked_candidates.append(candidates[idx])
                        used_ids.add(idx)
                        
                # Append the rest just in case
                for i, c in enumerate(candidates):
                    if i not in used_ids:
                        ranked_candidates.append(c)
                        
                return ranked_candidates
        except Exception as e:
            logger.exception(f"LLM ranking failed: {e}")
            
        # Fallback to returning original list
        return candidates
        
    async def _check_direct_pdf(self, url: str) -> tuple[bool, str | None, int | None]:
        """HEAD request, return (is_pdf, content_type, content_length)."""
        logger.info(f"Checking if direct PDF: {url}")
        
        is_pdf = False
        content_type = None
        content_length = None
        
        try:
            async with httpx.AsyncClient(verify=False, timeout=8.0) as client:
                response = await client.head(url, follow_redirects=True)
                
                content_type = response.headers.get("Content-Type", "").lower()
                length_str = response.headers.get("Content-Length")
                
                if length_str and length_str.isdigit():
                    content_length = int(length_str)
                    
                if "application/pdf" in content_type:
                    is_pdf = True
                elif url.lower().endswith(".pdf") and response.status_code in (200, 403, 405):
                    # Sometimes headers are wrong or servers block HEAD requests, but URL ends in PDF
                    is_pdf = True
                    
        except httpx.RequestError as e:
            logger.error(f"HTTP error during HEAD request to {url}: {e}")
            if url.lower().endswith(".pdf"):
                is_pdf = True
        except Exception as e:
            logger.exception(f"Unexpected error checking direct PDF {url}: {e}")
            
        return is_pdf, content_type, content_length
