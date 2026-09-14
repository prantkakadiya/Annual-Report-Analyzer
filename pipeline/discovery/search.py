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
    
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            logger.error("TAVILY_API_KEY not provided or found in environment.")
            raise ValueError("Tavily API key is required.")
        self.client = TavilyClient(api_key=self.api_key)
        
    async def search(self, company_name: str, fiscal_year: str | None = None) -> list[SearchCandidate]:
        """Main search method to find annual report candidates."""
        logger.info(f"Starting search for company: {company_name}, fiscal_year: {fiscal_year}")
        queries = self._build_queries(company_name, fiscal_year)
        
        all_candidates: dict[str, SearchCandidate] = {}
        
        for query in queries:
            try:
                logger.info(f"Executing search query: {query}")
                # Use synchronous tavily client in a thread pool if needed, or just block
                # Tavily doesn't have an async client in the standard package, so we just call it directly.
                # If we need truly async, we could use asyncio.to_thread
                response = self.client.search(query, search_depth="advanced", max_results=10)
                results = response.get("results", [])
                
                for result in results:
                    url = result.get("url")
                    if not url or url in all_candidates:
                        continue
                        
                    title = result.get("title", "")
                    snippet = result.get("content", "")
                    
                    score = self._score_candidate(url, title, snippet, company_name)
                    
                    candidate = SearchCandidate(
                        url=url,
                        title=title,
                        snippet=snippet,
                        score=score,
                        is_direct_pdf=False
                    )
                    all_candidates[url] = candidate
                    
            except Exception as e:
                logger.exception(f"Error during search query '{query}': {e}")
                
        # Check candidates for direct PDFs (concurrently or sequentially)
        sorted_candidates = sorted(all_candidates.values(), key=lambda c: c.score, reverse=True)
        top_candidates = sorted_candidates[:20]  # Limit checks to top 20
        
        for candidate in top_candidates:
            if candidate.score < 0:
                continue
            
            try:
                is_pdf, content_type, content_length = await self._check_direct_pdf(candidate.url)
                candidate.is_direct_pdf = is_pdf
                candidate.content_type = content_type
                candidate.content_length = content_length
            except Exception as e:
                logger.exception(f"Failed checking direct PDF for {candidate.url}")
                
        # Re-sort after checking direct PDFs (give bonus to direct PDFs)
        for c in top_candidates:
            if c.is_direct_pdf:
                c.score += 5.0
                
        return sorted(top_candidates, key=lambda c: c.score, reverse=True)
        
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
        
    def _score_candidate(self, url: str, title: str, snippet: str, company_name: str) -> float:
        """Score based on URL patterns, domain, title match."""
        score = 0.0
        
        url_lower = url.lower()
        title_lower = title.lower()
        company_lower = company_name.lower()
        
        # Domain heuristic
        if "bseindia.com" in url_lower or "nseindia.com" in url_lower:
            score += 2.0
        if "investor" in url_lower or "investors" in url_lower:
            score += 2.0
            
        # URL keywords
        if "annual-report" in url_lower or "annual_report" in url_lower or "annualreport" in url_lower:
            score += 3.0
            
        if "ar" in url_lower and ("2023" in url_lower or "2024" in url_lower):
            score += 1.0
            
        if url_lower.endswith(".pdf"):
            score += 2.0
            
        # Title match
        if company_lower in title_lower:
            score += 2.0
        if "annual report" in title_lower:
            score += 2.0
            
        logger.debug(f"Scored candidate {url} with score {score}")
        return score
        
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
                elif url.lower().endswith(".pdf") and response.status_code == 200:
                    # Sometimes headers are wrong but URL ends in PDF
                    is_pdf = True
                    
        except httpx.RequestError as e:
            logger.error(f"HTTP error during HEAD request to {url}: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error checking direct PDF {url}: {e}")
            
        return is_pdf, content_type, content_length
