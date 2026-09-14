"""Section splitting: regex heading detection with semantic RAG fallback."""
from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import BaseModel

from pipeline.parsing.parse import ParsedReport

logger = logging.getLogger(__name__)

__version__ = "1.0.0"


class SectionContent(BaseModel):
    """A contiguous section extracted from the report."""

    section_name: str
    text: str
    start_page: int
    end_page: int
    found_by: str  # "regex" or "semantic"


# ---------------------------------------------------------------------------
# Heading patterns – order matters (first match wins per page line).
# Each key is the canonical section name used everywhere downstream.
# ---------------------------------------------------------------------------

_SECTION_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "MD&A": [
        re.compile(
            r"(?im)^\s*#*\s*(?:MANAGEMENT(?:['\u2019]S)?\s+DISCUSSION\s+(?:AND|&)\s+ANALYSIS|MD\s*&\s*A)\b"
        ),
    ],
    "Directors' Report": [
        re.compile(r"(?im)^\s*#*\s*(?:DIRECTORS?['\u2019]?\s+REPORT|BOARD['\u2019]?S?\s+REPORT)\b"),
    ],
    "Balance Sheet": [
        re.compile(
            r"(?im)^\s*#*\s*(?:BALANCE\s+SHEET|STATEMENT\s+OF\s+FINANCIAL\s+POSITION)\b"
        ),
    ],
    "Statement of Profit & Loss": [
        re.compile(
            r"(?im)^\s*#*\s*(?:STATEMENT\s+OF\s+PROFIT\s+(?:AND|&)\s+LOSS|INCOME\s+STATEMENT)\b"
        ),
    ],
    "Cash Flow Statement": [
        re.compile(
            r"(?im)^\s*#*\s*(?:(?:STATEMENT\s+OF\s+)?CASH\s+FLOWS?|CASH\s+FLOW\s+STATEMENT)\b"
        ),
    ],
    "Notes to Accounts": [
        re.compile(
            r"(?im)^\s*#*\s*(?:NOTES\s+TO\s+(?:THE\s+)?(?:ACCOUNTS|FINANCIAL\s+STATEMENTS))\b"
        ),
    ],
    "Corporate Governance Report": [
        re.compile(r"(?im)^\s*#*\s*CORPORATE\s+GOVERNANCE(?:\s+REPORT)?\b"),
    ],
    "Auditor's Report": [
        re.compile(r"(?im)^\s*#*\s*(?:INDEPENDENT\s+)?AUDITORS?['\u2019]?\s+REPORT\b"),
    ],
}


class SectionSplitter:
    """Splits a parsed report into standard annual-report sections.

    **Pass 1 – regex**: scans every page for heading patterns and
    collects all text from the heading page up to (but not including)
    the next detected heading.

    **Pass 2 – semantic fallback**: for any standard section not found
    by regex, queries the RAG index (if provided) and synthesizes a
    pseudo-section from the top retrieved chunks.
    """

    def __init__(self) -> None:
        self._patterns = _SECTION_PATTERNS

    def split(
        self,
        parsed_report: ParsedReport,
        rag_index: Any | None = None,
    ) -> dict[str, SectionContent]:
        """Split *parsed_report* into named sections.

        Args:
            parsed_report: Output of ``ReportParser.parse()``.
            rag_index: An optional ``RAGEngine`` instance used for
                the semantic fallback when a section heading cannot
                be found by regex.

        Returns:
            Dict mapping canonical section names to their content.
        """
        logger.info("Starting section splitting (%d pages)", parsed_report.total_pages)

        try:
            sections = self._regex_split(parsed_report)
        except Exception:
            logger.exception("Regex split failed")
            sections = {}

        found_names = set(sections)
        missing = [s for s in self._patterns if s not in found_names]

        if missing and rag_index is not None:
            logger.info(
                "Sections not found by regex: %s – trying semantic fallback",
                missing,
            )
            try:
                import asyncio
                import nest_asyncio
                nest_asyncio.apply()
                sem_sections = asyncio.run(
                    self._semantic_fallback(missing, rag_index)
                )
                sections.update(sem_sections)
            except Exception:
                logger.exception("Semantic fallback failed")
        elif missing:
            logger.warning(
                "Sections missing and no RAG index for fallback: %s", missing
            )

        logger.info(
            "Section split complete – found %d/%d sections",
            len(sections),
            len(self._patterns),
        )
        return sections

    # ------------------------------------------------------------------
    # Pass 1 – regex
    # ------------------------------------------------------------------

    def _regex_split(
        self, parsed_report: ParsedReport
    ) -> dict[str, SectionContent]:
        """Detect section headings by regex and collect full text spans."""

        # Build list of (page_number, section_name) sorted by page order.
        heading_hits: list[tuple[int, str]] = []

        for page in parsed_report.pages:
            for sec_name, patterns in self._patterns.items():
                for pat in patterns:
                    if pat.search(page.text):
                        heading_hits.append((page.page_number, sec_name))
                        break  # one match per section per page is enough

        # Deduplicate: keep first occurrence of each section name.
        seen: set[str] = set()
        unique_hits: list[tuple[int, str]] = []
        for page_num, sec_name in heading_hits:
            if sec_name not in seen:
                seen.add(sec_name)
                unique_hits.append((page_num, sec_name))

        # Sort by page number.
        unique_hits.sort(key=lambda t: t[0])

        # Build a quick lookup: page_number -> text
        page_map: dict[int, str] = {
            p.page_number: p.text for p in parsed_report.pages
        }
        all_page_nums = sorted(page_map)

        sections: dict[str, SectionContent] = {}

        for i, (start_page, sec_name) in enumerate(unique_hits):
            # End page is one before the next section heading, or last page.
            if i + 1 < len(unique_hits):
                end_page = unique_hits[i + 1][0] - 1
            else:
                end_page = parsed_report.total_pages

            # Clamp
            end_page = max(end_page, start_page)

            # Gather text for pages in [start_page, end_page].
            text_parts: list[str] = []
            for pn in all_page_nums:
                if start_page <= pn <= end_page:
                    text_parts.append(
                        f"<!-- PAGE {pn} -->\n{page_map[pn]}"
                    )

            sections[sec_name] = SectionContent(
                section_name=sec_name,
                text="\n\n".join(text_parts),
                start_page=start_page,
                end_page=end_page,
                found_by="regex",
            )

        logger.info("Regex pass found %d sections: %s", len(sections), list(sections))
        return sections

    # ------------------------------------------------------------------
    # Pass 2 – semantic fallback via RAG
    # ------------------------------------------------------------------

    async def _semantic_fallback(
        self,
        missing_sections: list[str],
        rag_index: Any,
    ) -> dict[str, SectionContent]:
        """Retrieve chunks for each missing section name and synthesize."""
        sections: dict[str, SectionContent] = {}

        for sec_name in missing_sections:
            try:
                chunks = await rag_index.search(sec_name, top_k=8)
                if not chunks:
                    logger.info("No chunks found for section '%s'", sec_name)
                    continue

                # Merge chunk texts, derive page range.
                all_pages: set[int] = set()
                text_parts: list[str] = []
                for c in chunks:
                    all_pages.update(c.page_numbers)
                    text_parts.append(c.text)

                sorted_pages = sorted(all_pages)
                sections[sec_name] = SectionContent(
                    section_name=sec_name,
                    text="\n\n".join(text_parts),
                    start_page=sorted_pages[0],
                    end_page=sorted_pages[-1],
                    found_by="semantic",
                )
                logger.info(
                    "Semantic fallback found '%s' across pages %s",
                    sec_name,
                    sorted_pages,
                )
            except Exception:
                logger.exception(
                    "Semantic fallback error for section '%s'", sec_name
                )

        return sections
