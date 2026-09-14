"""PDF to page-tagged Markdown conversion using pymupdf4llm."""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import pymupdf4llm
from pydantic import BaseModel

logger = logging.getLogger(__name__)

__version__ = "1.0.0"


class PageContent(BaseModel):
    """Content of a single PDF page."""

    page_number: int
    text: str


class ParsedReport(BaseModel):
    """Result of parsing a PDF into page-tagged Markdown."""

    pages: list[PageContent]
    total_pages: int
    full_text: str  # concatenated with <!-- PAGE N --> markers


class ReportParser:
    """Converts a PDF file into page-tagged Markdown text.

    Uses ``pymupdf4llm`` for the actual conversion.  Each page is
    prefixed with an HTML comment ``<!-- PAGE N -->`` so that downstream
    consumers can recover page boundaries from the concatenated text.
    """

    def parse(self, pdf_path: str) -> ParsedReport:
        """Parse a PDF and return structured page-tagged Markdown.

        Args:
            pdf_path: Absolute or relative path to the PDF file.

        Returns:
            A ``ParsedReport`` with per-page text, total page count,
            and a concatenated full-text version with page markers.

        Raises:
            RuntimeError: If pymupdf4llm fails to extract any content.
        """
        start = time.perf_counter()
        logger.info("Parsing PDF: %s", pdf_path)

        try:
            pages = self._extract_pages(pdf_path)
        except Exception:
            logger.exception("Failed to extract pages from %s", pdf_path)
            raise

        if not pages:
            msg = f"No pages extracted from {pdf_path}"
            logger.error(msg)
            raise RuntimeError(msg)

        full_text = "\n".join(
            f"<!-- PAGE {p.page_number} -->\n{p.text}" for p in pages
        )

        elapsed = time.perf_counter() - start
        logger.info(
            "Parsed %s: %d pages in %.2fs",
            pdf_path,
            len(pages),
            elapsed,
        )

        return ParsedReport(
            pages=pages,
            total_pages=len(pages),
            full_text=full_text,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_pages(self, pdf_path: str) -> list[PageContent]:
        """Extract each page as Markdown via pymupdf4llm.

        ``page_chunks=True`` makes the library return a list of dicts,
        one per page, each containing ``text`` and ``metadata``.
        """
        md_chunks: list[dict] = pymupdf4llm.to_markdown(
            pdf_path, page_chunks=True
        )

        pages: list[PageContent] = []
        for chunk in md_chunks:
            page_num = chunk.get("metadata", {}).get("page", 0) + 1
            text = chunk.get("text", "")
            if not text.strip():
                logger.debug("Page %d is empty, skipping", page_num)
                continue
            pages.append(PageContent(page_number=page_num, text=text))

        logger.info("Extracted %d non-empty pages", len(pages))
        return pages
