"""Page-aware overlapping chunking of parsed Markdown for RAG."""
from __future__ import annotations

import hashlib
import logging
import re

from pipeline.parsing.parse import ParsedReport
from pipeline.parsing.sections import SectionContent
from pipeline.schema import DocumentChunk

logger = logging.getLogger(__name__)



# Regex to detect page markers injected by the parser / section splitter.
_PAGE_MARKER_RE = re.compile(r"<!--\s*PAGE\s+(\d+)\s*-->")


class DocumentChunker:
    """Splits a parsed report into page-sized chunks with adjacent page overlap.

    Since our embedding model (Qwen3-Embedding-0.6B) has a 32k token context
    limit, we can comfortably embed entire pages at a time. This prevents
    tearing tables or paragraphs apart. We add a small overlap from the 
    previous and next pages to maintain context across page boundaries.
    """

    def __init__(self, overlap_chars: int = 1000) -> None:
        self.overlap_chars = overlap_chars  # About 250 tokens overlap

    def chunk(
        self,
        parsed_report: ParsedReport,
        sections: dict[str, SectionContent] | None = None,
    ) -> list[DocumentChunk]:
        """Chunk the report text by page, including adjacent page overlaps.

        Args:
            parsed_report: Output of ``ReportParser.parse()``.
            sections: Optional dict of named sections.

        Returns:
            A list of ``DocumentChunk`` instances.
        """
        logger.info("Starting page-based chunking (overlap=%d chars)", self.overlap_chars)

        try:
            chunks: list[DocumentChunk] = []
            pages = parsed_report.pages
            total_pages = len(pages)

            if not total_pages:
                return []

            for i, page in enumerate(pages):
                page_text = page.text
                
                # Overlap from previous page
                prev_overlap = ""
                if i > 0 and self.overlap_chars > 0:
                    prev_text = pages[i-1].text
                    prev_overlap = prev_text[-self.overlap_chars:]
                    if prev_overlap:
                        # Try to snap to the first space to avoid cutting words
                        snap = prev_overlap.find(" ")
                        if snap != -1:
                            prev_overlap = prev_overlap[snap:]
                        prev_overlap = f"--- END OF PREVIOUS PAGE (p.{pages[i-1].page_number}) ---\n{prev_overlap}\n\n"

                # Overlap from next page
                next_overlap = ""
                if i < total_pages - 1 and self.overlap_chars > 0:
                    next_text = pages[i+1].text
                    next_overlap = next_text[:self.overlap_chars]
                    if next_overlap:
                        # Try to snap to the last space to avoid cutting words
                        snap = next_overlap.rfind(" ")
                        if snap != -1:
                            next_overlap = next_overlap[:snap]
                        next_overlap = f"\n\n--- START OF NEXT PAGE (p.{pages[i+1].page_number}) ---\n{next_overlap}"

                # Combine into single chunk
                chunk_text = f"{prev_overlap}{page_text}{next_overlap}"

                # Determine section if provided
                section_name = None
                if sections:
                    for sec_name, sec_content in sections.items():
                        if sec_content.start_page <= page.page_number <= sec_content.end_page:
                            section_name = sec_name
                            break

                chunk_id = hashlib.sha256(
                    f"page_{page.page_number}:{chunk_text[:200]}".encode()
                ).hexdigest()[:16]

                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        page_numbers=[page.page_number],
                        section=section_name,
                    )
                )

            logger.info(
                "Chunking complete: %d chunks, avg size %.0f chars",
                len(chunks),
                (sum(len(c.text) for c in chunks) / len(chunks)) if chunks else 0,
            )
            return chunks

        except Exception:
            logger.exception("Error during chunking")
            raise
