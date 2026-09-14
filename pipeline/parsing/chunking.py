"""Page-aware overlapping chunking of parsed Markdown for RAG."""
from __future__ import annotations

import hashlib
import logging
import re

from pipeline.parsing.parse import ParsedReport
from pipeline.parsing.sections import SectionContent
from pipeline.schema import DocumentChunk

logger = logging.getLogger(__name__)

__version__ = "1.0.0"

# Regex to detect page markers injected by the parser / section splitter.
_PAGE_MARKER_RE = re.compile(r"<!--\s*PAGE\s+(\d+)\s*-->")


class DocumentChunker:
    """Splits a parsed report into overlapping, page-tagged chunks.

    Default sizes are ~512 tokens chunk / ~100 tokens overlap (approximated
    at 4 chars ≈ 1 token).  Sentence boundaries are preferred as split
    points to avoid cutting mid-sentence.
    """

    def __init__(self, chunk_size: int = 2048, overlap: int = 400) -> None:
        self.chunk_size = chunk_size  # characters
        self.overlap = overlap

    def chunk(
        self,
        parsed_report: ParsedReport,
        sections: dict[str, SectionContent] | None = None,
    ) -> list[DocumentChunk]:
        """Chunk the full report text, preserving page numbers.

        If *sections* is provided, each section is chunked independently
        so that chunks don't span across section boundaries.  Otherwise
        the report is chunked as one continuous text.

        Args:
            parsed_report: Output of ``ReportParser.parse()``.
            sections: Optional dict of named sections.

        Returns:
            A list of ``DocumentChunk`` instances.
        """
        logger.info("Starting chunking (chunk_size=%d, overlap=%d)", self.chunk_size, self.overlap)

        try:
            chunks: list[DocumentChunk] = []

            if sections:
                for sec_name, sec in sections.items():
                    sec_chunks = self._split_text(sec.text, section=sec_name)
                    chunks.extend(sec_chunks)
                    logger.debug(
                        "Section '%s': %d chunks", sec_name, len(sec_chunks)
                    )
            else:
                chunks = self._split_text(parsed_report.full_text)

            logger.info(
                "Chunking complete: %d chunks, avg size %.0f chars",
                len(chunks),
                (sum(len(c.text) for c in chunks) / len(chunks)) if chunks else 0,
            )
            return chunks

        except Exception:
            logger.exception("Error during chunking")
            raise

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _split_text(
        self,
        text: str,
        section: str | None = None,
    ) -> list[DocumentChunk]:
        """Split *text* into overlapping chunks tracking page numbers."""
        chunks: list[DocumentChunk] = []
        text_len = len(text)
        start = 0

        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            # Try to break at a sentence boundary (". ") to avoid mid-sentence cuts.
            if end < text_len:
                # Look for the last sentence-ending period in the second half of the chunk.
                search_start = start + self.chunk_size // 2
                break_at = text.rfind(". ", search_start, end)
                if break_at == -1:
                    break_at = text.rfind(".\n", search_start, end)
                if break_at != -1:
                    end = break_at + 2  # include the period and space/newline

            chunk_text = text[start:end]

            # Derive page numbers from <!-- PAGE N --> markers inside this chunk.
            page_nums = [int(m) for m in _PAGE_MARKER_RE.findall(chunk_text)]
            if not page_nums:
                # Fall back: scan backwards from start for the most recent marker.
                prev_markers = _PAGE_MARKER_RE.findall(text[:start])
                page_nums = [int(prev_markers[-1])] if prev_markers else [1]

            chunk_id = hashlib.sha256(
                f"{chunk_text[:200]}:{sorted(set(page_nums))}".encode()
            ).hexdigest()[:16]

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    page_numbers=sorted(set(page_nums)),
                    section=section,
                )
            )

            if end >= text_len:
                break

            start = end - self.overlap

        return chunks
