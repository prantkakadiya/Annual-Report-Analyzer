"""PDF to page-tagged Markdown conversion using Docling with GPU acceleration."""
from __future__ import annotations

import logging
import time
import os
import tempfile
import pymupdf as fitz
from tqdm import tqdm

from pydantic import BaseModel

logger = logging.getLogger(__name__)

__version__ = "2.0.0"


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
    """Converts a PDF file into page-tagged Markdown text using Docling.

    Implements a GPU-accelerated pipeline with:
    1. In-memory auto-rotation of landscape pages (preventing table corruption).
    2. EasyOCR integration for scanned regions.
    3. 20-page chunking to optimize VRAM and provide accurate progress tracking.
    """

    def __init__(self):
        logger.info("Initializing Docling DocumentConverter with GPU support...")
        from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
        from docling.document_converter import PdfFormatOption, DocumentConverter
        from docling.datamodel.document import InputFormat
        import torch

        # Force CUDA if available, otherwise fallback to AUTO
        device = AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.AUTO
        accelerator_options = AcceleratorOptions(num_threads=8, device=device)
        pipeline_options = PdfPipelineOptions(accelerator_options=accelerator_options)
        
        # Enable EasyOCR for GPU-accelerated OCR on image pages/regions
        from docling.datamodel.pipeline_options import EasyOcrOptions
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = EasyOcrOptions(lang=["en"])
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def parse(self, pdf_path: str) -> ParsedReport:
        """Parse a PDF and return structured page-tagged Markdown.

        Args:
            pdf_path: Absolute or relative path to the PDF file.

        Returns:
            A ``ParsedReport`` with per-page text, total page count,
            and a concatenated full-text version with page markers.
        """
        start = time.perf_counter()
        logger.info("Parsing PDF: %s", pdf_path)

        try:
            pages, total_pages = self._extract_pages(pdf_path)
        except Exception:
            logger.exception("Failed to extract pages from %s", pdf_path)
            raise

        if not pages:
            raise RuntimeError("No text could be extracted from the PDF.")

        full_text = "\n\n".join(
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
            total_pages=total_pages,
            full_text=full_text,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_pages(self, pdf_path: str) -> tuple[list[PageContent], int]:
        """Extract each page as Markdown via chunked Docling conversion.
        
        Splits the document into 20-page chunks, auto-rotates landscape pages,
        and extracts per-page markdown from the Docling output to maintain
        compatibility with the downstream RAG chunking logic.
        """
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        chunk_size = 20
        extracted_pages: list[PageContent] = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            chunks = []
            # Phase 1: Chunking and Auto-Rotation
            for i in range(0, total_pages, chunk_size):
                end = min(i + chunk_size - 1, total_pages - 1)
                chunk_path = os.path.join(temp_dir, f"chunk_{i}_{end}.pdf")
                
                doc_chunk = fitz.open()
                doc_chunk.insert_pdf(doc, from_page=i, to_page=end)
                
                # Auto-detect and fix sideways pages in this chunk
                for page in doc_chunk:
                    blocks = page.get_text("dict").get("blocks", [])
                    upright, rotated = 0, 0
                    for b in blocks:
                        if b.get("type") == 0:  # text block
                            for l in b.get("lines", []):
                                dir_y = l["dir"][1]
                                if abs(dir_y) > 0.5:
                                    rotated += 1
                                else:
                                    upright += 1
                    
                    if rotated > upright and rotated > 0:
                        page.set_rotation(90)
                
                doc_chunk.save(chunk_path)
                doc_chunk.close()
                chunks.append((i, chunk_path))
                
            doc.close()
            logger.info("PDF split into %d chunks. Starting GPU Docling conversion...", len(chunks))
            
            # Phase 2: Docling Extraction
            for start_idx, chunk_path in tqdm(chunks, desc=f"Processing {total_pages} pages"):
                result = self.converter.convert(chunk_path)
                
                # Reconstruct per-page markdown to preserve downstream PageContent schema
                num_pages_in_chunk = len(result.document.pages)
                for local_page_idx in range(1, num_pages_in_chunk + 1):
                    global_page_num = start_idx + local_page_idx
                    try:
                        text = result.document.export_to_markdown(page_no=local_page_idx)
                        if text.strip():
                            extracted_pages.append(PageContent(page_number=global_page_num, text=text))
                    except Exception as e:
                        logger.warning("Failed to extract markdown for page %d: %s", global_page_num, e)
                        
        logger.info("Extracted %d non-empty pages out of %d total pages", len(extracted_pages), total_pages)
        return extracted_pages, total_pages
