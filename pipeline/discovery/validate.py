import json
import logging
import os
from typing import Optional

import pymupdf4llm
from openai import AsyncOpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

__version__ = "1.0.0"

class ValidationResult(BaseModel):
    is_valid: bool
    is_annual_report: bool
    company_match: bool
    detected_company: str | None = None
    fiscal_year: str | None = None
    confidence: float = 0.0
    reason: str | None = None

class ReportValidator:
    """Validates downloaded PDF files to confirm they are the correct annual report."""

    def __init__(self, llm_base_url: str = "http://localhost:8080"):
        self.llm_base_url = llm_base_url
        # Use dummy API key for local server
        self.client = AsyncOpenAI(base_url=f"{self.llm_base_url}/v1", api_key="not-needed")
        logger.info(f"ReportValidator initialized with LLM base URL: {self.llm_base_url}")

    async def validate(self, pdf_path: str, company_name: str | None = None) -> ValidationResult:
        """Main validation method."""
        logger.info(f"Starting validation for {pdf_path} (Company: {company_name or 'Auto-detect'})")
        
        if not os.path.exists(pdf_path):
            logger.error(f"File not found: {pdf_path}")
            return ValidationResult(
                is_valid=False, is_annual_report=False, company_match=False, 
                reason="File not found"
            )

        if not self._check_pdf_magic(pdf_path):
            logger.error(f"Invalid PDF magic bytes: {pdf_path}")
            return ValidationResult(
                is_valid=False, is_annual_report=False, company_match=False, 
                reason="Invalid PDF magic bytes"
            )

        try:
            extracted_text = self._extract_first_pages(pdf_path, num_pages=2)
            if not extracted_text.strip():
                logger.warning(f"No text extracted from first pages of {pdf_path}")
                return ValidationResult(
                    is_valid=False, is_annual_report=False, company_match=False, 
                    reason="No text on first pages"
                )

            validation_result = await self._llm_validate(extracted_text, company_name)
            logger.info(f"Validation result for {pdf_path}: {validation_result}")
            return validation_result

        except Exception as e:
            logger.exception(f"Exception during validation of {pdf_path}: {e}")
            return ValidationResult(
                is_valid=False, is_annual_report=False, company_match=False, 
                reason=f"Validation error: {e}"
            )

    def _check_pdf_magic(self, pdf_path: str) -> bool:
        """Check PDF magic bytes."""
        try:
            with open(pdf_path, 'rb') as f:
                header = f.read(4)
                return header == b'%PDF'
        except Exception as e:
            logger.exception(f"Error reading magic bytes for {pdf_path}: {e}")
            return False

    def _extract_first_pages(self, pdf_path: str, num_pages: int = 2) -> str:
        """Extract first N pages as text using pymupdf4llm."""
        logger.info(f"Extracting first {num_pages} pages from {pdf_path}")
        try:
            # We want just the first few pages
            page_numbers = list(range(num_pages))
            md_text = pymupdf4llm.to_markdown(pdf_path, pages=page_numbers)
            return md_text
        except Exception as e:
            logger.exception(f"Error extracting text from {pdf_path}: {e}")
            raise

    async def _llm_validate(self, text: str, company_name: str | None) -> ValidationResult:
        """Ask LLM to confirm company + fiscal year."""
        logger.info("Sending text to LLM for validation")
        
        target_check = f'for the company "{company_name}"' if company_name else "and identify the company"
        
        prompt = f"""
        You are an expert financial document validator. Analyze the following text extracted from the first pages of a document.
        
        Text:
        {text[:4000]}  # Limit text length just in case
        
        Determine if this document is an Annual Report {target_check}.
        Provide the output as JSON matching this schema:
        {{
            "is_annual_report": true/false,
            "company_match": true/false,  // Set true if you confidently identify a company, even if no target was given
            "detected_company": "Name of the company found in text",
            "fiscal_year": "e.g., 2023-24 or 2024",
            "confidence": 0.0 to 1.0,
            "reason": "Brief explanation"
        }}
        """

        schema = {
            "type": "object",
            "properties": {
                "is_annual_report": {"type": "boolean"},
                "company_match": {"type": "boolean"},
                "detected_company": {"type": "string"},
                "fiscal_year": {"type": "string"},
                "confidence": {"type": "number"},
                "reason": {"type": "string"}
            },
            "required": ["is_annual_report", "company_match", "detected_company", "fiscal_year", "confidence", "reason"],
            "additionalProperties": False
        }

        try:
            response = await self.client.chat.completions.create(
                model="qwen",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that outputs only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_schema", "json_schema": {"name": "validation", "schema": schema, "strict": True}},
                temperature=0.1,
                max_tokens=300,
                timeout=45.0
            )

            result_text = response.choices[0].message.content
            if not result_text:
                raise ValueError("Empty response from LLM")

            parsed = json.loads(result_text)
            
            is_annual_report = bool(parsed.get("is_annual_report", False))
            company_match = bool(parsed.get("company_match", False))
            
            return ValidationResult(
                is_valid=is_annual_report and company_match,
                is_annual_report=is_annual_report,
                company_match=company_match,
                detected_company=parsed.get("detected_company"),
                fiscal_year=parsed.get("fiscal_year"),
                confidence=float(parsed.get("confidence", 0.0)),
                reason=parsed.get("reason")
            )

        except Exception as e:
            logger.exception(f"Error during LLM validation: {e}")
            raise
