#!/usr/bin/env python3
"""
Debug script: Convert a PDF to Markdown using the project's internal ReportParser.
Usage:
    python debug_pdf_to_md.py path/to/report.pdf [output.md]
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure we can import from the pipeline module
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from pipeline.parsing.parse import ReportParser

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Convert PDF to Markdown for debugging.")
    parser.add_argument("pdf_path", type=str, help="Path to the input PDF file")
    parser.add_argument("output_path", type=str, nargs="?", help="Path to the output MD file (optional)")
    
    args = parser.parse_args()
    
    pdf_path = Path(args.pdf_path)
    if not pdf_path.is_file():
        logger.error(f"Input file does not exist: {pdf_path}")
        sys.exit(1)
        
    output_path = args.output_path
    if not output_path:
        # Default to same directory, same name, .md extension
        output_path = pdf_path.with_suffix(".md")
    
    logger.info(f"Initializing ReportParser...")
    report_parser = ReportParser()
    
    logger.info(f"Parsing PDF: {pdf_path}")
    try:
        # This uses the exact same parsing pipeline as the main app
        parsed_report = report_parser.parse(str(pdf_path))
        
        logger.info(f"Successfully parsed {parsed_report.total_pages} pages.")
        
        # Write the combined Markdown to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(parsed_report.full_text)
            
        logger.info(f"Markdown successfully written to: {output_path}")
        
    except Exception as e:
        logger.exception(f"Failed to parse PDF: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
