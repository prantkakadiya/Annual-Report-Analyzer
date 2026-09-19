"""Export analysis results as a styled PDF research note using fpdf2."""
from __future__ import annotations

import logging
from datetime import datetime

from fpdf import FPDF

from pipeline.schema import AnalysisResult

logger = logging.getLogger(__name__)


class _ResearchNotePDF(FPDF):
    """Custom FPDF subclass with header/footer for the research note."""

    company: str = ""
    fiscal_year: str = ""

    def header(self) -> None:
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, f"{self.company} — Annual Report Analysis", align="L")
        self.cell(0, 6, self.fiscal_year, align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), self.w - 10, self.get_y())
        self.ln(4)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 8, f"Page {self.page_no()}/{{nb}}", align="C")


def export_pdf(result: AnalysisResult, output_path: str) -> str:
    """Create a styled PDF research note from *result*.

    Args:
        result: Complete analysis result.
        output_path: Destination file path for the PDF.

    Returns:
        The *output_path* on success.

    Raises:
        RuntimeError: If PDF generation fails.
    """
    try:
        meta = result.metadata
        pdf = _ResearchNotePDF(orientation="P", unit="mm", format="A4")
        pdf.company = meta.company_name
        pdf.fiscal_year = meta.fiscal_year
        pdf.alias_nb_pages()
        pdf.set_auto_page_break(auto=True, margin=20)
        pdf.add_page()

        # ---- Title ----
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(20, 20, 80)
        pdf.cell(0, 12, meta.company_name, new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.set_font("Helvetica", "", 12)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(
            0, 8,
            f"Annual Report Analysis — {meta.fiscal_year}",
            new_x="LMARGIN", new_y="NEXT", align="C",
        )
        pdf.ln(6)

        # ---- Metadata ----
        _section_heading(pdf, "Report Metadata")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(0, 0, 0)
        for label, value in [
            ("Company", meta.company_name),
            ("Fiscal Year", meta.fiscal_year),
            ("Source", meta.source_url or "Manual upload"),
            ("Total Pages", str(meta.total_pages or "N/A")),
            ("Analysed", meta.discovered_at.strftime("%Y-%m-%d %H:%M")),
        ]:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(40, 7, f"{label}:", new_x="RIGHT")
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 7, str(value), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        # ---- Financial Metrics Table ----
        _section_heading(pdf, "Financial Metrics")

        if result.financial_data and result.financial_data.metrics:
            col_w = [38, 26, 26, 26, 26, 22, 16]  # total = 180
            headers = ["Metric", "CY Cons.", "CY Stand.", "PY Cons.", "PY Stand.", "Unit", "Page"]

            # Table header
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_fill_color(30, 30, 90)
            pdf.set_text_color(255, 255, 255)
            for i, h in enumerate(headers):
                pdf.cell(col_w[i], 7, h, border=1, fill=True, align="C")
            pdf.ln()

            # Table rows
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(0, 0, 0)

            def _fmt(v: float | None) -> str:
                if v is None:
                    return "—"
                if v == int(v):
                    return f"{int(v):,}"
                return f"{v:,.2f}"

            for row_idx, m in enumerate(result.financial_data.metrics):
                # Alternating row colour
                if row_idx % 2 == 0:
                    pdf.set_fill_color(240, 240, 250)
                else:
                    pdf.set_fill_color(255, 255, 255)

                vals = [
                    m.name,
                    _fmt(m.current_year_consolidated),
                    _fmt(m.current_year_standalone),
                    _fmt(m.previous_year_consolidated),
                    _fmt(m.previous_year_standalone),
                    m.unit,
                    f"[p.{m.page_number}]" if m.page_number else "—",
                ]
                for i, v in enumerate(vals):
                    align = "L" if i == 0 else "R" if i < 5 else "C"
                    pdf.cell(col_w[i], 6, v, border=1, fill=True, align=align)
                pdf.ln()
        else:
            pdf.set_font("Helvetica", "I", 10)
            pdf.cell(0, 8, "No financial metrics extracted.", new_x="LMARGIN", new_y="NEXT")

        pdf.ln(4)

        # ---- Business Insights ----
        _section_heading(pdf, "Business Insights")

        if result.insights and result.insights.insights:
            by_cat: dict[str, list[str]] = {}
            for ins in result.insights.insights:
                by_cat.setdefault(ins.category, []).append(ins.content)

            for cat, items in by_cat.items():
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_text_color(30, 30, 90)
                pdf.cell(0, 7, cat, new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 9)
                pdf.set_text_color(0, 0, 0)
                for item in items:
                    pdf.multi_cell(0, 5, f"  •  {item}")
                    pdf.ln(1)
                pdf.ln(2)
        else:
            pdf.set_font("Helvetica", "I", 10)
            pdf.cell(0, 8, "No insights generated.", new_x="LMARGIN", new_y="NEXT")

        # ---- Footer note ----
        pdf.ln(8)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(128, 128, 128)
        pdf.multi_cell(
            0, 4,
            f"Generated by Annual Report Analyzer v{result.analysis_version} "
            f"on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            align="C",
        )

        pdf.output(output_path)
        logger.info("PDF exported → %s", output_path)
        return output_path

    except Exception:
        logger.exception("Error exporting PDF to %s", output_path)
        raise


def _section_heading(pdf: FPDF, title: str) -> None:
    """Render a styled section heading."""
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 30, 90)
    pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(30, 30, 90)
    pdf.line(10, pdf.get_y(), pdf.w - 10, pdf.get_y())
    pdf.ln(3)
