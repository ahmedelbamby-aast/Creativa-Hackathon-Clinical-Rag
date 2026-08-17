"""Structure-aware document parser using PyMuPDF.

Extracts rich structural metadata from PDF files:
- Headings detected via font size heuristics
- Paragraphs as individual text blocks
- Tables extracted and serialised as Markdown
- Per-block metadata: page number, section title, subsection title, content type

Falls back to plain-text extraction via pypdf for documents that PyMuPDF
cannot open (corrupted files, encrypted PDFs without password, etc.).

Normalised output format per element:
    {
        "document_name": "IDF_Diabetes_Atlas_2025.pdf",
        "page_number": 12,
        "section_title": "Chapter 3: Prevention",
        "subsection_title": "Lifestyle Interventions",
        "content": "...",
        "content_type": "text"  # or "table" | "heading"
    }
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
DocumentElement = dict  # see module docstring for schema


# ---------------------------------------------------------------------------
# Heading detection heuristics
# ---------------------------------------------------------------------------

def _is_heading(span_flags: int, font_size: float, body_font_size: float) -> bool:
    """Return True when a span looks like a heading.

    PyMuPDF font flags:
        bit 4 = bold
        bit 1 = italic
    """
    is_bold = bool(span_flags & (1 << 4))
    is_larger = font_size >= body_font_size * 1.15
    return is_bold or is_larger


def _estimate_body_font_size(page_dict: dict) -> float:
    """Estimate the dominant (body) font size on a page from span statistics."""
    sizes: list[float] = []
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:  # 0 = text block
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                sizes.append(span.get("size", 12.0))
    if not sizes:
        return 12.0
    from collections import Counter
    rounded = [round(s) for s in sizes]
    most_common = Counter(rounded).most_common(1)
    return float(most_common[0][0]) if most_common else 12.0


# ---------------------------------------------------------------------------
# Table serialisation
# ---------------------------------------------------------------------------

def _table_to_markdown(table) -> str:
    """Serialise a PyMuPDF Table object to a Markdown pipe table."""
    try:
        rows = table.extract()
    except Exception:
        return ""

    if not rows:
        return ""

    clean_rows = []
    for row in rows:
        clean_row = [str(cell).strip() if cell is not None else "" for cell in row]
        clean_rows.append(clean_row)

    if not clean_rows:
        return ""

    lines: list[str] = []
    header = clean_rows[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in clean_rows[1:]:
        while len(row) < len(header):
            row.append("")
        lines.append("| " + " | ".join(row[: len(header)]) + " |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PyMuPDF extraction  — uses `pymupdf` (not deprecated `fitz`)
# ---------------------------------------------------------------------------

def _extract_with_fitz(file_path: str) -> list[DocumentElement]:
    """Extract structured elements from a PDF using PyMuPDF.

    Uses `import pymupdf` (the non-deprecated API).
    Iterates pages by index to avoid premature document closure.
    """
    try:
        import pymupdf  # newer, non-deprecated import
    except ImportError:
        import fitz as pymupdf  # fallback for older installations

    document_name = os.path.basename(file_path)
    elements: list[DocumentElement] = []

    try:
        doc = pymupdf.open(file_path)
    except Exception as e:
        logger.error("PyMuPDF failed to open %s: %s", file_path, e)
        return []

    num_pages = doc.page_count
    logger.debug("[%s] PyMuPDF opened: %d pages", document_name, num_pages)

    current_section = ""
    current_subsection = ""

    # Iterate by index — avoids generator/GC issues that cause "document closed"
    for page_index in range(num_pages):
        page_number = page_index + 1

        try:
            page = doc[page_index]
        except Exception as e:
            logger.warning("[%s] Could not load page %d: %s", document_name, page_number, e)
            continue

        # ── Get structured text dict ──────────────────────────────────
        try:
            page_dict = page.get_text("dict")
        except Exception as e:
            logger.debug("[%s] get_text('dict') failed on page %d: %s", document_name, page_number, e)
            # Fallback: get plain text for this page
            try:
                plain = page.get_text()
                if plain.strip():
                    elements.append({
                        "document_name": document_name,
                        "page_number": page_number,
                        "section_title": current_section,
                        "subsection_title": current_subsection,
                        "content": plain.strip(),
                        "content_type": "text",
                    })
            except Exception:
                pass
            continue

        body_font_size = _estimate_body_font_size(page_dict)

        # ── Extract tables ────────────────────────────────────────────
        table_bboxes: list[tuple] = []
        try:
            tabs = page.find_tables()
            for table in tabs:
                md = _table_to_markdown(table)
                if md:
                    elements.append({
                        "document_name": document_name,
                        "page_number": page_number,
                        "section_title": current_section,
                        "subsection_title": current_subsection,
                        "content": md,
                        "content_type": "table",
                        "_y0": table.bbox[1],
                    })
                    table_bboxes.append(table.bbox)
        except Exception as e:
            logger.debug("[%s] Table extraction failed p%d: %s", document_name, page_number, e)

        # ── Extract text blocks ───────────────────────────────────────
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue  # skip image blocks

            # Skip blocks that overlap with extracted tables
            bx0, by0, bx1, by1 = block["bbox"]
            overlaps = False
            for (tx0, ty0, tx1, ty1) in table_bboxes:
                # Check significant overlap
                ox = max(0, min(bx1, tx1) - max(bx0, tx0))
                oy = max(0, min(by1, ty1) - max(by0, ty0))
                block_area = max(1, (bx1 - bx0) * (by1 - by0))
                if (ox * oy) / block_area > 0.5:
                    overlaps = True
                    break
            if overlaps:
                continue

            # Collect text from all spans
            block_text_parts: list[str] = []
            is_block_heading = False
            block_font_size = body_font_size

            for line in block.get("lines", []):
                line_parts: list[str] = []
                for span in line.get("spans", []):
                    span_text = span.get("text", "").strip()
                    if not span_text:
                        continue
                    span_size = span.get("size", body_font_size)
                    span_flags = span.get("flags", 0)
                    line_parts.append(span_text)

                    if _is_heading(span_flags, span_size, body_font_size):
                        is_block_heading = True
                        block_font_size = span_size

                if line_parts:
                    block_text_parts.append(" ".join(line_parts))

            block_text = "\n".join(block_text_parts).strip()
            if not block_text:
                continue

            # Update section tracking
            if is_block_heading:
                if block_font_size >= body_font_size * 1.4:
                    current_section = block_text[:200]
                    current_subsection = ""
                else:
                    current_subsection = block_text[:200]

            elements.append({
                "document_name": document_name,
                "page_number": page_number,
                "section_title": current_section,
                "subsection_title": current_subsection,
                "content": block_text,
                "content_type": "heading" if is_block_heading else "text",
                "_y0": block["bbox"][1],
            })

    doc.close()

    # Sort by page then y-position
    elements.sort(key=lambda e: (e["page_number"], e.get("_y0", 0)))
    for el in elements:
        el.pop("_y0", None)

    logger.info(
        "[%s] PyMuPDF extracted %d elements from %d pages",
        document_name, len(elements), num_pages,
    )
    return elements


# ---------------------------------------------------------------------------
# Fallback: pypdf plain-text extraction
# ---------------------------------------------------------------------------

def _extract_with_pypdf(file_path: str) -> list[DocumentElement]:
    """Fallback text-only extraction using pypdf."""
    document_name = os.path.basename(file_path)
    elements: list[DocumentElement] = []
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                elements.append({
                    "document_name": document_name,
                    "page_number": i + 1,
                    "section_title": "",
                    "subsection_title": "",
                    "content": text,
                    "content_type": "text",
                })
        logger.info("[%s] pypdf fallback: %d pages", document_name, len(elements))
    except Exception as e:
        logger.error("[%s] pypdf extraction failed: %s", document_name, e)
    return elements


# ---------------------------------------------------------------------------
# DOCX extraction
# ---------------------------------------------------------------------------

def _extract_docx(file_path: str) -> list[DocumentElement]:
    """Extract paragraphs from a DOCX file with basic heading detection."""
    document_name = os.path.basename(file_path)
    elements: list[DocumentElement] = []
    try:
        from docx import Document
        doc = Document(file_path)
        current_section = ""
        current_subsection = ""
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            style_name = para.style.name if para.style else ""
            is_h1 = "Heading 1" in style_name
            is_h2 = "Heading 2" in style_name
            is_heading = is_h1 or is_h2 or "Heading" in style_name
            if is_h1:
                current_section = text[:200]
                current_subsection = ""
            elif is_h2:
                current_subsection = text[:200]
            elements.append({
                "document_name": document_name,
                "page_number": 1,
                "section_title": current_section,
                "subsection_title": current_subsection,
                "content": text,
                "content_type": "heading" if is_heading else "text",
            })
    except Exception as e:
        logger.error("[%s] DOCX extraction failed: %s", document_name, e)
    return elements


# ---------------------------------------------------------------------------
# TXT extraction
# ---------------------------------------------------------------------------

def _extract_txt(file_path: str) -> list[DocumentElement]:
    """Extract paragraphs from a plain text file."""
    document_name = os.path.basename(file_path)
    try:
        try:
            text = Path(file_path).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = Path(file_path).read_text(encoding="latin-1")
        elements: list[DocumentElement] = []
        for para in re.split(r"\n{2,}", text):
            para = para.strip()
            if para:
                elements.append({
                    "document_name": document_name,
                    "page_number": 1,
                    "section_title": "",
                    "subsection_title": "",
                    "content": para,
                    "content_type": "text",
                })
        return elements
    except Exception as e:
        logger.error("[%s] TXT extraction failed: %s", document_name, e)
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_document(file_path: str) -> list[DocumentElement]:
    """Parse a document and return structured elements.

    Tries PyMuPDF first for PDFs (rich structure + tables).
    Falls back to pypdf for plain text extraction.
    Handles DOCX and TXT natively.
    """
    file_path = str(file_path)
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        elements = _extract_with_fitz(file_path)
        if not elements:
            logger.warning(
                "[%s] PyMuPDF returned no elements -- falling back to pypdf",
                os.path.basename(file_path),
            )
            elements = _extract_with_pypdf(file_path)
        return elements
    elif ext == ".docx":
        return _extract_docx(file_path)
    elif ext == ".txt":
        return _extract_txt(file_path)
    else:
        logger.warning("Unsupported file type: %s", file_path)
        return []


def propagate_section_titles(elements: list[DocumentElement]) -> list[DocumentElement]:
    """Forward-fill section/subsection titles for elements that lack them."""
    current_section = ""
    current_subsection = ""
    result: list[DocumentElement] = []

    for el in elements:
        if el["content_type"] == "heading":
            if el["section_title"]:
                current_section = el["section_title"]
                current_subsection = ""
            if el["subsection_title"]:
                current_subsection = el["subsection_title"]
        else:
            if not el["section_title"]:
                el = {**el, "section_title": current_section}
            if not el["subsection_title"]:
                el = {**el, "subsection_title": current_subsection}
        result.append(el)

    return result
