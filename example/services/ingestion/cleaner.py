"""Document text cleaner and extractor."""

import os
from typing import Optional


def extract_text_from_pdf(file_path: str) -> Optional[str]:
    """Extract text from a PDF file."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception:
        return None


def extract_text_from_docx(file_path: str) -> Optional[str]:
    """Extract text from a DOCX file."""
    try:
        from docx import Document

        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        return text.strip()
    except Exception:
        return None


def extract_text_from_txt(file_path: str) -> Optional[str]:
    """Extract text from a plain text file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            return f.read().strip()
    except Exception:
        return None


def clean_text(text: str) -> str:
    """Clean extracted text: normalize whitespace, remove artifacts."""
    import re

    # Remove multiple blank lines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    # Remove leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return text.strip()


def extract_text(file_path: str, content_type: str) -> Optional[str]:
    """Extract and clean text from a document file.

    Args:
        file_path: Path to the document file.
        content_type: MIME type of the file.

    Returns:
        Cleaned text content, or None if extraction failed.
    """
    extractor_map = {
        "application/pdf": extract_text_from_pdf,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": extract_text_from_docx,
        "text/plain": extract_text_from_txt,
    }

    extractor = extractor_map.get(content_type)
    if not extractor:
        return None

    raw_text = extractor(file_path)
    if not raw_text:
        return None

    return clean_text(raw_text)
