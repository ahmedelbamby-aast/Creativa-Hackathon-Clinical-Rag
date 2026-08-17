"""General document text cleaner and extractor.

The extractor works page-by-page so that chunk metadata (e.g. page numbers)
stays accurate. Cleaning is file-agnostic and removes common PDF/text
artifacts: unicode spaces, control characters, hyphenation across line
breaks, standalone page numbers, repeated blank lines, and repeated
boilerplate header/footer lines.
"""

import os
import re
from collections import Counter
from typing import Optional


_DEFAULT_FILE_TYPE = ("text/plain", "plain-text")
_FILE_TYPES_BY_EXTENSION = {
    ".pdf": ("application/pdf", "pypdf"),
    ".docx": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "python-docx",
    ),
    ".txt": _DEFAULT_FILE_TYPE,
}


def _resolve_file_type(file_path: str) -> tuple[str, str]:
    """Return the content type and extractor label from one shared mapping."""
    extension = os.path.splitext(os.fspath(file_path))[1].lower()
    return _FILE_TYPES_BY_EXTENSION.get(extension, _DEFAULT_FILE_TYPE)


def detect_content_type(file_path: str) -> str:
    """Detect MIME-like content type from file extension.

    Args:
        file_path: Path to the document file.

    Returns:
        Content type string used to pick the right extractor.
    """
    return _resolve_file_type(file_path)[0]


def get_extractor_label(file_path: str) -> str:
    """Human-readable label of the extractor used for a file."""
    return _resolve_file_type(file_path)[1]


def _extract_pages_from_pdf(file_path: str) -> list[dict]:
    """Extract per-page text from a PDF file."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        pages = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            pages.append({"page_number": i + 1, "text": page_text})
        return pages
    except Exception:
        return []


def _extract_text_from_docx(file_path: str) -> Optional[str]:
    """Extract text from a DOCX file."""
    try:
        from docx import Document

        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        return text.strip()
    except Exception:
        return None


def _extract_text_from_txt(file_path: str) -> Optional[str]:
    """Extract text from a plain text file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            return f.read().strip()
    except Exception:
        return None


def extract_pages(file_path: str) -> list[dict]:
    """Extract per-page text from a document file.

    Content type is inferred from the file extension, so only the file
    path is needed (no database metadata required). Non-paginated formats
    (docx, txt) are returned as a single page.

    Args:
        file_path: Path to the document file.

    Returns:
        List of {"page_number", "text"} dicts, or [] if extraction failed.
    """
    content_type = detect_content_type(file_path)

    if content_type == "application/pdf":
        return _extract_pages_from_pdf(file_path)

    extractor_map = {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": _extract_text_from_docx,
        "text/plain": _extract_text_from_txt,
    }

    extractor = extractor_map.get(content_type)
    if not extractor:
        return []

    raw_text = extractor(file_path)
    if not raw_text:
        return []

    return [{"page_number": 1, "text": raw_text}]


def _normalize_text(text: str) -> str:
    """Normalize unicode: BOM, non-breaking/zero-width chars, control chars."""
    text = text.replace("\ufeff", "")
    text = text.replace("\u00a0", " ").replace("\u2007", " ").replace("\u202f", " ")
    text = re.sub(r"[\u200b\u200c\u200d\u2060]", "", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text


def _dehyphenate(text: str) -> str:
    """Join words broken by a hyphen at a line break (e.g. 'pre-\\nvent' -> 'prevent')."""
    return re.sub(r"-\s*\n\s*", "", text)


def _remove_standalone_page_numbers(text: str) -> tuple[str, int]:
    """Remove lines that contain only a page number.

    Returns:
        Tuple of (cleaned text, number of lines removed).
    """
    lines = text.split("\n")
    kept = []
    removed = 0
    for line in lines:
        if re.fullmatch(r"\s*\d{1,4}\s*", line):
            removed += 1
            continue
        kept.append(line)
    return "\n".join(kept), removed


def _collapse_blank_lines(text: str) -> str:
    """Collapse multiple consecutive blank lines into one."""
    return re.sub(r"\n\s*\n+", "\n\n", text)


def _strip_lines(text: str) -> str:
    """Strip leading/trailing whitespace on every line and on the whole text."""
    return "\n".join(line.strip() for line in text.split("\n")).strip()


# Lines that end with a page marker (pipe/dash + digits or roman numeral)
PAGE_MARKER_RE = re.compile(r"\s*[|·•\-–—]\s*[0-9IVXLCDM]{1,5}$")


def _strip_page_marker(line: str) -> tuple[str, bool]:
    """Strip a trailing page marker from a line.

    Running footers usually end with a page indicator that changes on every
    page (e.g. '...diabetesatlas.org | 42' or '...diabetesatlas.org | V').
    Stripping it lets such footers be recognized as the same repeated line.

    Args:
        line: A single stripped text line.

    Returns:
        Tuple of (line without trailing marker, whether a marker was found).
    """
    match = PAGE_MARKER_RE.search(line)
    if not match:
        return line, False
    return line[:match.start()].rstrip(), True


def _remove_repeated_lines(pages: list[str]) -> tuple[list[str], int, int]:
    """Remove boilerplate lines repeated across many pages.

    Two passes are used:

    1. Exact repetition: a line identical on many pages is boilerplate
       (running headers, document titles, etc.).
    2. Marker-aware repetition: a line that repeats once its trailing page
       marker is stripped is a running footer with per-page numbers.

    Short, identical column headers that merely repeat on each table page are
    left untouched because they carry content and do not end with a marker.

    Args:
        pages: List of raw page texts.

    Returns:
        Tuple of (cleaned page texts, exact lines removed, footer lines removed).
    """
    if len(pages) < 3:
        return pages, 0, 0

    # Pass 1: exact repetition
    exact_counter = Counter()
    for page in pages:
        for line in set(page.split("\n")):
            line = line.strip()
            if line and len(line) <= 100:
                exact_counter[line] += 1

    exact_threshold = max(3, int(len(pages) * 0.5))
    repeated_exact = {line for line, count in exact_counter.items() if count >= exact_threshold}

    # Pass 2: marker-aware repetition (running footers)
    marker_counter = Counter()
    for page in pages:
        for line in set(page.split("\n")):
            line = line.strip()
            if line and len(line) <= 100:
                base, has_marker = _strip_page_marker(line)
                if has_marker and base:
                    marker_counter[base] += 1

    marker_threshold = max(3, int(len(pages) * 0.25))
    repeated_marker = {base for base, count in marker_counter.items() if count >= marker_threshold}

    if not repeated_exact and not repeated_marker:
        return pages, 0, 0

    # Footers can also appear glued to other content on the same line
    # (e.g. after a table row or doubled at a page boundary), and the page
    # marker can even be missing (glued to the previous line). Removing them
    # as substrings keeps the surrounding content intact. Long bases only so
    # short words are never matched by accident.
    footer_patterns = [
        re.compile(re.escape(base) + r"\s*(?:[|·•\-–—]\s*[0-9IVXLCDM]{1,5})?")
        for base in repeated_marker
        if len(base) >= 15
    ]

    exact_removed = 0
    marker_removed = 0
    cleaned = []
    for page in pages:
        lines = []
        for line in page.split("\n"):
            stripped = line.strip()
            if stripped in repeated_exact:
                exact_removed += 1
                continue
            base, has_marker = _strip_page_marker(stripped)
            if has_marker and base in repeated_marker:
                marker_removed += 1
                continue
            cleaned_line = stripped
            for pattern in footer_patterns:
                new_line = pattern.sub("", cleaned_line).strip()
                if new_line != cleaned_line:
                    cleaned_line = new_line
            if cleaned_line != stripped:
                marker_removed += 1
            if cleaned_line:
                lines.append(cleaned_line)
        cleaned.append("\n".join(lines))
    return cleaned, exact_removed, marker_removed


def clean_pages(pages: list[dict]) -> tuple[list[dict], dict]:
    """Clean extracted pages in a general, file-agnostic way.

    Steps:
    1. Normalize unicode (BOM, non-breaking/zero-width chars, control chars).
    2. De-hyphenate words broken across line breaks.
    3. Remove standalone page-number lines.
    4. Collapse multiple blank lines.
    5. Remove repeated boilerplate lines (headers/footers) across pages.
    6. Strip leading/trailing whitespace per line.

    Args:
        pages: List of {"page_number", "text"} dicts.

    Returns:
        Tuple of (cleaned pages, cleaning stats dict).
    """
    chars_before = sum(len(p["text"]) for p in pages)

    normalized = [(p["page_number"], _normalize_text(p["text"])) for p in pages]
    dehyphenated = [(num, _dehyphenate(t)) for num, t in normalized]

    de_nummed = []
    removed_page_numbers = 0
    for num, text in dehyphenated:
        text, removed = _remove_standalone_page_numbers(text)
        removed_page_numbers += removed
        de_nummed.append((num, text))

    collapsed = [(num, _collapse_blank_lines(t)) for num, t in de_nummed]
    cleaned_texts, exact_removed, marker_removed = _remove_repeated_lines([t for _, t in collapsed])

    cleaned_pages = []
    for (num, _), text in zip(collapsed, cleaned_texts):
        text = _strip_lines(text)
        if text.strip():
            cleaned_pages.append({"page_number": num, "text": text})

    stats = {
        "pages_before": len(pages),
        "pages_after": len(cleaned_pages),
        "chars_before": chars_before,
        "chars_after": sum(len(p["text"]) for p in cleaned_pages),
        "page_number_lines_removed": removed_page_numbers,
        "repeated_boilerplate_lines_removed": exact_removed,
        "footer_lines_removed": marker_removed,
    }
    return cleaned_pages, stats


def extract_text(file_path: str) -> Optional[str]:
    """Extract and clean text from a document file as a single string.

    Convenience wrapper around :func:`extract_pages` and :func:`clean_pages`
    that joins all pages. Use :func:`extract_pages` when per-page metadata
    (e.g. page numbers) is required.

    Args:
        file_path: Path to the document file.

    Returns:
        Cleaned text content, or None if extraction failed.
    """
    pages = extract_pages(file_path)
    if not pages:
        return None
    cleaned, _ = clean_pages(pages)
    if not cleaned:
        return None
    return "\n\n".join(page["text"] for page in cleaned)
