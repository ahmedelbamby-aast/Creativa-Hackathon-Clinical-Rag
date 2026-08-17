"""Chunker adapter — bridges the SmartChunker with the RAG ingestion pipeline.

Takes parsed DocumentElements and produces RAG-ready chunk records with
full metadata. Each chunk record is ready to be embedded and stored in pgvector.
"""

import hashlib
import logging
import re
from typing import Optional

from src.config import config
from src.ingestion.core.chunker import SmartChunker
from src.ingestion.core.quality_filter import filter_chunks
from src.ingestion.core.language_detector import detect_language
from src.ingestion.category_classifier import classify_chunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
ChunkRecord = dict  # see build_chunk_record() for full schema


def _make_chunk_id(document_name: str, page_number: int, index: int) -> str:
    """Generate a deterministic, URL-safe chunk ID."""
    stem = re.sub(r"[^\w]", "_", document_name.rsplit(".", 1)[0])[:40]
    return f"{stem}_p{page_number}_c{index}"


def _clean_content(text: str) -> str:
    """Normalise whitespace without destroying structure."""
    # Collapse runs of 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove BOM and control chars (keep newlines and tabs)
    text = text.replace("\ufeff", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()


def build_chunk_record(
    chunk_text: str,
    quality_score: float,
    document_name: str,
    page_number: int,
    section_title: str,
    subsection_title: str,
    content_type: str,
    category: str,
    language: str,
    global_index: int,
) -> ChunkRecord:
    """Build a metadata-rich chunk record ready for pgvector storage."""
    chunk_id = _make_chunk_id(document_name, page_number, global_index)
    return {
        "chunk_id": chunk_id,
        "document_name": document_name,
        "page_number": page_number,
        "section_title": section_title,
        "subsection_title": subsection_title,
        "category": category,
        "content_type": content_type,
        "language": language,
        "text": chunk_text,
        "char_count": len(chunk_text),
        "word_count": len(chunk_text.split()),
        "quality_score": round(quality_score, 4),
    }


def chunk_elements(
    elements: list[dict],
    document_language: str = "en",
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    min_chunk_size: Optional[int] = None,
    min_quality_score: Optional[float] = None,
) -> list[ChunkRecord]:
    """Convert parsed DocumentElements into chunk records.

    Consecutive text elements on the same page, in the same category and
    language, are merged before chunking. PDF parsers commonly emit one tiny
    element per visual block; embedding those blocks independently produces
    weak context and quickly exhausts hosted embedding quotas. Page boundaries
    remain hard boundaries so citations stay exact. Tables remain standalone.

    Tables are passed as single chunks (the SmartChunker protects them
    via its BLOCK_PATTERNS if they are in Markdown table format).

    Args:
        elements: Parsed document elements from parser.parse_document().
        document_language: Language of the source document.
        chunk_size: Override config.chunk_size.
        chunk_overlap: Override config.chunk_overlap.
        min_chunk_size: Override config.min_chunk_size.
        min_quality_score: Override config.min_quality_score.

    Returns:
        List of ChunkRecord dicts with full metadata.
    """
    chunk_size = chunk_size or config.chunk_size
    chunk_overlap = chunk_overlap or config.chunk_overlap
    min_chunk_size = min_chunk_size or config.min_chunk_size
    min_quality_score = min_quality_score if min_quality_score is not None else config.min_quality_score

    chunker = SmartChunker(
        max_chunk_size=chunk_size,
        overlap_size=chunk_overlap,
        min_chunk_size=min_chunk_size,
    )

    semantic_units: list[dict] = []
    pending: dict | None = None

    def flush_pending() -> None:
        nonlocal pending
        if pending is not None:
            semantic_units.append(pending)
            pending = None

    for element in elements:
        raw_content = _clean_content(element.get("content", ""))
        if not raw_content:
            continue

        doc_name = element.get("document_name", "unknown")
        page_num = element.get("page_number", 1)
        section = element.get("section_title", "")
        subsection = element.get("subsection_title", "")
        content_type = element.get("content_type", "text")

        # Assign category
        category = classify_chunk(
            document_name=doc_name,
            section_title=section,
            subsection_title=subsection,
            content=raw_content,
            content_type=content_type,
        )

        prepared = {
            "content": raw_content,
            "document_name": doc_name,
            "page_number": page_num,
            "section_title": section,
            "subsection_title": subsection,
            "content_type": content_type,
            "category": category,
            "language": document_language,
        }

        if content_type == "table":
            flush_pending()
            semantic_units.append(prepared)
            continue

        # Short PDF blocks are too small for reliable language identification
        # and caused false Dutch/Swahili/etc. labels that fragmented pages.
        # Detect language after the page-scoped text has been assembled.
        unit_key = (doc_name, page_num, category)
        if pending is not None and pending["_key"] == unit_key:
            pending["content"] += "\n\n" + raw_content
            if pending["content_type"] == "heading" and content_type != "heading":
                pending["content_type"] = "text"
            if not pending["section_title"] and section:
                pending["section_title"] = section
            if not pending["subsection_title"] and subsection:
                pending["subsection_title"] = subsection
            continue

        flush_pending()
        pending = {**prepared, "_key": unit_key}

    flush_pending()

    records: list[ChunkRecord] = []
    global_index = 0
    for unit in semantic_units:
        raw_content = unit["content"]
        content_type = unit["content_type"]
        unit["language"] = detect_language(raw_content[:1500]) or document_language

        # Tables: pass as one chunk (don't split them)
        if content_type == "table":
            raw_chunks = [raw_content]
        else:
            raw_chunks = chunker.chunk(raw_content)

        if not raw_chunks:
            continue

        # Quality filter
        scored = filter_chunks(raw_chunks, min_score=min_quality_score)

        for chunk_text, score in scored:
            record = build_chunk_record(
                chunk_text=chunk_text,
                quality_score=score,
                document_name=unit["document_name"],
                page_number=unit["page_number"],
                section_title=unit["section_title"],
                subsection_title=unit["subsection_title"],
                content_type=content_type,
                category=unit["category"],
                language=unit["language"],
                global_index=global_index,
            )
            records.append(record)
            global_index += 1

    logger.info(
        "Chunked %d elements via %d page-scoped semantic units → %d chunk records",
        len(elements),
        len(semantic_units),
        len(records),
    )
    return records
