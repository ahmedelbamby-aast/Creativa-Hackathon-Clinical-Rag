"""Citation builder — generates source references from retrieved chunks.

Citations are built exclusively from chunk metadata — never from LLM output.
This guarantees that page numbers, document names, and sections are accurate.

Two citation modes:
1. Inline context markers: prepended to each chunk before it's sent to Gemini,
   so the LLM can reference sources naturally in its answer.
2. Formatted source list: appended to the final answer for the user to read.
"""

import logging
import re
from src.retriever import RetrievedChunk
from src.source_catalog import load_source_catalog

logger = logging.getLogger(__name__)


def normalize_inline_citations(answer: str, evidence_count: int) -> str:
    """Canonicalize provider citation variants to clickable ``[E#]`` markers."""
    pattern = re.compile(r"(?:\[|【)\s*E\s*(\d+)[^\]】]*(?:\]|】)", re.IGNORECASE)

    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        return f"[E{index}]" if 1 <= index <= evidence_count else match.group(0)

    return pattern.sub(replace, answer)


# ---------------------------------------------------------------------------
# Inline context labelling (for the prompt, not shown to user directly)
# ---------------------------------------------------------------------------

def label_chunk_for_context(chunk: RetrievedChunk, index: int) -> str:
    """Wrap a chunk's text with a source label for the LLM prompt.

    The label tells the LLM exactly where this information comes from so it
    can cite it correctly in its answer.

    Example output:
        [SOURCE 1: IDF Diabetes Atlas 2025 | Section: Prevention | Page 42]
        Physical activity interventions have shown...
    """
    parts = [f"E{index + 1}"]

    # Document name (strip common long prefixes)
    doc = chunk.document_name
    parts.append(doc)

    if chunk.section_title:
        parts.append(f"Section: {chunk.section_title[:80]}")
    if chunk.subsection_title:
        parts.append(f"Subsection: {chunk.subsection_title[:60]}")
    if chunk.page_number:
        parts.append(f"Page {chunk.page_number}")

    header = " | ".join(parts)
    return f"[{header}]\n{chunk.text}"


# ---------------------------------------------------------------------------
# Source list (shown to user after the answer)
# ---------------------------------------------------------------------------

def _format_document_name(name: str) -> str:
    """Make a document name more readable by removing file extension and underscores."""
    name = name.rsplit(".", 1)[0]           # Remove extension
    name = name.replace("_", " ")           # Underscores → spaces
    name = name.replace("-", " – ")         # Dashes → em-dashes for readability
    # Collapse multiple spaces
    import re
    name = re.sub(r" {2,}", " ", name)
    return name.strip()


def build_citation_list(
    chunks: list[RetrievedChunk],
    is_arabic: bool = False,
) -> str:
    """Build a formatted source citation list from retrieved chunks.

    Deduplicates by (document_name, page_number, section_title) so the same
    passage cited twice doesn't appear twice in the source list.

    Args:
        chunks: List of retrieved chunks used to generate the answer.
        is_arabic: If True, use Arabic labels.

    Returns:
        Formatted string with source citations, or empty string if no chunks.
    """
    if not chunks:
        return ""

    seen: set[tuple] = set()
    citations: list[str] = []
    catalog = load_source_catalog()

    for index, chunk in enumerate(chunks, start=1):
        key = (chunk.document_name, chunk.page_number, chunk.section_title)
        if key in seen:
            continue
        seen.add(key)

        doc_display = _format_document_name(chunk.document_name)
        linked_title = f"[{doc_display}]({chunk.source_url})" if chunk.source_url else doc_display
        parts = [f"**E{index} · {linked_title}**"]

        if chunk.section_title:
            parts.append(f"Section: *{chunk.section_title[:80]}*")
        if chunk.page_number:
            page_label = "صفحة" if is_arabic else "Page"
            parts.append(f"{page_label} {chunk.page_number}")

        source = catalog.get(chunk.document_name)
        publisher = chunk.publisher or (source.publisher if source else "")
        publication_date = chunk.publication_date or (source.publication_date if source else "")
        if publisher:
            parts.append(publisher)
        if publication_date:
            parts.append(publication_date)

        citations.append("• " + " — ".join(parts))

    if not citations:
        return ""

    header = "📚 **المصادر:**" if is_arabic else "📚 **Sources:**"
    return header + "\n" + "\n".join(citations)


def build_citation_records(chunks: list[RetrievedChunk]) -> list[dict[str, object]]:
    """Return structured citations whose IDs match the evidence sent to generation."""
    catalog = load_source_catalog()
    records: list[dict[str, object]] = []
    for index, chunk in enumerate(chunks, start=1):
        source = catalog.get(chunk.document_name)
        records.append(
            {
                "evidence_id": f"E{index}",
                "source_id": chunk.source_id,
                "document_name": chunk.document_name,
                "source_url": chunk.source_url,
                "publisher": chunk.publisher or (source.publisher if source else ""),
                "publication_date": chunk.publication_date or (
                    source.publication_date if source else ""
                ),
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
            }
        )
    return records


def build_debug_info(
    query: str,
    rewritten_query: str,
    routed_category: str,
    chunks: list[RetrievedChunk],
) -> str:
    """Build a debug information block showing retrieval internals.

    Only displayed when DEBUG=true in the selected environment.
    """
    lines = [
        "```",
        "── DEBUG INFO ──────────────────────────────────",
        f"Original query   : {query[:120]}",
        f"Rewritten query  : {rewritten_query[:120]}",
        f"Routed category  : {routed_category}",
        f"Chunks retrieved : {len(chunks)}",
        "",
    ]
    for i, c in enumerate(chunks):
        lines.append(
            f"  [{i+1}] score={c.score:.3f} | {c.document_name} | "
            f"p.{c.page_number} | cat={c.category} | type={c.content_type}"
        )
        if c.section_title:
            lines.append(f"       section: {c.section_title[:80]}")
        lines.append(f"       preview: {c.text[:100].strip()}...")
        lines.append("")
    lines.append("─────────────────────────────────────────────────")
    lines.append("```")
    return "\n".join(lines)
