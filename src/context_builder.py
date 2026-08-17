"""Context builder — assembles the final retrieval context for the LLM.

Takes retrieved chunks and formats them into a structured context block that:
1. Is clearly separated from the LLM's instructions
2. Preserves source attribution (document, section, page)
3. Limits total context length to avoid token limits
4. Prioritises higher-scoring chunks when truncation is needed
"""

import logging
from src.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

# Approximate max characters of retrieved context to include in prompt.
# ~12,000 chars ≈ ~3,000 tokens, leaving room for the prompt structure + output.
MAX_CONTEXT_CHARS = 12_000


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Build a single context string from retrieved chunks.

    Chunks are already sorted by score (descending). If the total context
    exceeds MAX_CONTEXT_CHARS, lower-scoring chunks are truncated.

    Args:
        chunks: Sorted list of RetrievedChunk objects.

    Returns:
        Formatted context string for inclusion in the LLM prompt.
        Empty string if no chunks provided.
    """
    if not chunks:
        return ""

    context_parts: list[str] = []
    total_chars = 0

    for i, chunk in enumerate(chunks):
        # Build source header
        header_parts = [f"[SOURCE {i + 1}]"]
        header_parts.append(f"Document: {chunk.document_name}")
        if chunk.section_title:
            header_parts.append(f"Section: {chunk.section_title[:100]}")
        if chunk.subsection_title:
            header_parts.append(f"Subsection: {chunk.subsection_title[:80]}")
        if chunk.page_number:
            header_parts.append(f"Page: {chunk.page_number}")
        header_parts.append(f"Category: {chunk.category}")

        header = " | ".join(header_parts)
        text = chunk.text.strip()

        block = f"{header}\n{text}"
        block_len = len(block)

        if total_chars + block_len > MAX_CONTEXT_CHARS and context_parts:
            # Truncate to fit within budget
            remaining = MAX_CONTEXT_CHARS - total_chars
            if remaining > 200:
                truncated_text = text[:remaining - len(header) - 50]
                # Snap to last word boundary
                last_space = truncated_text.rfind(" ")
                if last_space > 0:
                    truncated_text = truncated_text[:last_space]
                block = f"{header}\n{truncated_text}... [truncated]"
                context_parts.append(block)
            break

        context_parts.append(block)
        total_chars += block_len

    logger.debug(
        "Context built: %d/%d chunks, ~%d chars",
        len(context_parts), len(chunks), total_chars,
    )

    return "\n\n" + ("=" * 50) + "\n\n".join(context_parts)
