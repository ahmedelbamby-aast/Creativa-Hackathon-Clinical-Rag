"""Semantic retriever — queries pgvector and returns grounded chunks.

Retrieves the most relevant chunks for a user query, filtered by:
- Category (treatment / prevention / nutrition / all)
- Similarity threshold (drops irrelevant results)

Returns structured RetrievedChunk objects with all metadata needed
for citation generation.
"""

import logging
from dataclasses import dataclass

from src.config import config, CATEGORY_ALL
from src.embeddings import embedder
from src.vector_store import vector_store

logger = logging.getLogger(__name__)


def _is_reference_only(section_title: str, text: str) -> bool:
    """Return True for bibliography chunks, which are provenance—not answer evidence."""
    section = " ".join(section_title.casefold().split()).strip(" :.-")
    opening = " ".join(text.casefold().split())[:80]
    return section in {"references", "reference", "bibliography", "المراجع"} or opening.startswith(
        ("references ", "bibliography ", "المراجع ")
    )


class RetrievalProviderError(RuntimeError):
    """Embedding or vector-provider error that must not be mistaken for no evidence."""


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with full provenance metadata."""
    chunk_id: str
    text: str
    score: float           # Similarity score [0, 1] — higher = more relevant
    distance: float        # Raw pgvector cosine distance
    document_name: str
    page_number: int
    section_title: str
    subsection_title: str
    category: str
    content_type: str
    language: str
    # Source-level provenance fields (populated when ingested with manifest metadata)
    source_id: str = ""
    source_url: str = ""
    publisher: str = ""
    publication_date: str = ""
    source_checksum: str = ""
    chunk_profile: str = ""


def retrieve(
    query: str,
    category: str = CATEGORY_ALL,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
) -> list[RetrievedChunk]:
    """Retrieve the most relevant chunks for a query.

    Args:
        query: User question (English or Arabic).
        category: Category filter — one of ALL_CATEGORIES or "all".
        top_k: Number of results to fetch (defaults to config.top_k).
        similarity_threshold: Minimum score to keep (defaults to config.similarity_threshold).
            Set to 0.0 to disable threshold filtering.

    Returns:
        List of RetrievedChunk objects sorted by score descending.
        Empty list if no relevant results found.
    """
    top_k = top_k or config.top_k
    threshold = similarity_threshold if similarity_threshold is not None else config.similarity_threshold

    if not query or not query.strip():
        logger.warning("Empty query passed to retriever")
        return []

    # Embed the query
    try:
        query_vector = embedder.embed_query(query.strip())
    except Exception as e:
        logger.error("Failed to embed retrieval query: type=%s", type(e).__name__)
        raise RetrievalProviderError("query_embedding_failed") from e

    # Query vector store
    raw_results = vector_store.query(
        query_embedding=query_vector,
        category=category,
        top_k=top_k * 2,  # Over-fetch to allow for threshold filtering
    )

    if not raw_results:
        logger.info("No results returned from pgvector for query: %r", query[:80])
        return []

    # Convert to RetrievedChunk and apply threshold
    chunks: list[RetrievedChunk] = []
    for r in raw_results:
        score = r.get("score", 0.0)
        if score < threshold:
            logger.debug(
                "Filtered chunk (score=%.3f < threshold=%.3f): %r...",
                score, threshold, r.get("document", "")[:60],
            )
            continue

        meta = r.get("metadata", {})
        if _is_reference_only(meta.get("section_title", ""), r.get("document", "")):
            logger.debug("Filtered bibliography chunk: %s", meta.get("chunk_id", r.get("id", "")))
            continue
        chunk = RetrievedChunk(
            chunk_id=meta.get("chunk_id", r.get("id", "")),
            text=r.get("document", ""),
            score=score,
            distance=r.get("distance", 0.0),
            document_name=meta.get("document_name", "Unknown"),
            page_number=int(meta.get("page_number", 0)),
            section_title=meta.get("section_title", ""),
            subsection_title=meta.get("subsection_title", ""),
            category=meta.get("category", ""),
            content_type=meta.get("content_type", "text"),
            language=meta.get("language", "en"),
            source_id=meta.get("source_id", ""),
            source_url=meta.get("source_url", ""),
            publisher=meta.get("publisher", ""),
            publication_date=meta.get("publication_date", ""),
            source_checksum=meta.get("source_checksum", ""),
            chunk_profile=meta.get("chunk_profile", ""),
        )
        chunks.append(chunk)

        if len(chunks) >= top_k:
            break

    logger.info(
        "Retrieved %d chunks (category=%s, threshold=%.2f)",
        len(chunks), category, threshold,
    )

    if config.debug:
        for i, c in enumerate(chunks):
            logger.debug(
                "  [%d] score=%.3f | %s | p.%d | %s",
                i + 1, c.score, c.document_name, c.page_number,
                c.section_title[:50] if c.section_title else "(no section)",
            )

    return chunks


def is_retrieval_sufficient(chunks: list[RetrievedChunk]) -> bool:
    """Return True if retrieved chunks contain enough information to answer.

    Criteria:
    - At least 1 chunk was retrieved.
    - The best chunk score is above a minimum confidence level (0.25).
    """
    if not chunks:
        return False
    return chunks[0].score >= 0.25
