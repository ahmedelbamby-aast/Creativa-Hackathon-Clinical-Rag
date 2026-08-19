"""Retrieval-first evidence staging for the interactive RAG application."""

from __future__ import annotations

import html
import logging
import re

from src.config import config
from src.index_manifests import (
    index_manifest_hash,
    load_index_manifest,
    manifest_matches_runtime,
    runtime_index_hash,
)
from src.retrieval_contracts import EvidenceChunk, RetrievalEnvelope
from src.retriever import RetrievedChunk, is_retrieval_sufficient, retrieve
from src.rewriter import rewrite_query
from src.router import route_query
from src.safety import SafetyLevel, classify_safety, get_emergency_response
from src.gemini_errors import classify_gemini_error, gemini_user_message
from src.embedding_profiles import EmbeddingRuntime, get_embedding_runtime
from src.response_policy import is_out_of_domain, needs_clarification, response_text


logger = logging.getLogger(__name__)


def _directly_matching_chunks(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Apply condition/intent guardrails that vector similarity alone cannot enforce."""
    normalized_query = " ".join(query.casefold().replace("–", "-").split())
    required_groups: list[tuple[str, ...]] = []
    condition_groups = (
        (("type 1", "type i", "النوع الأول", "النوع الاول"), ("type 1", "type i")),
        (("type 2", "type ii", "النوع الثاني"), ("type 2", "type ii")),
        (("gestational", "pregnan", "الحمل", "الحوامل", "سكر الحمل"), ("gestational", "pregnan")),
    )
    for query_terms, evidence_terms in condition_groups:
        if any(term in normalized_query for term in query_terms):
            required_groups.append(evidence_terms)
            break
    if "risk factor" in normalized_query or "عوامل الخطر" in normalized_query:
        required_groups.append(("risk factor",))
    if "preventive cardiolog" in normalized_query or "طبيب القلب الوقائي" in normalized_query:
        required_groups.append(("preventive cardiolog",))
    if not required_groups:
        return chunks

    matching: list[RetrievedChunk] = []
    for chunk in chunks:
        evidence = " ".join(
            f"{chunk.section_title} {chunk.subsection_title} {chunk.text}".casefold().split()
        )
        if all(any(term in evidence for term in group) for group in required_groups):
            matching.append(chunk)
    return matching


def is_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text))


def _active_index_hash(runtime: EmbeddingRuntime) -> str:
    """Return a compatible local manifest hash or a hosted runtime fingerprint."""
    manifest = load_index_manifest(runtime.namespace)
    if manifest is not None:
        return (
            index_manifest_hash(manifest)
            if manifest_matches_runtime(
                manifest,
                runtime.namespace,
                runtime.dimension,
                runtime.provider,
                runtime.model,
            )
            else ""
        )
    if config.is_deployment:
        return runtime_index_hash(
            runtime.namespace,
            runtime.dimension,
            runtime.provider,
            runtime.model,
        )
    return ""


def stage_evidence(
    query: str,
    category: str,
    conversation_history: list[dict] | None = None,
    embedding_dimension: int | None = None,
) -> RetrievalEnvelope:
    """Retrieve once and return the exact evidence that may later reach generation."""
    query = query.strip()
    arabic = is_arabic(query)
    runtime = get_embedding_runtime(embedding_dimension)
    common = {
        "original_query": query,
        "requested_category": category,
        "routed_category": category,
        "namespace": runtime.namespace,
        "index_manifest_hash": "",
        "embedding_dimension": runtime.dimension,
        "embedding_provider": runtime.provider,
        "embedding_model": runtime.model,
        "embedding_table_family": runtime.table_family,
    }
    if not query:
        return RetrievalEnvelope(
            **common,
            rewritten_query="",
            status="needs_clarification",
            user_message=response_text("empty_question", is_arabic=arabic),
        )
    safety = classify_safety(query)
    if safety == SafetyLevel.EMERGENCY:
        return RetrievalEnvelope(
            **common,
            rewritten_query=query,
            status="safety_blocked",
            user_message=get_emergency_response(is_arabic=arabic),
        )
    if needs_clarification(query, conversation_history):
        return RetrievalEnvelope(
            **common,
            rewritten_query=query,
            status="needs_clarification",
            user_message=response_text("needs_clarification", is_arabic=arabic),
        )
    if is_out_of_domain(query, conversation_history):
        return RetrievalEnvelope(
            **common,
            rewritten_query=query,
            status="out_of_scope",
            user_message=response_text("out_of_scope", is_arabic=arabic),
        )
    try:
        active_hash = _active_index_hash(runtime)
        if not active_hash:
            return RetrievalEnvelope(
                **common,
                rewritten_query=query,
                status="stale_index",
                user_message=response_text("stale_index", is_arabic=arabic),
            )
        rewritten = rewrite_query(query, conversation_history=conversation_history)
        routed = route_query(rewritten, user_selected_category=category)
        chunks = retrieve(
            rewritten,
            category=routed,
            top_k=config.top_k,
            embedding_runtime=runtime,
        )
        chunks = _directly_matching_chunks(query, chunks)
        common.update(routed_category=routed, index_manifest_hash=active_hash)
        if not chunks:
            return RetrievalEnvelope(
                **common,
                rewritten_query=rewritten,
                status="out_of_scope",
                user_message=response_text("out_of_scope", is_arabic=arabic),
            )
        if not is_retrieval_sufficient(chunks):
            return RetrievalEnvelope(
                **common,
                rewritten_query=rewritten,
                status="out_of_scope",
                user_message=response_text("out_of_scope", is_arabic=arabic),
            )
        certified_chunks = [
            chunk
            for chunk in chunks
            if chunk.source_id and chunk.source_url.startswith("https://")
        ]
        skipped_count = len(chunks) - len(certified_chunks)
        if skipped_count:
            logger.warning(
                "Discarded %d retrieved chunk(s) without certified provenance",
                skipped_count,
            )
        if not certified_chunks or not is_retrieval_sufficient(certified_chunks):
            return RetrievalEnvelope(
                **common,
                rewritten_query=rewritten,
                status="invalid_provenance",
                user_message=response_text("invalid_provenance", is_arabic=arabic),
            )
        evidence = tuple(
            EvidenceChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=chunk.score,
                distance=chunk.distance,
                document_name=chunk.document_name,
                page_number=chunk.page_number,
                section_title=chunk.section_title,
                subsection_title=chunk.subsection_title,
                category=chunk.category,
                language=chunk.language,
                source_id=chunk.source_id,
                source_url=chunk.source_url,
                publisher=chunk.publisher,
                publication_date=chunk.publication_date,
            )
            for chunk in certified_chunks
        )
        return RetrievalEnvelope(**common, rewritten_query=rewritten, status="ready", chunks=evidence)
    except Exception as error:
        error_info = classify_gemini_error(error)
        logger.exception(
            "Evidence staging failed: code=%s error_type=%s",
            error_info.code,
            type(error).__name__,
        )
        return RetrievalEnvelope(
            **common,
            rewritten_query=query,
            status="infrastructure_failure",
            error_code=f"gemini:{error_info.code}",
            user_message=gemini_user_message(error, is_arabic=arabic, scope="retrieval"),
        )


def envelope_chunks(envelope: RetrievalEnvelope) -> list[RetrievedChunk]:
    """Convert immutable staged evidence to the legacy prompt/citation input type."""
    return [
        RetrievedChunk(
            chunk_id=item.chunk_id,
            text=item.text,
            score=item.score,
            distance=item.distance,
            document_name=item.document_name,
            page_number=item.page_number,
            section_title=item.section_title,
            subsection_title=item.subsection_title,
            category=item.category,
            content_type="text",
            language=item.language,
            source_id=item.source_id,
            source_url=item.source_url,
            publisher=item.publisher,
            publication_date=item.publication_date,
        )
        for item in envelope.chunks
    ]


def rehydrate_evidence(
    query: str,
    category: str,
    namespace: str,
    manifest_hash: str,
    chunk_ids: list[str],
    embedding_dimension: int | None = None,
) -> RetrievalEnvelope:
    """Rebuild previously displayed evidence by exact IDs without a new retrieval call."""
    arabic = is_arabic(query)
    runtime = get_embedding_runtime(embedding_dimension)
    common = {
        "original_query": query,
        "rewritten_query": query,
        "requested_category": category,
        "routed_category": category,
        "namespace": namespace,
        "index_manifest_hash": manifest_hash,
        "embedding_dimension": runtime.dimension,
        "embedding_provider": runtime.provider,
        "embedding_model": runtime.model,
        "embedding_table_family": runtime.table_family,
    }
    try:
        if namespace != runtime.namespace:
            raise ValueError("namespace changed")
        active_hash = _active_index_hash(runtime)
        if not active_hash or active_hash != manifest_hash:
            return RetrievalEnvelope(
                **common,
                status="stale_index",
                user_message=response_text("stale_index", is_arabic=arabic),
            )
        raw_chunks = runtime.vector_store.get_chunks(chunk_ids)
        if len(raw_chunks) != len(chunk_ids):
            return RetrievalEnvelope(
                **common,
                status="stale_index",
                user_message=response_text("stale_index", is_arabic=arabic),
            )
        chunks = tuple(
            EvidenceChunk(
                chunk_id=item["id"],
                text=item["document"],
                score=item["score"],
                distance=item["distance"],
                document_name=item["metadata"].get("document_name", ""),
                page_number=int(item["metadata"].get("page_number") or 0),
                section_title=item["metadata"].get("section_title", ""),
                subsection_title=item["metadata"].get("subsection_title", ""),
                category=item["metadata"].get("category", ""),
                language=item["metadata"].get("language", "en"),
                source_id=item["metadata"].get("source_id", ""),
                source_url=item["metadata"].get("source_url", ""),
                publisher=item["metadata"].get("publisher", ""),
                publication_date=item["metadata"].get("publication_date", ""),
            )
            for item in raw_chunks
        )
        if any(not item.source_id or not item.source_url.startswith("https://") for item in chunks):
            return RetrievalEnvelope(
                **common,
                status="invalid_provenance",
                user_message=response_text("invalid_provenance", is_arabic=arabic),
            )
        return RetrievalEnvelope(**common, status="ready", chunks=chunks)
    except Exception as error:
        error_info = classify_gemini_error(error)
        logger.exception(
            "Evidence rehydration failed: code=%s error_type=%s",
            error_info.code,
            type(error).__name__,
        )
        return RetrievalEnvelope(
            **common,
            status="infrastructure_failure",
            error_code=f"gemini:{error_info.code}",
            user_message=gemini_user_message(error, is_arabic=arabic, scope="retrieval"),
        )


def render_evidence(envelope: RetrievalEnvelope) -> str:
    """Render exactly the staged ranked text and provenance before generation."""
    if not envelope.is_ready:
        return envelope.user_message or "*No evidence is available for this question.*"
    lines = ["### Retrieved evidence", ""]
    for rank, item in enumerate(envelope.chunks, start=1):
        lines.extend(
            [
                f"#### {rank}. {html.escape(item.document_name)} · score {item.score:.3f}",
                f"Section: {html.escape(item.section_title or 'Not specified')} · Page {item.page_number}",
                f"Source: [{html.escape(item.source_id)}]({item.source_url})",
                "",
                "```text",
                item.text,
                "```",
                "",
            ]
        )
    return "\n".join(lines)
