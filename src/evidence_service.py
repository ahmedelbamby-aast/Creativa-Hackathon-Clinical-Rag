"""Retrieval-first evidence staging for the interactive RAG application."""

from __future__ import annotations

import html
import re

from src.config import config
from src.index_manifests import index_manifest_hash, load_index_manifest, manifest_matches_runtime
from src.retrieval_contracts import EvidenceChunk, RetrievalEnvelope
from src.retriever import RetrievedChunk, is_retrieval_sufficient, retrieve
from src.rewriter import rewrite_query
from src.router import route_query
from src.safety import SafetyLevel, classify_safety, get_emergency_response
from src.vector_store import vector_store
from src.gemini_errors import classify_gemini_error, gemini_user_message


def is_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text))


def _message(status: str, is_ar: bool, emergency: str = "") -> str:
    if emergency:
        return emergency
    messages = {
        "empty_evidence": (
            "أعتذر، لكن الوثائق المتاحة لا تحتوي على معلومات كافية للإجابة على هذا السؤال.",
            "The available documents do not contain sufficient information to answer this question.",
        ),
        "insufficient_evidence": (
            "أعتذر، لكن الأدلة المسترجعة غير كافية للإجابة بأمان على هذا السؤال.",
            "The retrieved evidence is insufficient to answer this question safely.",
        ),
        "invalid_provenance": (
            "أعتذر، لا يمكن استخدام الأدلة المسترجعة لأن مصدرها غير مكتمل.",
            "The retrieved evidence cannot be used because its source information is incomplete.",
        ),
        "stale_index": (
            "أعتذر، قاعدة المعرفة غير جاهزة حالياً. يرجى المحاولة لاحقاً.",
            "The knowledge base is not ready right now. Please try again later.",
        ),
        "infrastructure_failure": (
            "أعتذر، تعذر الوصول إلى قاعدة المعرفة حالياً. يرجى المحاولة لاحقاً.",
            "The knowledge base is temporarily unavailable. Please try again later.",
        ),
        "safety_blocked": ("", ""),
    }
    arabic, english = messages[status]
    return arabic if is_ar else english


def stage_evidence(
    query: str,
    category: str,
    conversation_history: list[dict] | None = None,
) -> RetrievalEnvelope:
    """Retrieve once and return the exact evidence that may later reach generation."""
    query = query.strip()
    arabic = is_arabic(query)
    common = {
        "original_query": query,
        "requested_category": category,
        "routed_category": category,
        "namespace": config.resolved_embedding_namespace,
        "index_manifest_hash": "",
    }
    if not query:
        return RetrievalEnvelope(**common, rewritten_query="", status="empty_evidence", user_message=_message("empty_evidence", arabic))
    safety = classify_safety(query)
    if safety == SafetyLevel.EMERGENCY:
        return RetrievalEnvelope(
            **common,
            rewritten_query=query,
            status="safety_blocked",
            user_message=get_emergency_response(is_arabic=arabic),
        )
    try:
        manifest = load_index_manifest(config.resolved_embedding_namespace)
        if manifest is None or not manifest_matches_runtime(manifest, config.resolved_embedding_namespace):
            return RetrievalEnvelope(
                **common,
                rewritten_query=query,
                status="stale_index",
                user_message=_message("stale_index", arabic),
            )
        rewritten = rewrite_query(query, conversation_history=conversation_history)
        routed = route_query(rewritten, user_selected_category=category)
        chunks = retrieve(rewritten, category=routed, top_k=config.top_k)
        common.update(routed_category=routed, index_manifest_hash=index_manifest_hash(manifest))
        if not chunks:
            return RetrievalEnvelope(
                **common,
                rewritten_query=rewritten,
                status="empty_evidence",
                user_message=_message("empty_evidence", arabic),
            )
        if not is_retrieval_sufficient(chunks):
            return RetrievalEnvelope(
                **common,
                rewritten_query=rewritten,
                status="insufficient_evidence",
                user_message=_message("insufficient_evidence", arabic),
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
            )
            for chunk in chunks
        )
        if any(not item.source_id or not item.source_url.startswith("https://") for item in evidence):
            return RetrievalEnvelope(
                **common,
                rewritten_query=rewritten,
                status="invalid_provenance",
                user_message=_message("invalid_provenance", arabic),
            )
        return RetrievalEnvelope(**common, rewritten_query=rewritten, status="ready", chunks=evidence)
    except Exception as error:
        return RetrievalEnvelope(
            **common,
            rewritten_query=query,
            status="infrastructure_failure",
            error_code=f"gemini:{classify_gemini_error(error).code}",
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
        )
        for item in envelope.chunks
    ]


def rehydrate_evidence(
    query: str,
    category: str,
    namespace: str,
    manifest_hash: str,
    chunk_ids: list[str],
) -> RetrievalEnvelope:
    """Rebuild previously displayed evidence by exact IDs without a new retrieval call."""
    arabic = is_arabic(query)
    common = {
        "original_query": query,
        "rewritten_query": query,
        "requested_category": category,
        "routed_category": category,
        "namespace": namespace,
        "index_manifest_hash": manifest_hash,
    }
    try:
        if namespace != config.resolved_embedding_namespace:
            raise ValueError("namespace changed")
        manifest = load_index_manifest(namespace)
        if (
            manifest is None
            or not manifest_matches_runtime(manifest, namespace)
            or index_manifest_hash(manifest) != manifest_hash
        ):
            return RetrievalEnvelope(
                **common,
                status="stale_index",
                user_message=_message("stale_index", arabic),
            )
        raw_chunks = vector_store.get_chunks(chunk_ids)
        if len(raw_chunks) != len(chunk_ids):
            return RetrievalEnvelope(
                **common,
                status="stale_index",
                user_message=_message("stale_index", arabic),
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
            )
            for item in raw_chunks
        )
        if any(not item.source_id or not item.source_url.startswith("https://") for item in chunks):
            return RetrievalEnvelope(
                **common,
                status="invalid_provenance",
                user_message=_message("invalid_provenance", arabic),
            )
        return RetrievalEnvelope(**common, status="ready", chunks=chunks)
    except Exception as error:
        return RetrievalEnvelope(
            **common,
            status="infrastructure_failure",
            error_code=f"gemini:{classify_gemini_error(error).code}",
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
