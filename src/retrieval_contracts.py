"""Typed, serialisable contracts for Phase 2 retrieval evaluation and evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Literal


RetrievalStatus = Literal[
    "ready",
    "needs_clarification",
    "out_of_scope",
    "empty_evidence",
    "insufficient_evidence",
    "invalid_provenance",
    "stale_index",
    "infrastructure_failure",
    "safety_blocked",
]
Relevance = Literal["relevant", "not_relevant", "unjudged"]


@dataclass(frozen=True)
class SourceManifestEntry:
    """Operational source identity used for traceable retrieval results."""

    source_id: str
    document_name: str
    source_url: str
    publisher: str
    publication_date: str
    license_note: str
    reuse_status: str
    checksum: str
    enabled: bool = True


@dataclass(frozen=True)
class IndexManifest:
    """Immutable description of one reproducible retrieval index."""

    namespace: str
    corpus_hash: str
    chunk_profile: str
    embedding_provider: str
    embedding_model: str
    dimension: int
    created_at: str
    source_catalog_hash: str = ""
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    char_chunk_size: int = 0
    char_chunk_overlap: int = 0
    token_count: int = 0

    def serializable(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalCase:
    """A stable retrieval case independent of configuration-specific chunk IDs."""

    case_id: str
    query: str
    language: Literal["en", "ar"]
    category: str
    expect_evidence: bool
    expected_source_id: str = ""
    expected_document_name: str = ""
    expected_page: int = 0
    expected_section: str = ""
    text_anchors: tuple[str, ...] = ()


@dataclass(frozen=True)
class RelevanceLabel:
    """Human-reviewable relevance decision for one ranked evidence result."""

    run_id: str
    case_id: str
    rank: int
    chunk_id: str
    relevance: Relevance = "unjudged"
    reviewer_a: str = ""
    reviewer_b: str = ""
    reviewer_a_label: Relevance = "unjudged"
    reviewer_b_label: Relevance = "unjudged"
    rationale: str = ""


@dataclass(frozen=True)
class EvidenceChunk:
    """Displayed evidence, copied from retrieval so generation cannot re-query it."""

    chunk_id: str
    text: str
    score: float
    distance: float
    document_name: str
    page_number: int
    section_title: str
    subsection_title: str
    category: str
    language: str
    source_id: str
    source_url: str


@dataclass(frozen=True)
class RetrievalEnvelope:
    """The complete, immutable input to a generation attempt."""

    original_query: str
    rewritten_query: str
    requested_category: str
    routed_category: str
    namespace: str
    index_manifest_hash: str
    status: RetrievalStatus
    chunks: tuple[EvidenceChunk, ...] = ()
    user_message: str = ""
    error_code: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def is_ready(self) -> bool:
        return self.status == "ready" and bool(self.chunks)
