"""Phase 2 retrieval contract and configuration tests."""

from dataclasses import FrozenInstanceError

import pytest

from src.config import AppConfig, CHUNK_PROFILES
from src.retrieval_contracts import EvidenceChunk, RetrievalEnvelope


def test_selected_profile_exposes_fixed_chunk_settings() -> None:
    config = AppConfig(retrieval_profile="small")

    assert config.selected_chunk_profile == CHUNK_PROFILES["small"]


def test_configuration_rejects_unknown_retrieval_profile() -> None:
    with pytest.raises(ValueError, match="RETRIEVAL_PROFILE"):
        AppConfig(retrieval_profile="unknown").validate()


def test_configuration_rejects_conflicting_namespace_names() -> None:
    with pytest.raises(ValueError, match="conflict"):
        AppConfig(
            active_index_namespace="phase2_local",
            embedding_namespace="legacy_local",
        ).validate()


def test_active_index_namespace_takes_precedence() -> None:
    config = AppConfig(
        active_index_namespace="phase2_local",
        embedding_namespace="phase2_local",
    )

    assert config.resolved_embedding_namespace == "phase2_local"


def test_retrieval_envelope_is_immutable_and_ready_only_with_evidence() -> None:
    chunk = EvidenceChunk(
        chunk_id="c1",
        text="Evidence",
        score=0.9,
        distance=0.1,
        document_name="guide.pdf",
        page_number=3,
        section_title="Care",
        subsection_title="",
        category="treatment",
        language="en",
        source_id="guide",
        source_url="https://example.test/guide",
    )
    envelope = RetrievalEnvelope(
        original_query="Question",
        rewritten_query="Question",
        requested_category="all",
        routed_category="treatment",
        namespace="phase2_local",
        index_manifest_hash="abc",
        status="ready",
        chunks=(chunk,),
    )

    assert envelope.is_ready is True
    with pytest.raises(FrozenInstanceError):
        envelope.status = "empty_evidence"  # type: ignore[misc]
