"""Evidence staging safety and provenance tests."""

from types import SimpleNamespace

from src import evidence_service
from src.retriever import RetrievedChunk
from src.retriever import RetrievalProviderError


def _chunk(*, source_url: str = "https://example.test/guide") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk-1", text="Evidence", score=0.8, distance=0.2,
        document_name="guide.pdf", page_number=4, section_title="Care",
        subsection_title="", category="treatment", content_type="text", language="en",
        source_id="guide", source_url=source_url,
    )


def _ready_manifest(monkeypatch) -> None:
    monkeypatch.setattr(evidence_service, "load_index_manifest", lambda _: SimpleNamespace())
    monkeypatch.setattr(evidence_service, "manifest_matches_runtime", lambda *args: True)
    monkeypatch.setattr(evidence_service, "index_manifest_hash", lambda _: "manifest")
    monkeypatch.setattr(evidence_service, "rewrite_query", lambda query, **_: query)
    monkeypatch.setattr(evidence_service, "route_query", lambda query, **_: "treatment")


def test_stage_rejects_missing_provenance(monkeypatch) -> None:
    _ready_manifest(monkeypatch)
    monkeypatch.setattr(evidence_service, "retrieve", lambda *args, **kwargs: [_chunk(source_url="")])

    envelope = evidence_service.stage_evidence("How can diabetes complications be prevented?", "all")

    assert envelope.status == "invalid_provenance"
    assert not envelope.chunks


def test_stage_and_render_preserve_exact_evidence(monkeypatch) -> None:
    _ready_manifest(monkeypatch)
    monkeypatch.setattr(evidence_service, "retrieve", lambda *args, **kwargs: [_chunk()])

    envelope = evidence_service.stage_evidence("How can diabetes complications be prevented?", "all")
    rendered = evidence_service.render_evidence(envelope)

    assert envelope.is_ready
    assert envelope_chunks_ids(envelope) == ["chunk-1"]
    assert "Evidence" in rendered and "https://example.test/guide" in rendered


def test_stage_discards_uncertified_chunks_but_keeps_certified_evidence(monkeypatch) -> None:
    _ready_manifest(monkeypatch)
    uncertified = _chunk(source_url="")
    certified = _chunk()
    certified.chunk_id = "chunk-2"
    monkeypatch.setattr(
        evidence_service,
        "retrieve",
        lambda *args, **kwargs: [uncertified, certified],
    )

    envelope = evidence_service.stage_evidence(
        "How can diabetes complications be prevented?",
        "all",
    )

    assert envelope.is_ready
    assert [chunk.chunk_id for chunk in envelope.chunks] == ["chunk-2"]


def test_rehydrate_does_not_embed_or_reretrieve(monkeypatch) -> None:
    _ready_manifest(monkeypatch)
    monkeypatch.setattr(evidence_service.vector_store, "get_chunks", lambda _: [{
        "id": "chunk-1", "document": "Evidence", "score": 0.0, "distance": 0.0,
        "metadata": {"document_name": "guide.pdf", "page_number": 4, "section_title": "Care",
                     "subsection_title": "", "category": "treatment", "language": "en",
                     "source_id": "guide", "source_url": "https://example.test/guide"},
    }])

    envelope = evidence_service.rehydrate_evidence(
        "Question", "all", evidence_service.config.resolved_embedding_namespace, "manifest", ["chunk-1"]
    )

    assert envelope.is_ready
    assert envelope.chunks[0].text == "Evidence"


def test_embedding_api_error_has_search_message_and_trace_code(monkeypatch) -> None:
    _ready_manifest(monkeypatch)

    def failed_retrieve(*args, **kwargs):
        raise RetrievalProviderError("query_embedding_failed") from RuntimeError("429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr(evidence_service, "retrieve", failed_retrieve)
    envelope = evidence_service.stage_evidence("How can diabetes complications be prevented?", "all")

    assert envelope.status == "infrastructure_failure"
    assert envelope.error_code == "gemini:rate_limited"
    assert envelope.user_message == "The knowledge search is busy. Please try again in a minute."


def envelope_chunks_ids(envelope) -> list[str]:
    return [chunk.chunk_id for chunk in evidence_service.envelope_chunks(envelope)]


def test_vague_question_requests_clarification_before_retrieval(monkeypatch) -> None:
    monkeypatch.setattr(
        evidence_service,
        "retrieve",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("retrieval must not run")),
    )

    envelope = evidence_service.stage_evidence("Tell me more", "all")

    assert envelope.status == "needs_clarification"
    assert "more specific diabetes question" in envelope.user_message


def test_vague_follow_up_is_allowed_when_history_has_context(monkeypatch) -> None:
    _ready_manifest(monkeypatch)
    monkeypatch.setattr(evidence_service, "retrieve", lambda *args, **kwargs: [_chunk()])

    envelope = evidence_service.stage_evidence(
        "Tell me more",
        "all",
        conversation_history=[{"role": "user", "content": "What are diabetes risk factors?"}],
    )

    assert envelope.is_ready


def test_no_relevant_chunks_returns_actionable_out_of_scope_message(monkeypatch) -> None:
    monkeypatch.setattr(
        evidence_service,
        "retrieve",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("retrieval must not run")),
    )

    envelope = evidence_service.stage_evidence("Who won the football World Cup?", "all")

    assert envelope.status == "out_of_scope"
    assert "indexed diabetes references" in envelope.user_message
    assert "don't know" in envelope.user_message


def test_condition_and_intent_guard_rejects_bibliographic_near_match(monkeypatch) -> None:
    _ready_manifest(monkeypatch)
    chunk = _chunk()
    chunk.text = "A cohort study reported cardiovascular outcomes among people with diabetes."
    monkeypatch.setattr(evidence_service, "retrieve", lambda *args, **kwargs: [chunk])

    envelope = evidence_service.stage_evidence(
        "What role do preventive cardiologists have in diabetes care?",
        "all",
    )

    assert envelope.status == "out_of_scope"
    assert not envelope.chunks


def test_hosted_index_uses_runtime_fingerprint_without_local_manifest(monkeypatch) -> None:
    monkeypatch.setattr(evidence_service.config, "app_env", "deployment")
    monkeypatch.setattr(evidence_service, "load_index_manifest", lambda _: None)
    monkeypatch.setattr(evidence_service, "runtime_index_hash", lambda _: "runtime-index")
    monkeypatch.setattr(evidence_service, "rewrite_query", lambda query, **_: query)
    monkeypatch.setattr(evidence_service, "route_query", lambda query, **_: "treatment")
    monkeypatch.setattr(evidence_service, "retrieve", lambda *args, **kwargs: [_chunk()])

    envelope = evidence_service.stage_evidence(
        "How can diabetes complications be prevented?",
        "all",
    )

    assert envelope.is_ready
    assert envelope.index_manifest_hash == "runtime-index"
