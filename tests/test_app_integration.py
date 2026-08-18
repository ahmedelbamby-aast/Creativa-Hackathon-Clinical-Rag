"""Integration behavior at the Gradio/backend boundary."""

from __future__ import annotations

import app
import pytest
from src.memory import ConversationMemory
from src.retrieval_contracts import EvidenceChunk, RetrievalEnvelope


def test_generation_error_keeps_retrieved_citations(monkeypatch):
    chunk = EvidenceChunk(
        chunk_id="chunk-1",
        text="Regular screening and risk-factor control can reduce complications.",
        score=0.8,
        distance=0.2,
        document_name="guideline.pdf",
        page_number=4,
        section_title="Prevention",
        subsection_title="Screening",
        category="prevention",
        language="en",
        source_id="guide",
        source_url="https://example.test/guide",
    )
    envelope = RetrievalEnvelope(
        original_query="How can complications be prevented?",
        rewritten_query="How can complications be prevented?",
        requested_category="prevention",
        routed_category="prevention",
        namespace="phase2_local",
        index_manifest_hash="manifest",
        status="ready",
        chunks=(chunk,),
    )
    monkeypatch.setattr(app, "stage_evidence", lambda *args, **kwargs: envelope)
    monkeypatch.setattr(app.generator, "generate", lambda _: (_ for _ in ()).throw(
        RuntimeError("GEMINI_API_KEY is not set")
    ))
    traces = []
    monkeypatch.setattr(app, "record_trace", lambda trace: traces.append(trace.serializable()))

    answer, citations, _ = app.rag_pipeline(
        "How can complications be prevented?", "prevention", ConversationMemory()
    )

    assert "not configured" in answer
    assert "GEMINI_API_KEY" not in answer
    assert "guideline" in citations
    assert "Page 4" in citations
    assert traces[0]["status"] == "generation_error"
    assert traces[0]["retrieval_count"] == 1
    assert set(traces[0]["stages_ms"]) >= {"retrieval", "generation"}


def test_ui_generation_uses_the_staged_envelope_only(monkeypatch):
    chunk = EvidenceChunk(
        chunk_id="chunk-1", text="Evidence", score=0.9, distance=0.1,
        document_name="guide.pdf", page_number=4, section_title="Care",
        subsection_title="", category="treatment", language="en",
        source_id="guide", source_url="https://example.test/guide",
    )
    envelope = RetrievalEnvelope(
        original_query="Question", rewritten_query="Question", requested_category="all",
        routed_category="treatment", namespace="phase2_local", index_manifest_hash="manifest",
        status="ready", chunks=(chunk,),
    )
    received = []
    monkeypatch.setattr(app, "stage_evidence", lambda *args, **kwargs: envelope)
    monkeypatch.setattr(app, "generate_from_evidence", lambda value, memory: (received.append(value) or ("Answer", "Sources", "Debug")))

    history, _, staged, _ = app.retrieve_for_ui("Question", [], "all", ConversationMemory())
    updated, citations, _, _ = app.generate_for_ui(history, staged, ConversationMemory())

    assert received == [envelope]
    assert citations == "Sources"
    assert updated[-1]["content"] == "Answer"


@pytest.mark.parametrize(
    ("error", "expected", "code"),
    [
        (RuntimeError("429 RESOURCE_EXHAUSTED"), "busy", "rate_limited"),
        (RuntimeError("401 UNAUTHENTICATED"), "temporarily unavailable", "authentication_failed"),
        (RuntimeError("504 timed out"), "took too long", "timeout"),
        (RuntimeError("400 INVALID_ARGUMENT"), "rephrase", "invalid_request"),
    ],
)
def test_generation_api_errors_have_simple_messages_and_trace_codes(monkeypatch, error, expected, code):
    chunk = EvidenceChunk(
        chunk_id="chunk-1", text="Evidence", score=0.9, distance=0.1,
        document_name="guide.pdf", page_number=4, section_title="Care", subsection_title="",
        category="treatment", language="en", source_id="guide", source_url="https://example.test/guide",
    )
    envelope = RetrievalEnvelope(
        original_query="Question", rewritten_query="Question", requested_category="all",
        routed_category="treatment", namespace="phase2_local", index_manifest_hash="manifest",
        status="ready", chunks=(chunk,),
    )
    monkeypatch.setattr(app.generator, "generate", lambda _: (_ for _ in ()).throw(error))
    traces = []
    trace = app.RequestTrace(query="Question", requested_category="all")
    monkeypatch.setattr(app, "record_trace", lambda value: traces.append(value.serializable()))

    answer, _, _ = app.generate_from_evidence(envelope, ConversationMemory(), trace)

    assert expected in answer
    assert "401" not in answer and "RESOURCE_EXHAUSTED" not in answer
    assert traces[0]["error"] == f"gemini:{code}"
