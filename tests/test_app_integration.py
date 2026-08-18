"""Integration behavior at the Gradio/backend boundary."""

from __future__ import annotations

import app
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

    assert "temporarily unavailable" in answer
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
