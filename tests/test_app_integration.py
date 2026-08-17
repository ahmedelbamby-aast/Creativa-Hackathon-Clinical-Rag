"""Integration behavior at the Gradio/backend boundary."""

from __future__ import annotations

import app
from src.memory import ConversationMemory
from src.retriever import RetrievedChunk


def test_generation_error_keeps_retrieved_citations(monkeypatch):
    chunk = RetrievedChunk(
        chunk_id="chunk-1",
        text="Regular screening and risk-factor control can reduce complications.",
        score=0.8,
        distance=0.2,
        document_name="guideline.pdf",
        page_number=4,
        section_title="Prevention",
        subsection_title="Screening",
        category="prevention",
        content_type="text",
        language="en",
    )
    monkeypatch.setattr(app, "retrieve", lambda *args, **kwargs: [chunk])
    monkeypatch.setattr(app.generator, "generate", lambda _: (_ for _ in ()).throw(
        RuntimeError("GEMINI_API_KEY is not set")
    ))
    traces = []
    monkeypatch.setattr(app, "record_trace", lambda trace: traces.append(trace.serializable()))

    answer, citations, _ = app.rag_pipeline(
        "How can complications be prevented?", "prevention", ConversationMemory()
    )

    assert "Configuration Error" in answer
    assert "guideline" in citations
    assert "Page 4" in citations
    assert traces[0]["status"] == "generation_error"
    assert traces[0]["retrieval_count"] == 1
    assert set(traces[0]["stages_ms"]) >= {"safety", "rewrite", "route", "retrieval", "generation"}
