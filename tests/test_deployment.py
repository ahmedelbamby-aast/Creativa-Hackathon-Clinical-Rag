"""Vercel ASGI entrypoint behavior without live external services."""

from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend import server
from src.memory import ConversationMemory
from src.retrieval_contracts import EvidenceChunk, RetrievalEnvelope


def _envelope() -> RetrievalEnvelope:
    return RetrievalEnvelope(
        original_query="Question", rewritten_query="Question", requested_category="all",
        routed_category="treatment", namespace="phase2_local", index_manifest_hash="manifest",
        status="ready", chunks=(EvidenceChunk(
            chunk_id="chunk-1", text="Evidence", score=0.9, distance=0.1,
            document_name="guide.pdf", page_number=4, section_title="Care", subsection_title="",
            category="treatment", language="en", source_id="guide",
            source_url="https://example.test/guide",
        ),),
    )


def test_health_reports_deployment_configuration() -> None:
    result = server.health()

    assert result["status"] == "ok"
    assert result["embedding_provider"] in {"local", "gemini"}
    assert result["embedding_namespace"]
    assert result["generation_provider"] in {"extractive", "gemini", "groq", "vercel_gateway", "auto"}
    assert result["active_generation_provider"] in {"extractive", "gemini", "groq", "vercel_gateway"}
    assert result["active_generation_model"]


def test_ready_reports_database_metadata(monkeypatch) -> None:
    runtime = server.get_embedding_runtime()
    monkeypatch.setattr(
        runtime.vector_store,
        "healthcheck",
        lambda: {"postgres": "16", "pgvector": "0.8.6"},
    )
    monkeypatch.setattr(
        runtime.vector_store,
        "collection_stats",
        lambda: {"treatment": 4, "prevention": 3, "nutrition": 2},
    )
    monkeypatch.setattr(
        runtime.vector_store,
        "namespace_audit",
        lambda: {
            "document_count": 2,
            "documents": {
                "a.pdf": {"chunk_count": 4},
                "b.pdf": {"chunk_count": 5},
            },
        },
    )

    result = server.ready()

    assert result["status"] == "ready"
    assert result["indexed_chunks"] == 9
    assert result["pgvector"] == "0.8.6"


def test_ready_query_dimension_is_parsed_and_invalid_values_are_rejected(monkeypatch) -> None:
    runtime = server.get_embedding_runtime(768)
    monkeypatch.setattr(runtime.vector_store, "healthcheck", lambda: {"postgres": "16", "pgvector": "0.8.6"})
    monkeypatch.setattr(runtime.vector_store, "collection_stats", lambda: {"treatment": 1})
    monkeypatch.setattr(
        runtime.vector_store,
        "namespace_audit",
        lambda: {"document_count": 1, "documents": {"a.pdf": {"chunk_count": 1}}},
    )
    client = TestClient(server.api)

    valid = client.get("/api/ready?embedding_dimension=768")
    invalid = client.get("/api/ready?embedding_dimension=512")

    assert valid.status_code == 200
    assert valid.json()["embedding_dimension"] == 768
    assert invalid.status_code == 422


def test_index_serves_serverless_client() -> None:
    response = server.index()

    assert response.status_code == 200
    assert b"Diabetes RAG Assistant" in response.body
    assert b"/api/retrieve" in response.body
    assert b"/api/generate" in response.body
    assert b"gradio_api" not in response.body
    assert b"Verified sample questions" in response.body
    assert b"Preventive cardiology and diabetes care" not in response.body
    assert b"/assets/katex/katex.min.js" in response.body


def test_metrics_guide_serves_plain_english_foundational_definitions() -> None:
    response = server.metrics_guide()
    content = response.body.decode("utf-8")
    assert response.status_code == 200
    assert "What do the quality numbers mean?" in content
    for metric_name in (
        "Hit Rate@k", "Precision@k", "Recall@k", "Mean Reciprocal Rank (MRR)",
        "Mean Average Precision (MAP)", "nDCG@k", "Exact Match",
        "Token Precision, Recall, and F1", "End-to-End Task Success",
        "Latency Percentiles", "Error Rate and Availability", "Token Usage and Cost per Request",
    ):
        assert metric_name in content
    assert "Not measured" in content


def test_dashboard_has_plain_english_unavailable_reason_rendering() -> None:
    content = server.index().body.decode("utf-8")
    assert "measured_count" in content
    assert "unavailable_reasons).map" not in content
    assert "this is a safe refusal case" not in content
    assert "this older trace lacks evaluation context" not in content
    assert "this chat has no reviewed label" not in content


def test_dashboard_separates_current_chat_from_cumulative_session_averages() -> None:
    content = server.index().body.decode("utf-8")
    assert "Current chat quality" in content
    assert "conversation_id=${encodeURIComponent(state.conversationId)}" in content
    assert "function buildCumulativeSessionAverages(traces)" in content
    assert "Average metrics across chat sessions" in content
    assert "Cumulative Precision@k" in content and "Cumulative Recall@k" in content
    assert "unmeasured quality values never count as zero" in content
    assert "improved ? 'delta-good' : 'delta-bad'" in content


def test_evidence_and_sources_panels_are_dynamic_and_expandable() -> None:
    content = server.index().body.decode("utf-8")
    assert 'id="evidence-panel" class="card panel resizable-panel"' in content
    assert 'id="sources-panel" class="card panel resizable-panel"' in content
    assert "function sizeResourcePanel(container, itemCount, pixelsPerItem)" in content
    assert "function toggleExpandedPanel(panelId)" in content
    assert "data-expand-panel=\"evidence-panel\"" in content
    assert "data-expand-panel=\"sources-panel\"" in content
    assert "if (event.key === 'Escape') closeExpandedPanel()" in content


def test_category_and_dimension_selectors_share_the_same_baseline() -> None:
    content = server.index().body.decode("utf-8")
    assert ".selector-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; align-items: end; }" in content
    assert ".selector-field select { height: 54px; }" in content
    assert "Choose once when starting a session. Clear the chat to switch." not in content


def test_self_hosted_math_assets_are_served() -> None:
    client = TestClient(server.api)

    script = client.get("/assets/katex/katex.min.js")
    stylesheet = client.get("/assets/katex/katex.min.css")

    assert script.status_code == 200
    assert script.headers["content-type"].startswith("text/javascript")
    assert script.content.startswith(b"!function")
    assert len(script.content) > 250_000
    assert stylesheet.status_code == 200
    assert stylesheet.headers["content-type"].startswith("text/css")
    assert b"@font-face" in stylesheet.content


def test_sample_catalog_is_balanced_across_all_four_scenarios() -> None:
    result = server.sample_questions()

    assert len(result["scenarios"]) == 4
    for scenario in result["scenarios"]:
        assert len(scenario["questions"]) == 6
        assert [item["language"] for item in scenario["questions"]].count("en") == 3
        assert [item["language"] for item in scenario["questions"]].count("ar") == 3


def test_chat_endpoint_rebuilds_bounded_memory(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_pipeline(message: str, category: str, memory: ConversationMemory, case_id: str = "", embedding_dimension=None):
        captured.update(message=message, category=category, history=memory.get_history(), case_id=case_id, embedding_dimension=embedding_dimension)
        return "Grounded answer", "Guideline, Page 4", "trace"

    monkeypatch.setattr(server.config, "generation_provider", "gemini")
    monkeypatch.setattr(server.config, "gemini_api_key", "configured")
    monkeypatch.setattr(server, "rag_pipeline", fake_pipeline)

    result = server.chat_endpoint(
        server.ChatRequest(
            message="How can complications be prevented?",
            category="prevention",
            history=[
                server.ChatMessage(role="user", content="Previous question"),
                server.ChatMessage(role="assistant", content="Previous answer"),
            ],
            case_id="case-1",
        )
    )

    assert result.answer == "Grounded answer"
    assert result.citations == "Guideline, Page 4"
    assert result.generation_provider
    assert result.generation_model
    assert captured["category"] == "prevention"
    assert captured["case_id"] == "case-1"
    assert len(captured["history"]) == 2


def test_chat_endpoint_keeps_deterministic_fallback_without_llm_credentials(monkeypatch) -> None:
    monkeypatch.setattr(server.config, "generation_provider", "vercel_gateway")
    monkeypatch.setattr(server.config, "ai_gateway_api_key", "")
    monkeypatch.setattr(server.config, "vercel_oidc_token", "")
    monkeypatch.setattr(server, "rag_pipeline", lambda *args, **kwargs: ("Evidence fallback", "Sources", ""))

    result = server.chat_endpoint(
        server.ChatRequest(message="What are diabetes risk factors?", category="all")
    )

    assert server.config.generation_configured is True
    assert result.answer == "Evidence fallback"


def test_retrieve_then_generate_uses_staged_chunk_ids(monkeypatch) -> None:
    envelope = _envelope()
    persisted = []
    monkeypatch.setattr(server, "record_trace", lambda trace: persisted.append(trace.serializable()))
    monkeypatch.setattr(
        server, "load_trace",
        lambda trace_id: server.RequestTrace.from_record(
            next(item for item in reversed(persisted) if item["trace_id"] == trace_id)
        ),
    )
    monkeypatch.setattr(server, "stage_evidence", lambda *args, **kwargs: envelope)
    retrieved = server.retrieve_endpoint(server.ChatRequest(message="Question", category="all"))

    assert retrieved.status == "ready"
    assert retrieved.trace_id
    assert retrieved.metrics["status"] == "running"
    assert retrieved.chunks[0].source_url == "https://example.test/guide"

    received = []
    monkeypatch.setattr(server.config, "generation_provider", "gemini")
    monkeypatch.setattr(server.config, "gemini_api_key", "configured")
    monkeypatch.setattr(server, "rehydrate_evidence", lambda *args, **kwargs: (received.append(args) or envelope))
    monkeypatch.setattr(server, "generate_from_evidence", lambda value, memory: ("Answer", "Sources", ""))
    result = server.generate_endpoint(server.GenerateRequest(
        message="Question", category="all", namespace="phase2_local", index_manifest_hash="manifest",
        chunk_ids=["chunk-1"], trace_id=retrieved.trace_id,
    ))

    assert result.answer == "Answer"
    assert received[0][-1] == ["chunk-1"]
    assert result.generation_provider
    assert result.sources[0].evidence_id == "E1"
    assert result.sources[0].source_url == "https://example.test/guide"
    assert persisted[-1]["status"] == "ok"


def test_retrieve_routes_the_requested_embedding_dimension(monkeypatch) -> None:
    envelope = _envelope()
    object.__setattr__(envelope, "embedding_dimension", 768)
    object.__setattr__(envelope, "embedding_provider", "gemini")
    object.__setattr__(envelope, "embedding_model", "gemini-embedding-2")
    object.__setattr__(envelope, "embedding_table_family", "rag_chunks_d768")
    captured = {}

    def fake_stage(*args, **kwargs):
        captured.update(kwargs)
        return envelope

    monkeypatch.setattr(server, "stage_evidence", fake_stage)
    monkeypatch.setattr(server, "record_trace", lambda *_: None)

    response = server.retrieve_endpoint(
        server.ChatRequest(
            message="Question",
            category="all",
            embedding_dimension=768,
        )
    )

    assert captured["embedding_dimension"] == 768
    assert response.embedding_dimension == 768
    assert response.metrics["embedding_dimension"] == 768
    assert response.metrics["embedding_table_family"] == "rag_chunks_d768"


def test_case_id_survives_retrieve_to_generate_trace(monkeypatch) -> None:
    envelope = _envelope()
    case = {
        "case_id": "case-1", "language": "en", "expect_evidence": True,
        "relevant_items": [], "review": {"status": "pending_human_review"},
    }
    persisted = []
    monkeypatch.setattr(server, "record_trace", lambda trace: persisted.append(trace.serializable()))
    monkeypatch.setattr(server, "load_trace", lambda trace_id: server.RequestTrace.from_record(persisted[-1]))
    monkeypatch.setattr(server, "stage_evidence", lambda *args, **kwargs: envelope)
    monkeypatch.setattr("src.observability.match_case", lambda *_: (case, "explicit_case_id"))
    monkeypatch.setattr("src.observability.gold_dataset", lambda: {"version": "v2", "cases": [case]})
    retrieved = server.retrieve_endpoint(server.ChatRequest(message="Question", category="all", case_id="case-1"))
    monkeypatch.setattr(server, "rehydrate_evidence", lambda *args, **kwargs: envelope)
    monkeypatch.setattr(server, "generate_from_evidence", lambda *args: ("Answer", "Sources", ""))
    server.generate_endpoint(server.GenerateRequest(message="Question", category="all", namespace="n", index_manifest_hash="h", chunk_ids=["chunk-1"], trace_id=retrieved.trace_id, case_id="case-1"))
    assert persisted[-1]["requested_case_id"] == "case-1"
    assert persisted[-1]["label_case_id"] == "case-1"
    assert persisted[-1]["case_match_method"] == "explicit_case_id"
