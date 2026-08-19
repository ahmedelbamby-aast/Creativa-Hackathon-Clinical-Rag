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
    monkeypatch.setattr(
        server.vector_store,
        "healthcheck",
        lambda: {"postgres": "16", "pgvector": "0.8.6"},
    )
    monkeypatch.setattr(
        server.vector_store,
        "collection_stats",
        lambda: {"treatment": 4, "prevention": 3, "nutrition": 2},
    )

    result = server.ready()

    assert result["status"] == "ready"
    assert result["indexed_chunks"] == 9
    assert result["pgvector"] == "0.8.6"


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

    def fake_pipeline(message: str, category: str, memory: ConversationMemory):
        captured.update(message=message, category=category, history=memory.get_history())
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
        )
    )

    assert result.answer == "Grounded answer"
    assert result.citations == "Guideline, Page 4"
    assert result.generation_provider
    assert result.generation_model
    assert captured["category"] == "prevention"
    assert len(captured["history"]) == 2


def test_chat_endpoint_keeps_deterministic_fallback_without_llm_credentials(monkeypatch) -> None:
    monkeypatch.setattr(server.config, "generation_provider", "vercel_gateway")
    monkeypatch.setattr(server.config, "ai_gateway_api_key", "")
    monkeypatch.setattr(server.config, "vercel_oidc_token", "")
    monkeypatch.setattr(server, "rag_pipeline", lambda *args: ("Evidence fallback", "Sources", ""))

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
    monkeypatch.setattr(server, "rehydrate_evidence", lambda *args: (received.append(args) or envelope))
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
