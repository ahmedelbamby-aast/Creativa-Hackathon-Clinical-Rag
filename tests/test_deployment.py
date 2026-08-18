"""Vercel ASGI entrypoint behavior without live external services."""

from fastapi import HTTPException

from backend import server
from src.memory import ConversationMemory


def test_health_reports_deployment_configuration() -> None:
    result = server.health()

    assert result["status"] == "ok"
    assert result["embedding_provider"] in {"local", "gemini"}
    assert result["embedding_namespace"]
    assert result["generation_provider"] in {"extractive", "gemini", "vercel_gateway"}


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
    assert b"/api/chat" in response.body
    assert b"gradio_api" not in response.body


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
    assert captured["category"] == "prevention"
    assert len(captured["history"]) == 2


def test_chat_endpoint_rejects_missing_generation_configuration(monkeypatch) -> None:
    monkeypatch.setattr(server.config, "generation_provider", "vercel_gateway")
    monkeypatch.setattr(server.config, "ai_gateway_api_key", "")
    monkeypatch.setattr(server.config, "vercel_oidc_token", "")

    try:
        server.chat_endpoint(server.ChatRequest(message="Question", category="all"))
    except HTTPException as exc:
        assert exc.status_code == 503
    else:
        raise AssertionError("Expected an unavailable generation configuration")
