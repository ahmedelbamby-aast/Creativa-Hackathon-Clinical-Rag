"""Vercel ASGI entrypoint behavior without live external services."""

from backend import server


def test_health_reports_deployment_configuration() -> None:
    result = server.health()

    assert result["status"] == "ok"
    assert result["embedding_provider"] in {"local", "gemini"}
    assert result["embedding_namespace"]


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
