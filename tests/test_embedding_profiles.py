"""Per-session embedding dimension routing and UI contract tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend import server
from src import embedding_profiles


@pytest.fixture(autouse=True)
def clear_profile_cache():
    embedding_profiles.get_embedding_runtime.cache_clear()
    yield
    embedding_profiles.get_embedding_runtime.cache_clear()


def test_each_dimension_has_an_isolated_runtime_and_table(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_NAMESPACE_768", "gemini_768_test")
    runtime_768 = embedding_profiles.get_embedding_runtime(768)
    runtime_1024 = embedding_profiles.get_embedding_runtime(1024)

    assert runtime_768 is not runtime_1024
    assert runtime_768.dimension == 768
    assert runtime_768.namespace == "gemini_768_test"
    assert runtime_768.table_family == "rag_chunks_d768"
    assert runtime_1024.dimension == 1024
    assert runtime_1024.namespace.endswith("_1024")
    assert runtime_1024.table_family == "rag_chunks_d1024"


def test_runtime_cache_is_scoped_by_dimension() -> None:
    assert embedding_profiles.get_embedding_runtime(768) is embedding_profiles.get_embedding_runtime(768)
    assert embedding_profiles.get_embedding_runtime(768) is not embedding_profiles.get_embedding_runtime(2048)


def test_request_dimension_is_strictly_validated() -> None:
    request = server.ChatRequest(message="Question", embedding_dimension=3072)
    assert request.embedding_dimension == 3072
    with pytest.raises(ValidationError):
        server.ChatRequest(message="Question", embedding_dimension=512)


def test_browser_ui_sends_and_locks_the_session_dimension() -> None:
    content = server.index().body.decode("utf-8")
    assert 'id="embedding-dimension"' in content
    assert "dimensionLocked" in content
    assert "embedding_dimension: embeddingDimension" in content
    assert "embedding_dimension: retrieved.embedding_dimension" in content
    for dimension in embedding_profiles.SUPPORTED_EMBEDDING_DIMENSIONS:
        assert f"rag_chunks_d{dimension}" not in content


def test_health_advertises_all_supported_dimensions() -> None:
    result = server.health()
    assert result["supported_embedding_dimensions"] == [384, 768, 1024, 2048, 3072]


def test_partial_dimension_is_not_selectable(monkeypatch) -> None:
    class Connection:
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def execute(self, query, params=None):
            class Result:
                def fetchone(self):
                    if "to_regclass" in str(query):
                        return ("rag_chunks_d1024",)
                    return (24, 1)
            return Result()

    monkeypatch.setattr(embedding_profiles.psycopg, "connect", lambda *_: Connection())
    catalog = embedding_profiles.embedding_profile_catalog()
    partial = next(profile for profile in catalog["profiles"] if profile["dimension"] == 1024)

    assert partial["document_count"] == 1
    assert partial["expected_document_count"] == 12
    assert partial["index_status"] == "ingesting"
    assert partial["available"] is False


def test_complete_but_unaccepted_dimension_is_not_selectable(monkeypatch) -> None:
    class Connection:
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def execute(self, query, params=None):
            class Result:
                def fetchone(self):
                    if "to_regclass" in str(query):
                        return ("present",)
                    return (946, 12)
            return Result()

    monkeypatch.setattr(embedding_profiles.psycopg, "connect", lambda *_: Connection())
    monkeypatch.setattr(embedding_profiles.config, "embedding_accepted_dimensions", (384, 768))
    catalog = embedding_profiles.embedding_profile_catalog()
    candidate = next(profile for profile in catalog["profiles"] if profile["dimension"] == 1024)

    assert catalog["accepted_dimensions"] == [384, 768]
    assert candidate["document_count"] == 12
    assert candidate["accepted"] is False
    assert candidate["index_status"] == "awaiting_acceptance"
    assert candidate["available"] is False
