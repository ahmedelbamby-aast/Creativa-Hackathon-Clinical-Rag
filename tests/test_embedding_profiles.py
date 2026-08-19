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
