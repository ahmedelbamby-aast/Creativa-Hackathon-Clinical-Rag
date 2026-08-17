"""Tests for pgvector storage behavior without a live database."""

from pathlib import Path

import pytest

from src.vector_store import VectorStore, normalize_namespace


class FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, rows):
        self.rows = rows
        self.parameters = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, statement, parameters=None):
        self.parameters = parameters
        return FakeResult(self.rows)


def test_namespace_is_safe_for_partition_names() -> None:
    assert normalize_namespace("Gemini Embedding/384") == "gemini_embedding_384"
    with pytest.raises(ValueError):
        normalize_namespace("---")


def test_query_maps_postgres_row_to_retrieved_result(monkeypatch) -> None:
    row = {
        "id": "chunk-1",
        "document": "Diabetes guidance",
        "document_name": "guide.pdf",
        "page_number": 4,
        "section_title": "Treatment",
        "subsection_title": "",
        "category": "treatment",
        "content_type": "text",
        "language": "en",
        "quality_score": 0.8,
        "distance": 0.2,
    }
    connection = FakeConnection([row])
    store = VectorStore(
        database_url="postgresql://example",
        namespace="local_2",
        dimension=2,
    )
    monkeypatch.setattr(store, "_connect", lambda: connection)

    results = store.query([1.0, 0.0], category="treatment", top_k=1)

    assert results[0]["id"] == "chunk-1"
    assert results[0]["score"] == 0.8
    assert results[0]["metadata"]["page_number"] == 4


def test_query_rejects_wrong_dimension() -> None:
    store = VectorStore(
        database_url="postgresql://example",
        namespace="local_2",
        dimension=2,
    )
    with pytest.raises(ValueError, match="2 dimensions"):
        store.query([1.0], top_k=1)


def test_schema_uses_partitioned_pgvector_table() -> None:
    schema = (
        Path(__file__).resolve().parents[1] / "database" / "schema.sql"
    ).read_text(encoding="utf-8")
    assert "CREATE EXTENSION IF NOT EXISTS vector" in schema
    assert "embedding vector(384)" in schema
    assert "PARTITION BY LIST (namespace)" in schema
