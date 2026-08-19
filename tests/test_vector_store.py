"""Tests for pgvector storage behavior without a live database."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.vector_store import VectorStore, _lexical_terms, normalize_namespace


class FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, rows):
        self.rows = rows
        self.parameters = None
        self.statement = ""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, statement, parameters=None):
        self.statement = str(statement)
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
        "source_id": "guide",
        "source_url": "https://example.test/guide",
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
    assert results[0]["metadata"]["source_url"] == "https://example.test/guide"


def test_query_rejects_wrong_dimension() -> None:
    store = VectorStore(
        database_url="postgresql://example",
        namespace="local_2",
        dimension=2,
    )
    with pytest.raises(ValueError, match="2 dimensions"):
        store.query([1.0], top_k=1)


def test_query_enriches_legacy_rows_from_source_catalog(monkeypatch) -> None:
    row = {
        "id": "chunk-1", "document": "Diabetes guidance",
        "document_name": "guide.pdf", "page_number": 4,
        "section_title": "Treatment", "subsection_title": "",
        "category": "treatment", "content_type": "text", "language": "en",
        "quality_score": 0.8, "distance": 0.2,
    }
    connection = FakeConnection([row])
    store = VectorStore(database_url="postgresql://example", namespace="local_2", dimension=2)
    monkeypatch.setattr(store, "_connect", lambda: connection)
    monkeypatch.setattr(
        "src.vector_store.load_source_catalog",
        lambda: {"guide.pdf": SimpleNamespace(
            source_id="guide", source_url="https://example.test/guide", enabled=True
        )},
    )

    result = store.query([1.0, 0.0], top_k=1)[0]

    assert result["metadata"]["source_id"] == "guide"
    assert result["metadata"]["source_url"] == "https://example.test/guide"
    assert "source_id," not in connection.statement


def test_get_chunks_preserves_requested_order(monkeypatch) -> None:
    rows = [
        {
            "id": "second", "document": "Second", "document_name": "guide.pdf",
            "page_number": 2, "section_title": "Care", "subsection_title": "",
            "category": "treatment", "content_type": "text", "language": "en",
            "source_id": "guide", "source_url": "https://example.test/guide", "quality_score": 1.0,
        },
        {
            "id": "first", "document": "First", "document_name": "guide.pdf",
            "page_number": 1, "section_title": "Care", "subsection_title": "",
            "category": "treatment", "content_type": "text", "language": "en",
            "source_id": "guide", "source_url": "https://example.test/guide", "quality_score": 1.0,
        },
    ]
    store = VectorStore(database_url="postgresql://example", namespace="local_2", dimension=2)
    monkeypatch.setattr(store, "_connect", lambda: FakeConnection(rows))

    results = store.get_chunks(["first", "second"])

    assert [result["id"] for result in results] == ["first", "second"]


def test_keyword_query_ranks_without_embedding_and_enriches_provenance(monkeypatch) -> None:
    row = {
        "id": "chunk-1", "document": "589 million adults with diabetes in 2024",
        "document_name": "guide.pdf", "page_number": 46,
        "section_title": "Key messages", "subsection_title": "",
        "category": "general", "content_type": "text", "language": "en",
        "quality_score": 0.9, "lexical_score": 0.75,
    }
    connection = FakeConnection([row])
    store = VectorStore(database_url="postgresql://example", namespace="gemini_384", dimension=384)
    monkeypatch.setattr(store, "_connect", lambda: connection)
    monkeypatch.setattr(
        "src.vector_store.load_source_catalog",
        lambda: {"guide.pdf": SimpleNamespace(
            source_id="guide", source_url="https://example.test/guide",
            publisher="Publisher", publication_date="2025", checksum="a" * 64,
            enabled=True,
        )},
    )

    result = store.keyword_query("589 million adults living with diabetes in 2024", top_k=1)[0]

    assert result["score"] == 0.875
    assert result["metadata"]["source_id"] == "guide"
    assert "ILIKE" in connection.statement
    assert _lexical_terms("589 million adults living with diabetes in 2024") == ["589", "2024"]


def test_schema_uses_partitioned_pgvector_table() -> None:
    schema = (
        Path(__file__).resolve().parents[1] / "database" / "schema.sql"
    ).read_text(encoding="utf-8")
    assert "CREATE EXTENSION IF NOT EXISTS vector" in schema
    assert "embedding vector(384)" in schema
    assert "PARTITION BY LIST (namespace)" in schema
    assert "source_url text NOT NULL DEFAULT ''" in schema
