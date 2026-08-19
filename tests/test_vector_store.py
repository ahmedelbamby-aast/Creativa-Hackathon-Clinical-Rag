"""Tests for pgvector storage behavior without a live database.

Covers:
- Namespace normalisation
- Dimension-to-parent-table routing (Critical Issue 1)
- Dimension validation for reads and writes
- Row-to-result mapping and source catalog enrichment
- Chunk ordering and keyword fallback
- Schema file consistency
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.vector_store import (
    VectorStore,
    _DIMENSION_TABLE_MAP,
    _lexical_terms,
    _resolve_parent_table,
    normalize_namespace,
)


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


# ---------------------------------------------------------------------------
# Namespace normalisation
# ---------------------------------------------------------------------------

def test_namespace_is_safe_for_partition_names() -> None:
    assert normalize_namespace("Gemini Embedding/384") == "gemini_embedding_384"
    with pytest.raises(ValueError):
        normalize_namespace("---")


# ---------------------------------------------------------------------------
# Critical Issue 1 — dimension-to-parent-table routing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dimension, expected_table", [
    (384,  "rag_chunks"),
    (768,  "rag_chunks_d768"),
    (1024, "rag_chunks_d1024"),
    (2048, "rag_chunks_d2048"),
    (3072, "rag_chunks_d3072"),
])
def test_dimension_routes_to_correct_parent_table(dimension, expected_table) -> None:
    """Every supported dimension must map to its own parent table."""
    table, schema_path = _resolve_parent_table(dimension)
    assert table == expected_table
    assert schema_path.exists(), (
        f"Schema file {schema_path} does not exist — "
        f"create database/schema_d{dimension}.sql before using this dimension"
    )


def test_unregistered_dimension_raises_value_error() -> None:
    """Attempting to use a dimension not in the map must fail loudly."""
    with pytest.raises(ValueError, match="No parent table registered"):
        _resolve_parent_table(512)


def test_vector_store_sets_parent_table_from_dimension() -> None:
    """VectorStore.__init__ must expose the correct parent_table attribute."""
    store_384  = VectorStore(database_url="postgresql://example", namespace="ns", dimension=384)
    store_768  = VectorStore(database_url="postgresql://example", namespace="ns", dimension=768)
    store_3072 = VectorStore(database_url="postgresql://example", namespace="ns", dimension=3072)

    assert store_384.parent_table  == "rag_chunks"
    assert store_768.parent_table  == "rag_chunks_d768"
    assert store_3072.parent_table == "rag_chunks_d3072"


def test_vector_store_partition_name_includes_parent_table() -> None:
    """Partition name must be '{parent_table}_{namespace}' so it is unique per dimension."""
    store = VectorStore(database_url="postgresql://example", namespace="gemini_768", dimension=768)
    assert store.partition_name == "rag_chunks_d768_gemini_768"


def test_baseline_partition_name_preserved_for_384() -> None:
    """The 384-d baseline partition naming must be backward-compatible."""
    store = VectorStore(database_url="postgresql://example", namespace="gemini_384", dimension=384)
    assert store.partition_name == "rag_chunks_gemini_384"


def test_all_dimension_map_entries_have_existing_schema_files() -> None:
    """Every entry in _DIMENSION_TABLE_MAP must have a corresponding SQL file on disk."""
    for dimension, (table, schema_path) in _DIMENSION_TABLE_MAP.items():
        assert schema_path.exists(), (
            f"_DIMENSION_TABLE_MAP[{dimension}] points to {schema_path} which does not exist"
        )


# ---------------------------------------------------------------------------
# Query dimension enforcement
# ---------------------------------------------------------------------------

def test_query_rejects_wrong_dimension() -> None:
    store = VectorStore(
        database_url="postgresql://example",
        namespace="local_384",
        dimension=384,
    )
    with pytest.raises(ValueError, match="384 dimensions"):
        store.query([1.0], top_k=1)  # only 1 value into a 384-d store


def test_add_chunks_rejects_wrong_dimension() -> None:
    store = VectorStore(
        database_url="postgresql://example",
        namespace="local_384",
        dimension=384,
    )
    with pytest.raises(ValueError, match="384 dimensions"):
        store.add_chunks(
            [{"chunk_id": "c1", "document_name": "d.pdf", "text": "hi"}],
            [[1.0, 2.0, 3.0]],  # 3-d vector into a 384-d store
        )


# ---------------------------------------------------------------------------
# Query result mapping
# ---------------------------------------------------------------------------

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
        namespace="local_384",
        dimension=384,
    )
    monkeypatch.setattr(store, "_connect", lambda: connection)

    results = store.query([0.0] * 384, category="treatment", top_k=1)

    assert results[0]["id"] == "chunk-1"
    assert results[0]["score"] == 0.8
    assert results[0]["metadata"]["page_number"] == 4
    assert results[0]["metadata"]["source_url"] == "https://example.test/guide"


def test_query_uses_correct_parent_table_in_sql(monkeypatch) -> None:
    """The SQL emitted by query() must reference the dimension-appropriate parent table."""
    row = {
        "id": "c1", "document": "text", "document_name": "d.pdf",
        "page_number": 1, "section_title": "", "subsection_title": "",
        "category": "general", "content_type": "text", "language": "en",
        "quality_score": 0.5, "distance": 0.1,
    }
    connection = FakeConnection([row])
    store = VectorStore(database_url="postgresql://example", namespace="gemini_768", dimension=768)
    monkeypatch.setattr(store, "_connect", lambda: connection)

    store.query([0.0] * 768, top_k=1)

    assert "rag_chunks_d768" in connection.statement


def test_query_enriches_legacy_rows_from_source_catalog(monkeypatch) -> None:
    row = {
        "id": "chunk-1", "document": "Diabetes guidance",
        "document_name": "guide.pdf", "page_number": 4,
        "section_title": "Treatment", "subsection_title": "",
        "category": "treatment", "content_type": "text", "language": "en",
        "quality_score": 0.8, "distance": 0.2,
    }
    connection = FakeConnection([row])
    store = VectorStore(database_url="postgresql://example", namespace="local_384", dimension=384)
    monkeypatch.setattr(store, "_connect", lambda: connection)
    monkeypatch.setattr(
        "src.vector_store.load_source_catalog",
        lambda: {"guide.pdf": SimpleNamespace(
            source_id="guide", source_url="https://example.test/guide", enabled=True
        )},
    )

    result = store.query([0.0] * 384, top_k=1)[0]

    assert result["metadata"]["source_id"] == "guide"
    assert result["metadata"]["source_url"] == "https://example.test/guide"
    assert "source_id," not in connection.statement


# ---------------------------------------------------------------------------
# get_chunks ordering
# ---------------------------------------------------------------------------

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
    store = VectorStore(database_url="postgresql://example", namespace="local_384", dimension=384)
    monkeypatch.setattr(store, "_connect", lambda: FakeConnection(rows))

    results = store.get_chunks(["first", "second"])

    assert [result["id"] for result in results] == ["first", "second"]


# ---------------------------------------------------------------------------
# Keyword query
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Schema file consistency
# ---------------------------------------------------------------------------

def test_schema_uses_partitioned_pgvector_table() -> None:
    """The 384-d baseline schema must remain intact for backward compatibility."""
    schema = (
        Path(__file__).resolve().parents[1] / "database" / "schema.sql"
    ).read_text(encoding="utf-8")
    assert "CREATE EXTENSION IF NOT EXISTS vector" in schema
    assert "embedding vector(384)" in schema
    assert "PARTITION BY LIST (namespace)" in schema
    assert "source_url text NOT NULL DEFAULT ''" in schema


@pytest.mark.parametrize("dimension, expected_col", [
    (768,  "vector(768)"),
    (1024, "vector(1024)"),
    (2048, "vector(2048)"),
    (3072, "vector(3072)"),
])
def test_higher_dimension_schemas_have_correct_vector_column(dimension, expected_col) -> None:
    """Each dimension-specific migration must declare the right vector column width."""
    _table, schema_path = _resolve_parent_table(dimension)
    schema = schema_path.read_text(encoding="utf-8")
    assert expected_col in schema, (
        f"{schema_path.name} must contain 'embedding {expected_col} NOT NULL'"
    )
    assert "PARTITION BY LIST (namespace)" in schema


@pytest.mark.parametrize("dimension", [768, 1024, 2048, 3072])
def test_higher_dimension_schemas_have_document_and_category_indexes(dimension) -> None:
    """Each migration must include the document_name and category indexes."""
    _table, schema_path = _resolve_parent_table(dimension)
    schema = schema_path.read_text(encoding="utf-8")
    assert "_document_idx" in schema
    assert "_category_idx" in schema
