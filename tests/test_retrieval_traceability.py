"""Tests for retrieval traceability.

Verifies that retrieved chunks expose complete metadata provenance fields
(source_id, source_url, publisher, publication_date, source_checksum, score, section, page).
"""

from __future__ import annotations

import pytest
from src.retriever import RetrievedChunk, retrieve


def test_retrieved_chunk_has_all_traceability_fields():
    """Verify RetrievedChunk structure exposes all expected traceability fields."""
    chunk = RetrievedChunk(
        chunk_id="chunk_1",
        text="A brief piece of diabetes information.",
        score=0.92,
        distance=0.08,
        document_name="IDF_Diabetes_Atlas_11th_Edition_2025_WEB.pdf",
        page_number=12,
        section_title="Prevention",
        subsection_title="",
        category="prevention",
        content_type="text",
        language="en",
        source_id="idf-atlas-11-2025",
        source_url="https://diabetesatlas.org/atlas/eleventh-edition/",
        publisher="International Diabetes Federation",
        publication_date="2025",
        source_checksum="81d01e73d486adbe7d4f14752644e109ed43151738e3e852303ebd3967a81d0d",
        chunk_profile="balanced",
    )
    
    assert chunk.source_id == "idf-atlas-11-2025"
    assert chunk.source_url == "https://diabetesatlas.org/atlas/eleventh-edition/"
    assert chunk.publisher == "International Diabetes Federation"
    assert chunk.publication_date == "2025"
    assert chunk.source_checksum == "81d01e73d486adbe7d4f14752644e109ed43151738e3e852303ebd3967a81d0d"
    assert chunk.page_number == 12
    assert chunk.section_title == "Prevention"
    assert chunk.score == 0.92


def test_retrieve_maps_all_provenance_fields(monkeypatch):
    """Verify retrieve() maps all fields from vector store query results to RetrievedChunk."""
    raw_db_row = {
        "id": "chunk_1",
        "document": "A brief piece of diabetes information.",
        "document_name": "IDF_Diabetes_Atlas_11th_Edition_2025_WEB.pdf",
        "page_number": 12,
        "section_title": "Prevention",
        "subsection_title": "",
        "category": "prevention",
        "content_type": "text",
        "language": "en",
        "quality_score": 0.9,
        "distance": 0.1,
        "score": 0.9,
        "metadata": {
            "chunk_id": "chunk_1",
            "document_name": "IDF_Diabetes_Atlas_11th_Edition_2025_WEB.pdf",
            "page_number": 12,
            "section_title": "Prevention",
            "subsection_title": "",
            "category": "prevention",
            "content_type": "text",
            "language": "en",
            "quality_score": 0.9,
            "source_id": "idf-atlas-11-2025",
            "source_url": "https://diabetesatlas.org/atlas/eleventh-edition/",
            "publisher": "International Diabetes Federation",
            "publication_date": "2025",
            "source_checksum": "81d01e73d486adbe7d4f14752644e109ed43151738e3e852303ebd3967a81d0d",
            "chunk_profile": "balanced",
        }
    }

    class FakeVectorStore:
        def query(self, query_embedding, category, top_k):
            return [raw_db_row]

    monkeypatch.setattr("src.retriever.embedder", type("FakeEmbedder", (), {"embed_query": lambda s, q: [0.1] * 384})())
    monkeypatch.setattr("src.retriever.vector_store", FakeVectorStore())

    results = retrieve("Prevention measures", similarity_threshold=0.0)
    assert len(results) == 1
    
    chunk = results[0]
    assert chunk.source_id == "idf-atlas-11-2025"
    assert chunk.source_url == "https://diabetesatlas.org/atlas/eleventh-edition/"
    assert chunk.publisher == "International Diabetes Federation"
    assert chunk.publication_date == "2025"
    assert chunk.source_checksum == "81d01e73d486adbe7d4f14752644e109ed43151738e3e852303ebd3967a81d0d"
    assert chunk.page_number == 12
    assert chunk.section_title == "Prevention"
    assert chunk.score == 0.9


def test_retrieve_excludes_reference_list_chunks(monkeypatch) -> None:
    raw_reference = {
        "id": "references-1",
        "document": "References 1. A cardiovascular outcomes cohort study.",
        "distance": 0.05,
        "score": 0.95,
        "metadata": {
            "chunk_id": "references-1",
            "document_name": "atlas.pdf",
            "page_number": 103,
            "section_title": "References",
            "subsection_title": "",
            "category": "general",
            "content_type": "text",
            "language": "en",
        },
    }

    class FakeVectorStore:
        def query(self, query_embedding, category, top_k):
            return [raw_reference]

    monkeypatch.setattr(
        "src.retriever.embedder",
        type("FakeEmbedder", (), {"embed_query": lambda self, query: [0.1] * 384})(),
    )
    monkeypatch.setattr("src.retriever.vector_store", FakeVectorStore())

    assert retrieve("preventive cardiologist", similarity_threshold=0.0) == []


def test_retrieve_uses_lexical_database_fallback_when_embedding_fails(monkeypatch) -> None:
    raw_result = {
        "id": "lexical-1",
        "document": "An estimated 589 million adults were living with diabetes in 2024.",
        "distance": 0.2,
        "score": 0.8,
        "metadata": {
            "chunk_id": "lexical-1",
            "document_name": "atlas.pdf",
            "page_number": 46,
            "section_title": "Key messages",
            "subsection_title": "",
            "category": "general",
            "content_type": "text",
            "language": "en",
            "source_id": "atlas",
            "source_url": "https://example.test/atlas",
        },
    }

    class FailedEmbedder:
        def embed_query(self, query):
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

    class LexicalVectorStore:
        def query(self, **kwargs):
            raise AssertionError("vector query must not run")

        def keyword_query(self, **kwargs):
            assert "589" in kwargs["query"]
            return [raw_result]

    monkeypatch.setattr("src.retriever.embedder", FailedEmbedder())
    monkeypatch.setattr("src.retriever.vector_store", LexicalVectorStore())

    chunks = retrieve("589 million adults with diabetes in 2024", similarity_threshold=0.0)

    assert [chunk.chunk_id for chunk in chunks] == ["lexical-1"]
    assert chunks[0].retrieval_mode == "lexical"
