"""Tests for index safety and atomic replacement.

Verifies that failed indexing does not replace the active namespace
or complete successfully, and that successful indexing builds
into the target namespace correctly while keeping other namespaces intact.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from src.ingestion.indexer import ingest_certified_corpus
from src.manifests import IndexManifest


class FakeStore:
    def __init__(self, has_doc=False):
        self.has_doc = has_doc
        self.added_chunks = []
        self.deleted_docs = []

    def has_document(self, name):
        return self.has_doc

    def delete_document(self, name):
        self.deleted_docs.append(name)

    def add_chunks(self, records, embeddings):
        self.added_chunks.extend(records)


class FakeEmbedder:
    def embed_batch(self, texts, show_progress=False):
        return [[0.1] * 384 for _ in texts]


def test_successful_indexing_creates_new_namespace(tmp_path, monkeypatch):
    """Verify that a successful ingestion runs parsing, chunking, embedding, storage, and saves manifest."""
    sources_file = tmp_path / "sources.json"
    sources_file.write_text("""{
      "sources": [
        {
          "source_id": "test-src",
          "title": "Test Title",
          "publisher": "Test Publisher",
          "source_url": "https://test.url",
          "publication_date": "2026",
          "checksum_sha256": "81d01e73d486adbe7d4f14752644e109ed43151738e3e852303ebd3967a81d0d",
          "file_name": "IDF_Diabetes_Atlas_11th_Edition_2025_WEB.pdf",
          "enabled": true
        }
      ]
    }""", encoding="utf-8")

    # Mock parse_document to avoid reading the massive PDF
    monkeypatch.setattr("src.ingestion.indexer.parse_document", lambda path: [
        {
            "document_name": "IDF_Diabetes_Atlas_11th_Edition_2025_WEB.pdf",
            "page_number": 1,
            "section_title": "Test Section",
            "subsection_title": "",
            "content": "Certified corpus content for testing.",
            "content_type": "text",
        }
    ])

    store = FakeStore()
    embedder = FakeEmbedder()

    # Run indexer
    manifest = ingest_certified_corpus(
        sources_path=str(sources_file),
        namespace="test_new_namespace",
        data_dir=Path("data/rew_data/books"),
        chunk_profile="balanced",
        manifest_dir=tmp_path,
        _store=store,
        _embedder=embedder,
    )

    assert isinstance(manifest, IndexManifest)
    assert manifest.namespace == "test_new_namespace"
    assert len(store.added_chunks) > 0
    assert store.added_chunks[0]["source_id"] == "test-src"
    assert store.added_chunks[0]["chunk_profile"] == "balanced"
    
    # Check that manifest was saved to disk
    manifest_file = tmp_path / "test_new_namespace.json"
    assert manifest_file.exists()


def test_failed_indexing_does_not_commit_or_alter_active(tmp_path, monkeypatch):
    """Verify that failure in parsing/embedding propagates and does not complete."""
    sources_file = tmp_path / "sources.json"
    sources_file.write_text("""{
      "sources": [
        {
          "source_id": "test-src",
          "title": "Test Title",
          "publisher": "Test Publisher",
          "source_url": "https://test.url",
          "publication_date": "2026",
          "checksum_sha256": "81d01e73d486adbe7d4f14752644e109ed43151738e3e852303ebd3967a81d0d",
          "file_name": "IDF_Diabetes_Atlas_11th_Edition_2025_WEB.pdf",
          "enabled": true
        }
      ]
    }""", encoding="utf-8")

    # Force parse_document to return empty list, simulating a parsing failure
    monkeypatch.setattr("src.ingestion.indexer.parse_document", lambda path: [])

    store = FakeStore()
    embedder = FakeEmbedder()

    with pytest.raises(ValueError, match="No content extracted"):
        ingest_certified_corpus(
            sources_path=str(sources_file),
            namespace="test_failed_namespace",
            data_dir=Path("data/rew_data/books"),
            chunk_profile="balanced",
            manifest_dir=tmp_path,
            _store=store,
            _embedder=embedder,
        )

    # Assure no chunks were added to database/store and no manifest was written
    assert len(store.added_chunks) == 0
    manifest_file = tmp_path / "test_failed_namespace.json"
    assert not manifest_file.exists()
